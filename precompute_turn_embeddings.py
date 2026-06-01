"""
Pre-compute per-turn Qwen embeddings for every conversation in the dataset.

Storage schema (saved as a single .pt file)
--------------------------------------------
Key format:  "{session_id}__{turn}_{role}"
             role: "user" | "system" | "history_avg"

  {session_id}__1_user         → emb of 1st user query        [128] fp16
  {session_id}__1_system       → emb of 1st system reply       [128] fp16
  {session_id}__2_user         → emb of 2nd user query         [128] fp16
  {session_id}__2_system       → emb of 2nd system reply       [128] fp16
  ...
  {session_id}__n_history_avg  → avg-pool of turns 1..(n-1)   [128] fp16

Processing modes
----------------
Default (--sessions_per_chunk N):
  Load N sessions at a time, encode immediately, save checkpoint.
  Low peak RAM, works well for large train splits.

Usage
-----
# train split, stream 8 sessions at a time (low memory)
python precompute_turn_embeddings.py \
    --split   train \
    --out     cache/turn_embeddings_train.pt \
    --batch   256 \
    --sessions_per_chunk 8

# test split (small, can do all at once)
python precompute_turn_embeddings.py \
    --split   test \
    --out     cache/turn_embeddings_test.pt \
    --batch   256
"""

import os
import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET_DIM = 128

# Qwen3-Embedding instruction prefix for query encoding
QWEN_TASK = "Given a music conversation query, retrieve relevant music tracks"


def _get_instruct(text: str) -> str:
    return f"Instruct: {QWEN_TASK}\nQuery: {text}"


# ── model ─────────────────────────────────────────────────────────────────────

def load_qwen(model_path: str, device: str = "cuda"):
    logger.info("Loading Qwen from %s (device=%s) …", model_path, device)
    dtype     = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=dtype,
    ).to(device).eval()
    logger.info("Qwen ready on %s (dtype=%s).", device, dtype)
    return tokenizer, model


def _last_token_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """Qwen3-Embedding uses the last non-padding token as the sentence embedding."""
    seq_len = attn_mask.sum(dim=1) - 1              # [B]
    batch   = last_hidden.shape[0]
    return last_hidden[torch.arange(batch, device=last_hidden.device), seq_len]


def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 256,
    add_instruct: bool = True,
) -> torch.Tensor:
    """Encode texts with Qwen3-Embedding; returns float16 CPU tensor [N, TARGET_DIM].

    Automatically halves batch_size on OOM and retries (down to batch_size=1).
    """
    if add_instruct:
        texts = [_get_instruct(t) for t in texts]

    all_embs = []
    start = 0
    current_batch = batch_size

    while start < len(texts):
        chunk = texts[start: start + current_batch]
        try:
            inputs = tokenizer(chunk, return_tensors="pt", padding=True,
                               truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model(**inputs)
                emb     = _last_token_pool(outputs.last_hidden_state,
                                           inputs["attention_mask"])
                emb     = F.normalize(emb, p=2, dim=1)
                all_embs.append(emb[:, :TARGET_DIM].cpu().half())
            start += current_batch
            # Recover batch size after success (up to original)
            current_batch = min(current_batch * 2, batch_size)
        except torch.cuda.OutOfMemoryError:
            if device == "cuda":
                torch.cuda.empty_cache()
            if current_batch <= 1:
                # Last resort: fall back to CPU for this single sample
                logger.warning("OOM even at batch_size=1; falling back to CPU for this chunk.")
                inputs = tokenizer(chunk, return_tensors="pt", padding=True,
                                   truncation=True, max_length=512)
                cpu_model = model.cpu()
                with torch.inference_mode():
                    outputs = cpu_model(**inputs)
                    emb     = _last_token_pool(outputs.last_hidden_state,
                                               inputs["attention_mask"])
                    emb     = F.normalize(emb, p=2, dim=1)
                    all_embs.append(emb[:, :TARGET_DIM].half())
                model.to(device)
                start += current_batch
            else:
                current_batch = max(1, current_batch // 2)
                logger.warning("OOM — reducing batch_size to %d and retrying.", current_batch)

    return torch.cat(all_embs, dim=0)  # [N, TARGET_DIM] fp16


# ── dataset helpers ───────────────────────────────────────────────────────────

def extract_turns(session: dict) -> List[Tuple[int, str, str]]:
    """(turn_number, role, content); role normalised to 'user'/'system'."""
    turns = []
    for conv in session["conversations"]:
        role    = conv["role"]
        content = str(conv.get("content", "") or "").strip()
        if not content:
            continue
        if role in ("assistant", "music"):
            role = "system"
        turns.append((int(conv["turn_number"]), role, content))
    turns.sort(key=lambda x: x[0])
    return turns


def process_sessions(
    sessions: List[dict],
    store: Dict[str, torch.Tensor],
    tokenizer,
    model,
    device: str,
    batch_size: int,
) -> int:
    """Encode all turns in a list of sessions, update store in-place.

    Returns number of NEW entries added.
    """
    # Collect unique texts not yet in store
    text_to_keys: Dict[str, List[str]] = defaultdict(list)
    session_turns_map: Dict[str, List[Tuple[int, str, str]]] = {}

    for session in sessions:
        sid   = session["session_id"]
        turns = extract_turns(session)
        session_turns_map[sid] = turns
        for turn_num, role, text in turns:
            key = f"{sid}__{turn_num}_{role}"
            if key not in store:
                text_to_keys[text].append(key)

    unique_texts = list(text_to_keys.keys())

    # Encode new texts
    if unique_texts:
        embs = encode_texts(unique_texts, tokenizer, model, device, batch_size)
        for i, text in enumerate(unique_texts):
            vec = embs[i]
            for key in text_to_keys[text]:
                store[key] = vec

    # Build history_avg for each session
    for sid, turns in session_turns_map.items():
        ordered = [(t, r) for t, r, _ in turns]
        for idx, (turn_num, role) in enumerate(ordered):
            if role != "user":
                continue
            avg_key = f"{sid}__{turn_num}_history_avg"
            if avg_key in store:
                continue
            source_keys = [
                f"{sid}__{t}_{r}"
                for t, r in ordered[:idx]
                if f"{sid}__{t}_{r}" in store
            ]
            if not source_keys:
                store[avg_key] = torch.zeros(TARGET_DIM, dtype=torch.float16)
            else:
                vecs = torch.stack([store[k].float() for k in source_keys])
                store[avg_key] = vecs.mean(dim=0).half()

    return len(unique_texts)


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    # Resume from existing store
    if os.path.exists(out_path):
        logger.info("Resuming from %s …", out_path)
        store: Dict[str, torch.Tensor] = torch.load(out_path, map_location="cpu")
        logger.info("  %d entries already cached.", len(store))
    else:
        store = {}

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    tokenizer, model = load_qwen(args.qwen, device)

    # Handle "all" split → process train + test (no validation split in this dataset)
    splits = ["train", "test"] if args.split == "all" else [args.split]

    for split in splits:
        logger.info("=" * 60)
        logger.info("Processing split: %s", split)
        ds = load_dataset(args.dataset, split=split)
        total_sessions = len(ds)
        logger.info("  %d sessions total.", total_sessions)

        chunk_size = args.sessions_per_chunk

        added_total = 0
        pbar = tqdm(range(0, total_sessions, chunk_size),
                    desc=f"{split}", unit="chunk")

        for chunk_start in pbar:
            chunk = [ds[i] for i in range(chunk_start,
                                           min(chunk_start + chunk_size, total_sessions))]
            added = process_sessions(chunk, store, tokenizer, model, device, args.batch)
            added_total += added
            pbar.set_postfix(store_size=len(store), new=added)

            # Save after EVERY chunk so progress is never lost on OOM/crash
            torch.save(store, out_path)
            # Release GPU memory fragments after each chunk
            if device == "cuda":
                torch.cuda.empty_cache()

        # Save after each split
        torch.save(store, out_path)
        logger.info("Split %s done. Added %d new entries. Store total: %d",
                    split, added_total, len(store))

    logger.info("=" * 60)
    logger.info("All done. Final store: %d entries → %s", len(store), out_path)

    # Quick sanity check
    example_keys = list(store.keys())[:4]
    for k in example_keys:
        logger.info("  %s  →  shape=%s dtype=%s", k, store[k].shape, store[k].dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute per-turn Qwen embeddings for all conversations."
    )
    parser.add_argument(
        "--dataset", type=str,
        default="talkpl-ai/TalkPlayData-Challenge-Dataset",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Split to process: train / test / validation / all",
    )
    parser.add_argument(
        "--qwen", type=str,
        default="/home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B",
    )
    parser.add_argument(
        "--out", type=str,
        default="qwen/turn_embeddings.pt",
        help="Output .pt file (shared across splits if you use --split all)",
    )
    parser.add_argument(
        "--batch", type=int, default=256,
        help="Qwen encoding batch size per GPU forward pass",
    )
    parser.add_argument(
        "--sessions_per_chunk", type=int, default=8,
        help="Number of sessions to load and encode at once "
             "(lower = less peak RAM, same speed on GPU)",
    )
    parser.add_argument(
        "--save_every", type=int, default=1000,
        help="Save checkpoint every N sessions processed",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="cuda / cpu / auto (default: auto-detect)",
    )
    args = parser.parse_args()
    main(args)
