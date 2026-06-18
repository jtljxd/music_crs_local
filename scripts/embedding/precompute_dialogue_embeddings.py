"""
precompute_dialogue_embeddings.py
===================================
Pre-compute per-turn Qwen3-Embedding dialogue embeddings in the NEW key format:
    {session_id}_{turn}_query   → current user query emb        [1024] fp16
    {session_id}_{turn}_history → concatenated previous turns    [1024] fp16

Compatible with:
    qwen/dialogue_embeddings_train_0.6b.pt
    qwen/dialogue_embeddings_test_0.6b.pt
    qwen/dialogue_embeddings_blindA_0.6b.pt

Usage:
    # Generate blindA embeddings
    nohup python precompute_dialogue_embeddings.py \\
        --dataset talkpl-ai/TalkPlayData-Challenge-Blind-A \\
        --split   test \\
        --out     qwen/dialogue_embeddings_blindA_0.6b.pt \\
        --qwen    /home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B \\
        --batch   64 \\
    > precompute_blindA.log 2>&1 &

    # Generate train embeddings
    python precompute_dialogue_embeddings.py \\
        --dataset talkpl-ai/TalkPlayData-Challenge-Dataset \\
        --split   train \\
        --out     qwen/dialogue_embeddings_train_0.6b.pt

    # Generate test embeddings
    python precompute_dialogue_embeddings.py \\
        --dataset talkpl-ai/TalkPlayData-Challenge-Dataset \\
        --split   test \\
        --out     qwen/dialogue_embeddings_test_0.6b.pt
"""

import argparse
import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET_DIM = 1024
QWEN_TASK  = "Given a music conversation query, retrieve relevant music tracks"


def _instruct(text: str) -> str:
    return f"Instruct: {QWEN_TASK}\nQuery: {text}"


# ─── model ────────────────────────────────────────────────────────────────────

def load_qwen(model_path: str, device: str = "cuda"):
    logger.info("Loading Qwen from %s (device=%s) …", model_path, device)
    dtype     = torch.float16 if "cuda" in device else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=dtype,
    ).to(device).eval()
    logger.info("Qwen ready (dtype=%s).", dtype)
    return tokenizer, model


def _last_token_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    seq_len = attn_mask.sum(dim=1) - 1
    batch   = last_hidden.shape[0]
    return last_hidden[torch.arange(batch, device=last_hidden.device), seq_len]


def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 64,
    add_instruct: bool = True,
) -> torch.Tensor:
    """Returns float16 CPU tensor [N, TARGET_DIM]."""
    if not texts:
        return torch.zeros(0, TARGET_DIM, dtype=torch.float16)

    all_embs = []
    i = 0
    while i < len(texts):
        batch_texts = texts[i: i + batch_size]
        if add_instruct:
            batch_texts = [_instruct(t) for t in batch_texts]
        try:
            inputs = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                out  = model(**inputs)
                embs = _last_token_pool(out.last_hidden_state, inputs["attention_mask"])
                embs = F.normalize(embs, p=2, dim=1)

            # Truncate / pad to TARGET_DIM
            if embs.shape[1] > TARGET_DIM:
                embs = embs[:, :TARGET_DIM]
            elif embs.shape[1] < TARGET_DIM:
                embs = F.pad(embs, (0, TARGET_DIM - embs.shape[1]))

            all_embs.append(embs.half().cpu())
            i += batch_size

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            logger.warning("OOM — halving batch_size to %d and retrying.", batch_size)

    return torch.cat(all_embs, dim=0)  # [N, TARGET_DIM]


# ─── session processing ───────────────────────────────────────────────────────

def extract_turns(session: dict) -> List[Tuple[int, str, str]]:
    """Return sorted list of (turn_number, role, content).
    role normalised: 'user' stays 'user'; 'assistant'/'music' → 'system'.
    """
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
    """Encode all turns and build query/history keys; update store in-place.

    New key format:
        {session_id}_{turn}_query    → user query at this turn
        {session_id}_{turn}_history  → concat text of all previous turns (up to max_length)
    """
    # Collect all NEW (key → text) pairs
    key_text_query:   Dict[str, str] = {}
    key_text_history: Dict[str, str] = {}

    for session in sessions:
        sid   = session["session_id"]
        turns = extract_turns(session)

        # Build per-turn context
        prev_texts: List[str] = []
        for turn_num, role, content in turns:
            if role != "user":
                prev_texts.append(content)
                continue

            q_key = f"{sid}_{turn_num}_query"
            h_key = f"{sid}_{turn_num}_history"

            if q_key not in store:
                key_text_query[q_key] = content

            if h_key not in store:
                history_str = " | ".join(prev_texts[-10:]) if prev_texts else ""
                key_text_history[h_key] = history_str

            prev_texts.append(content)

    new_count = 0

    # Encode query texts
    if key_text_query:
        keys  = list(key_text_query.keys())
        texts = [key_text_query[k] for k in keys]
        embs  = encode_texts(texts, tokenizer, model, device, batch_size, add_instruct=True)
        for j, k in enumerate(keys):
            store[k] = embs[j]
        new_count += len(keys)

    # Encode history texts (no instruct prefix — it's context, not a query)
    if key_text_history:
        keys  = list(key_text_history.keys())
        texts = [key_text_history[k] for k in keys]
        # Empty history → zero vector (skip encoding)
        non_empty = [(i, k, t) for i, (k, t) in enumerate(zip(keys, texts)) if t.strip()]
        zero_keys = [k for k, t in zip(keys, texts) if not t.strip()]

        for k in zero_keys:
            store[k] = torch.zeros(TARGET_DIM, dtype=torch.float16)

        if non_empty:
            idxs, ne_keys, ne_texts = zip(*non_empty)
            embs = encode_texts(list(ne_texts), tokenizer, model, device, batch_size, add_instruct=False)
            for j, k in enumerate(ne_keys):
                store[k] = embs[j]
        new_count += len(keys)

    return new_count


# ─── main ─────────────────────────────────────────────────────────────────────

def main(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    tokenizer, model = load_qwen(args.qwen, device)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    # Resume from existing store
    store: Dict[str, torch.Tensor] = {}
    if os.path.exists(out_path):
        logger.info("Resuming from %s …", out_path)
        store = torch.load(out_path, map_location="cpu", weights_only=True)
        logger.info("  %d entries already cached.", len(store))

    splits = ["train", "test"] if args.split == "all" else [args.split]

    for split in splits:
        logger.info("=" * 60)
        logger.info("Processing split: %s  dataset: %s", split, args.dataset)
        ds = load_dataset(args.dataset, split=split)
        total = len(ds)
        chunk = args.sessions_per_chunk
        added_total = 0

        pbar = tqdm(range(0, total, chunk), desc=split, unit="chunk")
        for start in pbar:
            sessions = [ds[j] for j in range(start, min(start + chunk, total))]
            added    = process_sessions(sessions, store, tokenizer, model, device, args.batch)
            added_total += added
            pbar.set_postfix(new=added, total=len(store))

            # Save after every chunk
            torch.save(store, out_path)

        torch.save(store, out_path)
        logger.info("Split '%s' done. Added %d entries. Store total: %d",
                    split, added_total, len(store))

    logger.info("=" * 60)
    logger.info("All done. %d entries → %s", len(store), out_path)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pre-compute Qwen3 dialogue embeddings (new 1024-dim key format)"
    )
    p.add_argument(
        "--dataset", type=str,
        default="talkpl-ai/TalkPlayData-Challenge-Dataset",
        help="HuggingFace dataset name",
    )
    p.add_argument(
        "--split", type=str, default="test",
        help="Dataset split: train / test / all",
    )
    p.add_argument(
        "--qwen", type=str,
        default="/home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B",
        help="Local path to Qwen3-Embedding-0.6B",
    )
    p.add_argument(
        "--out", type=str,
        default="qwen/dialogue_embeddings_blindA_0.6b.pt",
        help="Output .pt file path",
    )
    p.add_argument("--batch",              type=int, default=64)
    p.add_argument("--sessions_per_chunk", type=int, default=8)
    p.add_argument("--device",             type=str, default="auto")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
