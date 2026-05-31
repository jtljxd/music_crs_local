"""
Pre-compute per-turn Qwen embeddings for every conversation in the dataset.

Storage schema (saved as a single .pt / JSON file)
----------------------------------------------------
Key format:   "{session_id}__{turn}_{role}"
              role: "user" | "system" | "history_avg"

  {session_id}__1_user          → emb of the 1st user query      [128]
  {session_id}__1_system        → emb of the 1st system reply     [128]
  {session_id}__2_user          → emb of the 2nd user query       [128]
  {session_id}__2_system        → emb of the 2nd system reply     [128]
  ...
  {session_id}__n_history_avg   → avg-pool of turns 1..(n-1) user+system embs  [128]

All values are float16 tensors of shape [128] (first 128 dims of Qwen output).

Usage
-----
python precompute_turn_embeddings.py \
    --dataset talkpl-ai/TalkPlayData-Challenge-Dataset \
    --split   test \
    --qwen    /home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B \
    --out     cache/turn_embeddings.pt \
    --batch   128

The output file can then be loaded with:
    store = torch.load("cache/turn_embeddings.pt", map_location="cpu")
    emb   = store["{session_id}__3_history_avg"]   # shape [128]
"""

import os
import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

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

TARGET_DIM = 128        # store only first 128 dims
STORE_DIM  = 1024       # full Qwen hidden size (pad/truncate to this before slicing)


# ── model loader ──────────────────────────────────────────────────────────────

def load_qwen(model_path: str):
    logger.info("Loading Qwen tokenizer + model from %s …", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModel.from_pretrained(model_path).cpu().eval()
    logger.info("Qwen loaded.")
    return tokenizer, model


# ── batch encode ──────────────────────────────────────────────────────────────

def encode_batch(
    texts: List[str],
    tokenizer,
    model,
    batch_size: int = 128,
) -> torch.Tensor:
    """Encode a list of texts; returns float16 [N, TARGET_DIM] on CPU."""
    all_embs = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start: start + batch_size]
        with torch.no_grad():
            inputs  = tokenizer(
                chunk, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            outputs = model(**inputs)
            attn    = inputs["attention_mask"]
            tok_emb = outputs.last_hidden_state          # [B, T, H]
            mask    = attn.unsqueeze(-1).expand(tok_emb.size()).float()
            emb     = (
                torch.sum(tok_emb * mask, dim=1)
                / torch.clamp(mask.sum(dim=1), min=1e-9)
            )                                            # [B, H]
            emb = emb[:, :TARGET_DIM]                    # [B, 128]
        all_embs.append(emb.half())                      # store fp16
    return torch.cat(all_embs, dim=0)                    # [N, 128]


# ── dataset helpers ───────────────────────────────────────────────────────────

def extract_turns(session: dict) -> List[Tuple[int, str, str]]:
    """Return list of (turn_number, role, content) for a session.

    role is normalised to 'user' or 'system'.
    'music' turns are kept with role='system' (they represent assistant replies).
    """
    turns = []
    for conv in session["conversations"]:
        role    = conv["role"]
        content = str(conv.get("content", "") or "")
        if not content.strip():
            continue
        if role in ("assistant", "music"):
            role = "system"
        turns.append((conv["turn_number"], role, content))
    # Sort by turn_number
    turns.sort(key=lambda x: x[0])
    return turns


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    # Load existing store so we can resume / extend
    if os.path.exists(out_path):
        logger.info("Loading existing store from %s …", out_path)
        store: Dict[str, torch.Tensor] = torch.load(out_path, map_location="cpu")
        logger.info("  %d entries already cached.", len(store))
    else:
        store = {}

    # Load dataset
    logger.info("Loading dataset %s (split=%s) …", args.dataset, args.split)
    ds = load_dataset(args.dataset, split=args.split)
    logger.info("  %d sessions.", len(ds))

    # ── Pass 1: collect all individual turn texts that need encoding ──────────
    # We will encode each unique text exactly once, then build derived keys.

    # Maps text → list of (key, is_individual) so we can fill the store after encoding.
    text_to_keys: Dict[str, List[str]] = defaultdict(list)
    # session_id → ordered list of (turn_num, role, text) for history-avg computation
    session_turns: Dict[str, List[Tuple[int, str, str]]] = {}

    for session in tqdm(ds, desc="Collecting texts"):
        session_id = session["session_id"]
        turns      = extract_turns(session)
        session_turns[session_id] = turns

        for turn_num, role, text in turns:
            key = f"{session_id}__{turn_num}_{role}"
            if key not in store:
                text_to_keys[text].append(key)

    unique_texts = list(text_to_keys.keys())
    logger.info("%d unique individual turn texts to encode.", len(unique_texts))

    # ── Encode individual turns (skip already in store) ───────────────────────
    if unique_texts:
        tokenizer, model = load_qwen(args.qwen)
        logger.info("Encoding %d texts (batch_size=%d) …", len(unique_texts), args.batch)

        for start in tqdm(range(0, len(unique_texts), args.batch), desc="Encoding turns"):
            chunk = unique_texts[start: start + args.batch]
            with torch.no_grad():
                inputs  = tokenizer(
                    chunk, return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                )
                outputs = model(**inputs)
                attn    = inputs["attention_mask"]
                tok_emb = outputs.last_hidden_state
                mask    = attn.unsqueeze(-1).expand(tok_emb.size()).float()
                emb     = (
                    torch.sum(tok_emb * mask, dim=1)
                    / torch.clamp(mask.sum(dim=1), min=1e-9)
                )[:, :TARGET_DIM].half()   # [B, 128] fp16

            for i, text in enumerate(chunk):
                vec = emb[i]
                for key in text_to_keys[text]:
                    store[key] = vec

            # Periodic save every 10k entries added
            if (start // args.batch) % 100 == 0 and start > 0:
                torch.save(store, out_path)
                logger.info("  Checkpoint saved (%d entries).", len(store))

        logger.info("Individual turn encoding complete. %d entries in store.", len(store))
        torch.save(store, out_path)

    # ── Pass 2: build history_avg keys ────────────────────────────────────────
    # For the n-th turn query, history_avg = avg-pool of all user+system embs
    # from turns 1 … (n-1).

    logger.info("Building history_avg embeddings …")
    avg_todo = []   # (key, list_of_source_keys)

    for session_id, turns in session_turns.items():
        # Collect (turn_num, role) pairs in order
        ordered = [(t, r) for t, r, _ in turns]

        for idx, (turn_num, role) in enumerate(ordered):
            if role != "user":
                continue   # only generate history_avg for user turns
            avg_key = f"{session_id}__{turn_num}_history_avg"
            if avg_key in store:
                continue   # already computed

            # Source: all individual turn keys BEFORE this turn (turns 1..n-1)
            source_keys = [
                f"{session_id}__{t}_{r}"
                for t, r in ordered[:idx]
                if f"{session_id}__{t}_{r}" in store
            ]

            if not source_keys:
                # Turn 1 has no history → zero vector
                store[avg_key] = torch.zeros(TARGET_DIM, dtype=torch.float16)
            else:
                avg_todo.append((avg_key, source_keys))

    logger.info("  %d history_avg keys to compute.", len(avg_todo))
    for avg_key, source_keys in tqdm(avg_todo, desc="Building history_avg"):
        vecs = torch.stack([store[k].float() for k in source_keys])  # [M, 128]
        avg  = vecs.mean(dim=0).half()                                # [128] fp16
        store[avg_key] = avg

    torch.save(store, out_path)
    logger.info("Done. Final store: %d entries → %s", len(store), out_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    example_keys = list(store.keys())[:6]
    logger.info("Example keys: %s", example_keys)
    logger.info("Value shape:  %s  dtype: %s", store[example_keys[0]].shape, store[example_keys[0]].dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute per-turn Qwen embeddings for all conversations."
    )
    parser.add_argument(
        "--dataset", type=str,
        default="talkpl-ai/TalkPlayData-Challenge-Dataset",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Dataset split to process (e.g. 'test', 'train', 'all')",
    )
    parser.add_argument(
        "--qwen", type=str,
        default="/home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B",
        help="Path to Qwen3-Embedding-0.6B model",
    )
    parser.add_argument(
        "--out", type=str,
        default="cache/turn_embeddings.pt",
        help="Output .pt file path",
    )
    parser.add_argument(
        "--batch", type=int, default=128,
        help="Qwen encoding batch size",
    )
    args = parser.parse_args()
    main(args)
