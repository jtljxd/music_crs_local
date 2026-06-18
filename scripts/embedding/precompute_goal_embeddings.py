"""
precompute_goal_embeddings.py
==============================
Pre-compute Qwen3-Embedding-0.6B embeddings for the ``listener_goal`` field
inside ``conversation_goal`` for every session.

Key format (k-v store):
    key   : session_id  (str)
    value : 1024-dim fp16 CPU tensor

Output files (one per split / dataset):
    qwen/goal_embeddings_train_0.6b.pt
    qwen/goal_embeddings_test_0.6b.pt
    qwen/goal_embeddings_blindA_0.6b.pt

Data sources:
    train / test  → talkpl-ai/TalkPlayData-Challenge-Dataset
    blindA        → talkpl-ai/TalkPlayData-Challenge-Blind-A   (split="test")

``conversation_goal`` is a dict-like field; ``listener_goal`` is a free-text
string describing what kind of music the listener wants in this session.
If the field is missing or empty, a zero vector is stored.

Usage:
    # train
    python scripts/embedding/precompute_goal_embeddings.py \\
        --dataset talkpl-ai/TalkPlayData-Challenge-Dataset \\
        --split   train \\
        --out     qwen/goal_embeddings_train_0.6b.pt

    # test
    python scripts/embedding/precompute_goal_embeddings.py \\
        --dataset talkpl-ai/TalkPlayData-Challenge-Dataset \\
        --split   test \\
        --out     qwen/goal_embeddings_test_0.6b.pt

    # blind-A
    nohup python scripts/embedding/precompute_goal_embeddings.py \\
        --dataset talkpl-ai/TalkPlayData-Challenge-Blind-A \\
        --split   test \\
        --out     qwen/goal_embeddings_blindA_0.6b.pt \\
        --batch   64 \\
        > logs/goal_emb_blindA.log 2>&1 &
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

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
# Task instruction tailored for listener-goal retrieval
QWEN_TASK  = "Represent the music listener goal for retrieving relevant music tracks"


# ── model helpers ──────────────────────────────────────────────────────────────

def _instruct(text: str) -> str:
    return f"Instruct: {QWEN_TASK}\nQuery: {text}"


def load_qwen(model_path: str, device: str):
    logger.info("Loading Qwen3-Embedding from %s (device=%s) …", model_path, device)
    dtype     = torch.float16 if "cuda" in device else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=dtype,
    ).to(device).eval()
    logger.info("Model ready (dtype=%s).", dtype)
    return tokenizer, model


def _last_token_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    seq_len = attn_mask.sum(dim=1) - 1
    return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), seq_len]


def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode a list of texts; returns fp16 CPU tensor [N, TARGET_DIM]."""
    if not texts:
        return torch.zeros(0, TARGET_DIM, dtype=torch.float16)

    all_embs: List[torch.Tensor] = []
    i = 0
    while i < len(texts):
        batch = [_instruct(t) for t in texts[i : i + batch_size]]
        try:
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                out  = model(**inputs)
                embs = _last_token_pool(out.last_hidden_state, inputs["attention_mask"])
                embs = F.normalize(embs, p=2, dim=1)

            # Align to TARGET_DIM
            d = embs.shape[1]
            if d > TARGET_DIM:
                embs = embs[:, :TARGET_DIM]
            elif d < TARGET_DIM:
                embs = F.pad(embs, (0, TARGET_DIM - d))

            all_embs.append(embs.half().cpu())
            i += batch_size

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            logger.warning("OOM — halving batch_size to %d, retrying.", batch_size)

    return torch.cat(all_embs, dim=0)


# ── goal extraction ────────────────────────────────────────────────────────────

def extract_listener_goal(session: dict) -> Optional[str]:
    """Extract ``listener_goal`` from the session's ``conversation_goal`` field.

    The field may appear as:
        - dict: conversation_goal["listener_goal"]
        - string (JSON-encoded dict): parsed then accessed
        - None / missing

    Returns the goal text, or None if unavailable / empty.
    """
    raw = session.get("conversation_goal")
    if raw is None:
        return None

    goal_dict: Optional[dict] = None
    if isinstance(raw, dict):
        goal_dict = raw
    elif isinstance(raw, str) and raw.strip():
        import json
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                goal_dict = parsed
        except json.JSONDecodeError:
            # Treat the whole string as the goal
            return raw.strip() or None

    if goal_dict is None:
        return None

    # Try several common key names
    for key in ("listener_goal", "listenerGoal", "goal", "user_goal"):
        val = goal_dict.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()

    return None


# ── dataset processing ─────────────────────────────────────────────────────────

def collect_pending(
    ds,
    store: Dict[str, torch.Tensor],
) -> Tuple[List[str], List[str]]:
    """Return (keys, texts) for sessions not yet in the store."""
    keys:  List[str] = []
    texts: List[str] = []

    for idx in tqdm(range(len(ds)), desc="Scanning", unit="session"):
        item       = ds[idx]
        session_id = str(item["session_id"])
        if session_id in store:
            continue
        goal = extract_listener_goal(item)
        keys.append(session_id)
        texts.append(goal if goal else "")   # empty → zero vector later

    return keys, texts


def encode_and_store(
    keys:       List[str],
    texts:      List[str],
    store:      Dict[str, torch.Tensor],
    tokenizer,
    model,
    device:     str,
    batch_size: int,
    out_path:   str,
    save_every: int = 200,
) -> int:
    """Encode texts in batches, write zero vectors for empty texts, and save."""
    added = 0

    # Separate non-empty from empty
    non_empty_idx   = [i for i, t in enumerate(texts) if t.strip()]
    empty_idx       = [i for i, t in enumerate(texts) if not t.strip()]

    # Store zero vectors for sessions with no listener_goal
    for i in empty_idx:
        store[keys[i]] = torch.zeros(TARGET_DIM, dtype=torch.float16)
        added += 1

    if not non_empty_idx:
        torch.save(store, out_path)
        return added

    ne_keys  = [keys[i]  for i in non_empty_idx]
    ne_texts = [texts[i] for i in non_empty_idx]

    pbar = tqdm(range(0, len(ne_texts), batch_size), desc="Encoding", unit="batch")
    for start in pbar:
        batch_keys  = ne_keys [start : start + batch_size]
        batch_texts = ne_texts[start : start + batch_size]

        embs = encode_texts(batch_texts, tokenizer, model, device, batch_size)
        for j, k in enumerate(batch_keys):
            store[k] = embs[j]
            added += 1

        pbar.set_postfix(total=len(store))

        # Periodic checkpoint
        batches_done = start // batch_size + 1
        if batches_done % save_every == 0:
            torch.save(store, out_path)

    torch.save(store, out_path)
    return added


# ── main ───────────────────────────────────────────────────────────────────────

def main(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    tokenizer, model = load_qwen(args.qwen, device)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    # Resume
    store: Dict[str, torch.Tensor] = {}
    if os.path.exists(out_path):
        logger.info("Resuming from %s …", out_path)
        store = torch.load(out_path, map_location="cpu", weights_only=True)
        logger.info("  %d entries already cached.", len(store))

    logger.info("Loading dataset '%s' split='%s' …", args.dataset, args.split)
    ds = load_dataset(args.dataset, split=args.split)
    logger.info("Total sessions: %d", len(ds))

    keys, texts = collect_pending(ds, store)
    logger.info("%d sessions to encode.", len(keys))

    if keys:
        added = encode_and_store(
            keys, texts, store, tokenizer, model, device,
            args.batch, out_path, save_every=args.save_every,
        )
        logger.info("Added %d entries.", added)

    logger.info("Store total: %d  →  %s", len(store), out_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pre-compute Qwen3-Embedding-0.6B listener_goal embeddings (k=session_id)"
    )
    p.add_argument(
        "--dataset", type=str,
        default="talkpl-ai/TalkPlayData-Challenge-Dataset",
        help="HuggingFace dataset name",
    )
    p.add_argument(
        "--split", type=str, default="train",
        help="Dataset split: train / test",
    )
    p.add_argument(
        "--qwen", type=str,
        default="/home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B",
        help="Local path to Qwen3-Embedding-0.6B",
    )
    p.add_argument(
        "--out", type=str,
        default="qwen/goal_embeddings_train_0.6b.pt",
        help="Output .pt file path",
    )
    p.add_argument("--batch",      type=int, default=64,
                   help="Encoding batch size")
    p.add_argument("--save_every", type=int, default=50,
                   help="Save checkpoint every N batches")
    p.add_argument("--device",     type=str, default="auto")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
