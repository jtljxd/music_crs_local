"""
precompute_bge_embeddings.py
=============================
Pre-compute BGE-Small-EN-v1.5 embeddings for two purposes:

  A) Query-level: genre & decade fields from the structured query_split store
     Key format : {session_id}_{turn_number}_{field}
                  field ∈ {"genre", "decade"}
     Output     : bge/query_genre_embeddings_{split}.pt
                  bge/query_decade_embeddings_{split}.pt

  B) Item-level: tag_list from the track metadata catalog
     Key format : track_id  (str)
     Output     : bge/track_tag_embeddings.pt

BGE-Small-EN-v1.5 produces 384-dim embeddings; stored as fp16 on CPU.

Prerequisites:
    - query_split stores must exist under qwen/ (produced by
      scripts/retrieval/precompute_query_split.py)
    - Track metadata: talkpl-ai/TalkPlayData-Challenge-Track-Metadata

Usage examples:
    # Query embeddings — train split
    python scripts/embedding/precompute_bge_embeddings.py \\
        --mode          query \\
        --query_split   qwen/query_split_train.pt \\
        --out_dir       bge \\
        --split_name    train

    # Query embeddings — test split
    python scripts/embedding/precompute_bge_embeddings.py \\
        --mode          query \\
        --query_split   qwen/query_split_test.pt \\
        --out_dir       bge \\
        --split_name    test

    # Query embeddings — blind-A split
    python scripts/embedding/precompute_bge_embeddings.py \\
        --mode          query \\
        --query_split   qwen/query_split_blindA.pt \\
        --out_dir       bge \\
        --split_name    blindA

    # Track tag embeddings (run once)
    python scripts/embedding/precompute_bge_embeddings.py \\
        --mode          track \\
        --track_dataset talkpl-ai/TalkPlayData-Challenge-Track-Metadata \\
        --out_dir       bge

Notes:
    - Empty / missing values are stored as zero vectors.
    - For genre and decade, a list value is joined with ", " before encoding.
    - For tag_list, all tags are joined with ", ".
    - The script supports resuming: existing keys are skipped.
"""

from __future__ import annotations

import argparse
import json
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

TARGET_DIM = 384          # BGE-Small-EN-v1.5 output dim
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ── model helpers ──────────────────────────────────────────────────────────────

def load_bge(model_path: str, device: str):
    logger.info("Loading BGE model from %s (device=%s) …", model_path, device)
    dtype     = torch.float16 if "cuda" in device else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=dtype,
    ).to(device).eval()
    logger.info("BGE model ready (dtype=%s).", dtype)
    return tokenizer, model


def _cls_pool(last_hidden: torch.Tensor) -> torch.Tensor:
    """CLS token pooling — standard for BGE models."""
    return last_hidden[:, 0, :]


def encode_texts(
    texts:      List[str],
    tokenizer,
    model,
    device:     str,
    batch_size: int = 128,
    add_prefix: bool = True,
) -> torch.Tensor:
    """Encode texts; returns fp16 CPU tensor [N, TARGET_DIM].

    Empty strings are represented as zero vectors (not encoded).
    """
    if not texts:
        return torch.zeros(0, TARGET_DIM, dtype=torch.float16)

    all_embs: List[torch.Tensor] = []
    i = 0
    while i < len(texts):
        batch_raw  = texts[i : i + batch_size]
        batch_enc  = (
            [BGE_QUERY_PREFIX + t for t in batch_raw] if add_prefix else batch_raw
        )
        try:
            inputs = tokenizer(
                batch_enc, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                out  = model(**inputs)
                embs = _cls_pool(out.last_hidden_state)
                embs = F.normalize(embs, p=2, dim=1)

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


# ── text helpers ───────────────────────────────────────────────────────────────

def _to_text(value) -> str:
    """Convert a field value (str, list, None) to a plain text string."""
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v)
    return str(value).strip()


def _parse_query_split(raw) -> Optional[dict]:
    """Parse a query_split store value (JSON string or dict) → dict."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return None


# ── MODE A: query-level genre / decade embeddings ─────────────────────────────

QUERY_FIELDS = ["genre", "decade"]


def run_query_mode(args, tokenizer, model, device: str):
    """Encode genre and decade from query_split store.

    Produces two output files:
        {out_dir}/query_genre_embeddings_{split_name}.pt
        {out_dir}/query_decade_embeddings_{split_name}.pt

    Each file: { "{session_id}_{turn}_{field}": Tensor[384] fp16 }
    """
    logger.info("Loading query_split store from %s …", args.query_split)
    qs_store: Dict[str, str] = torch.load(
        args.query_split, map_location="cpu", weights_only=True
    )
    logger.info("  %d entries in query_split store.", len(qs_store))

    for field in QUERY_FIELDS:
        out_path = os.path.join(args.out_dir, f"query_{field}_embeddings_{args.split_name}.pt")
        os.makedirs(args.out_dir, exist_ok=True)

        # Resume
        store: Dict[str, torch.Tensor] = {}
        if os.path.exists(out_path):
            logger.info("Resuming %s from %s …", field, out_path)
            store = torch.load(out_path, map_location="cpu", weights_only=True)
            logger.info("  %d entries already cached.", len(store))

        # Collect pending
        keys:  List[str] = []
        texts: List[str] = []
        for qs_key, raw_val in qs_store.items():
            # qs_key format: {session_id}_{turn_number}
            store_key = f"{qs_key}_{field}"
            if store_key in store:
                continue
            parsed = _parse_query_split(raw_val)
            field_text = _to_text(parsed.get(field) if parsed else None)
            keys.append(store_key)
            texts.append(field_text)

        logger.info("[%s] %d keys to encode.", field, len(keys))
        if not keys:
            logger.info("[%s] Nothing to do.", field)
            continue

        # Separate empty from non-empty
        empty_idx    = [i for i, t in enumerate(texts) if not t.strip()]
        non_empty    = [(i, keys[i], texts[i]) for i in range(len(keys)) if texts[i].strip()]

        for i in empty_idx:
            store[keys[i]] = torch.zeros(TARGET_DIM, dtype=torch.float16)

        if non_empty:
            ne_indices, ne_keys, ne_texts = zip(*non_empty)
            pbar = tqdm(
                range(0, len(ne_texts), args.batch),
                desc=f"Encoding {field}",
                unit="batch",
            )
            for start in pbar:
                bk = ne_keys [start : start + args.batch]
                bt = ne_texts[start : start + args.batch]
                embs = encode_texts(list(bt), tokenizer, model, device, args.batch)
                for j, k in enumerate(bk):
                    store[k] = embs[j]
                pbar.set_postfix(total=len(store))
                if ((start // args.batch) + 1) % args.save_every == 0:
                    torch.save(store, out_path)

        torch.save(store, out_path)
        logger.info("[%s] Done. %d total entries → %s", field, len(store), out_path)


# ── MODE B: track tag_list embeddings ─────────────────────────────────────────

def run_track_mode(args, tokenizer, model, device: str):
    """Encode tag_list from track metadata.

    Output file: {out_dir}/track_tag_embeddings.pt
    Key: track_id  (str)
    Value: Tensor[384] fp16
    """
    out_path = os.path.join(args.out_dir, "track_tag_embeddings.pt")
    os.makedirs(args.out_dir, exist_ok=True)

    # Resume
    store: Dict[str, torch.Tensor] = {}
    if os.path.exists(out_path):
        logger.info("Resuming from %s …", out_path)
        store = torch.load(out_path, map_location="cpu", weights_only=True)
        logger.info("  %d entries already cached.", len(store))

    logger.info("Loading track metadata from %s …", args.track_dataset)
    ds = load_dataset(args.track_dataset, split=args.track_split)
    logger.info("  %d tracks.", len(ds))

    # Collect pending
    keys:  List[str] = []
    texts: List[str] = []
    for idx in tqdm(range(len(ds)), desc="Scanning tracks", unit="track"):
        item     = ds[idx]
        track_id = str(item["track_id"])
        if track_id in store:
            continue
        tag_list = item.get("tag_list", [])
        tag_text = _to_text(tag_list)
        keys.append(track_id)
        texts.append(tag_text)

    logger.info("%d tracks to encode.", len(keys))
    if not keys:
        logger.info("Nothing to do.")
        return

    # Zero vectors for tracks with no tags
    empty_idx = [i for i, t in enumerate(texts) if not t.strip()]
    non_empty = [(i, keys[i], texts[i]) for i in range(len(keys)) if texts[i].strip()]

    for i in empty_idx:
        store[keys[i]] = torch.zeros(TARGET_DIM, dtype=torch.float16)

    if non_empty:
        ne_indices, ne_keys, ne_texts = zip(*non_empty)
        pbar = tqdm(
            range(0, len(ne_texts), args.batch),
            desc="Encoding tag_list",
            unit="batch",
        )
        for start in pbar:
            bk = ne_keys [start : start + args.batch]
            bt = ne_texts[start : start + args.batch]
            # Track tags are passages (no query prefix)
            embs = encode_texts(list(bt), tokenizer, model, device, args.batch,
                                add_prefix=False)
            for j, k in enumerate(bk):
                store[k] = embs[j]
            pbar.set_postfix(total=len(store))
            if ((start // args.batch) + 1) % args.save_every == 0:
                torch.save(store, out_path)

    torch.save(store, out_path)
    logger.info("Done. %d total track-tag entries → %s", len(store), out_path)


# ── main ───────────────────────────────────────────────────────────────────────

def main(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    tokenizer, model = load_bge(args.bge, device)

    if args.mode == "query":
        run_query_mode(args, tokenizer, model, device)
    elif args.mode == "track":
        run_track_mode(args, tokenizer, model, device)
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Use 'query' or 'track'.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Pre-compute BGE-Small-EN-v1.5 embeddings for "
            "query genre/decade fields and track tag_list."
        )
    )
    p.add_argument(
        "--mode", type=str, required=True,
        choices=["query", "track"],
        help="'query': encode genre/decade from query_split store; "
             "'track': encode tag_list from track metadata.",
    )

    # ── query mode args ──
    q = p.add_argument_group("Query mode (--mode query)")
    q.add_argument(
        "--query_split", type=str,
        default="qwen/query_split_train.pt",
        help="Path to query_split .pt store (output of precompute_query_split.py)",
    )
    q.add_argument(
        "--split_name", type=str, default="train",
        help="Split label used in output filename (train / test / blindA)",
    )

    # ── track mode args ──
    t = p.add_argument_group("Track mode (--mode track)")
    t.add_argument(
        "--track_dataset", type=str,
        default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        help="HuggingFace track metadata dataset",
    )
    t.add_argument(
        "--track_split", type=str, default="all_tracks",
        help="Dataset split to load for tracks",
    )

    # ── shared args ──
    p.add_argument(
        "--bge", type=str,
        default="/home/lijiatong06/music-crs-baselines/bge-small-en-v1.5",
        help="Local path to bge-small-en-v1.5",
    )
    p.add_argument(
        "--out_dir", type=str, default="bge",
        help="Directory to write output .pt files",
    )
    p.add_argument("--batch",      type=int, default=256,
                   help="Encoding batch size (BGE is small, can use large batches)")
    p.add_argument("--save_every", type=int, default=50,
                   help="Save checkpoint every N batches")
    p.add_argument("--device",     type=str, default="auto")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
