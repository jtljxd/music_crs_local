"""
scripts/indexing/build_retrieval_indices.py
============================================
One-shot script to build (and cache) all track embedding indices.

This only needs to run ONCE per machine.  After completion the index is
persisted under ``--cache_dir/track_index/`` and all retrieval channels
load from cache automatically.

What it builds
--------------
  1.  Track embedding matrix index (all modalities)
        • cf_bpr       [N, 128]
        • metadata     [N, 1024]
        • lyrics       [N, 1024]
        • attributes   [N, 1024]
        • audio        [N, 512]
        • image        [N, 768+]
        • tag_bge      [N, 384]   (from bge/track_tag_embeddings.pt)
  2.  BM25 index   (built lazily by BM25_MODEL on first use; triggered here)
  3.  Summary JSON  (cache_dir/track_index/build_summary.json)

Usage
-----
    python scripts/indexing/build_retrieval_indices.py \
        --track_emb_dataset talkpl-ai/TalkPlayData-Challenge-Track-Embeddings \
        --track_metadata_dataset talkpl-ai/TalkPlayData-Challenge-Track-Metadata \
        --bge_tag_path bge/track_tag_embeddings.pt \
        --cache_dir qwen/retrieval_indices \
        --device cpu

nohup:
    nohup python scripts/indexing/build_retrieval_indices.py \
        --track_emb_dataset talkpl-ai/TalkPlayData-Challenge-Track-Embeddings \
        --track_metadata_dataset talkpl-ai/TalkPlayData-Challenge-Track-Metadata \
        --bge_tag_path bge/track_tag_embeddings.pt \
        --cache_dir qwen/retrieval_indices \
        > logs/build_indices.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

# Ensure project root is in sys.path so `mcrs` package is importable
# when the script is run from any directory (e.g. scripts/indexing/).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build track embedding indices (one-time)")
    p.add_argument(
        "--track_emb_dataset", type=str,
        default="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        help="HuggingFace dataset ID containing per-track embeddings.",
    )
    p.add_argument(
        "--track_metadata_dataset", type=str,
        default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        help="HuggingFace dataset ID for track metadata (used for BM25).",
    )
    p.add_argument(
        "--split_types", type=str, nargs="+", default=["all_tracks"],
        help="Track dataset split names to include.",
    )
    p.add_argument(
        "--bge_tag_path", type=str, default="bge/track_tag_embeddings.pt",
        help="Path to pre-computed BGE tag embeddings.",
    )
    p.add_argument(
        "--bge_rich_path", type=str, default="bge/track_rich_embeddings.pt",
        help="Path to pre-computed BGE rich track embeddings (name+artist+album+tags+date+duration+popularity).",
    )
    p.add_argument(
        "--cache_dir", type=str, default="qwen/retrieval_indices",
        help="Directory where the built index will be saved.",
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for tensor operations during index build.",
    )
    p.add_argument(
        "--force_rebuild", action="store_true",
        help="Force rebuild even if a cached index already exists.",
    )
    p.add_argument(
        "--skip_bm25", action="store_true",
        help="Skip BM25 index building (useful if already built).",
    )
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    from mcrs.retrieval_modules.index_store import IndexStore
    from mcrs.retrieval_modules.bm25 import BM25_MODEL

    os.makedirs("logs", exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    total_start = time.time()

    # ── 1. Track embedding index ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1: Building track embedding index …")
    logger.info("  Dataset : %s", args.track_emb_dataset)
    logger.info("  Splits  : %s", args.split_types)
    logger.info("  Cache   : %s", args.cache_dir)
    logger.info("  Device  : %s", args.device)
    logger.info("  Force   : %s", args.force_rebuild)

    t0 = time.time()
    store = IndexStore.build(
        track_emb_dataset=args.track_emb_dataset,
        split_types=args.split_types,
        cache_dir=args.cache_dir,
        bge_tag_path=args.bge_tag_path   if os.path.exists(args.bge_tag_path  or "") else None,
        bge_rich_path=args.bge_rich_path if os.path.exists(args.bge_rich_path or "") else None,
        device=args.device,
        force_rebuild=args.force_rebuild,
    )
    t1 = time.time()
    logger.info("  ✅ Index built in %.1f s", t1 - t0)
    logger.info("  Tracks     : %d", len(store.track_ids))
    logger.info("  Modalities : %s", list(store.matrices.keys()))

    # ── 2. BM25 index ─────────────────────────────────────────────────────────
    if not args.skip_bm25:
        logger.info("=" * 60)
        logger.info("Step 2: Building BM25 index …")
        logger.info("  Dataset : %s", args.track_metadata_dataset)
        t0 = time.time()
        try:
            bm25 = BM25_MODEL(
                dataset_name=args.track_metadata_dataset,
                split_types=args.split_types,
                corpus_types=["track_name", "artist_name", "album_name"],
                cache_dir=args.cache_dir,
            )
            t1 = time.time()
            logger.info("  ✅ BM25 index ready in %.1f s", t1 - t0)
            # Quick sanity check
            hits = bm25.text_to_item_retrieval("jazz piano relaxing", topk=5)
            logger.info("  Sanity check query → %s", hits[:3])
        except Exception as e:
            logger.warning("  ⚠️ BM25 build failed: %s", e)
    else:
        logger.info("Step 2: BM25 skipped (--skip_bm25).")

    # ── 3. Summary ────────────────────────────────────────────────────────────
    summary = {
        "n_tracks":   len(store.track_ids),
        "modalities": {k: list(v.shape) for k, v in store.matrices.items()},
        "cache_dir":  args.cache_dir,
        "bge_tag":    args.bge_tag_path,
        "build_time_s": round(time.time() - total_start, 1),
    }
    summary_path = os.path.join(args.cache_dir, "track_index", "build_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("✅ All done in %.1f s", time.time() - total_start)
    logger.info("Summary saved to %s", summary_path)
    logger.info("")
    logger.info("Index layout:")
    for mod, shp in summary["modalities"].items():
        logger.info("  %-14s  shape=%s  (%.1f MB)",
                    mod, shp,
                    (shp[0] * shp[1] * 4) / 1e6)


if __name__ == "__main__":
    main(parse_args())
