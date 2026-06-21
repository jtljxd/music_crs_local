"""
scripts/retrieval/precompute_multi_channel_candidates.py
=========================================================
Pre-compute MultiChannelRetrievalV2 retrieval results for all (session, turn) pairs
and save to a .pt file for use as hard negatives in DCNReranker training.

Output format (dict):
  key   : "{session_id}_{turn_number}"
  value : dict with keys:
            "merged"          : List[track_id]  RRF-merged top-K
            "CH01_CF_BPR"     : List[track_id]
            "CH02_Query_Meta" : List[track_id]
            ... (all per-channel lists)

Usage:
  # Train split
  python scripts/retrieval/precompute_multi_channel_candidates.py \\
      --split train \\
      --query_emb_path  qwen/dialogue_embeddings_train_0.6b.pt \\
      --goal_emb_path   qwen/goal_embeddings_train_0.6b.pt \\
      --bge_tag_path    bge/track_tag_embeddings.pt \\
      --cache_dir       qwen/retrieval_indices \\
      --topk            500 \\
      --out             qwen/retrieval_train_candidates.pt \\
      --device          cuda

  # Test split
  python scripts/retrieval/precompute_multi_channel_candidates.py \\
      --split test \\
      --query_emb_path  qwen/dialogue_embeddings_test_0.6b.pt \\
      --goal_emb_path   qwen/goal_embeddings_test_0.6b.pt \\
      --bge_tag_path    bge/track_tag_embeddings.pt \\
      --cache_dir       qwen/retrieval_indices \\
      --topk            500 \\
      --out             qwen/retrieval_test_candidates.pt \\
      --device          cuda

Server (nohup):
  nohup python scripts/retrieval/precompute_multi_channel_candidates.py \\
      --split train \\
      --query_emb_path  qwen/dialogue_embeddings_train_0.6b.pt \\
      --goal_emb_path   qwen/goal_embeddings_train_0.6b.pt \\
      --bge_tag_path    bge/track_tag_embeddings.pt \\
      --cache_dir       qwen/retrieval_indices \\
      --topk            500 \\
      --out             qwen/retrieval_train_candidates.pt \\
      --device          cuda \\
      > logs/precompute_retrieval_train.log 2>&1 &
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    from mcrs.retrieval_modules.multi_channel_v2 import MultiChannelRetrievalV2, MultiChannelConfig

    os.makedirs("logs", exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ── Build retrieval system ────────────────────────────────────────────
    logger.info("Building MultiChannelRetrievalV2 …")
    cfg = MultiChannelConfig(
        track_emb_dataset   = args.track_emb_dataset,
        track_metadata_name = args.track_metadata_dataset,
        user_metadata_name  = args.user_metadata_dataset,
        user_emb_dataset    = args.user_emb_dataset,
        split_types         = ["all_tracks"],
        cache_dir           = args.cache_dir,
        bge_tag_path        = args.bge_tag_path,
        bge_rich_path       = args.bge_rich_path,
        goal_emb_path       = args.goal_emb_path,
        query_emb_path      = args.query_emb_path,
        genre_emb_path      = args.turn_query_emb_path,
        decade_emb_path     = args.decade_emb_path,
        device              = args.device,
    )
    retrieval = MultiChannelRetrievalV2.build(cfg)

    # ── Load dataset ──────────────────────────────────────────────────────
    is_blinda = args.split.lower() in ("blinda", "blind_a", "blind-a")
    ds_name   = args.blind_dataset if is_blinda else args.conv_dataset
    hf_split  = "test" if is_blinda else args.split
    logger.info("Loading dataset: %s / %s …", ds_name, hf_split)
    ds = load_dataset(ds_name, split=hf_split)
    logger.info("  %d sessions", len(ds))

    n_sessions = min(args.max_sessions, len(ds)) if args.max_sessions > 0 else len(ds)

    # ── Try to load existing partial results (for resume) ─────────────────
    results: Dict[str, dict] = {}
    if os.path.exists(args.out) and not args.overwrite:
        logger.info("Resuming from existing file: %s (%d keys)",
                    args.out,
                    len(torch.load(args.out, map_location="cpu", weights_only=True)))
        results = torch.load(args.out, map_location="cpu", weights_only=True)

    # ── Iterate sessions / turns ───────────────────────────────────────────
    skipped, done, errors = 0, 0, 0
    for idx in tqdm(range(n_sessions), desc=f"Precompute ({args.split})"):
        item       = ds[idx]
        session_id = str(item.get("session_id") or item.get("id") or idx)
        user_id    = str(item.get("user_id", ""))

        # Collect all music turns for this session
        music_turns: List[int] = [
            int(c["turn_number"])
            for c in item.get("conversations", [])
            if c.get("role") == "music" and c.get("content")
        ]
        if not music_turns:
            continue

        for turn_number in music_turns:
            key = f"{session_id}_{turn_number}"

            # Skip if already computed
            if key in results and not args.overwrite:
                skipped += 1
                continue

            try:
                ctx     = retrieval.build_context(
                    session_id=session_id,
                    turn_number=turn_number,
                    session_data=item,
                    user_id=user_id,
                )
                ret_res = retrieval.retrieve(ctx, topk=args.topk)

                # Store merged + per-channel lists
                results[key] = {
                    ch: lst for ch, lst in ret_res.items()
                    if isinstance(lst, list)
                }
                done += 1

            except Exception as e:
                logger.warning("Failed %s t%d: %s", session_id, turn_number, e)
                errors += 1

        # Periodic save
        if (idx + 1) % args.save_every == 0:
            torch.save(results, args.out)
            logger.info("  [%d/%d] saved %d keys (skipped=%d, errors=%d)",
                        idx + 1, n_sessions, len(results), skipped, errors)

    # ── Final save ────────────────────────────────────────────────────────
    torch.save(results, args.out)
    logger.info(
        "Done. Total keys: %d  (new=%d, skipped=%d, errors=%d) → %s",
        len(results), done, skipped, errors, args.out,
    )

    # ── Quick stats ───────────────────────────────────────────────────────
    sample_keys = list(results.keys())[:3]
    for k in sample_keys:
        v = results[k]
        merged_len = len(v.get("merged", []))
        ch_count   = sum(1 for ck in v if ck != "merged" and isinstance(v[ck], list) and v[ck])
        logger.info("  Sample key=%s: merged=%d, active_channels=%d", k, merged_len, ch_count)


def parse_args():
    p = argparse.ArgumentParser(
        description="Pre-compute MultiChannelRetrievalV2 candidates for train/test/blinda"
    )
    p.add_argument("--split",  default="train",
                   help="'train', 'test', or 'blinda'")
    p.add_argument("--topk",   type=int, default=500,
                   help="Top-K merged candidates to store per turn")
    p.add_argument("--out",    required=True,
                   help="Output .pt path, e.g. qwen/retrieval_train_candidates.pt")
    p.add_argument("--max_sessions", type=int, default=0,
                   help="Limit number of sessions (0 = all)")
    p.add_argument("--save_every",   type=int, default=200,
                   help="Save checkpoint every N sessions")
    p.add_argument("--overwrite",    action="store_true",
                   help="Recompute even if key already exists")
    # Embedding paths
    p.add_argument("--query_emb_path",      required=True)
    p.add_argument("--goal_emb_path",       default=None)
    p.add_argument("--turn_query_emb_path", default=None)
    p.add_argument("--decade_emb_path",     default=None)
    p.add_argument("--bge_rich_path",       default="bge/track_rich_embeddings.pt")
    p.add_argument("--bge_tag_path",        default="bge/track_tag_embeddings.pt")
    p.add_argument("--cache_dir",           default="qwen/retrieval_indices")
    # Dataset names
    p.add_argument("--conv_dataset",           default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    p.add_argument("--blind_dataset",          default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
    p.add_argument("--track_emb_dataset",      default="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    p.add_argument("--track_metadata_dataset", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    p.add_argument("--user_metadata_dataset",  default="talkpl-ai/TalkPlayData-Challenge-User-Metadata")
    p.add_argument("--user_emb_dataset",       default="talkpl-ai/TalkPlayData-Challenge-User-Embeddings")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
