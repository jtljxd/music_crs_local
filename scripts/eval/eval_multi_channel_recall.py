"""
scripts/eval/eval_multi_channel_recall.py
==========================================
Evaluate all retrieval channels on Recall@K and NDCG@20.

For each (session, turn) with a ground-truth track:
  - Build a RetrievalContext
  - Run MultiChannelRetrievalV2.retrieve()
  - Compute per-channel Recall@{20,30,50,100,150,200} and NDCG@20

Output:
  - Printed table to stdout
  - exp/eval/multi_channel_recall_{split}.txt
  - exp/eval/multi_channel_recall_{split}.json  (machine-readable)

Usage:
    python scripts/eval/eval_multi_channel_recall.py \\
        --split test \\
        --query_emb_path  qwen/dialogue_embeddings_test_0.6b.pt \\
        --goal_emb_path   qwen/goal_embeddings_test_0.6b.pt \\
        --genre_emb_path  bge/query_genre_embeddings_test.pt \\
        --decade_emb_path bge/query_decade_embeddings_test.pt \\
        --topk 200

nohup:
    nohup python scripts/eval/eval_multi_channel_recall.py \\
        --split test \\
        --query_emb_path  qwen/dialogue_embeddings_test_0.6b.pt \\
        --goal_emb_path   qwen/goal_embeddings_test_0.6b.pt \\
        --genre_emb_path  bge/query_genre_embeddings_test.pt \\
        --decade_emb_path bge/query_decade_embeddings_test.pt \\
        > logs/eval_recall_test.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Ensure project root is importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Evaluation cutoffs
RECALL_KS = [20, 30, 50, 100, 150, 200, 300, 400, 500]
NDCG_K    = 20


# ── Metrics ───────────────────────────────────────────────────────────────────

def recall_at_k(candidates: List[str], gt: str, k: int) -> float:
    return 1.0 if gt in candidates[:k] else 0.0


def ndcg_at_k(candidates: List[str], gt: str, k: int) -> float:
    """NDCG@k for single relevant item."""
    for i, tid in enumerate(candidates[:k]):
        if tid == gt:
            return 1.0 / math.log2(i + 2)   # ideal DCG = 1/log2(2) = 1.0
    return 0.0


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    from mcrs.retrieval_modules.multi_channel_v2 import (
        MultiChannelRetrievalV2,
        MultiChannelConfig,
    )

    # ── Build retrieval system ────────────────────────────────────────────────
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
        # genre_emb_path doubles as turn_query BGE emb path (--turn_query_emb_path overrides)
        genre_emb_path      = args.turn_query_emb_path or args.genre_emb_path,
        decade_emb_path     = args.decade_emb_path,
        device              = args.device,
    )
    retrieval = MultiChannelRetrievalV2.build(cfg)
    channel_names = retrieval.channel_names() + ["merged"]
    logger.info("Channels: %s", channel_names)

    # ── Load conversation dataset ─────────────────────────────────────────────
    logger.info("Loading %s split of %s …", args.split, args.conv_dataset)
    ds = load_dataset(args.conv_dataset, split=args.split)
    n_sessions = len(ds)
    if args.max_sessions > 0:
        n_sessions = min(args.max_sessions, n_sessions)
    logger.info("%d sessions to evaluate.", n_sessions)

    # ── Accumulators ─────────────────────────────────────────────────────────
    # channel → metric_name → list of float
    stats: Dict[str, Dict[str, List[float]]] = {
        ch: defaultdict(list) for ch in channel_names
    }
    # Also track "coverage" = fraction of turns where channel returned ≥1 result
    coverage: Dict[str, List[int]] = defaultdict(list)

    total_turns = 0

    for idx in tqdm(range(n_sessions), desc="Eval", unit="session"):
        item  = ds[idx]
        convs = item.get("conversations", [])

        # session identifier
        session_id = str(item.get("session_id") or item.get("id") or idx)
        user_id    = str(item.get("user_id", ""))

        # Find all (turn_number, gt_track_id) pairs
        music_turns: Dict[int, str] = {
            int(c["turn_number"]): c["content"]
            for c in convs
            if c.get("role") == "music" and c.get("content")
        }
        if not music_turns:
            continue

        # --non_last_only: skip the last music turn in each session
        # (mirrors blind-A evaluation convention where the last turn is the target)
        # Single-turn sessions are skipped entirely (no non-last turn to evaluate)
        if args.non_last_only:
            if len(music_turns) <= 1:
                continue
            last_music_turn = max(music_turns.keys())
            music_turns = {t: tid for t, tid in music_turns.items() if t != last_music_turn}

        for turn_number, gt_track_id in music_turns.items():
            total_turns += 1

            # Build context
            ctx = retrieval.build_context(
                session_id=session_id,
                turn_number=turn_number,
                session_data=item,
                user_id=user_id,
            )

            # Retrieve
            try:
                results = retrieval.retrieve(ctx, topk=args.topk)
            except Exception as e:
                logger.warning("Retrieve failed for %s turn %d: %s",
                               session_id, turn_number, e)
                continue

            # Compute metrics per channel
            for ch_name in channel_names:
                candidates = results.get(ch_name, [])
                coverage[ch_name].append(1 if len(candidates) > 0 else 0)
                for k in RECALL_KS:
                    stats[ch_name][f"recall@{k}"].append(
                        recall_at_k(candidates, gt_track_id, k)
                    )
                stats[ch_name][f"ndcg@{NDCG_K}"].append(
                    ndcg_at_k(candidates, gt_track_id, NDCG_K)
                )

    logger.info("Evaluated %d turns.", total_turns)

    # ── Compute averages ──────────────────────────────────────────────────────
    metric_names = [f"recall@{k}" for k in RECALL_KS] + [f"ndcg@{NDCG_K}"]
    averages: Dict[str, Dict[str, float]] = {}
    for ch in channel_names:
        averages[ch] = {}
        for m in metric_names:
            vals = stats[ch][m]
            averages[ch][m] = sum(vals) / len(vals) if vals else 0.0
        cov = coverage[ch]
        averages[ch]["coverage"] = sum(cov) / len(cov) if cov else 0.0

    # ── Print table ───────────────────────────────────────────────────────────
    header_cols = ["Channel", "cov%"] + [f"R@{k}" for k in RECALL_KS] + [f"NDCG@{NDCG_K}"]
    col_w = max(max(len(ch) for ch in channel_names), len("Channel")) + 2

    lines: List[str] = []
    lines.append("")
    lines.append("=" * (col_w + len(header_cols) * 9))
    lines.append(
        f"Multi-Channel Recall Evaluation  |  split={args.split}  |  {total_turns} turns"
    )
    lines.append("=" * (col_w + len(header_cols) * 9))

    # Header
    header = f"{'Channel':<{col_w}}"
    header += f"{'cov%':>7}"
    for k in RECALL_KS:
        header += f"  R@{k:<5}"
    header += f"  NDCG@{NDCG_K}"
    lines.append(header)
    lines.append("-" * len(header))

    # Sort channels: merged last, others by ndcg@20 desc
    sorted_channels = sorted(
        [ch for ch in channel_names if ch != "merged"],
        key=lambda ch: averages[ch].get(f"ndcg@{NDCG_K}", 0),
        reverse=True,
    ) + ["merged"]

    for ch in sorted_channels:
        av = averages[ch]
        row  = f"{ch:<{col_w}}"
        row += f"{av['coverage']*100:>6.1f}%"
        for k in RECALL_KS:
            row += f"  {av.get(f'recall@{k}', 0)*100:>5.2f}%"
        row += f"  {av.get(f'ndcg@{NDCG_K}', 0):.5f}"
        lines.append(row)

    lines.append("=" * len(header))
    lines.append(f"Total evaluated turns: {total_turns}")
    report = "\n".join(lines)
    print(report)

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = os.path.join("exp", "eval")
    os.makedirs(out_dir, exist_ok=True)

    txt_path = os.path.join(out_dir, f"multi_channel_recall_{args.split}.txt")
    with open(txt_path, "w") as f:
        f.write(report + "\n")

    json_path = os.path.join(out_dir, f"multi_channel_recall_{args.split}.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "split":        args.split,
                "total_turns":  total_turns,
                "channels":     sorted_channels,
                "metrics":      averages,
            },
            f, indent=2,
        )

    logger.info("Report saved to %s", txt_path)
    logger.info("JSON saved to %s", json_path)


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate multi-channel retrieval (Recall@K + NDCG@20)"
    )
    p.add_argument("--split",  type=str, default="test",
                   help="Conversation dataset split to evaluate on.")
    p.add_argument("--conv_dataset", type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    p.add_argument("--track_emb_dataset", type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    p.add_argument("--track_metadata_dataset", type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    p.add_argument("--user_metadata_dataset", type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-User-Metadata")
    p.add_argument("--user_emb_dataset", type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
                   help="HF dataset with user CF-BPR embeddings (field: cf-bpr)")
    p.add_argument("--cache_dir", type=str, default="qwen/retrieval_indices")
    p.add_argument("--bge_tag_path", type=str,
                   default="bge/track_tag_embeddings.pt")
    # Per-split embedding paths (must be set appropriately for each split)
    p.add_argument("--query_emb_path",  type=str, default=None,
                   help="Path to qwen/dialogue_embeddings_{split}_0.6b.pt")
    p.add_argument("--goal_emb_path",   type=str, default=None,
                   help="Path to qwen/goal_embeddings_{split}_0.6b.pt")
    p.add_argument("--genre_emb_path",  type=str, default=None,
                   help="Path to bge/query_genre_embeddings_{split}.pt (legacy)")
    p.add_argument("--turn_query_emb_path", type=str, default=None,
                   help="Path to bge/turn_query_embeddings_{split}.pt (new CH22 query emb; overrides genre_emb_path for CH22)")
    p.add_argument("--bge_rich_path", type=str,
                   default="bge/track_rich_embeddings.pt",
                   help="Path to bge/track_rich_embeddings.pt (track-side BGE rich index for CH22)")
    p.add_argument("--decade_emb_path", type=str, default=None,
                   help="Path to bge/query_decade_embeddings_{split}.pt")
    p.add_argument("--topk", type=int, default=200)
    p.add_argument("--max_sessions", type=int, default=0,
                   help="Limit sessions for quick debug (0 = all).")
    p.add_argument("--non_last_only", action="store_true", default=False,
                   help="Only evaluate non-last music turns per session (for blind-A style eval).")
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "cpu"])
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
