"""
eval_retrieval_hitrate.py
=========================
Evaluate retrieval hit rate on the test split using BM25 or BERT retrieval.

For each (session, turn) where the ground-truth is a music track:
  - Run the selected retrieval channel (BM25 or BERT)
  - Check if the ground-truth track_id is in the retrieved results
  - Report hit rate and average recall size

Output: a text table printed to stdout + saved to
        exp/eval/retrieval_hitrate.txt

Usage:
    python eval_retrieval_hitrate.py \
        --retrieval_type bm25 \
        --dataset_name talkpl-ai/TalkPlayData-Challenge-Track-Metadata \
        --topk 200 \
        --split test \
        --max_sessions 0
"""

import argparse
import logging
import os
from collections import defaultdict
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

from mcrs.retrieval_modules import load_retrieval_module

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate(args):
    logger.info(
        "Initializing %s retrieval (topk=%d) …", args.retrieval_type, args.topk
    )
    retrieval = load_retrieval_module(
        retrieval_type=args.retrieval_type,
        dataset_name=args.dataset_name,
        track_split_types=["all_tracks"],
        corpus_types=["track_name", "artist_name", "album_name"],
        cache_dir=args.cache_dir,
    )
    logger.info("Retrieval ready.")

    # ── Load evaluation dataset ───────────────────────────────────────────────
    logger.info("Loading %s split of %s …", args.split, args.conv_dataset_name)
    ds = load_dataset(args.conv_dataset_name, split=args.split)
    total_sessions = len(ds)
    if args.max_sessions > 0:
        total_sessions = min(args.max_sessions, total_sessions)
    logger.info("Evaluating on %d sessions.", total_sessions)

    # ── Stats ────────────────────────────────────────────────────────────────
    total_queries = 0
    n_hits = 0
    recall_sizes = []

    for idx in tqdm(range(total_sessions), desc="Eval", unit="session"):
        item      = ds[idx]
        convs     = item["conversations"]

        music_turns = {
            int(c["turn_number"]): c["content"]
            for c in convs
            if c.get("role") == "music" and c.get("content")
        }
        if not music_turns:
            continue

        for turn_number, gt_track_id in music_turns.items():
            # Build query: concatenate all user turns up to and including this turn
            query_parts = []
            for c in convs:
                if int(c["turn_number"]) <= turn_number and c["role"] == "user":
                    query_parts.append(c.get("content", ""))
            user_query = " ".join(query_parts).strip()
            if not user_query:
                continue

            candidates = retrieval.text_to_item_retrieval(user_query, topk=args.topk)

            total_queries += 1
            recall_sizes.append(len(candidates))
            if gt_track_id in candidates:
                n_hits += 1

    # ── Report ───────────────────────────────────────────────────────────────
    avg_size = sum(recall_sizes) / len(recall_sizes) if recall_sizes else 0
    hit_pct  = 100.0 * n_hits / total_queries if total_queries else 0

    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(
        f"Retrieval Hit Rate Evaluation  "
        f"({args.split} split, {total_queries} queries)"
    )
    lines.append(f"{'='*60}")
    lines.append(f"Retrieval type : {args.retrieval_type}")
    lines.append(f"Top-K          : {args.topk}")
    lines.append(f"{'='*60}")
    lines.append(
        f"{'Channel':<20} {'Recall Size (avg)':>18} {'Hits':>8} {'Hit Rate':>10}"
    )
    lines.append("-" * 60)
    lines.append(
        f"{args.retrieval_type:<20} {avg_size:>18.1f} {n_hits:>8d} {hit_pct:>9.2f}%"
    )
    lines.append(f"{'='*60}")
    lines.append(f"Total queries evaluated: {total_queries}")
    report = "\n".join(lines)
    print(report)

    out_dir  = os.path.join("exp", "eval")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "retrieval_hitrate.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    logger.info("Report saved to %s", out_path)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate retrieval hit rate (BM25 / BERT)")
    p.add_argument(
        "--retrieval_type", type=str, default="bm25",
        choices=["bm25", "bert"],
        help="Retrieval backend to evaluate.",
    )
    p.add_argument(
        "--dataset_name", type=str,
        default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        help="Track metadata dataset used to build the retrieval index.",
    )
    p.add_argument(
        "--conv_dataset_name", type=str,
        default="talkpl-ai/TalkPlayData-Challenge-Dataset",
        help="Conversation dataset used for evaluation.",
    )
    p.add_argument(
        "--split", type=str, default="test",
        help="Dataset split to evaluate on (e.g. test, validation).",
    )
    p.add_argument(
        "--topk", type=int, default=200,
        help="Number of candidates to retrieve per query.",
    )
    p.add_argument(
        "--cache_dir", type=str, default="./cache",
        help="Cache directory for retrieval indices.",
    )
    p.add_argument(
        "--max_sessions", type=int, default=0,
        help="Maximum sessions to evaluate (0 = all).",
    )
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
