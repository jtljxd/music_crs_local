"""
eval_retrieval_hitrate.py
=========================
Evaluate per-channel hit rate on the test split.

For each (session, turn) where the ground-truth is a music track:
  - Run each of the 6 retrieval channels independently
  - Check if the ground-truth track_id is in the channel's results
  - Report hit rate per channel and for the union of all channels

Output: a Markdown table printed to stdout + saved to
        exp/eval/retrieval_hitrate.md

Usage:
    python eval_retrieval_hitrate.py \
        --config config/llama1b_multi_channel_devset.yaml \
        --turn_store qwen/dialogue_embeddings_test_0.6b.pt \
        --max_sessions 0          # 0 = all sessions
"""

import argparse
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set

import torch
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from mcrs.retrieval_modules.multi_channel_retrieval import (
    MultiChannelRetrieval,
    USER_MUSIC_TOWER_RECALL_NUM,
    QUERY_METADATA_RECALL_NUM,
    QUERY_ATTRIBUTES_RECALL_NUM,
    BM25_RECALL_NUM,
    BERT_RECALL_NUM,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EMB_DIM = 1024

CHANNEL_NAMES = [
    "ch1_CF-BPR",
    "ch2_SimilarUsers",
    "ch3_QwenMeta",
    "ch4_QwenAttr",
    "ch5_BM25",
    "ch6_Semantic",
    "ALL (union)",
]


def _get_emb(store, key, dim=EMB_DIM):
    if key in store:
        v = store[key].float()
        if v.shape[0] > dim:
            v = v[:dim]
        elif v.shape[0] < dim:
            v = F.pad(v, (0, dim - v.shape[0]))
        return v
    return torch.zeros(dim)


def per_channel_retrieve(
    retrieval: MultiChannelRetrieval,
    user_id: Optional[str],
    current_query: str,
    query_emb: torch.Tensor,
):
    """Run each channel independently; return list of 6 candidate lists."""
    ch1 = retrieval._retrieve_cf_bpr(user_id, USER_MUSIC_TOWER_RECALL_NUM)
    ch2 = retrieval._retrieve_similar_users_music(user_id)
    ch3 = retrieval._retrieve_query_metadata(query_emb, QUERY_METADATA_RECALL_NUM)
    ch4 = retrieval._retrieve_query_attributes(query_emb, QUERY_ATTRIBUTES_RECALL_NUM)
    ch5 = retrieval._retrieve_bm25(current_query, BM25_RECALL_NUM)
    ch6 = retrieval._retrieve_bert(current_query, BERT_RECALL_NUM)
    return [ch1, ch2, ch3, ch4, ch5, ch6]


def evaluate(args):
    config          = OmegaConf.load(args.config)
    cache_dir       = config.get("cache_dir", "./cache")
    qwen_model_path = config.get(
        "qwen_model_path",
        "/home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B",
    )
    dataset_name = config.get(
        "conversation_dataset_name", "talkpl-ai/TalkPlayData-Challenge-Dataset"
    )

    # ── Load turn store ──────────────────────────────────────────────────────
    logger.info("Loading turn store from %s …", args.turn_store)
    turn_store = torch.load(args.turn_store, map_location="cpu", weights_only=True)
    logger.info("  %d entries loaded.", len(turn_store))

    # ── Init retrieval (load-only, indices already built) ────────────────────
    logger.info("Initializing MultiChannelRetrieval (build_indices=False) …")
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
    qwen_model     = AutoModel.from_pretrained(qwen_model_path).cpu().eval()

    retrieval = MultiChannelRetrieval(
        dataset_name      = dataset_name,
        item_db_name      = config.get("item_db_name",
            "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"),
        user_db_name      = config.get("user_db_name",
            "talkpl-ai/TalkPlayData-Challenge-User-Metadata"),
        track_emb_db_name = config.get("track_emb_db_name",
            "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"),
        user_emb_db_name  = config.get("user_emb_db_name",
            "talkpl-ai/TalkPlayData-Challenge-User-Embeddings"),
        split_types       = list(config.get("track_split_types", ["all_tracks"])),
        cache_dir         = cache_dir,
        qwen_model_path   = qwen_model_path,
        device            = "cpu",   # eval on CPU (no LLM needed)
        qwen_model        = qwen_model,
        qwen_tokenizer    = qwen_tokenizer,
        build_indices     = False,   # indices already built during training
    )
    retrieval.set_turn_store(turn_store)
    logger.info("Retrieval ready.")

    # ── Load test dataset ────────────────────────────────────────────────────
    logger.info("Loading test dataset …")
    ds = load_dataset(dataset_name, split="test")
    total_sessions = len(ds)
    if args.max_sessions > 0:
        total_sessions = min(args.max_sessions, total_sessions)
    logger.info("Evaluating on %d sessions.", total_sessions)

    # ── Stats ────────────────────────────────────────────────────────────────
    total_queries   = 0
    hits            = defaultdict(int)   # channel_name → hit count
    ch_sizes        = defaultdict(list)  # channel_name → recall sizes

    for idx in tqdm(range(total_sessions), desc="Eval", unit="session"):
        item       = ds[idx]
        session_id = item["session_id"]
        user_id    = item.get("user_id", None)
        convs      = item["conversations"]

        # Find all (turn, gt_track) pairs
        music_turns = {
            int(c["turn_number"]): c["content"]
            for c in convs
            if c.get("role") == "music" and c.get("content")
        }
        if not music_turns:
            continue

        for turn_number, gt_track_id in music_turns.items():
            # Get user query text
            user_query = ""
            for c in convs:
                if int(c["turn_number"]) == turn_number and c["role"] == "user":
                    user_query = c.get("content", "")
                    break
            if not user_query:
                continue

            # Get query/history emb from turn store
            q_key = f"{session_id}_{turn_number}_query"
            h_key = f"{session_id}_{turn_number}_history"
            query_emb   = _get_emb(turn_store, q_key)
            hist_emb    = _get_emb(turn_store, h_key)
            blended_emb = F.normalize(0.7 * query_emb + 0.3 * hist_emb, p=2, dim=0)

            # Per-channel retrieval
            channels = per_channel_retrieve(
                retrieval, user_id, user_query, blended_emb
            )

            total_queries += 1
            union: Set[str] = set()

            for ch_idx, (name, cands) in enumerate(zip(CHANNEL_NAMES[:6], channels)):
                ch_sizes[name].append(len(cands))
                if gt_track_id in cands:
                    hits[name] += 1
                union.update(cands)

            if gt_track_id in union:
                hits["ALL (union)"] += 1
            ch_sizes["ALL (union)"].append(len(union))

    # ── Report ───────────────────────────────────────────────────────────────
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"Retrieval Hit Rate Evaluation  (test split, {total_queries} queries)")
    lines.append(f"{'='*70}")
    lines.append(
        f"{'Channel':<22} {'Recall Size (avg)':>18} {'Hits':>8} {'Hit Rate':>10}"
    )
    lines.append("-" * 70)

    for name in CHANNEL_NAMES:
        n_hits  = hits[name]
        sizes   = ch_sizes[name]
        avg_sz  = sum(sizes) / len(sizes) if sizes else 0
        hit_pct = 100.0 * n_hits / total_queries if total_queries else 0
        lines.append(
            f"{name:<22} {avg_sz:>18.1f} {n_hits:>8d} {hit_pct:>9.2f}%"
        )

    lines.append(f"{'='*70}")
    lines.append(f"Total queries evaluated: {total_queries}")
    report = "\n".join(lines)
    print(report)

    # Save to file
    out_dir = os.path.join("exp", "eval")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "retrieval_hitrate.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    logger.info("Report saved to %s", out_path)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate per-channel retrieval hit rate")
    p.add_argument("--config",       type=str,
                   default="config/llama1b_multi_channel_devset.yaml")
    p.add_argument("--turn_store",   type=str,
                   default="qwen/dialogue_embeddings_test_0.6b.pt")
    p.add_argument("--max_sessions", type=int, default=0,
                   help="0 = evaluate all sessions")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
