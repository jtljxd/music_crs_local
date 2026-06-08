"""
eval_retrieval_hitrate.py
=========================
Evaluate per-channel hit rate on the test split.

For each (session, turn) where the ground-truth is a music track:
  - Run each of the 3 retrieval channels independently
  - Check if the ground-truth track_id is in the channel's results
  - Report hit rate per channel and for the union of all channels

Output: a text table printed to stdout + saved to
        exp/eval/retrieval_hitrate.txt

Usage:
    python eval_retrieval_hitrate.py \
        --config config/llama1b_multi_channel_devset.yaml \
        --conv_emb_store qwen/hist_conversation_embeddings_test_0.6b.pt \
        --query_split_store qwen/query_split_test.pt \
        --max_sessions 0
"""

import argparse
import json
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
    BM25_RECALL_NUM,
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
    "ch3_QwenMeta",
    "ch5_BM25",
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
    session_id: Optional[str] = None,
    turn_number: Optional[int] = None,
):
    """Run each channel independently; return list of 3 candidate lists."""
    ch1 = retrieval._retrieve_cf_bpr(user_id, USER_MUSIC_TOWER_RECALL_NUM)
    ch3 = retrieval._retrieve_query_metadata(query_emb, QUERY_METADATA_RECALL_NUM)
    ch5 = retrieval._retrieve_bm25(
        current_query,
        BM25_RECALL_NUM,
        session_id=session_id,
        turn_number=turn_number,
    )
    return [ch1, ch3, ch5]


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

    # ── Load stores ──────────────────────────────────────────────────────────
    logger.info("Loading conv_emb_store from %s …", args.conv_emb_store)
    conv_emb_store = torch.load(args.conv_emb_store, map_location="cpu", weights_only=True)
    logger.info("  %d entries loaded.", len(conv_emb_store))

    query_split_store = None
    if args.query_split_store and os.path.exists(args.query_split_store):
        logger.info("Loading query_split_store from %s …", args.query_split_store)
        query_split_store = torch.load(
            args.query_split_store, map_location="cpu", weights_only=True
        )
        logger.info("  %d entries loaded.", len(query_split_store))
    else:
        logger.warning("No query_split_store provided; ch5 will use raw query text.")

    # ── Init retrieval ────────────────────────────────────────────────────────
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
        device            = "cpu",
        qwen_model        = qwen_model,
        qwen_tokenizer    = qwen_tokenizer,
        build_indices     = False,
        conv_emb_store    = conv_emb_store,
    )
    if query_split_store is not None:
        retrieval.set_query_split_store(query_split_store)
    logger.info("Retrieval ready.")

    # ── Load test dataset ────────────────────────────────────────────────────
    logger.info("Loading test dataset …")
    ds = load_dataset(dataset_name, split="test")
    total_sessions = len(ds)
    if args.max_sessions > 0:
        total_sessions = min(args.max_sessions, total_sessions)
    logger.info("Evaluating on %d sessions.", total_sessions)

    # ── Stats ────────────────────────────────────────────────────────────────
    total_queries = 0
    hits          = defaultdict(int)
    ch_sizes      = defaultdict(list)

    for idx in tqdm(range(total_sessions), desc="Eval", unit="session"):
        item       = ds[idx]
        session_id = item["session_id"]
        user_id    = item.get("user_id", None)
        convs      = item["conversations"]

        music_turns = {
            int(c["turn_number"]): c["content"]
            for c in convs
            if c.get("role") == "music" and c.get("content")
        }
        if not music_turns:
            continue

        for turn_number, gt_track_id in music_turns.items():
            user_query = ""
            for c in convs:
                if int(c["turn_number"]) == turn_number and c["role"] == "user":
                    user_query = c.get("content", "")
                    break
            if not user_query:
                continue

            # Resolve query_emb from conv_emb_store (ch3 uses 1024d directly)
            key = f"{session_id}_{turn_number}"
            query_emb = _get_emb(conv_emb_store, key)
            query_emb = F.normalize(query_emb.unsqueeze(0), p=2, dim=1).squeeze(0)

            channels = per_channel_retrieve(
                retrieval, user_id, user_query, query_emb,
                session_id=session_id, turn_number=turn_number,
            )

            total_queries += 1
            union: Set[str] = set()

            for ch_idx, (name, cands) in enumerate(zip(CHANNEL_NAMES[:3], channels)):
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

    out_dir  = os.path.join("exp", "eval")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "retrieval_hitrate.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    logger.info("Report saved to %s", out_path)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate per-channel retrieval hit rate (v2)")
    p.add_argument("--config",             type=str,
                   default="config/llama1b_multi_channel_devset.yaml")
    p.add_argument("--conv_emb_store",     type=str,
                   default="qwen/hist_conversation_embeddings_test_0.6b.pt",
                   help="hist_conversation_embeddings for ch3 (1024d, key={sid}_{turn})")
    p.add_argument("--query_split_store",  type=str,
                   default="qwen/query_split_test.pt",
                   help="query_split store for ch5 BM25 keywords")
    p.add_argument("--max_sessions",       type=int, default=0,
                   help="0 = all sessions")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
