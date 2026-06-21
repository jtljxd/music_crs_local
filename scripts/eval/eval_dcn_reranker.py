"""
scripts/eval/eval_dcn_reranker.py
==================================
Evaluate DCNReranker v5:
  1. MultiChannelRetrievalV2 → top-500
  2. DCNReranker → top-20
  3. NDCG@20 / Recall@{20,50,100}

Feature extraction uses shared functions from dcn_reranker.py [P6].

Splits:
  --split test   : first --max_sessions sessions, all music turns
  --split blinda : all sessions, non-last music turns only

Usage:
  python scripts/eval/eval_dcn_reranker.py \\
      --split test --max_sessions 100 \\
      --checkpoint     checkpoints/dcn_reranker_best.pt \\
      --query_emb_path qwen/dialogue_embeddings_test_0.6b.pt \\
      --goal_emb_path  qwen/goal_embeddings_test_0.6b.pt \\
      --bge_tag_path   bge/track_tag_embeddings.pt \\
      --cache_dir      qwen/retrieval_indices \\
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from typing import Dict, List

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RECALL_KS = [20, 50, 100]
NDCG_K    = 20


# ── Metrics ────────────────────────────────────────────────────────────────────

def ndcg_at_k(candidates: List[str], gt: str, k: int) -> float:
    for i, tid in enumerate(candidates[:k]):
        if tid == gt:
            return 1.0 / math.log2(i + 2)
    return 0.0


def recall_at_k(candidates: List[str], gt: str, k: int) -> float:
    return 1.0 if gt in candidates[:k] else 0.0


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_candidates(
    model, tag_proj, device,
    user_profile_t, user_cf_t, goal_emb_t, query_emb_t,
    cands: List[str],
    index, track_meta, bge_tag_store,
    score_batch_size: int = 512,
) -> List[tuple]:
    """Score all candidates, return sorted [(score, track_id)] descending.
    Uses shared extract_track_features from dcn_reranker [P6].
    """
    from mcrs.reranking_modules.dcn_reranker import extract_track_features

    proj_device = next(tag_proj.parameters()).device
    scored = []
    with torch.no_grad():
        for i in range(0, len(cands), score_batch_size):
            chunk = cands[i: i + score_batch_size]
            B     = len(chunk)
            feats = [extract_track_features(
                tid, index, track_meta, bge_tag_store, tag_proj, proj_device
            ) for tid in chunk]
            # unpack: audio[512], image[768], attr[1024], lyrics[1024],
            #         meta[1024], context[137], cf[128]
            t_audio, t_image, t_attr, t_lyrics, t_meta, t_ctx, t_cf = zip(*feats)
            sc = model.encode(
                user_profile_t.expand(B, -1).to(device),
                user_cf_t.expand(B, -1).to(device),
                goal_emb_t.expand(B, -1).to(device),
                query_emb_t.expand(B, -1).to(device),
                torch.stack(t_audio).to(device),
                torch.stack(t_image).to(device),
                torch.stack(t_attr).to(device),
                torch.stack(t_lyrics).to(device),
                torch.stack(t_meta).to(device),
                torch.stack(t_ctx).to(device),
                torch.stack(t_cf).to(device),
            ).squeeze(1).cpu()
            for tid, s in zip(chunk, sc.tolist()):
                scored.append((s, tid))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    from mcrs.retrieval_modules.multi_channel_v2 import MultiChannelRetrievalV2, MultiChannelConfig
    from mcrs.reranking_modules.dcn_reranker import (
        DCNReranker, extract_user_profile, extract_goal_emb, extract_query_emb,
    )

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    logger.info("Device: %s", device)

    # ── Load model ────────────────────────────────────────────────────────
    logger.info("Loading DCNReranker …")
    ckpt  = torch.load(args.checkpoint, map_location=str(device), weights_only=False)
    model = DCNReranker(
        cross_layers=args.cross_layers,
        deep_dims=tuple(args.deep_dims), dropout=0.0,
    )
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval().to(device)

    tag_proj = nn.Linear(384, 32, bias=False).to(device)
    if "tag_proj" in ckpt:
        tag_proj.load_state_dict(ckpt["tag_proj"])
    tag_proj.eval()
    logger.info("Model loaded. Params: %d", model.count_params())

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
        device              = str(device),
    )
    retrieval    = MultiChannelRetrievalV2.build(cfg)
    index        = retrieval.index
    valid_tracks = set(index.track_ids)

    # ── Feature stores ────────────────────────────────────────────────────
    def _lpt(p, name):
        if p and os.path.exists(p):
            d = torch.load(p, map_location="cpu", weights_only=True)
            logger.info("%s: %d entries", name, len(d))
            return d
        return {}

    goal_store    = _lpt(args.goal_emb_path,  "goal_emb")
    query_store   = _lpt(args.query_emb_path, "query_emb")
    bge_tag_store = _lpt(args.bge_tag_path,   "bge_tag")

    # ── Metadata ──────────────────────────────────────────────────────────
    logger.info("Loading track metadata …")
    track_meta: Dict = {}
    for sp, rows in load_dataset(args.track_metadata_dataset).items():
        for row in rows:
            tid = str(row.get("track_id", ""))
            if tid:
                track_meta[tid] = {"popularity": row.get("popularity"),
                                   "duration_ms": row.get("duration_ms")}

    logger.info("Loading user metadata …")
    user_meta: Dict = {}
    try:
        for sp, rows in load_dataset(args.user_metadata_dataset).items():
            for row in rows:
                uid = str(row.get("user_id", ""))
                if uid: user_meta[uid] = dict(row)
    except Exception as e:
        logger.warning("user meta: %s", e)

    logger.info("Loading user CF-BPR …")
    user_cf_store: Dict[str, torch.Tensor] = {}
    try:
        for sp, rows in load_dataset(args.user_emb_dataset).items():
            for row in rows:
                uid = str(row.get("user_id", ""))
                v   = row.get("cf-bpr")
                if uid and v is not None:
                    t = torch.tensor(v, dtype=torch.float32)
                    if t.numel() > 0: user_cf_store[uid] = t
        logger.info("  %d users with CF emb", len(user_cf_store))
    except Exception as e:
        logger.warning("user CF: %s", e)

    # ── Load evaluation dataset ───────────────────────────────────────────
    is_blinda = args.split.lower() in ("blinda", "blind_a", "blind-a")
    ds = load_dataset(
        args.blind_dataset if is_blinda else args.conv_dataset,
        split="test",
    )
    n_sessions = len(ds) if is_blinda else min(args.max_sessions, len(ds))
    logger.info("Evaluating %d sessions (%s) …", n_sessions, args.split)

    # ── Metrics ───────────────────────────────────────────────────────────
    ret_ndcg, ret_recalls  = [], {k: [] for k in RECALL_KS}
    rnk_ndcg, rnk_recalls  = [], {k: [] for k in RECALL_KS}
    total_turns = 0
    missing_query_cnt = 0

    for idx in tqdm(range(n_sessions), desc=f"Eval ({args.split})"):
        item       = ds[idx]
        session_id = str(item.get("session_id") or item.get("id") or idx)
        user_id    = str(item.get("user_id", ""))

        music_turns: Dict[int, str] = {
            int(c["turn_number"]): c["content"]
            for c in item.get("conversations", [])
            if c.get("role") == "music" and c.get("content")
        }
        if not music_turns:
            continue

        if is_blinda:
            if len(music_turns) <= 1:
                continue
            last_t = max(music_turns.keys())
            music_turns = {t: tid for t, tid in music_turns.items() if t != last_t}

        for turn_number, gt_track_id in music_turns.items():
            total_turns += 1

            # ── Step 1: Retrieval → top-500 ───────────────────────────────
            cands = []
            try:
                ctx     = retrieval.build_context(
                    session_id=session_id, turn_number=turn_number,
                    session_data=item, user_id=user_id,
                )
                ret_res = retrieval.retrieve(ctx, topk=args.retrieval_topk)
                cands   = [t for t in ret_res.get("merged", []) if t in valid_tracks]
            except Exception as e:
                logger.debug("Retrieval failed %s t%d: %s", session_id, turn_number, e)

            for k in RECALL_KS:
                ret_recalls[k].append(recall_at_k(cands, gt_track_id, k))
            ret_ndcg.append(ndcg_at_k(cands, gt_track_id, NDCG_K))

            if not cands:
                for k in RECALL_KS: rnk_recalls[k].append(0.0)
                rnk_ndcg.append(0.0)
                continue

            # ── Step 2: User context features (shared functions) [P6] ─────
            user_profile_t = extract_user_profile(user_id, user_meta).unsqueeze(0)
            ucf = user_cf_store.get(user_id)
            user_cf_t = (ucf.float() if ucf is not None else torch.zeros(128)).unsqueeze(0)

            # [P4] Pure goal emb
            goal_emb_t = extract_goal_emb(goal_store, session_id).unsqueeze(0)

            # [P5] Strict per-turn query key
            query_vec = extract_query_emb(query_store, session_id, turn_number)
            if query_vec.abs().sum() == 0:
                missing_query_cnt += 1
            query_emb_t = query_vec.unsqueeze(0)

            # ── Step 3: Reranker → top-20 ─────────────────────────────────
            scored   = score_candidates(
                model, tag_proj, device,
                user_profile_t, user_cf_t, goal_emb_t, query_emb_t,
                cands, index, track_meta, bge_tag_store,
                args.score_batch_size,
            )
            reranked = [tid for _, tid in scored]

            for k in RECALL_KS:
                rnk_recalls[k].append(recall_at_k(reranked, gt_track_id, k))
            rnk_ndcg.append(ndcg_at_k(reranked, gt_track_id, NDCG_K))

    # ── Print results ─────────────────────────────────────────────────────
    if total_turns == 0:
        logger.error("No turns evaluated!"); return

    def _avg(lst): return sum(lst) / len(lst) if lst else 0.0

    logger.info("Missing per-turn query keys: %d / %d", missing_query_cnt, total_turns)

    lines = [
        f"\n{'='*62}",
        f"Split: {args.split}  |  Sessions: {n_sessions}  |  Turns: {total_turns}",
        f"{'='*62}",
        f"{'Metric':<22} {'Retrieval(500)':>16} {'Reranker(top20)':>16}",
        f"{'-'*56}",
        f"{'NDCG@20':<22} {_avg(ret_ndcg):>16.4f} {_avg(rnk_ndcg):>16.4f}",
        *[f"{'Recall@'+str(k):<22} {_avg(ret_recalls[k]):>16.4f} {_avg(rnk_recalls[k]):>16.4f}"
          for k in RECALL_KS],
        f"{'='*62}\n",
    ]
    report = "\n".join(lines)
    print(report)

    os.makedirs("exp/eval", exist_ok=True)
    prefix = f"exp/eval/dcn_reranker_{args.split}"
    with open(f"{prefix}.txt", "w") as f:
        f.write(report)
    with open(f"{prefix}.json", "w") as f:
        json.dump({
            "split": args.split, "n_sessions": n_sessions,
            "total_turns": total_turns,
            "missing_query_pct": missing_query_cnt / max(total_turns, 1),
            "retrieval": {"ndcg@20": _avg(ret_ndcg),
                          **{f"recall@{k}": _avg(ret_recalls[k]) for k in RECALL_KS}},
            "reranker":  {"ndcg@20": _avg(rnk_ndcg),
                          **{f"recall@{k}": _avg(rnk_recalls[k]) for k in RECALL_KS}},
        }, f, indent=2)
    logger.info("Saved → %s.txt / .json", prefix)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Eval DCNReranker v5")
    p.add_argument("--split",         default="test",
                   help="'test' or 'blinda'")
    p.add_argument("--max_sessions",  type=int, default=100)
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--cross_layers",  type=int, default=3)
    p.add_argument("--deep_dims",     type=int, nargs="+", default=[256, 256, 128])
    p.add_argument("--query_emb_path",      required=True)
    p.add_argument("--goal_emb_path",       default=None)
    p.add_argument("--turn_query_emb_path", default=None)
    p.add_argument("--decade_emb_path",     default=None)
    p.add_argument("--bge_rich_path",       default="bge/track_rich_embeddings.pt")
    p.add_argument("--bge_tag_path",        default="bge/track_tag_embeddings.pt")
    p.add_argument("--cache_dir",           default="qwen/retrieval_indices")
    p.add_argument("--retrieval_topk",      type=int, default=500)
    p.add_argument("--score_batch_size",    type=int, default=512)
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
