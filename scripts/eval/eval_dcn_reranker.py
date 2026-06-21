"""
scripts/eval/eval_dcn_reranker.py
==================================
评估 DCNReranker 精排效果：
  1. 用 MultiChannelRetrievalV2 召回 top-500 候选
  2. 用 DCNReranker 精排，输出 top-20
  3. 计算 NDCG@20 / Recall@{20,50,100}

评估集：
  --split test       : test split 前 100 个 session 的所有 music turn
  --split blinda     : Blind-A 每个 session 的非最后一轮 music turn

Usage:
  python scripts/eval/eval_dcn_reranker.py \\
      --split test \\
      --checkpoint      checkpoints/dcn_reranker_best.pt \\
      --tag_proj        checkpoints/dcn_reranker_best_tag_proj.pt \\
      --query_emb_path  qwen/dialogue_embeddings_test_0.6b.pt \\
      --goal_emb_path   qwen/goal_embeddings_test_0.6b.pt \\
      --turn_query_emb_path bge/turn_query_embeddings_test.pt \\
      --bge_rich_path   bge/track_rich_embeddings.pt \\
      --bge_tag_path    bge/track_tag_embeddings.pt \\
      --cache_dir       qwen/retrieval_indices \\
      --retrieval_topk  500 \\
      --max_sessions    100 \\
      --device          cuda
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


# ── Feature helpers ────────────────────────────────────────────────────────────

def _bucket_emb(value, n_bins: int, lo: float, hi: float) -> torch.Tensor:
    t = torch.zeros(n_bins)
    if value is None:
        return t
    v = float(value)
    idx = int((v - lo) / (hi - lo + 1e-9) * n_bins)
    t[max(0, min(n_bins - 1, idx))] = 1.0
    return t


def build_user_profile(user_id: str, user_meta: Dict) -> torch.Tensor:
    um = user_meta.get(user_id, {})
    age = um.get("age")
    age_bucket = (
        _bucket_emb(math.log1p(float(age)), 16, 0, math.log1p(100))
        if age is not None else torch.zeros(16)
    )
    country_emb  = _bucket_emb(um.get("country_code_hash"),              16, 0, 1)
    gender_emb   = torch.zeros(2)
    if um.get("gender") == "male":     gender_emb[0] = 1.0
    elif um.get("gender") == "female": gender_emb[1] = 1.0
    lang_emb     = _bucket_emb(um.get("preferred_language_hash"),         4, 0, 1)
    culture_emb  = _bucket_emb(um.get("preferred_musical_culture_hash"), 32, 0, 1)
    listen_cnt   = _bucket_emb(math.log1p(float(um.get("listen_count",  0) or 0)),
                               8, 0, math.log1p(10000))
    session_cnt  = _bucket_emb(math.log1p(float(um.get("session_count", 0) or 0)),
                               8, 0, math.log1p(1000))
    return torch.cat([age_bucket, country_emb, gender_emb,
                      lang_emb, culture_emb, listen_cnt, session_cnt])  # [86]


def build_track_features(
    track_id: str,
    index,
    track_meta: Dict,
    bge_tag_store: Dict,
    tag_proj: nn.Linear,
    proj_device,
    ch_results: Dict = None,
) -> tuple:
    """Return (audio[512], image[768], attr[1024], lyrics[1024], meta[1024], context[137], cf[128], ret[N_CH*2])."""
    from mcrs.reranking_modules.dcn_reranker import build_retrieval_feat

    def _gv(mod, dim):
        v = index.get_vec(mod, track_id)
        return v.float() if v is not None else torch.zeros(dim)

    audio    = _gv("audio",      512)
    image    = _gv("image",      768)
    attr     = _gv("attributes", 1024)
    lyrics   = _gv("lyrics",     1024)
    meta_emb = _gv("metadata",   1024)

    bge_v = bge_tag_store.get(track_id, torch.zeros(384)).float()
    with torch.no_grad():
        tag32 = F.normalize(
            tag_proj(bge_v.unsqueeze(0).to(proj_device)).squeeze(0), p=2, dim=0
        ).cpu()
    tm = track_meta.get(track_id, {})
    log_pop    = torch.tensor([math.log1p(float(tm.get("popularity",  0) or 0))])
    dur_bucket = _bucket_emb(tm.get("duration_ms"), 8, 30000, 600000)
    context = torch.cat([tag32.clone(), tag32, tag32.clone(), tag32.clone(),
                         log_pop, dur_bucket])  # [137]

    tcf = index.get_vec("cf_bpr", track_id)
    cf  = tcf.float() if tcf is not None else torch.zeros(128)

    ret_feat = build_retrieval_feat(track_id, ch_results or {})

    return audio, image, attr, lyrics, meta_emb, context, cf, ret_feat


def score_candidates(
    model, tag_proj, device,
    user_profile, user_cf, conv_goal, query_emb_vec,
    cands: List[str],
    index, track_meta, bge_tag_store,
    ch_results: Dict = None,
    score_batch_size: int = 512,
) -> List[tuple]:
    """Score all candidates, return sorted [(score, track_id)] descending."""
    proj_device = next(tag_proj.parameters()).device
    scored = []
    with torch.no_grad():
        for i in range(0, len(cands), score_batch_size):
            chunk = cands[i: i + score_batch_size]
            B = len(chunk)
            feats = [build_track_features(tid, index, track_meta,
                                          bge_tag_store, tag_proj, proj_device,
                                          ch_results)
                     for tid in chunk]
            t_audio, t_image, t_attr, t_lyrics, t_meta, t_ctx, t_cf, t_ret = zip(*feats)
            sc = model.encode(
                user_profile.expand(B, -1).to(device),
                user_cf.expand(B, -1).to(device),
                conv_goal.expand(B, -1).to(device),
                query_emb_vec.expand(B, -1).to(device),
                torch.stack(t_audio).to(device),
                torch.stack(t_image).to(device),
                torch.stack(t_attr).to(device),
                torch.stack(t_lyrics).to(device),
                torch.stack(t_meta).to(device),
                torch.stack(t_ctx).to(device),
                torch.stack(t_cf).to(device),
                torch.stack(t_ret).to(device),
            ).squeeze(1).cpu()
            for tid, s in zip(chunk, sc.tolist()):
                scored.append((s, tid))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    from mcrs.retrieval_modules.multi_channel_v2 import MultiChannelRetrievalV2, MultiChannelConfig
    from mcrs.reranking_modules.dcn_reranker import DCNReranker

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    logger.info("Device: %s", device)

    # ── Load model ────────────────────────────────────────────────────────
    logger.info("Loading DCNReranker …")
    model = DCNReranker.load(
        args.checkpoint, device=str(device),
        cross_layers=args.cross_layers,
        deep_dims=tuple(args.deep_dims), dropout=0.0,
    )
    model.eval()
    tag_proj = nn.Linear(384, 32, bias=False).to(device)
    if args.tag_proj and os.path.exists(args.tag_proj):
        ckpt = torch.load(args.tag_proj, map_location=str(device), weights_only=True)
        tag_proj.load_state_dict(ckpt["state_dict"])
    tag_proj.eval()

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

    # ── Aux feature stores ────────────────────────────────────────────────
    def _lpt(p, name):
        if p and os.path.exists(p):
            d = torch.load(p, map_location="cpu", weights_only=True)
            logger.info("%s: %d entries", name, len(d))
            return d
        return {}

    goal_store    = _lpt(args.goal_emb_path,  "goal_emb")
    query_store   = _lpt(args.query_emb_path, "query_emb")
    bge_tag_store = _lpt(args.bge_tag_path,   "bge_tag")

    # ── Track / user metadata ─────────────────────────────────────────────
    logger.info("Loading track metadata …")
    track_meta: Dict = {}
    try:
        for sp in load_dataset(args.track_metadata_dataset):
            for row in load_dataset(args.track_metadata_dataset)[sp]:
                tid = str(row.get("track_id", ""))
                if tid:
                    track_meta[tid] = {"popularity": row.get("popularity"),
                                       "duration_ms": row.get("duration_ms")}
    except Exception as e:
        logger.warning("track meta: %s", e)

    logger.info("Loading user metadata …")
    user_meta: Dict = {}
    try:
        for sp in load_dataset(args.user_metadata_dataset):
            for row in load_dataset(args.user_metadata_dataset)[sp]:
                uid = str(row.get("user_id", ""))
                if uid:
                    user_meta[uid] = dict(row)
    except Exception as e:
        logger.warning("user meta: %s", e)

    logger.info("Loading user CF-BPR …")
    user_cf_store: Dict[str, torch.Tensor] = {}
    try:
        for sp in load_dataset(args.user_emb_dataset):
            for row in load_dataset(args.user_emb_dataset)[sp]:
                uid = str(row.get("user_id", ""))
                v   = row.get("cf-bpr")
                if uid and v is not None:
                    t = torch.tensor(v, dtype=torch.float32)
                    if t.numel() > 0:
                        user_cf_store[uid] = t
        logger.info("  %d users with CF emb", len(user_cf_store))
    except Exception as e:
        logger.warning("user CF: %s", e)

    # ── Load dataset & determine turns to evaluate ────────────────────────
    is_blinda = args.split.lower() in ("blinda", "blind_a", "blind-a")
    if is_blinda:
        ds_name = args.blind_dataset
        ds      = load_dataset(ds_name, split="test")
        logger.info("Blind-A: %d sessions, eval non-last music turns", len(ds))
    else:
        ds_name = args.conv_dataset
        ds      = load_dataset(ds_name, split="test")
        logger.info("Test split: %d sessions total, using first %d",
                    len(ds), args.max_sessions)

    # ── Metrics accumulators ──────────────────────────────────────────────
    # Both channels: retrieval (recall@500) and reranker (top-20)
    ret_ndcg, ret_recalls = [], {k: [] for k in RECALL_KS}
    rnk_ndcg, rnk_recalls = [], {k: [] for k in RECALL_KS}
    total_turns = 0

    n_sessions = min(args.max_sessions, len(ds)) if not is_blinda else len(ds)

    for idx in tqdm(range(n_sessions), desc=f"Eval ({args.split})"):
        item       = ds[idx]
        session_id = str(item.get("session_id") or item.get("id") or idx)
        user_id    = str(item.get("user_id", ""))
        convs      = item.get("conversations", [])

        # Collect (turn_number → gt_track_id) pairs to evaluate
        music_turns: Dict[int, str] = {
            int(c["turn_number"]): c["content"]
            for c in convs
            if c.get("role") == "music" and c.get("content")
        }
        if not music_turns:
            continue

        if is_blinda:
            # Blind-A: skip the last music turn (that's the prediction target)
            if len(music_turns) <= 1:
                continue
            last_t = max(music_turns.keys())
            music_turns = {t: tid for t, tid in music_turns.items() if t != last_t}
        else:
            # Test: evaluate all music turns (or only last, by flag)
            pass

        for turn_number, gt_track_id in music_turns.items():
            total_turns += 1

            # ── Step 1: Multi-channel retrieval → top-500 ─────────────────
            ch_results = {}
            try:
                ctx       = retrieval.build_context(
                    session_id=session_id, turn_number=turn_number,
                    session_data=item, user_id=user_id,
                )
                ret_res   = retrieval.retrieve(ctx, topk=args.retrieval_topk)
                ch_results = {k: v for k, v in ret_res.items()}  # keep per-channel info
                cands     = [tid for tid in ret_res.get("merged", [])
                             if tid in valid_tracks]
            except Exception as e:
                logger.debug("Retrieval failed %s t%d: %s", session_id, turn_number, e)
                cands = []

            # Retrieval metrics (before reranking)
            for k in RECALL_KS:
                ret_recalls[k].append(recall_at_k(cands, gt_track_id, k))
            ret_ndcg.append(ndcg_at_k(cands, gt_track_id, NDCG_K))

            if not cands:
                # No candidates → reranker gets nothing
                for k in RECALL_KS:
                    rnk_recalls[k].append(0.0)
                rnk_ndcg.append(0.0)
                continue

            # ── Step 2: Build user context features ───────────────────────
            user_profile = build_user_profile(user_id, user_meta).unsqueeze(0)  # [1,86]
            ucf          = user_cf_store.get(user_id)
            user_cf_t    = (ucf.float() if ucf is not None else torch.zeros(128)).unsqueeze(0)

            ge = goal_store.get(session_id)
            ge = ge.float() if ge is not None else torch.zeros(1024)
            conv_goal = torch.cat([torch.zeros(8), ge, torch.zeros(4)]).unsqueeze(0)  # [1,1036]

            q = query_store.get(f"{session_id}_{turn_number}_query")
            if q is None:
                q = query_store.get(f"{session_id}_{turn_number}")
            query_emb_vec = (q.float() if q is not None else torch.zeros(1024)).unsqueeze(0)

            # ── Step 3: DCNReranker → top-20 ──────────────────────────────
            scored = score_candidates(
                model, tag_proj, device,
                user_profile, user_cf_t, conv_goal, query_emb_vec,
                cands, index, track_meta, bge_tag_store,
                ch_results=ch_results,
                score_batch_size=args.score_batch_size,
            )
            reranked = [tid for _, tid in scored]

            for k in RECALL_KS:
                rnk_recalls[k].append(recall_at_k(reranked, gt_track_id, k))
            rnk_ndcg.append(ndcg_at_k(reranked, gt_track_id, NDCG_K))

    # ── Print results ─────────────────────────────────────────────────────
    if total_turns == 0:
        logger.error("No turns evaluated!")
        return

    def _avg(lst): return sum(lst) / len(lst) if lst else 0.0

    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"Split: {args.split}  |  Sessions: {n_sessions}  |  Turns: {total_turns}")
    lines.append(f"{'='*60}")
    lines.append(f"{'Metric':<20} {'Retrieval(500)':>16} {'Reranker(top20)':>16}")
    lines.append(f"{'-'*54}")
    lines.append(f"{'NDCG@20':<20} {_avg(ret_ndcg):>16.4f} {_avg(rnk_ndcg):>16.4f}")
    for k in RECALL_KS:
        label = f"Recall@{k}"
        lines.append(f"{label:<20} {_avg(ret_recalls[k]):>16.4f} {_avg(rnk_recalls[k]):>16.4f}")
    lines.append(f"{'='*60}\n")
    report = "\n".join(lines)
    print(report)

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs("exp/eval", exist_ok=True)
    out_prefix = f"exp/eval/dcn_reranker_{args.split}"
    with open(f"{out_prefix}.txt", "w") as f:
        f.write(report)

    result_dict = {
        "split":        args.split,
        "n_sessions":   n_sessions,
        "total_turns":  total_turns,
        "retrieval": {
            "ndcg@20": _avg(ret_ndcg),
            **{f"recall@{k}": _avg(ret_recalls[k]) for k in RECALL_KS},
        },
        "reranker": {
            "ndcg@20": _avg(rnk_ndcg),
            **{f"recall@{k}": _avg(rnk_recalls[k]) for k in RECALL_KS},
        },
    }
    with open(f"{out_prefix}.json", "w") as f:
        json.dump(result_dict, f, indent=2)
    logger.info("Saved → %s.txt / .json", out_prefix)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Eval DCNReranker: recall@500 → rerank → NDCG@20")
    # Split
    p.add_argument("--split",         type=str, default="test",
                   help="'test' (first --max_sessions sessions) or 'blinda' (non-last turns)")
    p.add_argument("--max_sessions",  type=int, default=100,
                   help="Max sessions for test split")
    # Model
    p.add_argument("--checkpoint",    required=True, type=str)
    p.add_argument("--tag_proj",      type=str, default=None)
    p.add_argument("--cross_layers",  type=int, default=3)
    p.add_argument("--deep_dims",     type=int, nargs="+", default=[512, 256, 128])
    # Retrieval
    p.add_argument("--query_emb_path",      required=True, type=str)
    p.add_argument("--goal_emb_path",       type=str, default=None)
    p.add_argument("--turn_query_emb_path", type=str, default=None)
    p.add_argument("--decade_emb_path",     type=str, default=None)
    p.add_argument("--bge_rich_path",       type=str, default="bge/track_rich_embeddings.pt")
    p.add_argument("--bge_tag_path",        type=str, default="bge/track_tag_embeddings.pt")
    p.add_argument("--cache_dir",           type=str, default="qwen/retrieval_indices")
    p.add_argument("--retrieval_topk",      type=int, default=500)
    p.add_argument("--score_batch_size",    type=int, default=512)
    # Datasets
    p.add_argument("--conv_dataset",           type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    p.add_argument("--blind_dataset",          type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
    p.add_argument("--track_emb_dataset",      type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    p.add_argument("--track_metadata_dataset", type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    p.add_argument("--user_metadata_dataset",  type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-User-Metadata")
    p.add_argument("--user_emb_dataset",       type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-User-Embeddings")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
