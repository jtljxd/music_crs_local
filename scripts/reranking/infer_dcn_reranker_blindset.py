"""
scripts/reranking/infer_dcn_reranker_blindset.py
=================================================
用训练好的 DCNReranker 对 Blind-A 所有 session 的最后一个 turn 进行推理，
返回 top-20 music_id，输出格式与官方提交格式一致。

流程：
  1. 加载 DCNReranker checkpoint
  2. 用 MultiChannelRetrievalV2（20+路 RRF 合并）对每个 session 在线召回
  3. 对召回候选组装 7 个特征 → DCNReranker 打分
  4. 取 top-20，输出 JSON

召回数量：multi_channel 默认每路 topk=200，RRF 合并后约 350-500 个候选，
          由 --retrieval_topk 控制（传给 retrieval.retrieve()）。

Usage:
  python scripts/reranking/infer_dcn_reranker_blindset.py \\
      --checkpoint      checkpoints/dcn_reranker_best.pt \\
      --tag_proj        checkpoints/dcn_reranker_best_tag_proj.pt \\
      --query_emb_path  qwen/hist_conversation_embeddings_blindA_0.6b.pt \\
      --goal_emb_path   qwen/goal_embeddings_blindA_0.6b.pt \\
      --turn_query_emb_path bge/turn_query_embeddings_blinda.pt \\
      --bge_rich_path   bge/track_rich_embeddings.pt \\
      --bge_tag_path    bge/track_tag_embeddings.pt \\
      --cache_dir       qwen/retrieval_indices \\
      --retrieval_topk  500 \\
      --topk            20 \\
      --out             exp/inference/blindset_A/dcn_reranker_top20.json \\
      --device          cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from typing import Dict, List, Optional

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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _bucket_emb(value, n_bins: int, lo: float, hi: float) -> torch.Tensor:
    t = torch.zeros(n_bins)
    if value is None:
        return t
    v = float(value)
    idx = int((v - lo) / (hi - lo + 1e-9) * n_bins)
    t[max(0, min(n_bins - 1, idx))] = 1.0
    return t


def build_user_profile(user_id: str, user_meta: Dict[str, dict]) -> torch.Tensor:
    um = user_meta.get(user_id, {})
    age = um.get("age")
    age_bucket = (
        _bucket_emb(math.log1p(float(age)), 16, 0, math.log1p(100))
        if age is not None else torch.zeros(16)
    )
    country_emb = _bucket_emb(um.get("country_code_hash"), 16, 0, 1)
    gender_emb  = torch.zeros(2)
    if um.get("gender") == "male":     gender_emb[0] = 1.0
    elif um.get("gender") == "female": gender_emb[1] = 1.0
    lang_emb    = _bucket_emb(um.get("preferred_language_hash"),        4,  0, 1)
    culture_emb = _bucket_emb(um.get("preferred_musical_culture_hash"), 32, 0, 1)
    listen_cnt  = _bucket_emb(math.log1p(float(um.get("listen_count",  0) or 0)), 8, 0, math.log1p(10000))
    session_cnt = _bucket_emb(math.log1p(float(um.get("session_count", 0) or 0)), 8, 0, math.log1p(1000))
    return torch.cat([age_bucket, country_emb, gender_emb,
                      lang_emb, culture_emb, listen_cnt, session_cnt])  # [86]


def build_track_features(
    track_id:      str,
    index,                                     # IndexStore
    track_meta:    Dict[str, dict],
    bge_tag_store: Dict[str, torch.Tensor],
    tag_proj:      nn.Linear,
    proj_device:   torch.device,
) -> tuple:
    """Return (track_semantic[4352], track_context[137], track_cf[128])."""
    def _gv(mod, dim):
        v = index.get_vec(mod, track_id)
        return v.float() if v is not None else torch.zeros(dim)

    clap_emb   = _gv("audio",      512)
    siglip_emb = _gv("image",      768)
    attr_emb   = _gv("attributes", 1024)
    lyr_emb    = _gv("lyrics",     1024)
    meta_emb   = _gv("metadata",   1024)
    track_semantic = torch.cat([clap_emb, siglip_emb, attr_emb, lyr_emb, meta_emb])  # [4352]

    bge_v = bge_tag_store.get(track_id, torch.zeros(384)).float()
    with torch.no_grad():
        tag32 = F.normalize(
            tag_proj(bge_v.unsqueeze(0).to(proj_device)).squeeze(0), p=2, dim=0
        ).cpu()
    isrc32   = tag32.clone()
    artist32 = tag32.clone()
    album32  = tag32.clone()
    tm = track_meta.get(track_id, {})
    log_pop    = torch.tensor([math.log1p(float(tm.get("popularity",  0) or 0))])
    dur_bucket = _bucket_emb(tm.get("duration_ms"), 8, 30000, 600000)
    track_context = torch.cat([isrc32, tag32, artist32, album32, log_pop, dur_bucket])  # [137]

    tcf = index.get_vec("cf_bpr", track_id)
    track_cf = tcf.float() if tcf is not None else torch.zeros(128)

    return track_semantic, track_context, track_cf


# ── Main inference ─────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    from mcrs.retrieval_modules.multi_channel_v2 import (
        MultiChannelRetrievalV2,
        MultiChannelConfig,
    )
    from mcrs.reranking_modules.dcn_reranker import DCNReranker

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    logger.info("Device: %s", device)

    # ── Load DCNReranker ──────────────────────────────────────────────────
    logger.info("Loading DCNReranker from %s …", args.checkpoint)
    model = DCNReranker.load(
        args.checkpoint,
        device=str(device),
        cross_layers=args.cross_layers,
        deep_dims=tuple(args.deep_dims),
        dropout=0.0,
    )
    model.eval()

    tag_proj = nn.Linear(384, 32, bias=False).to(device)
    if args.tag_proj and os.path.exists(args.tag_proj):
        ckpt = torch.load(args.tag_proj, map_location=str(device), weights_only=True)
        tag_proj.load_state_dict(ckpt["state_dict"])
        logger.info("tag_proj loaded from %s", args.tag_proj)
    tag_proj.eval()

    # ── Build MultiChannelRetrievalV2 (20+ 路 RRF 合并) ──────────────────
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
        genre_emb_path      = args.turn_query_emb_path,  # turn-query BGE path
        decade_emb_path     = args.decade_emb_path,
        device              = str(device),
    )
    retrieval = MultiChannelRetrievalV2.build(cfg)
    # IndexStore is shared inside the retrieval system
    index = retrieval.index
    valid_tracks = set(index.track_ids)
    logger.info("Index: %d tracks, channels: %s",
                len(valid_tracks), retrieval.channel_names())

    # ── Aux stores for DCN features ───────────────────────────────────────
    def _lpt(p, name):
        if p and os.path.exists(p):
            d = torch.load(p, map_location="cpu", weights_only=True)
            logger.info("%s: %d entries", name, len(d))
            return d
        logger.warning("Not found: %s → %s", name, p)
        return {}

    goal_store    = _lpt(args.goal_emb_path,  "goal_emb")
    query_store   = _lpt(args.query_emb_path, "query_emb")
    bge_tag_store = _lpt(args.bge_tag_path,   "bge_tag")

    # ── Track metadata ─────────────────────────────────────────────────────
    logger.info("Loading track metadata …")
    track_meta: Dict[str, dict] = {}
    try:
        tm_ds = load_dataset(args.track_metadata_dataset)
        for sp in tm_ds:
            for row in tm_ds[sp]:
                tid = str(row.get("track_id", ""))
                if tid:
                    track_meta[tid] = {
                        "popularity":  row.get("popularity"),
                        "duration_ms": row.get("duration_ms"),
                    }
    except Exception as e:
        logger.warning("Track metadata load failed: %s", e)

    # ── User metadata ──────────────────────────────────────────────────────
    logger.info("Loading user metadata …")
    user_meta: Dict[str, dict] = {}
    try:
        u_ds = load_dataset(args.user_metadata_dataset)
        for sp in u_ds:
            for row in u_ds[sp]:
                uid = str(row.get("user_id", ""))
                if uid:
                    user_meta[uid] = dict(row)
    except Exception as e:
        logger.warning("User metadata load failed: %s", e)

    # ── User CF-BPR ────────────────────────────────────────────────────────
    logger.info("Loading user CF-BPR embeddings …")
    user_cf_store: Dict[str, torch.Tensor] = {}
    try:
        ue_ds = load_dataset(args.user_emb_dataset)
        for sp in ue_ds:
            for row in ue_ds[sp]:
                uid = str(row.get("user_id", ""))
                v   = row.get("cf-bpr")
                if uid and v is not None:
                    t = torch.tensor(v, dtype=torch.float32)
                    if t.numel() > 0:
                        user_cf_store[uid] = t
        logger.info("  %d users with CF emb", len(user_cf_store))
    except Exception as e:
        logger.warning("User CF load failed: %s", e)

    # ── Load Blind-A ───────────────────────────────────────────────────────
    logger.info("Loading Blind-A dataset …")
    blind_ds = load_dataset(args.blind_dataset, split="test")

    fallback_tids: List[str] = list(index.track_ids)[:args.topk]
    results: List[dict] = []
    proj_device = next(tag_proj.parameters()).device

    for item in tqdm(blind_ds, desc="Blind-A inference"):
        session_id = str(item["session_id"])
        user_id    = str(item.get("user_id", ""))
        convs      = item["conversations"]

        # ── Find last user turn ───────────────────────────────────────────
        user_turns = sorted(
            [int(c["turn_number"]) for c in convs if c.get("role") == "user"]
        )
        if not user_turns:
            results.append({
                "session_id":          session_id,
                "user_id":             user_id,
                "turn_number":         1,
                "predicted_track_ids": fallback_tids,
                "predicted_response":  "",
            })
            continue
        target_turn = user_turns[-1]

        # ── Multi-channel online retrieval (RRF merged) ───────────────────
        try:
            ctx = retrieval.build_context(
                session_id   = session_id,
                turn_number  = target_turn,
                session_data = item,
                user_id      = user_id,
            )
            ret_results = retrieval.retrieve(ctx, topk=args.retrieval_topk)
            cands = [tid for tid in ret_results.get("merged", []) if tid in valid_tracks]
        except Exception as e:
            logger.warning("Retrieval failed for %s turn %d: %s",
                           session_id, target_turn, e)
            cands = []

        if not cands:
            logger.debug("Fallback for %s (no retrieval).", session_id)
            cands = fallback_tids

        logger.debug("%s turn %d: %d candidates", session_id, target_turn, len(cands))

        # ── User context features (shared across all candidates) ──────────
        user_profile = build_user_profile(user_id, user_meta).unsqueeze(0)  # [1, 86]

        ucf = user_cf_store.get(user_id)
        user_cf = (ucf.float() if ucf is not None else torch.zeros(128)).unsqueeze(0)

        goal_emb = goal_store.get(session_id)
        goal_emb = goal_emb.float() if goal_emb is not None else torch.zeros(1024)
        conv_goal = torch.cat([torch.zeros(8), goal_emb, torch.zeros(4)]).unsqueeze(0)  # [1, 1036]

        q = None
        for t in range(target_turn, -1, -1):
            q = query_store.get(f"{session_id}_{t}_query")
            if q is None:
                q = query_store.get(f"{session_id}_{t}")
            if q is not None:
                break
        query_emb_vec = (q.float() if q is not None else torch.zeros(1024)).unsqueeze(0)  # [1, 1024]

        # ── Score all candidates in mini-batches ──────────────────────────
        scored: List[tuple] = []
        with torch.no_grad():
            for i in range(0, len(cands), args.score_batch_size):
                chunk = cands[i: i + args.score_batch_size]
                B = len(chunk)

                t_sem_list, t_ctx_list, t_cf_list = [], [], []
                for tid in chunk:
                    ts, tc, tf = build_track_features(
                        tid, index, track_meta, bge_tag_store, tag_proj, proj_device
                    )
                    t_sem_list.append(ts)
                    t_ctx_list.append(tc)
                    t_cf_list.append(tf)

                scores = model.encode(
                    user_profile.expand(B, -1).to(device),
                    user_cf.expand(B, -1).to(device),
                    conv_goal.expand(B, -1).to(device),
                    query_emb_vec.expand(B, -1).to(device),
                    torch.stack(t_sem_list).to(device),
                    torch.stack(t_ctx_list).to(device),
                    torch.stack(t_cf_list).to(device),
                ).squeeze(1).cpu()  # [B]

                for tid, sc in zip(chunk, scores.tolist()):
                    scored.append((sc, tid))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_ids = [tid for _, tid in scored[: args.topk]]

        results.append({
            "session_id":          session_id,
            "user_id":             user_id,
            "turn_number":         target_turn,
            "predicted_track_ids": top_ids,
            "predicted_response":  "",
        })

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d sessions → %s", len(results), args.out)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DCNReranker inference on Blind-A (multi-channel retrieval)")
    # Model
    p.add_argument("--checkpoint",   required=True, type=str)
    p.add_argument("--tag_proj",     type=str, default=None)
    p.add_argument("--cross_layers", type=int, default=3)
    p.add_argument("--deep_dims",    type=int, nargs="+", default=[512, 256, 128])
    # Retrieval (multi-channel)
    p.add_argument("--query_emb_path",      required=True, type=str,
                   help="Blind-A dialogue embeddings .pt")
    p.add_argument("--goal_emb_path",       type=str, default=None)
    p.add_argument("--turn_query_emb_path", type=str, default=None)
    p.add_argument("--decade_emb_path",     type=str, default=None)
    p.add_argument("--bge_rich_path",       type=str, default="bge/track_rich_embeddings.pt")
    p.add_argument("--bge_tag_path",        type=str, default="bge/track_tag_embeddings.pt")
    p.add_argument("--cache_dir",           type=str, default="qwen/retrieval_indices")
    p.add_argument("--retrieval_topk",      type=int, default=500,
                   help="Max candidates per session from multi-channel RRF merge")
    # Output
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--out",  type=str,
                   default="exp/inference/blindset_A/dcn_reranker_top20.json")
    p.add_argument("--score_batch_size", type=int, default=512)
    # Datasets
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
    run(parse_args())
