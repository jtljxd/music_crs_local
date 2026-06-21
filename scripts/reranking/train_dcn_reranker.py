"""
scripts/reranking/train_dcn_reranker.py
========================================
Training script for DCNReranker.

Loss: Listwise softmax cross-entropy with hard negatives from retrieval candidates.
  - 每个 sample = (session, turn, gt_track, [neg_track_1, ..., neg_track_K])
  - 负样本从该 (session, turn) 的召回候选里随机采 K 个（排除 gt_track）
  - score_list[i] = score(user, track_i), shape [1+K]
  - loss = cross_entropy(score_list, label=0)   正样本在第 0 位

Usage:
  python scripts/reranking/train_dcn_reranker.py \\
      --query_emb_path     qwen/hist_conversation_embeddings_train_0.6b.pt \\
      --goal_emb_path      qwen/goal_embeddings_train_0.6b.pt \\
      --retrieval_path     qwen/retrieval_train_candidates.pt \\
      --bge_tag_path       bge/track_tag_embeddings.pt \\
      --cache_dir          qwen/retrieval_indices \\
      --out                checkpoints/dcn_reranker_best.pt \\
      --epochs 20 --batch_size 64 --num_neg 15 --lr 1e-3 \\
      --device cuda
"""

from __future__ import annotations

import sys, os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import json
import logging
import math
import random
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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


def _get_track_features(
    track_id: str,
    index,
    track_meta: Dict[str, dict],
    bge_tag_store: Dict[str, torch.Tensor],
    tag_proj: nn.Linear,
    proj_device,
) -> tuple:
    """Return (semantic[4352], context[137], cf[128])."""
    def _gv(mod, dim):
        v = index.get_vec(mod, track_id)
        return v.float() if v is not None else torch.zeros(dim)

    clap_emb   = _gv("audio",      512)
    siglip_emb = _gv("image",      768)
    attr_emb   = _gv("attributes", 1024)
    lyr_emb    = _gv("lyrics",     1024)
    meta_emb   = _gv("metadata",   1024)
    semantic = torch.cat([clap_emb, siglip_emb, attr_emb, lyr_emb, meta_emb])  # [4352]

    bge_v = bge_tag_store.get(track_id, torch.zeros(384)).float()
    with torch.no_grad():
        tag32 = F.normalize(
            tag_proj(bge_v.unsqueeze(0).to(proj_device)).squeeze(0), p=2, dim=0
        ).cpu()
    tm = track_meta.get(track_id, {})
    log_pop    = torch.tensor([math.log1p(float(tm.get("popularity", 0) or 0))])
    dur_bucket = _bucket_emb(tm.get("duration_ms"), 8, 30000, 600000)
    context = torch.cat([tag32.clone(), tag32, tag32.clone(), tag32.clone(),
                         log_pop, dur_bucket])  # [137]

    tcf = index.get_vec("cf_bpr", track_id)
    cf = tcf.float() if tcf is not None else torch.zeros(128)
    return semantic, context, cf


# ── Dataset ────────────────────────────────────────────────────────────────────

class DCNDataset(Dataset):
    """
    Each sample returns:
      user features [86/128/1036/1024]
      pos track features [4352/137/128]
      K neg track features [K, 4352], [K, 137], [K, 128]

    Loss: cross_entropy(scores[0:1+K], label=0)
    """

    def __init__(
        self,
        samples:          List[dict],           # {session_id, user_id, turn_number, gt_track_id}
        index_store,
        query_store:      Dict,
        goal_store:       Dict,
        track_meta:       Dict[str, dict],
        user_meta:        Dict[str, dict],
        user_cf_store:    Dict[str, torch.Tensor],
        bge_tag_store:    Dict[str, torch.Tensor],
        tag_proj:         nn.Linear,
        retrieval_store:  Dict,                 # {"{sid}_{turn}" → list[track_id]}
        all_track_ids:    List[str],            # full track pool for fallback negatives
        num_neg:          int = 15,
    ):
        self.samples         = samples
        self.index           = index_store
        self.query_store     = query_store
        self.goal_store      = goal_store
        self.track_meta      = track_meta
        self.user_meta       = user_meta
        self.user_cf_store   = user_cf_store
        self.bge_tag_store   = bge_tag_store
        self.tag_proj        = tag_proj
        self.retrieval_store = retrieval_store
        self.all_track_ids   = all_track_ids
        self.num_neg         = num_neg

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_negatives(self, sid: str, turn: int, gt_track: str) -> List[str]:
        """Sample num_neg hard negatives from retrieval candidates for this (sid, turn)."""
        cands: List[str] = []
        # Try keys: exact match first, then scan backwards
        for t in range(turn, -1, -1):
            raw = self.retrieval_store.get(f"{sid}_{t}")
            if raw is not None:
                if isinstance(raw, dict):
                    cands = raw.get("union", raw.get("merged", []))
                elif isinstance(raw, list):
                    cands = raw
                break

        # Filter out ground truth
        cands = [tid for tid in cands if tid != gt_track]

        if len(cands) >= self.num_neg:
            return random.sample(cands, self.num_neg)

        # Fallback: fill remaining from global pool
        neg_ids = list(cands)
        tried   = set(neg_ids) | {gt_track}
        while len(neg_ids) < self.num_neg:
            c = random.choice(self.all_track_ids)
            if c not in tried:
                neg_ids.append(c)
                tried.add(c)
        return neg_ids

    def __getitem__(self, idx: int) -> dict:
        s        = self.samples[idx]
        sid      = s["session_id"]
        user_id  = s["user_id"]
        turn     = s["turn_number"]
        gt_track = s["gt_track_id"]

        proj_device = next(self.tag_proj.parameters()).device

        # ── user_profile [86] ──────────────────────────────────────────────
        um = self.user_meta.get(user_id, {})
        age = um.get("age")
        age_bucket = (
            _bucket_emb(math.log1p(float(age)), 16, 0, math.log1p(100))
            if age is not None else torch.zeros(16)
        )
        country_emb  = _bucket_emb(um.get("country_code_hash"),              16, 0, 1)
        gender_emb   = torch.zeros(2)
        if um.get("gender") == "male":    gender_emb[0] = 1.0
        elif um.get("gender") == "female": gender_emb[1] = 1.0
        lang_emb     = _bucket_emb(um.get("preferred_language_hash"),         4, 0, 1)
        culture_emb  = _bucket_emb(um.get("preferred_musical_culture_hash"), 32, 0, 1)
        listen_cnt   = _bucket_emb(math.log1p(float(um.get("listen_count",  0) or 0)),
                                   8, 0, math.log1p(10000))
        session_cnt  = _bucket_emb(math.log1p(float(um.get("session_count", 0) or 0)),
                                   8, 0, math.log1p(1000))
        user_profile = torch.cat([age_bucket, country_emb, gender_emb,
                                  lang_emb, culture_emb, listen_cnt, session_cnt])  # [86]

        # ── user_cf [128] ──────────────────────────────────────────────────
        ucf = self.user_cf_store.get(user_id)
        user_cf = ucf.float() if ucf is not None else torch.zeros(128)

        # ── conv_goal [1036] ───────────────────────────────────────────────
        ge = self.goal_store.get(sid)
        ge = ge.float() if ge is not None else torch.zeros(1024)
        conv_goal = torch.cat([torch.zeros(8), ge, torch.zeros(4)])  # [1036]

        # ── query_emb [1024] ───────────────────────────────────────────────
        q = self.query_store.get(f"{sid}_{turn}_query")
        if q is None:
            q = self.query_store.get(f"{sid}_{turn}")
        query_emb = q.float() if q is not None else torch.zeros(1024)

        # ── positive track features ────────────────────────────────────────
        pos_sem, pos_ctx, pos_cf = _get_track_features(
            gt_track, self.index, self.track_meta,
            self.bge_tag_store, self.tag_proj, proj_device
        )

        # ── negative track features [K, dim] ──────────────────────────────
        neg_ids = self._sample_negatives(sid, turn, gt_track)
        neg_sem_list, neg_ctx_list, neg_cf_list = [], [], []
        for nid in neg_ids:
            ns, nc, nf = _get_track_features(
                nid, self.index, self.track_meta,
                self.bge_tag_store, self.tag_proj, proj_device
            )
            neg_sem_list.append(ns)
            neg_ctx_list.append(nc)
            neg_cf_list.append(nf)

        return {
            "user_profile":   user_profile,                        # [86]
            "user_cf":        user_cf,                             # [128]
            "conv_goal":      conv_goal,                           # [1036]
            "query_emb":      query_emb,                           # [1024]
            "pos_semantic":   pos_sem,                             # [4352]
            "pos_context":    pos_ctx,                             # [137]
            "pos_cf":         pos_cf,                              # [128]
            "neg_semantic":   torch.stack(neg_sem_list),           # [K, 4352]
            "neg_context":    torch.stack(neg_ctx_list),           # [K, 137]
            "neg_cf":         torch.stack(neg_cf_list),            # [K, 128]
        }


# ── Build samples ──────────────────────────────────────────────────────────────

def build_samples(conv_dataset_name: str, split: str) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset(conv_dataset_name, split=split)
    samples = []
    for item in ds:
        sid     = str(item.get("session_id") or item.get("id") or "")
        user_id = str(item.get("user_id", ""))
        convs   = item.get("conversations", [])
        for c in convs:
            if c.get("role") == "music" and c.get("content"):
                samples.append({
                    "session_id":  sid,
                    "user_id":     user_id,
                    "turn_number": int(c["turn_number"]),
                    "gt_track_id": c["content"],
                })
    logger.info("Built %d samples from %s/%s.", len(samples), conv_dataset_name, split)
    return samples


# ── Forward with hard-negative loss ───────────────────────────────────────────

def compute_loss(model, batch: dict, device: str) -> torch.Tensor:
    """
    batch keys: user_profile, user_cf, conv_goal, query_emb,
                pos_semantic, pos_context, pos_cf,
                neg_semantic [B,K,4352], neg_context [B,K,137], neg_cf [B,K,128]

    For each sample in batch:
      scores = [score(user, pos), score(user, neg_1), ..., score(user, neg_K)]  [1+K]
      loss   = cross_entropy(scores, label=0)
    """
    B = batch["user_profile"].size(0)
    K = batch["neg_semantic"].size(1)

    # User features → [B, *]
    u_prof  = batch["user_profile"].to(device)   # [B, 86]
    u_cf    = batch["user_cf"].to(device)         # [B, 128]
    c_goal  = batch["conv_goal"].to(device)       # [B, 1036]
    q_emb   = batch["query_emb"].to(device)       # [B, 1024]

    # Positive score → [B, 1]
    pos_score = model.encode(
        u_prof, u_cf, c_goal, q_emb,
        batch["pos_semantic"].to(device),
        batch["pos_context"].to(device),
        batch["pos_cf"].to(device),
    )  # [B, 1]

    # Negative scores → [B, K]
    # Flatten [B,K,dim] → [B*K, dim], score, reshape
    def _flat(t): return t.reshape(B * K, -1).to(device)

    neg_score = model.encode(
        u_prof.repeat_interleave(K, dim=0),
        u_cf.repeat_interleave(K, dim=0),
        c_goal.repeat_interleave(K, dim=0),
        q_emb.repeat_interleave(K, dim=0),
        _flat(batch["neg_semantic"]),
        _flat(batch["neg_context"]),
        _flat(batch["neg_cf"]),
    ).reshape(B, K)  # [B, K]

    # Concat: [B, 1+K],  label=0 (positive is first)
    logits = torch.cat([pos_score, neg_score], dim=1)   # [B, 1+K]
    labels = torch.zeros(B, dtype=torch.long, device=device)
    return F.cross_entropy(logits, labels)


# ── Training loop ──────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from mcrs.retrieval_modules.index_store import IndexStore
    from mcrs.reranking_modules.dcn_reranker import DCNReranker

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    device = args.device

    # ── Index ─────────────────────────────────────────────────────────────
    logger.info("Loading IndexStore …")
    index = IndexStore.build(
        track_emb_dataset=args.track_emb_dataset,
        split_types=["all_tracks"],
        cache_dir=args.cache_dir,
        bge_tag_path=args.bge_tag_path,
        device="cpu",
    )
    valid_tracks  = set(index.track_ids)
    all_track_ids = list(index.track_ids)

    # ── Embedding stores ──────────────────────────────────────────────────
    def _lpt(p, name):
        if p and os.path.exists(p):
            d = torch.load(p, map_location="cpu", weights_only=True)
            logger.info("Loaded %s: %d entries", name, len(d))
            return d
        logger.warning("Not found: %s", p)
        return {}

    query_store      = _lpt(args.query_emb_path,  "query_store")
    goal_store       = _lpt(args.goal_emb_path,   "goal_store")
    retrieval_store  = _lpt(args.retrieval_path,  "retrieval_store")
    bge_tag_store    = _lpt(args.bge_tag_path,    "bge_tag")

    # ── Track metadata ────────────────────────────────────────────────────
    logger.info("Loading track metadata …")
    track_meta: Dict[str, dict] = {}
    tm_ds = load_dataset(args.track_metadata_dataset)
    for sp in tm_ds:
        for row in tm_ds[sp]:
            tid = str(row.get("track_id", ""))
            if tid:
                track_meta[tid] = {
                    "popularity":  row.get("popularity"),
                    "duration_ms": row.get("duration_ms"),
                }

    # ── User metadata ─────────────────────────────────────────────────────
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
        logger.warning("User metadata: %s", e)

    # ── User CF store ─────────────────────────────────────────────────────
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
        logger.warning("User CF: %s", e)

    # ── Shared tag projection ─────────────────────────────────────────────
    tag_proj = nn.Linear(384, 32, bias=False).to(device)

    # ── Samples ───────────────────────────────────────────────────────────
    def _filter(samples, q_store):
        out = []
        for s in samples:
            if s["gt_track_id"] not in valid_tracks:
                continue
            sid, t = s["session_id"], s["turn_number"]
            if (q_store.get(f"{sid}_{t}_query") is not None
                    or q_store.get(f"{sid}_{t}") is not None):
                out.append(s)
        return out

    train_samples = _filter(build_samples(args.conv_dataset, "train"), query_store)

    val_query_store = _lpt(args.val_query_emb_path, "val_query") if args.val_query_emb_path else query_store
    val_goal_store  = _lpt(args.val_goal_emb_path,  "val_goal")  if args.val_goal_emb_path  else goal_store
    val_ret_store   = _lpt(args.val_retrieval_path,  "val_retrieval") if args.val_retrieval_path else retrieval_store

    val_samples = _filter(build_samples(args.conv_dataset, "test"), val_query_store)
    logger.info("Filtered → train=%d, val=%d", len(train_samples), len(val_samples))

    # ── DataLoaders ───────────────────────────────────────────────────────
    def _make_ds(samples, q_store, g_store, r_store):
        return DCNDataset(
            samples, index, q_store, g_store,
            track_meta, user_meta, user_cf_store,
            bge_tag_store, tag_proj, r_store,
            all_track_ids, args.num_neg,
        )

    train_loader = DataLoader(
        _make_ds(train_samples, query_store, goal_store, retrieval_store),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        _make_ds(val_samples, val_query_store, val_goal_store, val_ret_store),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    ) if val_samples else None

    # ── Model + optimizer ─────────────────────────────────────────────────
    model = DCNReranker(
        cross_layers=args.cross_layers,
        deep_dims=tuple(args.deep_dims),
        dropout=args.dropout,
    ).to(device)

    params    = list(model.parameters()) + list(tag_proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    patience_cnt  = 0
    history       = []
    start_epoch   = 1

    # ── Resume ────────────────────────────────────────────────────────────
    resume_path = args.resume or (args.out if args.resume_best else None)
    if resume_path and os.path.exists(resume_path):
        logger.info("Resuming from %s …", resume_path)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer"     in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler"     in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
        if "epoch"         in ckpt: start_epoch   = ckpt["epoch"] + 1
        if "best_val_loss" in ckpt: best_val_loss = ckpt["best_val_loss"]
        if "history"       in ckpt: history       = ckpt["history"]
        tp_path = resume_path.replace(".pt", "_tag_proj.pt")
        if os.path.exists(tp_path):
            tag_proj.load_state_dict(torch.load(tp_path, map_location=device,
                                                weights_only=True)["state_dict"])
        logger.info("Resumed: epoch=%d, best_val_loss=%.4f", start_epoch, best_val_loss)

    # ── Training epochs ───────────────────────────────────────────────────
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train(); tag_proj.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            optimizer.zero_grad()
            loss = compute_loss(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        tr_l = sum(train_losses) / len(train_losses)

        # ── Validation ────────────────────────────────────────────────────
        if val_loader:
            model.eval(); tag_proj.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    val_losses.append(compute_loss(model, batch, device).item())
            vl_l = sum(val_losses) / len(val_losses)
            logger.info("Epoch %d  train=%.4f  val=%.4f  (random_baseline=%.4f)",
                        epoch, tr_l, vl_l, math.log(1 + args.num_neg))
            history.append({"epoch": epoch, "train_loss": tr_l, "val_loss": vl_l})

            if vl_l < best_val_loss:
                best_val_loss = vl_l
                patience_cnt  = 0
                _save(model, tag_proj, optimizer, scheduler,
                      epoch, best_val_loss, history, args.out)
                logger.info("  ✅ Saved epoch=%d val=%.4f → %s", epoch, vl_l, args.out)
            else:
                patience_cnt += 1
                logger.info("  patience %d/%d", patience_cnt, args.patience)
                if patience_cnt >= args.patience:
                    logger.info("Early stopping at epoch %d.", epoch)
                    break
        else:
            logger.info("Epoch %d  train=%.4f  (no val)", epoch, tr_l)
            history.append({"epoch": epoch, "train_loss": tr_l, "val_loss": float("nan")})
            _save(model, tag_proj, optimizer, scheduler,
                  epoch, float("nan"), history, args.out)

    with open(args.out.replace(".pt", "_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Done. best_val_loss=%.4f", best_val_loss)


def _save(model, tag_proj, optimizer, scheduler, epoch, best_val_loss, history, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "state_dict":    model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "epoch":         epoch,
        "best_val_loss": best_val_loss,
        "history":       history,
    }, path)
    torch.save({"state_dict": tag_proj.state_dict()},
               path.replace(".pt", "_tag_proj.pt"))


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DCN-V2 Reranker (hard-negative)")
    p.add_argument("--conv_dataset",           type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    p.add_argument("--track_emb_dataset",      type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    p.add_argument("--track_metadata_dataset", type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    p.add_argument("--user_metadata_dataset",  type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-User-Metadata")
    p.add_argument("--user_emb_dataset",       type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-User-Embeddings")
    # Embeddings
    p.add_argument("--query_emb_path",     required=True, type=str)
    p.add_argument("--goal_emb_path",      required=True, type=str)
    p.add_argument("--val_query_emb_path", type=str, default=None)
    p.add_argument("--val_goal_emb_path",  type=str, default=None)
    p.add_argument("--bge_tag_path",       type=str, default="bge/track_tag_embeddings.pt")
    # Retrieval candidates (hard negatives)
    p.add_argument("--retrieval_path",     type=str, default=None,
                   help="Path to pre-computed retrieval candidates .pt for train split")
    p.add_argument("--val_retrieval_path", type=str, default=None,
                   help="Path to pre-computed retrieval candidates .pt for val/test split")
    # Training
    p.add_argument("--cache_dir",    type=str,  default="qwen/retrieval_indices")
    p.add_argument("--out",          type=str,  default="checkpoints/dcn_reranker_best.pt")
    p.add_argument("--epochs",       type=int,  default=20)
    p.add_argument("--batch_size",   type=int,  default=64)
    p.add_argument("--num_neg",      type=int,  default=15,
                   help="Hard negative count per sample (from retrieval candidates)")
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--cross_layers", type=int,  default=3)
    p.add_argument("--deep_dims",    type=int,  nargs="+", default=[512, 256, 128])
    p.add_argument("--patience",     type=int,  default=5)
    p.add_argument("--num_workers",  type=int,  default=0)
    p.add_argument("--resume",       type=str,  default=None)
    p.add_argument("--resume_best",  action="store_true")
    p.add_argument("--device",       type=str,  default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
