"""
scripts/reranking/train_dcn_reranker.py
========================================
Training script for DCNReranker v5.

Key design decisions:
  [P1] Listwise softmax CE loss (positive at index 0)
  [P2] Hard negatives from retrieval candidates (same distribution as inference)
  [P4] conv_goal replaced by pure goal_emb [1024→128]
  [P5] Query embedding: strict per-turn key {sid}_{turn}_query, zero if absent
  [P6] Feature extraction via shared functions in dcn_reranker.py

Usage:
  python scripts/reranking/train_dcn_reranker.py \\
      --query_emb_path     qwen/dialogue_embeddings_train_0.6b.pt \\
      --goal_emb_path      qwen/goal_embeddings_train_0.6b.pt \\
      --val_query_emb_path qwen/dialogue_embeddings_test_0.6b.pt \\
      --val_goal_emb_path  qwen/goal_embeddings_test_0.6b.pt \\
      --retrieval_path     qwen/retrieval_train_candidates.pt \\
      --val_retrieval_path qwen/retrieval_test_candidates.pt \\
      --bge_tag_path       bge/track_tag_embeddings.pt \\
      --cache_dir          qwen/retrieval_indices \\
      --out                checkpoints/dcn_reranker_best.pt \\
      --epochs 30 --batch_size 64 --num_neg 15 --lr 3e-4 \\
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


# ── Dataset ────────────────────────────────────────────────────────────────────

class DCNDataset(Dataset):

    def __init__(
        self,
        samples:         List[dict],
        index_store,
        query_store:     Dict,
        goal_store:      Dict,
        track_meta:      Dict,
        user_meta:       Dict,
        user_cf_store:   Dict[str, torch.Tensor],
        bge_tag_store:   Dict[str, torch.Tensor],
        tag_proj:        nn.Linear,
        retrieval_store: Dict,       # {"{sid}_{turn}" → dict-of-channels or list}
        all_track_ids:   List[str],  # fallback pool for random negatives
        num_neg:         int = 15,
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

    def __len__(self): return len(self.samples)

    def _get_merged_candidates(self, sid: str, turn: int) -> List[str]:
        """Get merged retrieval list for (sid, turn). Tries exact key, else empty."""
        raw = self.retrieval_store.get(f"{sid}_{turn}")
        if raw is None:
            return []
        if isinstance(raw, dict):
            return raw.get("merged", raw.get("union", []))
        if isinstance(raw, list):
            return raw
        return []

    def _sample_negatives(self, candidates: List[str], gt_track: str) -> List[str]:
        """[P2] Hard negatives from retrieval candidates, fallback to random."""
        negs = [t for t in candidates if t != gt_track]
        if len(negs) >= self.num_neg:
            return random.sample(negs, self.num_neg)
        # Fill with random from pool
        tried = set(negs) | {gt_track}
        while len(negs) < self.num_neg:
            c = random.choice(self.all_track_ids)
            if c not in tried:
                negs.append(c); tried.add(c)
        return negs

    def __getitem__(self, idx: int) -> dict:
        from mcrs.reranking_modules.dcn_reranker import (
            extract_query_emb, extract_goal_emb,
            extract_user_profile, extract_track_features,
        )

        s        = self.samples[idx]
        sid      = s["session_id"]
        user_id  = s["user_id"]
        turn     = s["turn_number"]
        gt_track = s["gt_track_id"]

        proj_device = next(self.tag_proj.parameters()).device

        # ── User features ──────────────────────────────────────────────────
        user_profile = extract_user_profile(user_id, self.user_meta)   # [86]
        ucf          = self.user_cf_store.get(user_id)
        user_cf      = ucf.float() if ucf is not None else torch.zeros(128)

        # [P4] Pure goal embedding — no padding junk
        goal_emb  = extract_goal_emb(self.goal_store, sid)             # [1024]

        # [P5] Strict per-turn query key only
        query_emb = extract_query_emb(self.query_store, sid, turn)     # [1024] or zeros

        # ── Retrieval candidates for hard negatives ────────────────────────
        cands = self._get_merged_candidates(sid, turn)
        neg_ids = self._sample_negatives(cands, gt_track)

        # ── Track features ─────────────────────────────────────────────────
        def _tfeat(tid):
            return extract_track_features(
                tid, self.index, self.track_meta,
                self.bge_tag_store, self.tag_proj, proj_device,
            )

        pa, pi, pattr, pl, pm, pctx, pcf = _tfeat(gt_track)

        neg_feats = [_tfeat(nid) for nid in neg_ids]
        neg_audio   = torch.stack([f[0] for f in neg_feats])   # [K, 512]
        neg_image   = torch.stack([f[1] for f in neg_feats])   # [K, 768]
        neg_attr    = torch.stack([f[2] for f in neg_feats])   # [K, 1024]
        neg_lyrics  = torch.stack([f[3] for f in neg_feats])   # [K, 1024]
        neg_meta    = torch.stack([f[4] for f in neg_feats])   # [K, 1024]
        neg_context = torch.stack([f[5] for f in neg_feats])   # [K, 137]
        neg_cf      = torch.stack([f[6] for f in neg_feats])   # [K, 128]

        return {
            "user_profile": user_profile,   # [86]
            "user_cf":      user_cf,         # [128]
            "goal_emb":     goal_emb,        # [1024]
            "query_emb":    query_emb,       # [1024]
            "pos_audio":    pa,              # [512]
            "pos_image":    pi,              # [768]
            "pos_attr":     pattr,           # [1024]
            "pos_lyrics":   pl,              # [1024]
            "pos_meta":     pm,              # [1024]
            "pos_context":  pctx,            # [137]
            "pos_cf":       pcf,             # [128]
            "neg_audio":    neg_audio,
            "neg_image":    neg_image,
            "neg_attr":     neg_attr,
            "neg_lyrics":   neg_lyrics,
            "neg_meta":     neg_meta,
            "neg_context":  neg_context,
            "neg_cf":       neg_cf,
        }


# ── Build samples ──────────────────────────────────────────────────────────────

def build_samples(conv_dataset_name: str, split: str) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset(conv_dataset_name, split=split)
    samples = []
    for item in ds:
        sid     = str(item.get("session_id") or item.get("id") or "")
        user_id = str(item.get("user_id", ""))
        for c in item.get("conversations", []):
            if c.get("role") == "music" and c.get("content"):
                samples.append({
                    "session_id":  sid,
                    "user_id":     user_id,
                    "turn_number": int(c["turn_number"]),
                    "gt_track_id": c["content"],
                })
    logger.info("Built %d samples from %s/%s.", len(samples), conv_dataset_name, split)
    return samples


def _filter_samples(samples: List[dict], query_store: Dict,
                    valid_tracks: set) -> List[dict]:
    """
    [P5] Keep only samples where:
      1. gt_track is in the index
      2. Strict per-turn query key exists: {sid}_{turn}_query
    Log a summary so we know how many are missing.
    """
    kept, missing_query, missing_track = [], 0, 0
    for s in samples:
        if s["gt_track_id"] not in valid_tracks:
            missing_track += 1
            continue
        k = f"{s['session_id']}_{s['turn_number']}_query"
        if query_store.get(k) is None:
            missing_query += 1
            continue
        kept.append(s)
    logger.info("  Filtered: kept=%d  missing_track=%d  missing_query=%d",
                len(kept), missing_track, missing_query)
    return kept


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _save(model, tag_proj, optimizer, scheduler,
          epoch, best_val_loss, history, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "state_dict":    model.state_dict(),
        "tag_proj":      tag_proj.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "epoch":         epoch,
        "best_val_loss": best_val_loss,
        "history":       history,
    }, path)


# ── Training ───────────────────────────────────────────────────────────────────

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
    logger.info("Index: %d tracks", len(valid_tracks))

    # ── Embedding stores ──────────────────────────────────────────────────
    def _lpt(p, name):
        if p and os.path.exists(p):
            d = torch.load(p, map_location="cpu", weights_only=True)
            logger.info("Loaded %s: %d entries", name, len(d))
            return d
        logger.warning("Not found: %s (%s)", p, name)
        return {}

    query_store     = _lpt(args.query_emb_path, "query_store")
    goal_store      = _lpt(args.goal_emb_path,  "goal_store")
    retrieval_store = _lpt(args.retrieval_path,  "retrieval_store")
    bge_tag_store   = _lpt(args.bge_tag_path,   "bge_tag")

    val_q = _lpt(args.val_query_emb_path, "val_query") if args.val_query_emb_path else {}
    val_g = _lpt(args.val_goal_emb_path,  "val_goal")  if args.val_goal_emb_path  else {}
    val_r = _lpt(args.val_retrieval_path, "val_ret")   if args.val_retrieval_path else {}

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
        logger.warning("User metadata: %s", e)

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
        logger.warning("User CF: %s", e)

    # ── Tag projection (shared MLP 384→32) ───────────────────────────────
    tag_proj = nn.Linear(384, 32, bias=False).to(device)

    # ── Samples ───────────────────────────────────────────────────────────
    logger.info("Building train samples …")
    train_raw = build_samples(args.conv_dataset, "train")
    train_samples = _filter_samples(train_raw, query_store, valid_tracks)

    val_samples: List[dict] = []
    if val_q:
        logger.info("Building val samples …")
        val_raw = build_samples(args.conv_dataset, "test")
        val_samples = _filter_samples(val_raw, val_q, valid_tracks)

    logger.info("Final → train=%d, val=%d", len(train_samples), len(val_samples))

    def _make_ds(samples, q_store, g_store, r_store):
        return DCNDataset(
            samples, index, q_store, g_store, track_meta, user_meta,
            user_cf_store, bge_tag_store, tag_proj, r_store, all_track_ids, args.num_neg,
        )

    train_loader = DataLoader(
        _make_ds(train_samples, query_store, goal_store, retrieval_store),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        _make_ds(val_samples, val_q, val_g, val_r),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    ) if val_samples else None

    # ── Model ─────────────────────────────────────────────────────────────
    model = DCNReranker(
        cross_layers=args.cross_layers,
        deep_dims=tuple(args.deep_dims),
        dropout=args.dropout,
    ).to(device)
    logger.info("Model params: %d", model.count_params())

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
        if "tag_proj"     in ckpt: tag_proj.load_state_dict(ckpt["tag_proj"])
        if "optimizer"    in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler"    in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
        if "epoch"        in ckpt: start_epoch   = ckpt["epoch"] + 1
        if "best_val_loss"in ckpt: best_val_loss = ckpt["best_val_loss"]
        if "history"      in ckpt: history       = ckpt["history"]
        logger.info("Resumed: epoch=%d, best_val_loss=%.4f", start_epoch, best_val_loss)

    random_baseline = math.log(1 + args.num_neg)

    # ── Training loop ─────────────────────────────────────────────────────
    def _run_batch(batch):
        return model(
            batch["user_profile"].to(device),
            batch["user_cf"].to(device),
            batch["goal_emb"].to(device),
            batch["query_emb"].to(device),
            batch["pos_audio"].to(device),
            batch["pos_image"].to(device),
            batch["pos_attr"].to(device),
            batch["pos_lyrics"].to(device),
            batch["pos_meta"].to(device),
            batch["pos_context"].to(device),
            batch["pos_cf"].to(device),
            batch["neg_audio"].to(device),
            batch["neg_image"].to(device),
            batch["neg_attr"].to(device),
            batch["neg_lyrics"].to(device),
            batch["neg_meta"].to(device),
            batch["neg_context"].to(device),
            batch["neg_cf"].to(device),
        )

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train(); tag_proj.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            optimizer.zero_grad()
            loss = _run_batch(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()
        tr_l = sum(train_losses) / len(train_losses)

        if val_loader:
            model.eval(); tag_proj.eval()
            with torch.no_grad():
                vl_l = sum(_run_batch(b).item() for b in val_loader) / len(val_loader)
            logger.info("Epoch %d  train=%.4f  val=%.4f  (baseline=%.4f)",
                        epoch, tr_l, vl_l, random_baseline)
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
            history.append({"epoch": epoch, "train_loss": tr_l})
            _save(model, tag_proj, optimizer, scheduler,
                  epoch, float("nan"), history, args.out)

    with open(args.out.replace(".pt", "_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Done. best_val_loss=%.4f", best_val_loss)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DCN-V2 Reranker v5")
    p.add_argument("--conv_dataset",           default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    p.add_argument("--track_emb_dataset",      default="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    p.add_argument("--track_metadata_dataset", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    p.add_argument("--user_metadata_dataset",  default="talkpl-ai/TalkPlayData-Challenge-User-Metadata")
    p.add_argument("--user_emb_dataset",       default="talkpl-ai/TalkPlayData-Challenge-User-Embeddings")
    p.add_argument("--query_emb_path",     required=True)
    p.add_argument("--goal_emb_path",      required=True)
    p.add_argument("--val_query_emb_path", default=None)
    p.add_argument("--val_goal_emb_path",  default=None)
    p.add_argument("--bge_tag_path",       default="bge/track_tag_embeddings.pt")
    p.add_argument("--retrieval_path",     default=None,
                   help="Pre-computed retrieval candidates for train (hard negatives)")
    p.add_argument("--val_retrieval_path", default=None)
    p.add_argument("--cache_dir",          default="qwen/retrieval_indices")
    p.add_argument("--out",                default="checkpoints/dcn_reranker_best.pt")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--num_neg",      type=int,   default=15)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--cross_layers", type=int,   default=3)
    p.add_argument("--deep_dims",    type=int,   nargs="+", default=[256, 256, 128])
    p.add_argument("--patience",     type=int,   default=8)
    p.add_argument("--num_workers",  type=int,   default=0)
    p.add_argument("--resume",       default=None)
    p.add_argument("--resume_best",  action="store_true")
    p.add_argument("--device",       default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
