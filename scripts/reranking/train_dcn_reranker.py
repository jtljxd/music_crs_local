"""
scripts/reranking/train_dcn_reranker.py
========================================
Training script for DCNReranker (v4: truncate-concat + retrieval signals).

Loss: Listwise softmax CE with hard negatives from retrieval candidates.

Usage:
  python scripts/reranking/train_dcn_reranker.py \\
      --query_emb_path     qwen/hist_conversation_embeddings_train_0.6b.pt \\
      --goal_emb_path      qwen/goal_embeddings_train_0.6b.pt \\
      --retrieval_path     qwen/retrieval_train_candidates.pt \\
      --bge_tag_path       bge/track_tag_embeddings.pt \\
      --cache_dir          qwen/retrieval_indices \\
      --out                checkpoints/dcn_reranker_best.pt \\
      --epochs 20 --batch_size 64 --num_neg 15 --lr 3e-4 \\
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
from typing import Dict, List

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


def _get_track_embs(track_id: str, index) -> tuple:
    """Return (audio[512], image[768], attr[1024], lyrics[1024], meta[1024])."""
    def _gv(mod, dim):
        v = index.get_vec(mod, track_id)
        return v.float() if v is not None else torch.zeros(dim)
    return (
        _gv("audio",      512),
        _gv("image",      768),
        _gv("attributes", 1024),
        _gv("lyrics",     1024),
        _gv("metadata",   1024),
    )


def _get_track_context(
    track_id: str,
    track_meta: Dict,
    bge_tag_store: Dict,
    tag_proj: nn.Linear,
    proj_device,
) -> torch.Tensor:
    """Build 137-dim context feature."""
    bge_v = bge_tag_store.get(track_id, torch.zeros(384)).float()
    with torch.no_grad():
        tag32 = F.normalize(
            tag_proj(bge_v.unsqueeze(0).to(proj_device)).squeeze(0), p=2, dim=0
        ).cpu()
    tm = track_meta.get(track_id, {})
    log_pop    = torch.tensor([math.log1p(float(tm.get("popularity", 0) or 0))])
    dur_bucket = _bucket_emb(tm.get("duration_ms"), 8, 30000, 600000)
    return torch.cat([tag32, tag32, tag32, tag32, log_pop, dur_bucket])  # [137]


def _get_track_cf(track_id: str, index) -> torch.Tensor:
    v = index.get_vec("cf_bpr", track_id)
    return v.float() if v is not None else torch.zeros(128)


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
        retrieval_store: Dict,           # {"{sid}_{turn}" → dict-of-channels or list}
        all_track_ids:   List[str],
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

    def _get_channel_results(self, sid: str, turn: int) -> Dict[str, List[str]]:
        """Return per-channel dict for this (sid, turn), falling back to empty."""
        for t in range(turn, -1, -1):
            raw = self.retrieval_store.get(f"{sid}_{t}")
            if raw is not None:
                if isinstance(raw, dict):
                    return raw
                if isinstance(raw, list):
                    return {"merged": raw}
        return {}

    def _sample_negatives(self, ch_results: Dict, gt_track: str) -> List[str]:
        merged = ch_results.get("merged") or ch_results.get("union", [])
        if not merged:
            # flatten all channels
            seen, merged = set(), []
            for v in ch_results.values():
                for tid in (v if isinstance(v, list) else []):
                    if tid not in seen:
                        seen.add(tid); merged.append(tid)
        cands = [t for t in merged if t != gt_track]
        if len(cands) >= self.num_neg:
            return random.sample(cands, self.num_neg)
        neg_ids = list(cands)
        tried   = set(neg_ids) | {gt_track}
        while len(neg_ids) < self.num_neg:
            c = random.choice(self.all_track_ids)
            if c not in tried:
                neg_ids.append(c); tried.add(c)
        return neg_ids

    def __getitem__(self, idx: int) -> dict:
        from mcrs.reranking_modules.dcn_reranker import build_retrieval_feat

        s        = self.samples[idx]
        sid      = s["session_id"]
        user_id  = s["user_id"]
        turn     = s["turn_number"]
        gt_track = s["gt_track_id"]

        proj_device = next(self.tag_proj.parameters()).device

        # ── user_profile [86] ──────────────────────────────────────────────
        um = self.user_meta.get(user_id, {})
        age = um.get("age")
        age_b = (_bucket_emb(math.log1p(float(age)), 16, 0, math.log1p(100))
                 if age is not None else torch.zeros(16))
        gender = torch.zeros(2)
        if um.get("gender") == "male":    gender[0] = 1.0
        elif um.get("gender") == "female": gender[1] = 1.0
        user_profile = torch.cat([
            age_b,
            _bucket_emb(um.get("country_code_hash"),              16, 0, 1),
            gender,
            _bucket_emb(um.get("preferred_language_hash"),         4, 0, 1),
            _bucket_emb(um.get("preferred_musical_culture_hash"), 32, 0, 1),
            _bucket_emb(math.log1p(float(um.get("listen_count",  0) or 0)),
                        8, 0, math.log1p(10000)),
            _bucket_emb(math.log1p(float(um.get("session_count", 0) or 0)),
                        8, 0, math.log1p(1000)),
        ])  # [86]

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

        # ── Retrieval channel results for this (sid, turn) ─────────────────
        ch_results = self._get_channel_results(sid, turn)

        # ── Positive track ─────────────────────────────────────────────────
        pos_audio, pos_image, pos_attr, pos_lyrics, pos_meta_emb = _get_track_embs(gt_track, self.index)
        pos_context = _get_track_context(gt_track, self.track_meta, self.bge_tag_store, self.tag_proj, proj_device)
        pos_cf      = _get_track_cf(gt_track, self.index)
        pos_ret     = build_retrieval_feat(gt_track, ch_results)

        # ── Negative tracks [K, dim] ───────────────────────────────────────
        neg_ids = self._sample_negatives(ch_results, gt_track)
        neg_audio_l, neg_image_l, neg_attr_l = [], [], []
        neg_lyrics_l, neg_meta_l, neg_ctx_l, neg_cf_l, neg_ret_l = [], [], [], [], []
        for nid in neg_ids:
            na, ni, nattr, nl, nm = _get_track_embs(nid, self.index)
            neg_audio_l.append(na); neg_image_l.append(ni); neg_attr_l.append(nattr)
            neg_lyrics_l.append(nl); neg_meta_l.append(nm)
            neg_ctx_l.append(_get_track_context(nid, self.track_meta, self.bge_tag_store, self.tag_proj, proj_device))
            neg_cf_l.append(_get_track_cf(nid, self.index))
            neg_ret_l.append(build_retrieval_feat(nid, ch_results))

        return {
            "user_profile": user_profile,
            "user_cf":      user_cf,
            "conv_goal":    conv_goal,
            "query_emb":    query_emb,
            # positive
            "pos_audio":    pos_audio,
            "pos_image":    pos_image,
            "pos_attr":     pos_attr,
            "pos_lyrics":   pos_lyrics,
            "pos_meta_emb": pos_meta_emb,
            "pos_context":  pos_context,
            "pos_cf":       pos_cf,
            "pos_ret_feat": pos_ret,
            # negatives [K, dim]
            "neg_audio":    torch.stack(neg_audio_l),
            "neg_image":    torch.stack(neg_image_l),
            "neg_attr":     torch.stack(neg_attr_l),
            "neg_lyrics":   torch.stack(neg_lyrics_l),
            "neg_meta_emb": torch.stack(neg_meta_l),
            "neg_context":  torch.stack(neg_ctx_l),
            "neg_cf":       torch.stack(neg_cf_l),
            "neg_ret_feat": torch.stack(neg_ret_l),
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


# ── Training ───────────────────────────────────────────────────────────────────

def _save(model, optimizer, scheduler, epoch, best_val_loss, history, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "state_dict":    model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "epoch":         epoch,
        "best_val_loss": best_val_loss,
        "history":       history,
    }, path)


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

    # ── Stores ────────────────────────────────────────────────────────────
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

    # ── Metadata ──────────────────────────────────────────────────────────
    logger.info("Loading track metadata …")
    track_meta: Dict = {}
    tm_ds = load_dataset(args.track_metadata_dataset)
    for sp in tm_ds:
        for row in tm_ds[sp]:
            tid = str(row.get("track_id", ""))
            if tid:
                track_meta[tid] = {"popularity": row.get("popularity"),
                                   "duration_ms": row.get("duration_ms")}

    logger.info("Loading user metadata …")
    user_meta: Dict = {}
    try:
        u_ds = load_dataset(args.user_metadata_dataset)
        for sp in u_ds:
            for row in u_ds[sp]:
                uid = str(row.get("user_id", ""))
                if uid: user_meta[uid] = dict(row)
    except Exception as e:
        logger.warning("User metadata: %s", e)

    logger.info("Loading user CF-BPR …")
    user_cf_store: Dict[str, torch.Tensor] = {}
    try:
        ue_ds = load_dataset(args.user_emb_dataset)
        for sp in ue_ds:
            for row in ue_ds[sp]:
                uid = str(row.get("user_id", ""))
                v   = row.get("cf-bpr")
                if uid and v is not None:
                    t = torch.tensor(v, dtype=torch.float32)
                    if t.numel() > 0: user_cf_store[uid] = t
        logger.info("  %d users with CF emb", len(user_cf_store))
    except Exception as e:
        logger.warning("User CF: %s", e)

    # ── Tag projection ────────────────────────────────────────────────────
    tag_proj = nn.Linear(384, 32, bias=False).to(device)

    # ── Samples ───────────────────────────────────────────────────────────
    def _filter(samples, q_store):
        out = []
        for s in samples:
            if s["gt_track_id"] not in valid_tracks: continue
            sid, t = s["session_id"], s["turn_number"]
            if (q_store.get(f"{sid}_{t}_query") is not None
                    or q_store.get(f"{sid}_{t}") is not None):
                out.append(s)
        return out

    train_samples = _filter(build_samples(args.conv_dataset, "train"), query_store)

    val_q = _lpt(args.val_query_emb_path, "val_query") if args.val_query_emb_path else query_store
    val_g = _lpt(args.val_goal_emb_path,  "val_goal")  if args.val_goal_emb_path  else goal_store
    val_r = _lpt(args.val_retrieval_path, "val_ret")   if args.val_retrieval_path else retrieval_store

    val_samples = _filter(build_samples(args.conv_dataset, "test"), val_q)
    logger.info("Filtered → train=%d, val=%d", len(train_samples), len(val_samples))

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
        logger.info("Resumed: epoch=%d, best_val_loss=%.4f", start_epoch, best_val_loss)

    random_baseline = math.log(1 + args.num_neg)

    # ── Epochs ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train(); tag_proj.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            optimizer.zero_grad()
            loss = model(
                batch["user_profile"].to(device),
                batch["user_cf"].to(device),
                batch["conv_goal"].to(device),
                batch["query_emb"].to(device),
                batch["pos_audio"].to(device),
                batch["pos_image"].to(device),
                batch["pos_attr"].to(device),
                batch["pos_lyrics"].to(device),
                batch["pos_meta_emb"].to(device),
                batch["pos_context"].to(device),
                batch["pos_cf"].to(device),
                batch["pos_ret_feat"].to(device),
                batch["neg_audio"].to(device),
                batch["neg_image"].to(device),
                batch["neg_attr"].to(device),
                batch["neg_lyrics"].to(device),
                batch["neg_meta_emb"].to(device),
                batch["neg_context"].to(device),
                batch["neg_cf"].to(device),
                batch["neg_ret_feat"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        tr_l = sum(train_losses) / len(train_losses)

        if val_loader:
            model.eval(); tag_proj.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    val_losses.append(model(
                        batch["user_profile"].to(device),
                        batch["user_cf"].to(device),
                        batch["conv_goal"].to(device),
                        batch["query_emb"].to(device),
                        batch["pos_audio"].to(device),
                        batch["pos_image"].to(device),
                        batch["pos_attr"].to(device),
                        batch["pos_lyrics"].to(device),
                        batch["pos_meta_emb"].to(device),
                        batch["pos_context"].to(device),
                        batch["pos_cf"].to(device),
                        batch["pos_ret_feat"].to(device),
                        batch["neg_audio"].to(device),
                        batch["neg_image"].to(device),
                        batch["neg_attr"].to(device),
                        batch["neg_lyrics"].to(device),
                        batch["neg_meta_emb"].to(device),
                        batch["neg_context"].to(device),
                        batch["neg_cf"].to(device),
                        batch["neg_ret_feat"].to(device),
                    ).item())
            vl_l = sum(val_losses) / len(val_losses)
            logger.info("Epoch %d  train=%.4f  val=%.4f  (random_baseline=%.4f)",
                        epoch, tr_l, vl_l, random_baseline)
            history.append({"epoch": epoch, "train_loss": tr_l, "val_loss": vl_l})

            if vl_l < best_val_loss:
                best_val_loss = vl_l
                patience_cnt  = 0
                _save(model, optimizer, scheduler, epoch, best_val_loss, history, args.out)
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
            _save(model, optimizer, scheduler, epoch, float("nan"), history, args.out)

    with open(args.out.replace(".pt", "_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Done. best_val_loss=%.4f", best_val_loss)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DCN-V2 Reranker (v4: trunc-concat + ret signals)")
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
    p.add_argument("--retrieval_path",     default=None)
    p.add_argument("--val_retrieval_path", default=None)
    p.add_argument("--cache_dir",          default="qwen/retrieval_indices")
    p.add_argument("--out",                default="checkpoints/dcn_reranker_best.pt")
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--num_neg",      type=int,   default=15)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--cross_layers", type=int,   default=3)
    p.add_argument("--deep_dims",    type=int,   nargs="+", default=[512, 256, 128])
    p.add_argument("--patience",     type=int,   default=8)
    p.add_argument("--num_workers",  type=int,   default=0)
    p.add_argument("--resume",       default=None)
    p.add_argument("--resume_best",  action="store_true")
    p.add_argument("--device",       default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
