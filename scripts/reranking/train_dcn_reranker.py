"""
scripts/reranking/train_dcn_reranker.py
========================================
Training script for DCNReranker.

Each sample = one (session, turn) positive pair.
Features assembled:
  user_profile  [86]   : age/country/gender/lang/culture/listen_cnt/session_cnt
  user_cf       [128]  : CF-BPR user embedding (zeros if unavailable)
  conv_goal     [1036] : category8 + goal_emb1024 + specificity4
  query_emb     [1024] : current-turn dialogue embedding
  track_semantic[4352] : CLAP512 + SigLIP768 + attr1024 + lyrics1024 + meta1024
  track_context [137]  : ISRC32 + tag32 + artist32 + album32 + logpop1 + dur_bucket8
  track_cf      [128]  : CF-BPR track embedding (zeros if unavailable)

Loss: in-batch softmax cross-entropy (InfoNCE style)
      score_matrix[i,j] = score(user_i, track_j)  [B, B]
      label[i] = i  (diagonal = positive)

Usage example:
  python scripts/reranking/train_dcn_reranker.py \
      --query_emb_path  qwen/hist_conversation_embeddings_train_0.6b.pt \
      --goal_emb_path   qwen/goal_embeddings_train_0.6b.pt \
      --bge_tag_path    bge/track_tag_embeddings.pt \
      --cache_dir       qwen/retrieval_indices \
      --out             checkpoints/dcn_reranker_best.pt \
      --epochs 10 --batch_size 256 --lr 1e-3 \
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bucket_emb(value, n_bins: int, lo: float, hi: float) -> torch.Tensor:
    t = torch.zeros(n_bins)
    if value is None:
        return t
    v = float(value)
    idx = int((v - lo) / (hi - lo + 1e-9) * n_bins)
    t[max(0, min(n_bins - 1, idx))] = 1.0
    return t


# ── Dataset ───────────────────────────────────────────────────────────────────

class DCNDataset(Dataset):
    """One sample = one (session, turn) → positive track."""

    def __init__(
        self,
        samples:        List[dict],
        index_store,                          # IndexStore (for track embeddings)
        query_store:    Dict,                 # {key → Tensor[1024]}
        goal_store:     Dict,                 # {session_id → Tensor[1024]}
        track_meta:     Dict[str, dict],      # track_id → {popularity, release_year, ...}
        user_meta:      Dict[str, dict],      # user_id → metadata dict
        user_cf_store:  Dict[str, torch.Tensor],  # user_id → Tensor[128]
        bge_tag_store:  Dict[str, torch.Tensor],  # track_id → Tensor[384]
        tag_proj:       nn.Linear,            # 384 → 32 shared projection
    ):
        self.samples       = samples
        self.index         = index_store
        self.query_store   = query_store
        self.goal_store    = goal_store
        self.track_meta    = track_meta
        self.user_meta     = user_meta
        self.user_cf_store = user_cf_store
        self.bge_tag_store = bge_tag_store
        self.tag_proj      = tag_proj

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s        = self.samples[idx]
        sid      = s["session_id"]
        user_id  = s["user_id"]
        turn     = s["turn_number"]
        gt_track = s["gt_track_id"]

        # ── user_profile [86] ──────────────────────────────────────────────
        um = self.user_meta.get(user_id, {})
        age = um.get("age")
        if age is not None:
            age_bucket = _bucket_emb(math.log1p(float(age)), 16, 0, math.log1p(100))
        else:
            age_bucket = torch.zeros(16)
        country_emb = _bucket_emb(um.get("country_code_hash"), 16, 0, 1)
        gender_emb  = torch.zeros(2)
        if um.get("gender") == "male":    gender_emb[0] = 1.0
        elif um.get("gender") == "female": gender_emb[1] = 1.0
        lang_emb    = _bucket_emb(um.get("preferred_language_hash"),       4,  0, 1)
        culture_emb = _bucket_emb(um.get("preferred_musical_culture_hash"), 32, 0, 1)
        listen_cnt  = _bucket_emb(
            math.log1p(float(um.get("listen_count",  0) or 0)), 8, 0, math.log1p(10000))
        session_cnt = _bucket_emb(
            math.log1p(float(um.get("session_count", 0) or 0)), 8, 0, math.log1p(1000))
        user_profile = torch.cat([
            age_bucket, country_emb, gender_emb, lang_emb, culture_emb,
            listen_cnt, session_cnt
        ])  # [86]

        # ── user_cf [128] ──────────────────────────────────────────────────
        ucf = self.user_cf_store.get(user_id)
        user_cf = ucf.float() if ucf is not None else torch.zeros(128)

        # ── conv_goal [1036] = category8 + goal_emb1024 + specificity4 ────
        goal_emb = self.goal_store.get(sid)
        goal_emb = goal_emb.float() if goal_emb is not None else torch.zeros(1024)
        category_emb   = torch.zeros(8)
        specificity_emb = torch.zeros(4)
        conv_goal = torch.cat([category_emb, goal_emb, specificity_emb])  # [1036]

        # ── query_emb [1024] ───────────────────────────────────────────────
        qk = f"{sid}_{turn}_query"
        q  = self.query_store.get(qk)
        if q is None:
            q = self.query_store.get(f"{sid}_{turn}")
        query_emb = q.float() if q is not None else torch.zeros(1024)

        # ── track features ─────────────────────────────────────────────────
        def _gv(mod, dim):
            v = self.index.get_vec(mod, gt_track)
            return v.float() if v is not None else torch.zeros(dim)

        clap_emb  = _gv("audio",      512)   # CLAP
        siglip_emb = _gv("image",     768)   # SigLIP
        attr_emb  = _gv("attributes", 1024)
        lyr_emb   = _gv("lyrics",     1024)
        meta_emb  = _gv("metadata",   1024)
        track_semantic = torch.cat([clap_emb, siglip_emb, attr_emb, lyr_emb, meta_emb])  # [4352]

        # track_context [137] = ISRC32+tag32+artist32+album32+logpop1+dur_bucket8
        # ISRC/artist/album: use 32-dim BGE tag projections (approximation)
        proj_device = next(self.tag_proj.parameters()).device
        bge_v = self.bge_tag_store.get(gt_track, torch.zeros(384)).float()
        with torch.no_grad():
            tag32 = F.normalize(
                self.tag_proj(bge_v.unsqueeze(0).to(proj_device)).squeeze(0), p=2, dim=0
            ).cpu()
        isrc32   = tag32.clone()    # placeholder: same projection
        artist32 = tag32.clone()
        album32  = tag32.clone()
        tm = self.track_meta.get(gt_track, {})
        log_pop = torch.tensor([math.log1p(float(tm.get("popularity", 0) or 0))])
        dur_bucket = _bucket_emb(tm.get("duration_ms"), 8, 30000, 600000)
        track_context = torch.cat([isrc32, tag32, artist32, album32, log_pop, dur_bucket])  # [137]

        # track_cf [128]
        tcf = self.index.get_vec("cf_bpr", gt_track)
        track_cf = tcf.float() if tcf is not None else torch.zeros(128)

        return {
            "user_profile":   user_profile,    # [86]
            "user_cf":        user_cf,          # [128]
            "conv_goal":      conv_goal,        # [1036]
            "query_emb":      query_emb,        # [1024]
            "track_semantic": track_semantic,   # [4352]
            "track_context":  track_context,    # [137]
            "track_cf":       track_cf,         # [128]
        }


# ── Build samples ─────────────────────────────────────────────────────────────

def build_samples(conv_dataset_name: str, split: str) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset(conv_dataset_name, split=split)
    samples = []
    for item in ds:
        sid     = str(item.get("session_id") or item.get("id") or "")
        user_id = str(item.get("user_id", ""))
        convs   = item.get("conversations", [])
        session_year = session_month = session_weekday = None
        for c in convs:
            ts = c.get("timestamp") or c.get("created_at")
            if ts:
                try:
                    import datetime
                    dt = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    session_year, session_month, session_weekday = dt.year, dt.month, dt.weekday()
                    break
                except Exception:
                    pass
        for c in convs:
            if c.get("role") == "music" and c.get("content"):
                samples.append({
                    "session_id":      sid,
                    "user_id":         user_id,
                    "turn_number":     int(c["turn_number"]),
                    "gt_track_id":     c["content"],
                    "session_year":    session_year,
                    "session_month":   session_month,
                    "session_weekday": session_weekday,
                })
    logger.info("Built %d samples from %s/%s.", len(samples), conv_dataset_name, split)
    return samples


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from mcrs.retrieval_modules.index_store import IndexStore
    from mcrs.reranking_modules.dcn_reranker import DCNReranker

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    device = args.device

    # ── Index ────────────────────────────────────────────────────────────
    logger.info("Loading IndexStore …")
    index = IndexStore.build(
        track_emb_dataset=args.track_emb_dataset,
        split_types=["all_tracks"],
        cache_dir=args.cache_dir,
        bge_tag_path=args.bge_tag_path,
        device="cpu",
    )
    valid_tracks = set(index.track_ids)

    # ── Embedding stores ─────────────────────────────────────────────────
    def _lpt(p, name):
        if p and os.path.exists(p):
            s = torch.load(p, map_location="cpu", weights_only=True)
            logger.info("Loaded %s: %d entries", name, len(s))
            return s
        logger.warning("Not found: %s", p)
        return {}

    query_store = _lpt(args.query_emb_path, "query_store")
    goal_store  = _lpt(args.goal_emb_path,  "goal_store")

    # ── Track metadata ────────────────────────────────────────────────────
    logger.info("Loading track metadata …")
    track_meta: Dict[str, dict] = {}
    tm_ds = load_dataset(args.track_metadata_dataset)
    for sp in tm_ds:
        for row in tm_ds[sp]:
            tid = str(row.get("track_id", ""))
            if tid:
                track_meta[tid] = {
                    "popularity":   row.get("popularity"),
                    "release_year": row.get("release_year"),
                    "duration_ms":  row.get("duration_ms"),
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
        logger.warning("User metadata load failed: %s", e)

    # ── User CF store ─────────────────────────────────────────────────────
    logger.info("Loading user CF-BPR embeddings …")
    user_cf_store: Dict[str, torch.Tensor] = {}
    try:
        ue_ds = load_dataset(args.user_emb_dataset)
        for sp in ue_ds:
            for row in ue_ds[sp]:
                uid = str(row.get("user_id", ""))
                v = row.get("cf-bpr")
                if uid and v is not None:
                    t = torch.tensor(v, dtype=torch.float32)
                    if t.numel() > 0:
                        user_cf_store[uid] = t
        logger.info("  %d users with CF embeddings.", len(user_cf_store))
    except Exception as e:
        logger.warning("User CF load failed: %s", e)

    # ── BGE tag store + shared projection ─────────────────────────────────
    bge_tag_store: Dict[str, torch.Tensor] = {}
    if args.bge_tag_path and os.path.exists(args.bge_tag_path):
        bge_tag_store = torch.load(args.bge_tag_path, map_location="cpu", weights_only=True)
        logger.info("BGE tag store: %d entries", len(bge_tag_store))
    tag_proj = nn.Linear(384, 32, bias=False).to(device)   # shared, trained together

    # ── Samples ───────────────────────────────────────────────────────────
    def _filter(samples):
        out = []
        for s in samples:
            if s["gt_track_id"] not in valid_tracks:
                continue
            qk = f"{s['session_id']}_{s['turn_number']}_query"
            if (query_store.get(qk) is not None
                    or query_store.get(f"{s['session_id']}_{s['turn_number']}") is not None):
                out.append(s)
        return out

    train_samples = _filter(build_samples(args.conv_dataset, "train"))

    # Validation: use separate emb stores if provided
    val_query_store = _lpt(args.val_query_emb_path, "val_query") if args.val_query_emb_path else query_store
    val_goal_store  = _lpt(args.val_goal_emb_path,  "val_goal")  if args.val_goal_emb_path  else goal_store

    def _filter_val(samples):
        out = []
        for s in samples:
            if s["gt_track_id"] not in valid_tracks:
                continue
            qk = f"{s['session_id']}_{s['turn_number']}_query"
            if (val_query_store.get(qk) is not None
                    or val_query_store.get(f"{s['session_id']}_{s['turn_number']}") is not None):
                out.append(s)
        return out

    val_samples = _filter_val(build_samples(args.conv_dataset, "test"))
    logger.info("Filtered → train=%d, val=%d", len(train_samples), len(val_samples))

    # ── DataLoaders ───────────────────────────────────────────────────────
    def _make_ds(samples, q_store, g_store):
        return DCNDataset(
            samples, index, q_store, g_store,
            track_meta, user_meta, user_cf_store,
            bge_tag_store, tag_proj,
        )

    train_loader = DataLoader(
        _make_ds(train_samples, query_store, goal_store),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        _make_ds(val_samples, val_query_store, val_goal_store),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    ) if val_samples else None

    # ── Model + optimizer ─────────────────────────────────────────────────
    model = DCNReranker(
        cross_layers=args.cross_layers,
        deep_dims=tuple(args.deep_dims),
        dropout=args.dropout,
    ).to(device)

    # Include tag_proj params in optimizer
    params = list(model.parameters()) + list(tag_proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    patience_cnt  = 0
    history       = []

    # ── Training epochs ───────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        tag_proj.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            out = model(
                batch["user_profile"],
                batch["user_cf"],
                batch["conv_goal"],
                batch["query_emb"],
                batch["track_semantic"],
                batch["track_context"],
                batch["track_cf"],
            )
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        tr_l = sum(train_losses) / len(train_losses)

        # Validation
        if val_loader:
            model.eval()
            tag_proj.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    out = model(
                        batch["user_profile"], batch["user_cf"],
                        batch["conv_goal"],    batch["query_emb"],
                        batch["track_semantic"], batch["track_context"], batch["track_cf"],
                    )
                    val_losses.append(out["loss"].item())
            vl_l = sum(val_losses) / len(val_losses)
            logger.info("Epoch %d/%d  train=%.4f  val=%.4f", epoch, args.epochs, tr_l, vl_l)
            history.append({"epoch": epoch, "train_loss": tr_l, "val_loss": vl_l})

            if vl_l < best_val_loss:
                best_val_loss = vl_l
                patience_cnt  = 0
                model.save(args.out)
                torch.save({"state_dict": tag_proj.state_dict()},
                           args.out.replace(".pt", "_tag_proj.pt"))
                logger.info("  ✅ Saved → %s", args.out)
            else:
                patience_cnt += 1
                logger.info("  patience %d/%d", patience_cnt, args.patience)
                if patience_cnt >= args.patience:
                    logger.info("Early stopping at epoch %d.", epoch)
                    break
        else:
            logger.info("Epoch %d/%d  train=%.4f  (no val, saving)", epoch, args.epochs, tr_l)
            history.append({"epoch": epoch, "train_loss": tr_l, "val_loss": float("nan")})
            model.save(args.out)

    hist_path = args.out.replace(".pt", "_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Done. Best val_loss=%.4f", best_val_loss)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DCN-V2 Reranker")
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
    p.add_argument("--query_emb_path",   required=True,  type=str)
    p.add_argument("--goal_emb_path",    required=True,  type=str)
    p.add_argument("--val_query_emb_path", type=str, default=None)
    p.add_argument("--val_goal_emb_path",  type=str, default=None)
    p.add_argument("--bge_tag_path",     type=str,  default="bge/track_tag_embeddings.pt")
    p.add_argument("--cache_dir",        type=str,  default="qwen/retrieval_indices")
    p.add_argument("--out",              type=str,  default="checkpoints/dcn_reranker_best.pt")
    p.add_argument("--epochs",           type=int,  default=10)
    p.add_argument("--batch_size",       type=int,  default=256)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--weight_decay",     type=float, default=1e-4)
    p.add_argument("--dropout",          type=float, default=0.2)
    p.add_argument("--cross_layers",     type=int,  default=3)
    p.add_argument("--deep_dims",        type=int,  nargs="+", default=[512, 256, 128])
    p.add_argument("--patience",         type=int,  default=5)
    p.add_argument("--num_workers",      type=int,  default=0)
    p.add_argument("--device",           type=str,  default="cuda",
                   choices=["cuda", "cpu"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
