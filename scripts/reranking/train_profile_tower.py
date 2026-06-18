"""
scripts/reranking/train_profile_tower.py
=========================================
Train the User Profile Tower (demographic-only two-tower model).

User side input (76-dim total):
    age_bucket_8 + country_code_16 + gender_2 + preferred_language_4
    + preferred_musical_culture_32 + session_year_8 + session_month_4 + weekday_2

Track side input (112-dim total):
    tag_avg_32 + artist_avg_32 + album_avg_32
    + popularity_bucket_8 + release_year_bucket_8 + duration_bucket_8

Note: tag_avg/artist_avg/album_avg are derived from BGE tag embeddings reduced to 32-dim
      via a learned linear projection (trained jointly).

Usage:
    python scripts/reranking/train_profile_tower.py \\
        --bge_tag_path  bge/track_tag_embeddings.pt \\
        --cache_dir     qwen/retrieval_indices \\
        --out           checkpoints/profile_tower_best.pt \\
        --epochs 30 --batch_size 512

nohup:
    nohup python scripts/reranking/train_profile_tower.py \\
        --bge_tag_path  bge/track_tag_embeddings.pt \\
        --cache_dir     qwen/retrieval_indices \\
        --out           checkpoints/profile_tower_best.pt \\
        > logs/train_profile_tower.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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


# ── Bucket helper ─────────────────────────────────────────────────────────────

def _bucket_emb(value, n_bins: int, lo: float, hi: float) -> torch.Tensor:
    v = torch.zeros(n_bins)
    if value is None:
        return v
    try:
        idx = int((float(value) - lo) / (hi - lo) * n_bins)
        v[max(0, min(idx, n_bins - 1))] = 1.0
    except (ValueError, TypeError):
        pass
    return v


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_samples(conv_dataset_name: str, split: str) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset(conv_dataset_name, split=split)
    samples = []
    for item in ds:
        sid     = str(item.get("session_id") or item.get("id") or "")
        user_id = str(item.get("user_id", ""))
        convs   = item.get("conversations", [])
        # Date features
        session_year = session_month = session_weekday = None
        for c in convs:
            ts = c.get("timestamp") or c.get("created_at")
            if ts:
                try:
                    import datetime
                    dt = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    session_year = dt.year; session_month = dt.month
                    session_weekday = dt.weekday()
                    break
                except Exception:
                    pass
        for c in convs:
            if c.get("role") == "music" and c.get("content"):
                samples.append({
                    "session_id":       sid,
                    "user_id":          user_id,
                    "turn_number":      int(c["turn_number"]),
                    "gt_track_id":      c["content"],
                    "session_year":     session_year,
                    "session_month":    session_month,
                    "session_weekday":  session_weekday,
                })
    logger.info("Built %d samples from %s/%s.", len(samples), conv_dataset_name, split)
    return samples


class ProfileTowerDataset(Dataset):
    def __init__(
        self,
        samples:      List[dict],
        user_meta:    Dict[str, dict],
        track_meta:   Dict[str, dict],
        bge_tag_store: Dict[str, torch.Tensor],   # track_id → [384]
        tag_proj:     nn.Linear,                   # 384 → 32 (shared)
    ):
        self.samples       = samples
        self.user_meta     = user_meta
        self.track_meta    = track_meta
        self.bge_tag_store = bge_tag_store
        self.tag_proj      = tag_proj

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        user_id  = s.get("user_id", "")
        track_id = s["gt_track_id"]

        # ── User side ─────────────────────────────────────────────────────────
        um = self.user_meta.get(user_id, {})
        age_emb     = _bucket_emb(um.get("age"),                         8,   0,   80)
        country_emb = _bucket_emb(um.get("country_code_hash"),          16,   0,    1)
        gender_emb  = torch.zeros(2)
        g = str(um.get("gender", "")).lower()
        if "male" in g and "female" not in g:
            gender_emb[0] = 1.0
        elif "female" in g:
            gender_emb[1] = 1.0
        lang_emb    = _bucket_emb(um.get("preferred_language_hash"),     4,   0,    1)
        culture_emb = _bucket_emb(um.get("preferred_musical_culture_hash"), 32, 0,  1)
        year_emb    = _bucket_emb(s.get("session_year"),                 8, 2018, 2026)
        month_emb   = _bucket_emb(s.get("session_month"),               4,   1,   13)
        weekday_emb = _bucket_emb(s.get("session_weekday"),             2,   0,    7)

        # ── Track side ────────────────────────────────────────────────────────
        tm = self.track_meta.get(track_id, {})
        pop_b   = _bucket_emb(tm.get("popularity"),   8,     0, 100)
        year_b  = _bucket_emb(tm.get("release_year"), 8,  1950, 2030)
        dur_b   = _bucket_emb(tm.get("duration_ms"),  8, 30000, 600000)

        # tag_avg, artist_avg, album_avg: project BGE [384] → [32]
        bge_v = self.bge_tag_store.get(track_id, torch.zeros(384)).float()
        with torch.no_grad():
            tag32 = F.normalize(self.tag_proj(bge_v.unsqueeze(0)).squeeze(0), p=2, dim=0)
        # artist_avg and album_avg reuse tag projection as approximation
        artist32 = tag32.clone()
        album32  = tag32.clone()

        return {
            # User
            "age_emb":     age_emb,
            "country_emb": country_emb,
            "gender_emb":  gender_emb,
            "lang_emb":    lang_emb,
            "culture_emb": culture_emb,
            "year_emb":    year_emb,
            "month_emb":   month_emb,
            "weekday_emb": weekday_emb,
            # Track
            "tag_avg":     tag32,
            "artist_avg":  artist32,
            "album_avg":   album32,
            "pop_bucket":  pop_b,
            "year_bucket": year_b,
            "dur_bucket":  dur_b,
        }


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    from mcrs.tower_models.profile_tower import ProfileTower
    from datasets import load_dataset

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    device = args.device

    # ── Load data ─────────────────────────────────────────────────────────────
    bge_tag_store: Dict[str, torch.Tensor] = {}
    if args.bge_tag_path and os.path.exists(args.bge_tag_path):
        bge_tag_store = torch.load(args.bge_tag_path, map_location="cpu", weights_only=True)
        logger.info("Loaded BGE tag store: %d entries", len(bge_tag_store))

    logger.info("Loading track metadata …")
    tm_ds = load_dataset(args.track_metadata_dataset)
    track_meta: Dict[str, dict] = {}
    for sn in tm_ds:
        for row in tm_ds[sn]:
            tid = str(row.get("track_id", ""))
            if tid:
                track_meta[tid] = {k: row.get(k) for k in
                                   ("popularity", "release_year", "duration_ms")}

    logger.info("Loading user metadata …")
    user_meta: Dict[str, dict] = {}
    try:
        u_ds = load_dataset(args.user_metadata_dataset)
        for sn in u_ds:
            for row in u_ds[sn]:
                uid = str(row.get("user_id", ""))
                if uid:
                    user_meta[uid] = dict(row)
    except Exception as e:
        logger.warning("User metadata load failed: %s", e)

    # ── Build samples ─────────────────────────────────────────────────────────
    valid_tracks = set(track_meta.keys())
    def _filter(samples):
        return [s for s in samples if s["gt_track_id"] in valid_tracks]

    train_samples = _filter(build_samples(args.conv_dataset, "train"))
    val_samples   = _filter(build_samples(args.conv_dataset, "test"))
    logger.info("Filtered → train=%d, val=%d", len(train_samples), len(val_samples))

    # Shared linear projection 384 → 32 (trained jointly with the model)
    tag_proj = nn.Linear(384, 32, bias=False).to(device)

    def _make_loader(samples, shuffle):
        ds = ProfileTowerDataset(samples, user_meta, track_meta, bge_tag_store, tag_proj)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True)

    train_loader = _make_loader(train_samples, shuffle=True)
    val_loader   = _make_loader(val_samples,   shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ProfileTower(dropout=args.dropout).to(device)
    params = list(model.parameters()) + list(tag_proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    patience_cnt  = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train(); tag_proj.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(
                batch["age_emb"],    batch["country_emb"], batch["gender_emb"],
                batch["lang_emb"],   batch["culture_emb"],
                batch["year_emb"],   batch["month_emb"],   batch["weekday_emb"],
                batch["tag_avg"],    batch["artist_avg"],  batch["album_avg"],
                batch["pop_bucket"], batch["year_bucket"], batch["dur_bucket"],
            )
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            train_losses.append(out["loss"].item())
        scheduler.step()

        model.eval(); tag_proj.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out   = model(
                    batch["age_emb"],    batch["country_emb"], batch["gender_emb"],
                    batch["lang_emb"],   batch["culture_emb"],
                    batch["year_emb"],   batch["month_emb"],   batch["weekday_emb"],
                    batch["tag_avg"],    batch["artist_avg"],  batch["album_avg"],
                    batch["pop_bucket"], batch["year_bucket"], batch["dur_bucket"],
                )
                val_losses.append(out["loss"].item())

        tr_l = sum(train_losses) / len(train_losses)
        vl_l = sum(val_losses)   / len(val_losses)
        logger.info("Epoch %d/%d  train=%.4f  val=%.4f", epoch, args.epochs, tr_l, vl_l)
        history.append({"epoch": epoch, "train_loss": tr_l, "val_loss": vl_l})

        if vl_l < best_val_loss:
            best_val_loss = vl_l
            patience_cnt  = 0
            torch.save({
                "model_state":    model.state_dict(),
                "tag_proj_state": tag_proj.state_dict(),
            }, args.out)
            logger.info("  ✅ Saved → %s", args.out)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info("Early stopping at epoch %d.", epoch)
                break

    hist_path = args.out.replace(".pt", "_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Done. Best val_loss=%.4f", best_val_loss)


def parse_args():
    p = argparse.ArgumentParser(description="Train Profile Tower")
    p.add_argument("--conv_dataset",           type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    p.add_argument("--track_metadata_dataset", type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    p.add_argument("--user_metadata_dataset",  type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-User-Metadata")
    p.add_argument("--bge_tag_path",  type=str, default="bge/track_tag_embeddings.pt")
    p.add_argument("--cache_dir",     type=str, default="qwen/retrieval_indices")
    p.add_argument("--out",           type=str, default="checkpoints/profile_tower_best.pt")
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch_size",    type=int,   default=512)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--patience",      type=int,   default=5)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--device",        type=str,   default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
