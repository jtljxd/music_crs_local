"""
scripts/reranking/train_intent_tower.py
========================================
Train the Query+Goal Intent Tower (two-tower contrastive model).

Dataset construction:
  - Positive pair:  (turn context, ground-truth track)
  - Negatives:      other tracks in the same batch (in-batch negatives)

Each training sample requires:
  User side:
    - query_emb [1024]          from qwen/dialogue_embeddings_{split}.pt
    - goal_emb  [1024]          from qwen/goal_embeddings_{split}_0.6b.pt
    - category_emb [8]          one-hot / bucket from query_split store
    - specificity_emb [16]      from query_split store
    - session_date_emb [14]     year_emb8 + month_emb4 + weekday_emb2
    - user_profile_emb [62]     from user metadata (age8+country16+gender2+lang4+culture32)

  Track side:
    - metadata_emb, lyrics_emb, attributes_emb, audio_emb, image_emb from IndexStore
    - cf_bpr [128]              from IndexStore
    - pop_bucket, year_bucket, duration_bucket [8 each]  from track metadata

Usage:
    python scripts/reranking/train_intent_tower.py \\
        --query_emb_path  qwen/dialogue_embeddings_train_0.6b.pt \\
        --goal_emb_path   qwen/goal_embeddings_train_0.6b.pt \\
        --cache_dir       qwen/retrieval_indices \\
        --out             checkpoints/intent_tower_best.pt \\
        --epochs 30 --batch_size 512 --lr 1e-3 --patience 5

nohup:
    nohup python scripts/reranking/train_intent_tower.py \\
        --query_emb_path  qwen/dialogue_embeddings_train_0.6b.pt \\
        --goal_emb_path   qwen/goal_embeddings_train_0.6b.pt \\
        --cache_dir       qwen/retrieval_indices \\
        --out             checkpoints/intent_tower_best.pt \\
        > logs/train_intent_tower.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

# Ensure project root is importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────

def _bucket_emb(value: Optional[float], n_bins: int,
                lo: float, hi: float) -> torch.Tensor:
    """Soft one-hot bucket embedding."""
    v = torch.zeros(n_bins)
    if value is None or (isinstance(value, float) and (value != value)):
        return v
    idx = int((float(value) - lo) / (hi - lo) * n_bins)
    idx = max(0, min(idx, n_bins - 1))
    v[idx] = 1.0
    return v


class IntentTowerDataset(Dataset):
    """One sample = one (session, turn) with known ground-truth track."""

    def __init__(
        self,
        samples:          List[dict],       # list of sample dicts
        index_store,                        # IndexStore
        query_store:      Dict,
        goal_store:       Dict,
        track_meta:       Dict[str, dict],  # track_id → metadata dict
        user_meta:        Dict[str, dict],  # user_id → metadata dict
    ):
        self.samples     = samples
        self.index       = index_store
        self.query_store = query_store
        self.goal_store  = goal_store
        self.track_meta  = track_meta
        self.user_meta   = user_meta

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        sid        = s["session_id"]
        turn       = s["turn_number"]
        gt_track   = s["gt_track_id"]
        user_id    = s.get("user_id", "")

        # ── User side ─────────────────────────────────────────────────────────
        query_emb = (
            self.query_store.get(f"{sid}_{turn}_query")
            or self.query_store.get(f"{sid}_{turn}")
            or torch.zeros(1024)
        )
        goal_emb = self.goal_store.get(sid, torch.zeros(1024))

        # Profile features from user metadata
        um = self.user_meta.get(user_id, {})
        age_emb     = _bucket_emb(um.get("age"),     8,   0,  80)
        country_emb = _bucket_emb(um.get("country_code_hash"), 16, 0, 1)
        gender_emb  = torch.zeros(2)
        if um.get("gender") == "male":
            gender_emb[0] = 1.0
        elif um.get("gender") == "female":
            gender_emb[1] = 1.0
        lang_emb    = _bucket_emb(um.get("preferred_language_hash"), 4, 0, 1)
        culture_emb = _bucket_emb(um.get("preferred_musical_culture_hash"), 32, 0, 1)
        profile_emb = torch.cat([age_emb, country_emb, gender_emb, lang_emb, culture_emb])  # 62

        # Session date features
        year_emb    = _bucket_emb(s.get("session_year"),    8, 2018, 2026)
        month_emb   = _bucket_emb(s.get("session_month"),   4,    1,   13)
        weekday_emb = _bucket_emb(s.get("session_weekday"), 2,    0,    7)
        date_emb    = torch.cat([year_emb, month_emb, weekday_emb])  # 14

        # category / specificity from query split (placeholder zeros if missing)
        category_emb   = torch.zeros(8)
        specificity_emb = torch.zeros(16)

        # ── Track side ────────────────────────────────────────────────────────
        def _get_track_vecs(track_id: str):
            meta_emb  = self.index.get_vec("metadata",   track_id) or torch.zeros(1024)
            lyr_emb   = self.index.get_vec("lyrics",     track_id) or torch.zeros(1024)
            attr_emb  = self.index.get_vec("attributes", track_id) or torch.zeros(1024)
            audio_emb = self.index.get_vec("audio",      track_id) or torch.zeros(512)
            image_emb = self.index.get_vec("image",      track_id) or torch.zeros(1152)
            cf_emb    = self.index.get_vec("cf_bpr",     track_id) or torch.zeros(128)
            tm = self.track_meta.get(track_id, {})
            pop_b  = _bucket_emb(tm.get("popularity"),     8,   0, 100)
            year_b = _bucket_emb(tm.get("release_year"),   8, 1950, 2030)
            dur_b  = _bucket_emb(tm.get("duration_ms"),    8,  30000, 600000)
            return (meta_emb.float(), lyr_emb.float(), attr_emb.float(),
                    audio_emb.float(), image_emb.float(), cf_emb.float(),
                    pop_b, year_b, dur_b)

        track_vecs = _get_track_vecs(gt_track)

        return {
            # User
            "query_emb":    query_emb.float(),
            "goal_emb":     goal_emb.float(),
            "category_emb": category_emb,
            "spec_emb":     specificity_emb,
            "date_emb":     date_emb,
            "profile_emb":  profile_emb,
            # Track (positive)
            "t_meta":   track_vecs[0],
            "t_lyrics": track_vecs[1],
            "t_attr":   track_vecs[2],
            "t_audio":  track_vecs[3],
            "t_image":  track_vecs[4],
            "t_cf":     track_vecs[5],
            "t_pop":    track_vecs[6],
            "t_year":   track_vecs[7],
            "t_dur":    track_vecs[8],
        }


# ── Build samples from conversation dataset ───────────────────────────────────

def build_samples(conv_dataset_name: str, split: str) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset(conv_dataset_name, split=split)
    samples = []
    for item in ds:
        sid     = str(item.get("session_id") or item.get("id") or "")
        user_id = str(item.get("user_id", ""))
        convs   = item.get("conversations", [])
        # Extract date info from first conversation
        session_year = None; session_month = None; session_weekday = None
        for c in convs:
            ts = c.get("timestamp") or c.get("created_at")
            if ts:
                try:
                    import datetime
                    dt = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    session_year    = dt.year
                    session_month   = dt.month
                    session_weekday = dt.weekday()
                    break
                except Exception:
                    pass

        for c in convs:
            if c.get("role") == "music" and c.get("content"):
                samples.append({
                    "session_id":    sid,
                    "user_id":       user_id,
                    "turn_number":   int(c["turn_number"]),
                    "gt_track_id":   c["content"],
                    "session_year":  session_year,
                    "session_month": session_month,
                    "session_weekday": session_weekday,
                })
    logger.info("Built %d training samples from %s/%s.", len(samples), conv_dataset_name, split)
    return samples


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    from mcrs.retrieval_modules.index_store import IndexStore
    from mcrs.tower_models.intent_tower import IntentTower
    from datasets import load_dataset

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    device = args.device

    # ── Load index ───────────────────────────────────────────────────────────
    logger.info("Loading IndexStore …")
    index = IndexStore.build(
        track_emb_dataset=args.track_emb_dataset,
        split_types=["all_tracks"],
        cache_dir=args.cache_dir,
        bge_tag_path=None,
        device="cpu",
    )

    # ── Load embedding stores ─────────────────────────────────────────────────
    def _lpt(p):
        if p and os.path.exists(p):
            s = torch.load(p, map_location="cpu", weights_only=True)
            logger.info("Loaded %s: %d entries", p, len(s))
            return s
        logger.warning("Not found: %s", p)
        return {}

    query_store = _lpt(args.query_emb_path)
    goal_store  = _lpt(args.goal_emb_path)

    # ── Load track metadata for bucket features ───────────────────────────────
    logger.info("Loading track metadata …")
    tm_ds = load_dataset(args.track_metadata_dataset)
    track_meta: Dict[str, dict] = {}
    for split_name in tm_ds:
        for row in tm_ds[split_name]:
            tid = str(row.get("track_id", ""))
            if tid:
                track_meta[tid] = {
                    "popularity":    row.get("popularity"),
                    "release_year":  row.get("release_year"),
                    "duration_ms":   row.get("duration_ms"),
                }

    # ── Load user metadata ────────────────────────────────────────────────────
    logger.info("Loading user metadata …")
    user_meta: Dict[str, dict] = {}
    try:
        u_ds = load_dataset(args.user_metadata_dataset)
        for split_name in u_ds:
            for row in u_ds[split_name]:
                uid = str(row.get("user_id", ""))
                if uid:
                    user_meta[uid] = dict(row)
    except Exception as e:
        logger.warning("User metadata load failed: %s", e)

    # ── Build samples ─────────────────────────────────────────────────────────
    train_samples = build_samples(args.conv_dataset, "train")
    val_samples   = build_samples(args.conv_dataset, "test")

    # Filter samples to those with valid embeddings and tracks in index
    valid_tracks = set(index.track_ids)
    def _filter(samples):
        out = []
        for s in samples:
            if s["gt_track_id"] in valid_tracks:
                qk = f"{s['session_id']}_{s['turn_number']}_query"
                if query_store.get(qk) is not None or query_store.get(f"{s['session_id']}_{s['turn_number']}") is not None:
                    out.append(s)
        return out

    train_samples = _filter(train_samples)
    val_samples   = _filter(val_samples)
    logger.info("Filtered → train=%d, val=%d", len(train_samples), len(val_samples))

    # ── DataLoaders ────────────────────────────────────────────────────────────
    def _make_loader(samples, shuffle):
        ds = IntentTowerDataset(samples, index, query_store, goal_store,
                                track_meta, user_meta)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True)

    train_loader = _make_loader(train_samples, shuffle=True)
    val_loader   = _make_loader(val_samples,   shuffle=False)

    # ── Model, optimiser ──────────────────────────────────────────────────────
    model = IntentTower(dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_loss = float("inf")
    patience_cnt  = 0
    history = []

    # ── Training ──────────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(
                batch["query_emb"], batch["goal_emb"], batch["category_emb"],
                batch["spec_emb"],  batch["date_emb"], batch["profile_emb"],
                batch["t_meta"],    batch["t_lyrics"],  batch["t_attr"],
                batch["t_audio"],   batch["t_image"],   batch["t_cf"],
                batch["t_pop"],     batch["t_year"],    batch["t_dur"],
            )
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out   = model(
                    batch["query_emb"], batch["goal_emb"], batch["category_emb"],
                    batch["spec_emb"],  batch["date_emb"], batch["profile_emb"],
                    batch["t_meta"],    batch["t_lyrics"],  batch["t_attr"],
                    batch["t_audio"],   batch["t_image"],   batch["t_cf"],
                    batch["t_pop"],     batch["t_year"],    batch["t_dur"],
                )
                val_losses.append(out["loss"].item())

        tr_l = sum(train_losses) / len(train_losses)
        vl_l = sum(val_losses)   / len(val_losses)
        logger.info("Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
                    epoch, args.epochs, tr_l, vl_l)
        history.append({"epoch": epoch, "train_loss": tr_l, "val_loss": vl_l})

        if vl_l < best_val_loss:
            best_val_loss = vl_l
            patience_cnt  = 0
            model.save(args.out)
            logger.info("  ✅ Saved best model → %s", args.out)
        else:
            patience_cnt += 1
            logger.info("  patience %d/%d", patience_cnt, args.patience)
            if patience_cnt >= args.patience:
                logger.info("Early stopping at epoch %d.", epoch)
                break

    # Save training history
    hist_path = args.out.replace(".pt", "_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)


def parse_args():
    p = argparse.ArgumentParser(description="Train Intent Tower (Query+Goal two-tower)")
    p.add_argument("--conv_dataset",          type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    p.add_argument("--track_emb_dataset",     type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    p.add_argument("--track_metadata_dataset",type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    p.add_argument("--user_metadata_dataset", type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-User-Metadata")
    p.add_argument("--query_emb_path",  type=str, required=True)
    p.add_argument("--goal_emb_path",   type=str, required=True)
    p.add_argument("--cache_dir",       type=str, default="qwen/retrieval_indices")
    p.add_argument("--out",             type=str, default="checkpoints/intent_tower_best.pt")
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--batch_size",      type=int,   default=512)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--dropout",         type=float, default=0.3)
    p.add_argument("--patience",        type=int,   default=5)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--device",          type=str,   default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
