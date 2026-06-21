"""
scripts/reranking/train_cf_tower.py
=====================================
Train the CF-BPR Two-Tower model.

Input: pre-trained user_cf-bpr [128] and track_cf-bpr [128]
Output: compressed 32-dim user/track embeddings

Training: BPR loss with in-batch negatives (each positive paired with all
other tracks in the batch as negatives), plus L2 regularisation.

Requires: user_cf and track_cf embeddings in the Track/User metadata datasets.

Usage:
    python scripts/reranking/train_cf_tower.py \\
        --cache_dir qwen/retrieval_indices \\
        --out       checkpoints/cf_tower_best.pt \\
        --epochs 50 --batch_size 1024 --lr 1e-3 --patience 8

nohup:
    nohup python scripts/reranking/train_cf_tower.py \\
        --cache_dir qwen/retrieval_indices \\
        --out       checkpoints/cf_tower_best.pt \\
        > logs/train_cf_tower.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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

class CFBPRDataset(Dataset):
    """Each sample = (user_cf_128, pos_track_cf_128) pair.

    Negatives are all other tracks in the same batch (in-batch BPR).
    """

    def __init__(
        self,
        samples:      List[Tuple[str, str]],    # (user_id, track_id)
        user_cf_store:  Dict[str, torch.Tensor],  # user_id → [128]
        track_cf_store: Dict[str, torch.Tensor],  # track_id → [128]
    ):
        self.samples       = samples
        self.user_cf_store  = user_cf_store
        self.track_cf_store = track_cf_store

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        user_id, track_id = self.samples[idx]
        u_vec = self.user_cf_store.get(user_id, torch.zeros(128)).float()
        t_vec = self.track_cf_store.get(track_id, torch.zeros(128)).float()
        return {"user_cf": u_vec, "track_cf": t_vec}


def build_cf_samples(
    conv_dataset_name: str,
    user_cf_store: Dict[str, torch.Tensor],
    track_cf_store: Dict[str, torch.Tensor],
    split: str,
) -> List[Tuple[str, str]]:
    """Collect (user_id, gt_track_id) pairs where both CF vecs exist."""
    from datasets import load_dataset
    ds = load_dataset(conv_dataset_name, split=split)
    samples = []
    missing_u = missing_t = 0
    for item in ds:
        uid = str(item.get("user_id", ""))
        if uid not in user_cf_store:
            missing_u += 1
            continue
        for c in item.get("conversations", []):
            if c.get("role") == "music" and c.get("content"):
                tid = c["content"]
                if tid in track_cf_store:
                    samples.append((uid, tid))
                else:
                    missing_t += 1
    logger.info("CF samples %s: %d valid  (missing_user=%d, missing_track=%d)",
                split, len(samples), missing_u, missing_t)
    return samples


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    from mcrs.tower_models.cf_tower import CFTower
    from mcrs.retrieval_modules.index_store import IndexStore
    from datasets import load_dataset

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    device = args.device

    # ── Load user CF store ────────────────────────────────────────────────────
    logger.info("Loading user CF-BPR embeddings from %s …", args.user_emb_dataset)
    user_cf_store: Dict[str, torch.Tensor] = {}
    try:
        u_ds = load_dataset(args.user_emb_dataset)
        for sn in u_ds:
            for row in u_ds[sn]:
                uid = str(row.get("user_id", ""))
                v   = row.get("cf-bpr")
                if uid and v is not None:
                    try:
                        t = torch.tensor(v, dtype=torch.float32)
                        if t.numel() > 0:
                            user_cf_store[uid] = t
                    except Exception:
                        pass
        logger.info("  %d users with CF-BPR vecs.", len(user_cf_store))
    except Exception as e:
        logger.error("Failed to load user CF-BPR: %s", e)
        raise

    # ── Load track CF store from IndexStore ───────────────────────────────────
    logger.info("Loading track CF-BPR via IndexStore …")
    index = IndexStore.build(
        track_emb_dataset=args.track_emb_dataset,
        split_types=["all_tracks"],
        cache_dir=args.cache_dir,
        bge_tag_path=None,
        device="cpu",
    )
    if not index.has_modality("cf_bpr"):
        raise ValueError("IndexStore has no 'cf_bpr' modality. Run build_retrieval_indices first.")

    # Build track_cf_store dict from index matrices (CPU)
    cf_mat = index.matrices["cf_bpr"]  # [N, 128]
    track_cf_store: Dict[str, torch.Tensor] = {
        tid: cf_mat[i] for i, tid in enumerate(index.track_ids)
    }
    logger.info("  %d tracks with CF-BPR vecs.", len(track_cf_store))

    # ── Build samples ─────────────────────────────────────────────────────────
    train_samples = build_cf_samples(args.conv_dataset, user_cf_store, track_cf_store, "train")
    val_samples   = build_cf_samples(args.conv_dataset, user_cf_store, track_cf_store, "test")

    if not train_samples:
        logger.error("No training samples found. Check that user_cf-bpr and cf_bpr columns exist.")
        return

    def _make_loader(samples, shuffle):
        ds = CFBPRDataset(samples, user_cf_store, track_cf_store)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=True)   # drop_last avoids batch-size-1 BPR edge case

    train_loader = _make_loader(train_samples, shuffle=True)
    val_loader   = _make_loader(val_samples,   shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CFTower(dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    patience_cnt  = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            optimizer.zero_grad()
            user_cf  = batch["user_cf"].to(device)    # [B, 128]
            track_cf = batch["track_cf"].to(device)   # [B, 128]

            # Build in-batch negatives: for each sample, negatives = all OTHER tracks
            # Use contrastive NT-Xent instead of BPR for larger batch efficiency
            u_vec = F.normalize(model.user_tower(user_cf),   p=2, dim=1)
            t_vec = F.normalize(model.track_tower(track_cf), p=2, dim=1)
            logits = torch.matmul(u_vec, t_vec.T)             # [B, B]
            labels = torch.arange(logits.size(0), device=device)
            loss   = F.cross_entropy(logits / args.temperature, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                user_cf  = batch["user_cf"].to(device)
                track_cf = batch["track_cf"].to(device)
                u_vec = F.normalize(model.user_tower(user_cf),   p=2, dim=1)
                t_vec = F.normalize(model.track_tower(track_cf), p=2, dim=1)
                logits = torch.matmul(u_vec, t_vec.T) / args.temperature
                labels = torch.arange(logits.size(0), device=device)
                val_losses.append(F.cross_entropy(logits, labels).item())

        tr_l = sum(train_losses) / len(train_losses)
        vl_l = sum(val_losses)   / len(val_losses) if val_losses else float("inf")
        logger.info("Epoch %d/%d  train=%.4f  val=%.4f", epoch, args.epochs, tr_l, vl_l)
        history.append({"epoch": epoch, "train_loss": tr_l, "val_loss": vl_l})

        if vl_l < best_val_loss:
            best_val_loss = vl_l
            patience_cnt  = 0
            model.save(args.out)
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
    p = argparse.ArgumentParser(description="Train CF-BPR Two-Tower (128→32)")
    p.add_argument("--conv_dataset",           type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    p.add_argument("--track_emb_dataset",      type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    p.add_argument("--user_emb_dataset",        type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
                   help="HF dataset with user CF-BPR vectors (field: cf-bpr)")
    p.add_argument("--cache_dir",    type=str, default="qwen/retrieval_indices")
    p.add_argument("--out",          type=str, default="checkpoints/cf_tower_best.pt")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=1024)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--temperature",  type=float, default=0.05)
    p.add_argument("--patience",     type=int,   default=8)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--device",       type=str,   default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
