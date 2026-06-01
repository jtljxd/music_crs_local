"""
Three-Tower Reranker Training Script
=====================================
Constructs training samples from TalkPlayData-Challenge-Dataset:

  For each (session, turn_n):
    - query_emb      = turn store lookup  {session_id}__{n}_user       [128]
    - history_emb    = turn store lookup  {session_id}__{n}_history_avg [128]
    - intent_features = cat([query_emb, history_emb])                   [256]

  Label assignment (from goal_progress_assessments):
    - MOVES_TOWARD_GOAL      → 1.0  (positive)
    - DOES_NOT_MOVE_TOWARD_GOAL → 0.3 (weak positive / soft label)

  Per-batch negatives:
    - 1 hard negative: the track recommended in a DIFFERENT session's
      same-indexed turn (in-batch hard neg)
    - 3 easy negatives: randomly sampled from all tracks              → label 0

Loss: weighted BCEWithLogitsLoss over all (positive + negative) samples in the batch.

Usage:
    python train_three_tower.py \
        --config config/llama1b_multi_channel_devset.yaml \
        --epochs 3 \
        --batch_size 64 \
        --lr 1e-3 \
        --checkpoint_dir qwen/three_tower_ckpt

    # checkpoint is also saved to cache/three_tower_reranker/model.pt (inference path)
"""

import argparse
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from mcrs.reranking_modules.three_tower_reranker import (
    ThreeTowerReranker,
    ThreeTowerRerankerWrapper,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def build_training_samples(
    dataset_name: str,
    turn_store: Dict[str, torch.Tensor],
) -> List[Dict]:
    """
    Returns a list of sample dicts, one per (session, turn):
    {
        session_id, turn_number, user_id,
        track_id,          # the recommended track at this turn
        label,             # 1.0 or 0.3
        query_emb,         # [128] float32
        history_emb,       # [128] float32
    }
    """
    logger.info("Loading conversation dataset …")
    ds = load_dataset(dataset_name, split="train")

    # goal_progress_assessments is a list of dicts per item;
    # keyed by turn_number (str) or int
    samples = []
    skipped = 0

    for item in tqdm(ds, desc="Building samples", unit="session"):
        session_id = item["session_id"]
        user_id    = item.get("user_id", None)
        convs      = item["conversations"]          # list of {turn_number, role, content}
        goals      = item.get("goal_progress_assessments", [])  # list of {turn_number, ...}

        # Build turn→label map from goal_progress_assessments
        goal_map: Dict[int, str] = {}
        for g in (goals or []):
            tn  = int(g.get("turn_number", -1))
            val = g.get("goal_progress", g.get("assessment", ""))
            goal_map[tn] = val

        # Build turn→track_id map (role == "music")
        music_turns: Dict[int, str] = {}
        for c in convs:
            if c.get("role") == "music":
                music_turns[int(c["turn_number"])] = c["content"]

        for turn_number, track_id in music_turns.items():
            if not track_id:
                skipped += 1
                continue

            # Look up embeddings
            q_key   = f"{session_id}__{turn_number}_user"
            h_key   = f"{session_id}__{turn_number}_history_avg"
            if q_key not in turn_store:
                skipped += 1
                continue

            query_emb   = turn_store[q_key].float()
            history_emb = (
                turn_store[h_key].float()
                if h_key in turn_store
                else torch.zeros_like(query_emb)
            )

            # Determine label
            assessment = goal_map.get(turn_number, "")
            if "MOVES_TOWARD_GOAL" in assessment:
                label = 1.0
            elif "DOES_NOT_MOVE_TOWARD_GOAL" in assessment:
                label = 0.3
            else:
                # No assessment available – treat as positive (system recommended)
                label = 1.0

            samples.append({
                "session_id":   session_id,
                "turn_number":  turn_number,
                "user_id":      user_id,
                "track_id":     track_id,
                "label":        label,
                "query_emb":    query_emb,      # [128]
                "history_emb":  history_emb,    # [128]
            })

    logger.info(
        "Built %d training samples (skipped %d).", len(samples), skipped
    )
    return samples


# ─────────────────────────────────────────────────────────────────────────────
#  Feature building helpers (mirrors ThreeTowerRerankerWrapper.rerank())
# ─────────────────────────────────────────────────────────────────────────────

TRACK_MODAL_COLS = ThreeTowerRerankerWrapper.TRACK_MODAL_COLS
TRACK_DIM        = ThreeTowerRerankerWrapper.TRACK_MODAL_TARGET_DIM
USER_DIM         = ThreeTowerRerankerWrapper.USER_CF_BPR_TARGET_DIM


def _raw_to_tensor(value, dim: int) -> Tuple[torch.Tensor, bool]:
    return ThreeTowerRerankerWrapper._raw_to_tensor(value, dim)


def build_track_features(
    track_id: str,
    track_embeddings: Dict,
) -> Tuple[List[torch.Tensor], List[bool]]:
    """Returns (modal_embs [6×dim], missing_flags [6])."""
    if track_id in track_embeddings:
        embs    = track_embeddings[track_id]
        missing = set(embs["__missing__"])
    else:
        embs    = {}
        missing = set(TRACK_MODAL_COLS)

    modal_embs    = []
    missing_flags = []
    for col in TRACK_MODAL_COLS:
        if col in missing or col not in embs:
            modal_embs.append(torch.zeros(TRACK_DIM))
            missing_flags.append(True)
        else:
            modal_embs.append(embs[col])
            missing_flags.append(False)
    return modal_embs, missing_flags


def build_user_features(
    user_id: Optional[str],
    user_embeddings: Dict,
) -> Tuple[torch.Tensor, bool]:
    """Returns (cf_bpr [128], is_missing)."""
    if user_id and user_id in user_embeddings:
        data = user_embeddings[user_id]
        return data["cf-bpr"], ("cf-bpr" in data["__missing__"])
    return torch.zeros(USER_DIM), True


# ─────────────────────────────────────────────────────────────────────────────
#  Batch collation
# ─────────────────────────────────────────────────────────────────────────────

def collate_batch(
    samples: List[Dict],
    all_track_ids: List[str],
    track_embeddings: Dict,
    user_embeddings: Dict,
    device: str,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    For each sample in the mini-batch:
      - 1 positive/soft-positive (label from dataset)
      - 1 hard negative: track from a different session in the same batch
      - 3 easy negatives: random tracks from all_track_ids

    Returns (model_inputs, labels) where model_inputs is a flat batch of
    len(samples) * 5 rows.
    """
    rows_intent     = []
    rows_modal      = [[] for _ in TRACK_MODAL_COLS]
    rows_modal_mask = []
    rows_user_bpr   = []
    rows_cf_missing = []
    labels          = []

    # Collect the tracks recommended in this mini-batch (for hard negatives)
    batch_track_ids = [s["track_id"] for s in samples]

    for i, s in enumerate(samples):
        intent = torch.cat([s["query_emb"], s["history_emb"]])   # [256]
        u_bpr, u_miss = build_user_features(s["user_id"], user_embeddings)

        def _add_row(track_id: str, lbl: float):
            modal_embs, miss_flags = build_track_features(track_id, track_embeddings)
            rows_intent.append(intent)
            for j, me in enumerate(modal_embs):
                rows_modal[j].append(me)
            rows_modal_mask.append(miss_flags)
            rows_user_bpr.append(u_bpr)
            rows_cf_missing.append(u_miss)
            labels.append(lbl)

        # Positive / soft-positive
        _add_row(s["track_id"], s["label"])

        # Hard negative: pick a track from another sample in this batch
        hard_neg_idx = (i + 1 + random.randint(0, len(samples) - 2)) % len(samples)
        hard_neg_tid = batch_track_ids[hard_neg_idx]
        if hard_neg_tid != s["track_id"]:
            _add_row(hard_neg_tid, 0.0)
        else:
            _add_row(random.choice(all_track_ids), 0.0)

        # 3 easy negatives
        for _ in range(3):
            easy_neg = random.choice(all_track_ids)
            _add_row(easy_neg, 0.0)

    # Stack everything
    intent_tensor  = torch.stack(rows_intent).to(device)          # [B*5, 256]
    modal_tensors  = [
        torch.stack(rows_modal[j]).to(device) for j in range(len(TRACK_MODAL_COLS))
    ]                                                               # 6 × [B*5, 128]
    modal_mask     = torch.tensor(rows_modal_mask, dtype=torch.bool, device=device)
    user_bpr       = torch.stack(rows_user_bpr).to(device)
    cf_missing     = torch.tensor(rows_cf_missing, dtype=torch.bool, device=device)
    label_tensor   = torch.tensor(labels, dtype=torch.float32, device=device)

    model_inputs = dict(
        intent_features    = intent_tensor,
        modal_embs         = modal_tensors,
        item_categorical   = {},
        user_cf_bpr        = user_bpr,
        user_categorical   = {},
        modal_missing_mask = modal_mask,
        cf_bpr_missing     = cf_missing,
    )
    return model_inputs, label_tensor


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    config          = OmegaConf.load(f"config/{args.config}" if not args.config.endswith(".yaml")
                                      else args.config)
    device          = config.get("device", "cuda") if args.device is None else args.device
    cache_dir       = config.get("cache_dir", "./cache")
    qwen_model_path = config.get(
        "qwen_model_path",
        "/home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B",
    )

    # ── Load turn store ──────────────────────────────────────────────────────
    turn_store_path = args.turn_store
    logger.info("Loading turn store from %s …", turn_store_path)
    turn_store: Dict[str, torch.Tensor] = torch.load(
        turn_store_path, map_location="cpu", weights_only=True
    )
    logger.info("  %d entries loaded.", len(turn_store))

    # ── Build training samples ───────────────────────────────────────────────
    dataset_name = config.get(
        "conversation_dataset_name", "talkpl-ai/TalkPlayData-Challenge-Dataset"
    )
    samples = build_training_samples(dataset_name, turn_store)
    if not samples:
        logger.error("No training samples built. Check turn store path and dataset.")
        return

    # ── Load reranker wrapper (track/user embeddings + vocab) ────────────────
    logger.info("Loading ThreeTowerRerankerWrapper for feature lookup …")
    wrapper = ThreeTowerRerankerWrapper(
        dataset_name          = dataset_name,
        track_emb_db_name     = config.get(
            "track_emb_db_name",
            "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        ),
        user_emb_db_name      = config.get(
            "user_emb_db_name",
            "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
        ),
        track_metadata_db_name = config.get("item_db_name",
            "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"),
        user_metadata_db_name  = config.get("user_db_name",
            "talkpl-ai/TalkPlayData-Challenge-User-Metadata"),
        split_types   = list(config.get("track_split_types", ["all_tracks"])),
        cache_dir     = cache_dir,
        qwen_model_path = qwen_model_path,
        device        = device,
        lr            = args.lr,
    )

    model       = wrapper.model
    optimizer   = wrapper.optimizer
    track_embs  = wrapper.track_embeddings
    user_embs   = wrapper.user_embeddings
    all_tids    = list(track_embs.keys())

    # Loss: weighted BCE
    loss_fn = torch.nn.BCEWithLogitsLoss()

    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info("Starting training: %d samples, %d epochs, batch=%d",
                len(samples), args.epochs, args.batch_size)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        random.shuffle(samples)
        model.train()
        epoch_loss   = 0.0
        epoch_steps  = 0

        pbar = tqdm(
            range(0, len(samples), args.batch_size),
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="batch",
        )
        for batch_start in pbar:
            batch_samples = samples[batch_start: batch_start + args.batch_size]
            if not batch_samples:
                continue

            model_inputs, label_tensor = collate_batch(
                batch_samples, all_tids, track_embs, user_embs, device
            )

            scores = model(**model_inputs)          # [B*5]
            loss   = loss_fn(scores, label_tensor)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss  += loss.item()
            epoch_steps += 1
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % args.save_every == 0:
                wrapper.save_checkpoint()
                _extra = os.path.join(checkpoint_dir, f"model_step{global_step}.pt")
                torch.save({"model_state_dict": model.state_dict()}, _extra)
                logger.info("Step %d: checkpoint saved.", global_step)

        avg_loss = epoch_loss / max(epoch_steps, 1)
        logger.info("Epoch %d done. avg_loss=%.4f", epoch, avg_loss)

        # Save end-of-epoch checkpoint
        wrapper.save_checkpoint()
        ep_path = os.path.join(checkpoint_dir, f"model_epoch{epoch}.pt")
        torch.save({"model_state_dict": model.state_dict()}, ep_path)
        logger.info("Epoch %d checkpoint → %s", epoch, ep_path)

    logger.info("Training complete. Final checkpoint at %s/model.pt",
                os.path.join(cache_dir, "three_tower_reranker"))


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Three-Tower Reranker")
    p.add_argument(
        "--config", type=str,
        default="config/llama1b_multi_channel_devset.yaml",
        help="Path to OmegaConf yaml config",
    )
    p.add_argument(
        "--turn_store", type=str,
        default="qwen/turn_embeddings.pt",
        help="Pre-computed turn embedding store (train split)",
    )
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument(
        "--checkpoint_dir", type=str,
        default="qwen/three_tower_ckpt",
        help="Directory for per-epoch/step checkpoints",
    )
    p.add_argument(
        "--save_every", type=int, default=500,
        help="Save checkpoint every N steps",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Override device (default: from config yaml)",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
