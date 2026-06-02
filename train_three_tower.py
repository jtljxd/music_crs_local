"""
Three-Tower Reranker Training Script (v2)
==========================================

Sample construction:
  For each (session, turn_n):
    Positive: ALL tracks that appear in conversations (any turn) → label = 1.0
    Hard negatives (in-batch): 5 tracks from DIFFERENT session_ids in the same batch
    Super-hard negatives (in-recall): 5 tracks randomly sampled from retrieval candidates
      (BM25 + Qwen channel results, excluding the positive)
    Easy negatives: 10 tracks randomly sampled from the full track pool

Loss: BCEWithLogitsLoss over all rows in the batch.

Usage:
    python train_three_tower.py \
        --config config/llama1b_multi_channel_devset.yaml \
        --epochs 10 \
        --batch_size 32 \
        --lr 1e-3
"""

import argparse
import logging
import os
import random
from typing import Dict, List, Optional, Tuple, Set

import bm25s
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
#  BM25 helper (lightweight, no BERT for recall during training)
# ─────────────────────────────────────────────────────────────────────────────

def load_bm25(cache_dir: str) -> Tuple[Optional[object], Optional[List[str]]]:
    """Load pre-built BM25 index from multi_channel_retrieval cache."""
    bm25_dir = os.path.join(cache_dir, "multi_channel_retrieval", "bm25_index")
    ids_path = os.path.join(bm25_dir, "track_ids.json")
    if not os.path.exists(ids_path):
        return None, None
    import json
    model = bm25s.BM25.load(bm25_dir, load_corpus=True)
    with open(ids_path) as f:
        track_ids = json.load(f)
    return model, track_ids


def bm25_recall(model, track_ids: List[str], query: str, topk: int = 50) -> List[str]:
    tokens  = bm25s.tokenize([query.lower()])
    results = model.retrieve(tokens, k=min(topk, len(track_ids)), return_as="tuple")
    return [track_ids[item["id"]] for item in results.documents[0]]


# ─────────────────────────────────────────────────────────────────────────────
#  Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def build_training_samples(
    dataset_name: str,
    turn_store: Dict[str, torch.Tensor],
) -> List[Dict]:
    """
    Returns a list of sample dicts, one per (session, turn).
    All tracks appearing in conversations are positives (label=1.0).
    """
    logger.info("Loading conversation dataset …")
    ds = load_dataset(dataset_name, split="train")

    samples = []
    skipped = 0

    for item in tqdm(ds, desc="Building samples", unit="session"):
        session_id = item["session_id"]
        user_id    = item.get("user_id", None)
        convs      = item["conversations"]

        # Collect ALL music tracks in this session → all are positives
        music_turns: Dict[int, str] = {}
        for c in convs:
            if c.get("role") == "music" and c.get("content"):
                music_turns[int(c["turn_number"])] = c["content"]

        if not music_turns:
            continue

        # Collect all positive track ids in this session (for super-hard neg pool)
        session_positive_ids: Set[str] = set(music_turns.values())

        for turn_number, track_id in music_turns.items():
            q_key = f"{session_id}__{turn_number}_user"
            h_key = f"{session_id}__{turn_number}_history_avg"
            if q_key not in turn_store:
                skipped += 1
                continue

            query_emb   = turn_store[q_key].float()
            history_emb = (
                turn_store[h_key].float()
                if h_key in turn_store
                else torch.zeros_like(query_emb)
            )

            # user_query text (for BM25 recall)
            user_query = ""
            for c in convs:
                if int(c["turn_number"]) == turn_number and c["role"] == "user":
                    user_query = c.get("content", "")
                    break

            samples.append({
                "session_id":            session_id,
                "turn_number":           turn_number,
                "user_id":               user_id,
                "track_id":              track_id,
                "session_positive_ids":  session_positive_ids,
                "user_query":            user_query,
                "query_emb":             query_emb,
                "history_emb":           history_emb,
            })

    logger.info("Built %d training samples (skipped %d).", len(samples), skipped)
    return samples


# ─────────────────────────────────────────────────────────────────────────────
#  Feature building (mirrors ThreeTowerRerankerWrapper)
# ─────────────────────────────────────────────────────────────────────────────

TRACK_MODAL_COLS = ThreeTowerRerankerWrapper.TRACK_MODAL_COLS
TRACK_DIM        = ThreeTowerRerankerWrapper.TRACK_MODAL_TARGET_DIM
USER_DIM         = ThreeTowerRerankerWrapper.USER_CF_BPR_TARGET_DIM


def build_track_features(
    track_id: str,
    track_embeddings: Dict,
) -> Tuple[List[torch.Tensor], List[bool]]:
    if track_id in track_embeddings:
        embs    = track_embeddings[track_id]
        missing = set(embs["__missing__"])
    else:
        embs    = {}
        missing = set(TRACK_MODAL_COLS)

    modal_embs, missing_flags = [], []
    for col in TRACK_MODAL_COLS:
        if col in missing or col not in embs:
            modal_embs.append(torch.zeros(TRACK_DIM))
            missing_flags.append(True)
        else:
            modal_embs.append(embs[col])
            missing_flags.append(False)
    return modal_embs, missing_flags


def build_user_features(user_id, user_embeddings) -> Tuple[torch.Tensor, bool]:
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
    bm25_model=None,
    bm25_track_ids: Optional[List[str]] = None,
    num_hard_neg: int = 5,
    num_recall_neg: int = 5,
    num_easy_neg: int = 10,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    For each sample:
      1 positive (label=1.0)
      num_hard_neg   in-batch hard negatives (diff session_id)
      num_recall_neg super-hard negatives (from BM25 recall, excl. positives)
      num_easy_neg   easy negatives (random from all tracks)
    """
    rows_intent     = []
    rows_modal      = [[] for _ in TRACK_MODAL_COLS]
    rows_modal_mask = []
    rows_user_bpr   = []
    rows_cf_missing = []
    labels          = []

    batch_session_ids = [s["session_id"] for s in samples]
    batch_track_ids   = [s["track_id"]   for s in samples]

    def _add_row(s: Dict, track_id: str, lbl: float):
        intent  = torch.cat([s["query_emb"], s["history_emb"]])
        u_bpr, u_miss = build_user_features(s["user_id"], user_embeddings)
        me, mf = build_track_features(track_id, track_embeddings)

        rows_intent.append(intent)
        for j, emb in enumerate(me):
            rows_modal[j].append(emb)
        rows_modal_mask.append(mf)
        rows_user_bpr.append(u_bpr)
        rows_cf_missing.append(u_miss)
        labels.append(lbl)

    for i, s in enumerate(samples):
        # ── positive ──────────────────────────────────────────────────────
        _add_row(s, s["track_id"], 1.0)

        # ── in-batch hard negatives (different session) ────────────────────
        diff_indices = [
            j for j, sid in enumerate(batch_session_ids) if sid != s["session_id"]
        ]
        hard_picks = random.sample(
            diff_indices, min(num_hard_neg, len(diff_indices))
        ) if diff_indices else []
        for j in hard_picks:
            _add_row(s, batch_track_ids[j], 0.0)

        # ── super-hard negatives from recall ──────────────────────────────
        if bm25_model is not None and s["user_query"]:
            recall = bm25_recall(bm25_model, bm25_track_ids, s["user_query"], topk=50)
            # Exclude all session positives
            recall = [tid for tid in recall if tid not in s["session_positive_ids"]]
        else:
            recall = []
        recall_picks = random.sample(recall, min(num_recall_neg, len(recall))) if recall else []
        for tid in recall_picks:
            _add_row(s, tid, 0.0)

        # ── easy negatives (random) ────────────────────────────────────────
        for _ in range(num_easy_neg):
            easy_tid = random.choice(all_track_ids)
            _add_row(s, easy_tid, 0.0)

    # Stack tensors
    intent_tensor  = torch.stack(rows_intent).to(device)
    modal_tensors  = [torch.stack(rows_modal[j]).to(device) for j in range(len(TRACK_MODAL_COLS))]
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
    config = OmegaConf.load(args.config)
    device = args.device or config.get("device", "cuda")
    cache_dir       = config.get("cache_dir", "./cache")
    qwen_model_path = config.get(
        "qwen_model_path",
        "/home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B",
    )
    dataset_name = config.get(
        "conversation_dataset_name", "talkpl-ai/TalkPlayData-Challenge-Dataset"
    )

    # ── Load turn store ──────────────────────────────────────────────────────
    logger.info("Loading turn store from %s …", args.turn_store)
    turn_store: Dict[str, torch.Tensor] = torch.load(
        args.turn_store, map_location="cpu", weights_only=True
    )
    logger.info("  %d entries loaded.", len(turn_store))

    # ── Build training samples ───────────────────────────────────────────────
    samples = build_training_samples(dataset_name, turn_store)
    if not samples:
        logger.error("No training samples built.")
        return

    # ── Load reranker wrapper ────────────────────────────────────────────────
    logger.info("Initializing ThreeTowerRerankerWrapper …")
    wrapper = ThreeTowerRerankerWrapper(
        dataset_name           = dataset_name,
        track_emb_db_name      = config.get("track_emb_db_name",
            "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"),
        user_emb_db_name       = config.get("user_emb_db_name",
            "talkpl-ai/TalkPlayData-Challenge-User-Embeddings"),
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

    model      = wrapper.model
    optimizer  = wrapper.optimizer
    track_embs = wrapper.track_embeddings
    user_embs  = wrapper.user_embeddings
    all_tids   = list(track_embs.keys())

    # ── Load BM25 for super-hard negatives ───────────────────────────────────
    bm25_model, bm25_tids = load_bm25(cache_dir)
    if bm25_model is None:
        logger.warning(
            "BM25 index not found in %s; super-hard negatives will be skipped.\n"
            "Run inference once first to build the BM25 index, then retrain.",
            cache_dir,
        )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info("Training: %d samples, %d epochs, batch=%d", len(samples), args.epochs, args.batch_size)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        random.shuffle(samples)
        model.train()
        epoch_loss  = 0.0
        epoch_steps = 0

        pbar = tqdm(range(0, len(samples), args.batch_size),
                    desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for batch_start in pbar:
            batch_samples = samples[batch_start: batch_start + args.batch_size]
            if not batch_samples:
                continue

            model_inputs, label_tensor = collate_batch(
                batch_samples, all_tids, track_embs, user_embs, device,
                bm25_model=bm25_model, bm25_track_ids=bm25_tids,
                num_hard_neg=5, num_recall_neg=5, num_easy_neg=10,
            )

            scores = model(**model_inputs)
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
                logger.info("Step %d: checkpoint saved.", global_step)

        avg_loss = epoch_loss / max(epoch_steps, 1)
        logger.info("Epoch %d done. avg_loss=%.4f", epoch, avg_loss)
        wrapper.save_checkpoint()
        ep_path = os.path.join(checkpoint_dir, f"model_epoch{epoch}.pt")
        torch.save({"model_state_dict": model.state_dict()}, ep_path)
        logger.info("Epoch %d checkpoint → %s", epoch, ep_path)

    logger.info("Training complete. Model saved to qwen/three_tower_reranker/model.pt")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Three-Tower Reranker v2")
    p.add_argument("--config",      type=str,   default="config/llama1b_multi_channel_devset.yaml")
    p.add_argument("--turn_store",  type=str,   default="qwen/turn_embeddings.pt")
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--checkpoint_dir", type=str, default="qwen/three_tower_ckpt")
    p.add_argument("--save_every",  type=int,   default=500)
    p.add_argument("--device",      type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
