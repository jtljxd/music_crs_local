"""
Three-Tower Reranker Training Script (v3) — DCN + PPNet
========================================================

New features vs v2:
  - emb key format: {session_id}_{turn}_query / {session_id}_{turn}_history (1024-dim, 0.6B)
  - Model: DCN (3 cross layers) + PPNet-DNN gated by 1024-dim query emb
  - Label: any track in conversations = positive (label=1.0)
  - Negatives per positive:
      5  in-batch hard (different session_id)
      5  super-hard from recall (BM25, excl. session positives)
      10 easy (random from full track pool)
  - Training schedule:
      Phase 1 — train split,  5 epochs
      Phase 2 — test  split,  2 epochs  (fine-tune, lower lr)

Usage (background):
    nohup python train_three_tower.py \
        --config config/llama1b_multi_channel_devset.yaml \
        --train_store qwen/dialogue_embeddings_train_0.6b.pt \
        --test_store  qwen/dialogue_embeddings_test_0.6b.pt \
        --train_epochs 5 \
        --test_epochs  2 \
        --batch_size 32 \
        --lr 1e-3 \
        --lr_finetune 3e-4 \
    > train_three_tower.log 2>&1 &
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

# emb dimension (0.6B model output)
EMB_DIM = 1024


# ─────────────────────────────────────────────────────────────────────────────
#  BM25 helper
# ─────────────────────────────────────────────────────────────────────────────

def load_or_build_bm25(
    cache_dir: str,
    track_metadata_dict: Dict,
) -> Tuple[object, List[str]]:
    """Load BM25 index from cache, or build it from track_metadata_dict."""
    import json
    bm25_dir = os.path.join(cache_dir, "multi_channel_retrieval", "bm25_index")
    ids_path = os.path.join(bm25_dir, "track_ids.json")

    if os.path.exists(ids_path):
        logger.info("Loading cached BM25 index from %s …", bm25_dir)
        model = bm25s.BM25.load(bm25_dir, load_corpus=True)
        with open(ids_path) as f:
            track_ids = json.load(f)
        logger.info("  BM25 index loaded (%d tracks).", len(track_ids))
        return model, track_ids

    logger.info("BM25 index not found. Building from track metadata …")
    os.makedirs(bm25_dir, exist_ok=True)

    track_ids = list(track_metadata_dict.keys())
    corpus = []
    for tid in track_ids:
        meta  = track_metadata_dict[tid]
        parts = []
        for field in ["track_name", "artist_name", "album_name", "tag_list", "release_date"]:
            val = meta.get(field, "")
            if isinstance(val, list):
                val = " ".join(str(v) for v in val)
            if val:
                parts.append(str(val))
        corpus.append(" ".join(parts))

    tokens  = bm25s.tokenize(corpus)
    bm25_m  = bm25s.BM25()
    bm25_m.index(tokens)
    bm25_m.save(bm25_dir, corpus=corpus)
    with open(ids_path, "w") as f:
        json.dump(track_ids, f)

    logger.info("BM25 index built and saved (%d tracks).", len(track_ids))
    bm25_m = bm25s.BM25.load(bm25_dir, load_corpus=True)
    return bm25_m, track_ids


def bm25_recall(model, track_ids: List[str], query: str, topk: int = 50) -> List[str]:
    tokens  = bm25s.tokenize([query.lower()])
    results = model.retrieve(tokens, k=min(topk, len(track_ids)), return_as="tuple")
    return [track_ids[item["id"]] for item in results.documents[0]]


# ─────────────────────────────────────────────────────────────────────────────
#  Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def _get_emb(store: Dict, key: str, dim: int = EMB_DIM) -> torch.Tensor:
    """Fetch tensor from store, truncate/pad to dim, float32."""
    if key in store:
        v = store[key].float()
        if v.shape[0] > dim:
            v = v[:dim]
        elif v.shape[0] < dim:
            v = F.pad(v, (0, dim - v.shape[0]))
        return v
    return torch.zeros(dim, dtype=torch.float32)


def build_training_samples(
    dataset_name: str,
    split: str,
    turn_store: Dict[str, torch.Tensor],
) -> List[Dict]:
    """
    Returns a list of sample dicts, one per (session, music-turn).
    All tracks appearing in conversations are positives (label=1.0).
    New key format: {session_id}_{turn}_query / {session_id}_{turn}_history
    """
    logger.info("Loading conversation dataset '%s' split='%s' …", dataset_name, split)
    ds = load_dataset(dataset_name, split=split)

    samples = []
    skipped = 0

    for item in tqdm(ds, desc=f"Building samples [{split}]", unit="session"):
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

        session_positive_ids: Set[str] = set(music_turns.values())

        for turn_number, track_id in music_turns.items():
            q_key = f"{session_id}_{turn_number}_query"
            h_key = f"{session_id}_{turn_number}_history"
            if q_key not in turn_store:
                skipped += 1
                continue

            query_emb   = _get_emb(turn_store, q_key)
            history_emb = _get_emb(turn_store, h_key)

            # user query text for BM25 recall
            user_query = ""
            for c in convs:
                if int(c["turn_number"]) == turn_number and c["role"] == "user":
                    user_query = c.get("content", "")
                    break

            samples.append({
                "session_id":           session_id,
                "turn_number":          turn_number,
                "user_id":              user_id,
                "track_id":             track_id,
                "session_positive_ids": session_positive_ids,
                "user_query":           user_query,
                "query_emb":            query_emb,   # [1024]
                "history_emb":          history_emb, # [1024]
            })

    logger.info("Built %d samples from '%s' (skipped %d).", len(samples), split, skipped)
    return samples


# ─────────────────────────────────────────────────────────────────────────────
#  Feature building
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
    Per sample:
      1  positive  (label=1.0)
      num_hard_neg  in-batch hard negatives  (different session_id)
      num_recall_neg super-hard recall negatives (BM25, excl. session positives)
      num_easy_neg   easy negatives (random)
    """
    rows_intent     = []
    rows_gate       = []   # PPNet gate: raw query_emb [1024]
    rows_modal      = [[] for _ in TRACK_MODAL_COLS]
    rows_modal_mask = []
    rows_user_bpr   = []
    rows_cf_missing = []
    labels          = []

    batch_session_ids = [s["session_id"] for s in samples]
    batch_track_ids   = [s["track_id"]   for s in samples]

    def _add_row(s: Dict, track_id: str, lbl: float):
        # intent = concat(query_emb, history_emb) [2048]
        intent = torch.cat([s["query_emb"], s["history_emb"]])
        gate   = s["query_emb"]   # [1024] for PPNet gate
        u_bpr, u_miss = build_user_features(s["user_id"], user_embeddings)
        me, mf = build_track_features(track_id, track_embeddings)

        rows_intent.append(intent)
        rows_gate.append(gate)
        for j, emb in enumerate(me):
            rows_modal[j].append(emb)
        rows_modal_mask.append(mf)
        rows_user_bpr.append(u_bpr)
        rows_cf_missing.append(u_miss)
        labels.append(lbl)

    for i, s in enumerate(samples):
        # positive
        _add_row(s, s["track_id"], 1.0)

        # in-batch hard negatives (different session)
        diff_idx = [j for j, sid in enumerate(batch_session_ids) if sid != s["session_id"]]
        for j in (random.sample(diff_idx, min(num_hard_neg, len(diff_idx))) if diff_idx else []):
            _add_row(s, batch_track_ids[j], 0.0)

        # super-hard recall negatives (BM25)
        recall = []
        if bm25_model is not None and s["user_query"]:
            recall_raw = bm25_recall(bm25_model, bm25_track_ids, s["user_query"], topk=50)
            recall = [t for t in recall_raw if t not in s["session_positive_ids"]]
        for tid in (random.sample(recall, min(num_recall_neg, len(recall))) if recall else []):
            _add_row(s, tid, 0.0)

        # easy negatives (random)
        for _ in range(num_easy_neg):
            _add_row(s, random.choice(all_track_ids), 0.0)

    intent_tensor  = torch.stack(rows_intent).to(device)     # [B, 2048]
    gate_tensor    = torch.stack(rows_gate).to(device)        # [B, 1024]
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
        query_gate_emb     = gate_tensor,
        modal_missing_mask = modal_mask,
        cf_bpr_missing     = cf_missing,
    )
    return model_inputs, label_tensor


# ─────────────────────────────────────────────────────────────────────────────
#  Core training loop (single phase)
# ─────────────────────────────────────────────────────────────────────────────

def run_phase(
    phase_name: str,
    samples: List[Dict],
    model: ThreeTowerReranker,
    optimizer: torch.optim.Optimizer,
    track_embs: Dict,
    user_embs: Dict,
    all_tids: List[str],
    bm25_model,
    bm25_tids: List[str],
    wrapper: "ThreeTowerRerankerWrapper",
    args,
    epochs: int,
    device: str,
    checkpoint_dir: str,
):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    logger.info("[%s] %d samples, %d epochs, batch=%d", phase_name, len(samples), epochs, args.batch_size)

    global_step = 0
    for epoch in range(1, epochs + 1):
        random.shuffle(samples)
        model.train()
        epoch_loss, epoch_steps = 0.0, 0

        pbar = tqdm(
            range(0, len(samples), args.batch_size),
            desc=f"[{phase_name}] Epoch {epoch}/{epochs}",
            unit="batch",
        )
        for batch_start in pbar:
            batch = samples[batch_start: batch_start + args.batch_size]
            if not batch:
                continue

            model_inputs, label_tensor = collate_batch(
                batch, all_tids, track_embs, user_embs, device,
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
                logger.info("[%s] Step %d: checkpoint saved.", phase_name, global_step)

        avg_loss = epoch_loss / max(epoch_steps, 1)
        logger.info("[%s] Epoch %d done. avg_loss=%.4f", phase_name, epoch, avg_loss)

        # Save per-epoch checkpoint
        wrapper.save_checkpoint()
        ep_path = os.path.join(checkpoint_dir, f"model_{phase_name}_epoch{epoch}.pt")
        torch.save({"model_state_dict": model.state_dict()}, ep_path)
        logger.info("[%s] Epoch %d checkpoint → %s", phase_name, epoch, ep_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry
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

    # ── Load turn stores ─────────────────────────────────────────────────────
    logger.info("Loading train turn store from %s …", args.train_store)
    train_store: Dict[str, torch.Tensor] = torch.load(
        args.train_store, map_location="cpu", weights_only=True
    )
    logger.info("  train: %d entries.", len(train_store))

    logger.info("Loading test turn store from %s …", args.test_store)
    test_store: Dict[str, torch.Tensor] = torch.load(
        args.test_store, map_location="cpu", weights_only=True
    )
    logger.info("  test:  %d entries.", len(test_store))

    # ── Build samples for both splits ────────────────────────────────────────
    train_samples = build_training_samples(dataset_name, "train", train_store)
    test_samples  = build_training_samples(dataset_name, "test",  test_store)

    if not train_samples:
        logger.error("No training samples built.")
        return

    # ── Initialize wrapper ───────────────────────────────────────────────────
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
    track_embs = wrapper.track_embeddings
    user_embs  = wrapper.user_embeddings
    all_tids   = list(track_embs.keys())

    # ── BM25 for super-hard negatives ────────────────────────────────────────
    bm25_model, bm25_tids = load_or_build_bm25(cache_dir, wrapper.track_metadata)

    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Phase 1: train split, 5 epochs ───────────────────────────────────────
    optimizer = wrapper.optimizer   # Adam, lr=args.lr
    run_phase(
        phase_name     = "train",
        samples        = train_samples,
        model          = model,
        optimizer      = optimizer,
        track_embs     = track_embs,
        user_embs      = user_embs,
        all_tids       = all_tids,
        bm25_model     = bm25_model,
        bm25_tids      = bm25_tids,
        wrapper        = wrapper,
        args           = args,
        epochs         = args.train_epochs,
        device         = device,
        checkpoint_dir = checkpoint_dir,
    )

    # ── Phase 2: test split, 2 epochs (lower lr) ─────────────────────────────
    if test_samples and args.test_epochs > 0:
        logger.info("Phase 2: fine-tuning on test split (lr=%.2e) …", args.lr_finetune)
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr_finetune
        run_phase(
            phase_name     = "test_ft",
            samples        = test_samples,
            model          = model,
            optimizer      = optimizer,
            track_embs     = track_embs,
            user_embs      = user_embs,
            all_tids       = all_tids,
            bm25_model     = bm25_model,
            bm25_tids      = bm25_tids,
            wrapper        = wrapper,
            args           = args,
            epochs         = args.test_epochs,
            device         = device,
            checkpoint_dir = checkpoint_dir,
        )

    logger.info("All phases done. Final model: qwen/three_tower_reranker/model.pt")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Three-Tower Reranker v3 (DCN+PPNet)")
    p.add_argument("--config",         type=str,   default="config/llama1b_multi_channel_devset.yaml")
    p.add_argument("--train_store",    type=str,   default="qwen/dialogue_embeddings_train_0.6b.pt",
                   help="turn store for train split")
    p.add_argument("--test_store",     type=str,   default="qwen/dialogue_embeddings_test_0.6b.pt",
                   help="turn store for test split")
    p.add_argument("--train_epochs",   type=int,   default=5)
    p.add_argument("--test_epochs",    type=int,   default=2)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--lr_finetune",    type=float, default=3e-4,
                   help="learning rate for test-split fine-tuning phase")
    p.add_argument("--checkpoint_dir", type=str,   default="qwen/three_tower_ckpt")
    p.add_argument("--save_every",     type=int,   default=500)
    p.add_argument("--device",         type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
