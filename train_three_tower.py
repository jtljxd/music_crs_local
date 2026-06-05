"""
Three-Tower Reranker Training Script (v3) — DCN + PPNet
========================================================

- emb key format: {session_id}_{turn}_query / {session_id}_{turn}_history (1024-dim, 0.6B)
- Model: DCN (3 cross layers) + PPNet-DNN gated by 1024-dim query emb
- Label: any track in conversations = positive (label=1.0)
- Negatives per positive:
    5  in-batch hard (different session_id)
    5  super-hard from FULL recall (all 6 channels, 600 candidates, randomly pick 5)
    10 easy (random from full track pool)
- Training: builds all retrieval indices (BM25, semantic, track matrices) on first run,
  persisted to qwen/retrieval_indices/ so inference never needs to rebuild.
- Training schedule:
    Phase 1 — train split,  5 epochs
    Phase 2 — test  split,  2 epochs  (fine-tune, lower lr)

Usage (background):
    nohup python train_three_tower.py \\
        --config config/llama1b_multi_channel_devset.yaml \\
        --train_store qwen/dialogue_embeddings_train_0.6b.pt \\
        --test_store  qwen/dialogue_embeddings_test_0.6b.pt \\
        --train_epochs 5 \\
        --test_epochs  2 \\
        --batch_size 32 \\
        --lr 1e-3 \\
        --lr_finetune 3e-4 \\
    > train_three_tower.log 2>&1 &
"""

import argparse
import logging
import os
import random
from typing import Dict, List, Optional, Tuple, Set

import torch
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from mcrs.reranking_modules.three_tower_reranker import (
    ThreeTowerReranker,
    ThreeTowerRerankerWrapper,
)
from mcrs.retrieval_modules.multi_channel_retrieval import MultiChannelRetrieval

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EMB_DIM = 1024
RECALL_TOPK = 600   # full recall pool; randomly pick 5 super-hard negatives from it


# ─────────────────────────────────────────────────────────────────────────────
#  Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def _get_emb(store: Dict, key: str, dim: int = EMB_DIM) -> torch.Tensor:
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
    """One sample per (session, music-turn).

    Label assignment (with turn offset fix):
      - conversation turn N recommends a track
      - assessment at turn N+1 gives the feedback
      - MOVES_TOWARD_GOAL     → label = 1.0  (positive)
      - DOES_NOT_MOVE_TOWARD_GOAL → label = 0.0 (negative signal, still included)
      - no assessment (last turn, no feedback) → skip
    """
    logger.info("Loading '%s' split='%s' …", dataset_name, split)
    ds = load_dataset(dataset_name, split=split)

    samples, skipped = [], 0
    label_dist = {"1.0": 0, "0.0": 0, "skipped_no_feedback": 0}

    for item in tqdm(ds, desc=f"Building samples [{split}]", unit="session"):
        session_id = item["session_id"]
        user_id    = item.get("user_id", None)
        convs      = item["conversations"]
        assessments = item.get("goal_progress_assessments", [])

        # Build assessment lookup: feedback_turn → gpa value
        # feedback_turn = music_turn + 1
        asmt_map = {
            int(a["turn_number"]): a.get("goal_progress_assessment", None)
            for a in assessments
        }

        # Collect music turns: turn_number → track_id
        music_turns: Dict[int, str] = {}
        for c in convs:
            if c.get("role") == "music" and c.get("content"):
                music_turns[int(c["turn_number"])] = c["content"]

        if not music_turns:
            continue

        # Only truly positive tracks for negative-sampling exclusion
        session_positive_ids: Set[str] = {
            tid for tn, tid in music_turns.items()
            if asmt_map.get(tn + 1) == "MOVES_TOWARD_GOAL"
        }

        for turn_number, track_id in music_turns.items():
            # Feedback for turn N is in assessment at turn N+1
            feedback_turn = turn_number + 1
            if feedback_turn not in asmt_map:
                # Last turn or no assessment → skip (no reliable label)
                label_dist["skipped_no_feedback"] += 1
                skipped += 1
                continue

            gpa = asmt_map[feedback_turn]
            if gpa == "MOVES_TOWARD_GOAL":
                label = 1.0
                label_dist["1.0"] += 1
            elif gpa == "DOES_NOT_MOVE_TOWARD_GOAL":
                label = 0.0
                label_dist["0.0"] += 1
            else:
                # None or unknown value → skip
                label_dist["skipped_no_feedback"] += 1
                skipped += 1
                continue

            q_key = f"{session_id}_{turn_number}_query"
            if q_key not in turn_store:
                skipped += 1
                continue

            query_emb   = _get_emb(turn_store, q_key)
            history_emb = _get_emb(turn_store, f"{session_id}_{turn_number}_history")

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
                "label":                label,
                "session_positive_ids": session_positive_ids,
                "user_query":           user_query,
                "query_emb":            query_emb,
                "history_emb":          history_emb,
            })

    logger.info(
        "Built %d samples from '%s' | label=1.0: %d  label=0.0: %d  skipped: %d",
        len(samples), split,
        label_dist["1.0"], label_dist["0.0"], label_dist["skipped_no_feedback"],
    )
    return samples


# ─────────────────────────────────────────────────────────────────────────────
#  Feature building
# ─────────────────────────────────────────────────────────────────────────────

TRACK_MODAL_COLS = ThreeTowerRerankerWrapper.TRACK_MODAL_COLS
TRACK_DIM        = ThreeTowerRerankerWrapper.TRACK_MODAL_TARGET_DIM
USER_DIM         = ThreeTowerRerankerWrapper.USER_CF_BPR_TARGET_DIM


def build_track_features(track_id, track_embeddings):
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


def build_user_features(user_id, user_embeddings):
    if user_id and user_id in user_embeddings:
        data = user_embeddings[user_id]
        return data["cf-bpr"], ("cf-bpr" in data["__missing__"])
    return torch.zeros(USER_DIM), True


# ─────────────────────────────────────────────────────────────────────────────
#  Full-channel recall helper
# ─────────────────────────────────────────────────────────────────────────────

def full_recall(
    retrieval: MultiChannelRetrieval,
    sample: Dict,
    topk: int = RECALL_TOPK,
) -> List[str]:
    """Run all 6 retrieval channels, return up to topk deduplicated candidates."""
    try:
        candidates = retrieval.retrieve(
            user_id       = sample["user_id"],
            current_query = sample["user_query"],
            history_queries = [],
            session_id    = sample["session_id"],
            turn_number   = sample["turn_number"],
        )
        return candidates[:topk]
    except Exception as e:
        logger.debug("Recall failed for session %s: %s", sample["session_id"], e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  Batch collation
# ─────────────────────────────────────────────────────────────────────────────

def collate_batch(
    samples: List[Dict],
    all_track_ids: List[str],
    track_embeddings: Dict,
    user_embeddings: Dict,
    device: str,
    retrieval: MultiChannelRetrieval,
    num_hard_neg: int = 5,
    num_recall_neg: int = 5,
    num_easy_neg: int = 10,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    rows_intent, rows_gate = [], []
    rows_modal  = [[] for _ in TRACK_MODAL_COLS]
    rows_modal_mask, rows_user_bpr, rows_cf_missing, labels = [], [], [], []

    batch_session_ids = [s["session_id"] for s in samples]
    batch_track_ids   = [s["track_id"]   for s in samples]

    def _add_row(s, track_id, lbl):
        intent = torch.cat([s["query_emb"], s["history_emb"]])
        gate   = s["query_emb"]
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
        # positive or labeled negative from assessment
        _add_row(s, s["track_id"], s["label"])

        # in-batch hard negatives (different session)
        diff_idx = [j for j, sid in enumerate(batch_session_ids) if sid != s["session_id"]]
        for j in (random.sample(diff_idx, min(num_hard_neg, len(diff_idx))) if diff_idx else []):
            _add_row(s, batch_track_ids[j], 0.0)

        # super-hard recall negatives (all 6 channels, 600 candidates → random 5)
        if s["user_query"]:
            recall_pool = full_recall(retrieval, s, topk=RECALL_TOPK)
            recall_pool = [t for t in recall_pool if t not in s["session_positive_ids"]]
        else:
            recall_pool = []
        for tid in (random.sample(recall_pool, min(num_recall_neg, len(recall_pool)))
                    if recall_pool else []):
            _add_row(s, tid, 0.0)

        # easy negatives (random)
        for _ in range(num_easy_neg):
            _add_row(s, random.choice(all_track_ids), 0.0)

    intent_tensor = torch.stack(rows_intent).to(device)
    gate_tensor   = torch.stack(rows_gate).to(device)
    modal_tensors = [torch.stack(rows_modal[j]).to(device) for j in range(len(TRACK_MODAL_COLS))]
    modal_mask    = torch.tensor(rows_modal_mask, dtype=torch.bool, device=device)
    user_bpr      = torch.stack(rows_user_bpr).to(device)
    cf_missing    = torch.tensor(rows_cf_missing, dtype=torch.bool, device=device)
    label_tensor  = torch.tensor(labels, dtype=torch.float32, device=device)

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
    phase_name, samples, model, optimizer,
    track_embs, user_embs, all_tids,
    retrieval, wrapper, args, epochs, device, checkpoint_dir,
):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    logger.info("[%s] %d samples, %d epochs, batch=%d",
                phase_name, len(samples), epochs, args.batch_size)

    global_step = 0
    for epoch in range(1, epochs + 1):
        random.shuffle(samples)
        model.train()
        epoch_loss, epoch_steps = 0.0, 0

        pbar = tqdm(range(0, len(samples), args.batch_size),
                    desc=f"[{phase_name}] Epoch {epoch}/{epochs}", unit="batch")
        for batch_start in pbar:
            batch = samples[batch_start: batch_start + args.batch_size]
            if not batch:
                continue

            model_inputs, label_tensor = collate_batch(
                batch, all_tids, track_embs, user_embs, device,
                retrieval=retrieval,
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

        avg_loss = epoch_loss / max(epoch_steps, 1)
        logger.info("[%s] Epoch %d done. avg_loss=%.4f", phase_name, epoch, avg_loss)
        wrapper.save_checkpoint()
        ep_path = os.path.join(checkpoint_dir, f"model_{phase_name}_epoch{epoch}.pt")
        torch.save({"model_state_dict": model.state_dict()}, ep_path)
        logger.info("[%s] Checkpoint → %s", phase_name, ep_path)


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
    train_store = torch.load(args.train_store, map_location="cpu", weights_only=True)
    logger.info("  train: %d entries.", len(train_store))

    logger.info("Loading test turn store from %s …", args.test_store)
    test_store = torch.load(args.test_store, map_location="cpu", weights_only=True)
    logger.info("  test:  %d entries.", len(test_store))

    # ── Build samples ────────────────────────────────────────────────────────
    train_samples = build_training_samples(dataset_name, "train", train_store)
    test_samples  = build_training_samples(dataset_name, "test",  test_store)
    if not train_samples:
        logger.error("No training samples built.")
        return

    # ── Initialize MultiChannelRetrieval (build_indices=True) ────────────────
    logger.info("Initializing MultiChannelRetrieval (build_indices=True) …")
    from transformers import AutoTokenizer, AutoModel
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
    qwen_model     = AutoModel.from_pretrained(qwen_model_path).cpu().eval()

    retrieval = MultiChannelRetrieval(
        dataset_name      = dataset_name,
        item_db_name      = config.get("item_db_name",
            "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"),
        user_db_name      = config.get("user_db_name",
            "talkpl-ai/TalkPlayData-Challenge-User-Metadata"),
        track_emb_db_name = config.get("track_emb_db_name",
            "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"),
        user_emb_db_name  = config.get("user_emb_db_name",
            "talkpl-ai/TalkPlayData-Challenge-User-Embeddings"),
        split_types       = list(config.get("track_split_types", ["all_tracks"])),
        cache_dir         = cache_dir,
        qwen_model_path   = qwen_model_path,
        device            = device,
        qwen_model        = qwen_model,
        qwen_tokenizer    = qwen_tokenizer,
        build_indices     = True,   # ← build all indices on first run
    )
    logger.info("All retrieval indices ready.")

    # ── Initialize ThreeTowerRerankerWrapper ──────────────────────────────────
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
        qwen_model    = qwen_model,
        qwen_tokenizer = qwen_tokenizer,
    )

    # Inject turn store so retrieval can use pre-computed embs
    retrieval.set_turn_store(train_store)

    model      = wrapper.model
    track_embs = wrapper.track_embeddings
    user_embs  = wrapper.user_embeddings
    all_tids   = list(track_embs.keys())

    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Phase 1: train split, 5 epochs ───────────────────────────────────────
    retrieval.set_turn_store(train_store)
    optimizer = wrapper.optimizer
    run_phase(
        phase_name     = "train",
        samples        = train_samples,
        model          = model,
        optimizer      = optimizer,
        track_embs     = track_embs,
        user_embs      = user_embs,
        all_tids       = all_tids,
        retrieval      = retrieval,
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
        retrieval.set_turn_store(test_store)
        run_phase(
            phase_name     = "test_ft",
            samples        = test_samples,
            model          = model,
            optimizer      = optimizer,
            track_embs     = track_embs,
            user_embs      = user_embs,
            all_tids       = all_tids,
            retrieval      = retrieval,
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
    p.add_argument("--train_store",    type=str,   default="qwen/dialogue_embeddings_train_0.6b.pt")
    p.add_argument("--test_store",     type=str,   default="qwen/dialogue_embeddings_test_0.6b.pt")
    p.add_argument("--train_epochs",   type=int,   default=5)
    p.add_argument("--test_epochs",    type=int,   default=2)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--lr_finetune",    type=float, default=3e-4)
    p.add_argument("--checkpoint_dir", type=str,   default="qwen/three_tower_ckpt")
    p.add_argument("--save_every",     type=int,   default=500)
    p.add_argument("--device",         type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
