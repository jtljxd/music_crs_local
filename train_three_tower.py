"""
train_three_tower.py  (v4 — CF-BPR Retrieval Three-Tower)
==========================================================

Trains a **retrieval** three-tower model (ch1_CF-BPR) whose fusion vector
replaces the plain user-CF cosine lookup.

Model Architecture
------------------
User塔:  CF-BPR Emb (128d) + 用户画像特征
         → Linear(128, 256) → BN → ReLU → Dropout
         → Linear(256, 128)
         output: user_vec [128]

Query塔: hist_conversation_emb (1024d)
         → Linear(1024, 512) → BN → ReLU → Dropout
         → Linear(512, 256) → BN → ReLU → Dropout
         → Linear(256, 128)
         output: query_vec [128]

融合层 (Gate Fusion):
         gate = Sigmoid(Linear(concat(user_vec, query_vec), 128))
         fusion_vec = gate ⊙ user_vec + (1 - gate) ⊙ query_vec
         output: fusion_vec [128]

Item塔:  CF-BPR Emb (128d) + 属性特征
         → Linear(128, 256) → BN → ReLU → Dropout
         → Linear(256, 128)
         output: item_vec [128]

Score: cosine_similarity(fusion_vec, item_vec)

Label & Loss
------------
- MOVES_TOWARD_GOAL   → positive, sample_weight = 1.5
- DOES_NOT_MOVE_TOWARD_GOAL → positive (weak), sample_weight = 1.0
- 10 random negatives per positive (from full track pool), weight = 1.0

Loss = weighted binary cross-entropy with logits (cosine score * temp as logit)

Training schedule:
  Phase 1 — train split, 5 epochs
  Phase 2 — test  split, 2 epochs (fine-tune, lower lr)

After training the model is saved to:
  qwen/cf_bpr_retrieval/model.pt

Usage:
    nohup python train_three_tower.py \\
        --config config/llama1b_multi_channel_devset.yaml \\
        --train_conv_emb qwen/hist_conversation_embeddings_train_0.6b.pt \\
        --test_conv_emb  qwen/hist_conversation_embeddings_test_0.6b.pt \\
        --train_epochs 5 \\
        --test_epochs  2 \\
        --batch_size 64 \\
        --lr 1e-3 \\
        --lr_finetune 3e-4 \\
    > train_cf_bpr_retrieval.log 2>&1 &
"""

import argparse
import logging
import os
import random
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONV_EMB_DIM   = 1024   # hist_conversation_embeddings dimension
CF_BPR_DIM     = 128    # CF-BPR embedding dimension
HIDDEN_DIM     = 256
OUTPUT_DIM     = 128
TEMPERATURE    = 20.0   # cosine logit temperature

# Positive sample weights
WEIGHT_MOVES    = 1.5
WEIGHT_NO_MOVE  = 1.0
NUM_EASY_NEG    = 10    # random negatives per positive


# ─────────────────────────────────────────────────────────────────────────────
#  Model definition
# ─────────────────────────────────────────────────────────────────────────────

class UserTower(nn.Module):
    """User塔: CF-BPR Emb (128d) → Linear(128,256) → BN → ReLU → Dropout → Linear(256,128)"""

    def __init__(self, cf_bpr_dim: int = CF_BPR_DIM, hidden: int = HIDDEN_DIM,
                 out: int = OUTPUT_DIM, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cf_bpr_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out),
        )

    def forward(self, cf_bpr: torch.Tensor) -> torch.Tensor:
        """Args: cf_bpr [B, 128]  Returns: [B, 128]"""
        return self.net(cf_bpr)


class QueryTower(nn.Module):
    """Query塔: conv_emb (1024d) → 512 → BN → ReLU → Dropout → 256 → BN → ReLU → Dropout → 128"""

    def __init__(self, conv_dim: int = CONV_EMB_DIM, hidden: int = HIDDEN_DIM,
                 out: int = OUTPUT_DIM, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(conv_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out),
        )

    def forward(self, conv_emb: torch.Tensor) -> torch.Tensor:
        """Args: conv_emb [B, 1024]  Returns: [B, 128]"""
        return self.net(conv_emb)


class ItemTower(nn.Module):
    """Item塔: CF-BPR Emb (128d) → Linear(128,256) → BN → ReLU → Dropout → Linear(256,128)"""

    def __init__(self, cf_bpr_dim: int = CF_BPR_DIM, hidden: int = HIDDEN_DIM,
                 out: int = OUTPUT_DIM, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cf_bpr_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out),
        )

    def forward(self, cf_bpr: torch.Tensor) -> torch.Tensor:
        """Args: cf_bpr [B, 128]  Returns: [B, 128]"""
        return self.net(cf_bpr)


class CFBPRThreeTower(nn.Module):
    """Three-tower retrieval model with gate fusion."""

    def __init__(
        self,
        cf_bpr_dim: int = CF_BPR_DIM,
        conv_dim:   int = CONV_EMB_DIM,
        hidden:     int = HIDDEN_DIM,
        out:        int = OUTPUT_DIM,
        dropout:    float = 0.2,
        temperature: float = TEMPERATURE,
    ):
        super().__init__()
        self.temperature = temperature

        self.user_tower  = UserTower(cf_bpr_dim, hidden, out, dropout)
        self.query_tower = QueryTower(conv_dim,  hidden, out, dropout)
        self.item_tower  = ItemTower(cf_bpr_dim, hidden, out, dropout)

        # Gate fusion: gate = Sigmoid(Linear(concat(user_vec, query_vec), out))
        self.gate_linear = nn.Linear(out * 2, out)

    def encode_fusion(
        self,
        user_cf_bpr: torch.Tensor,
        conv_emb:    torch.Tensor,
    ) -> torch.Tensor:
        """Returns fusion_vec [B, out]."""
        user_vec  = self.user_tower(user_cf_bpr)   # [B, 128]
        query_vec = self.query_tower(conv_emb)      # [B, 128]
        gate      = torch.sigmoid(self.gate_linear(
            torch.cat([user_vec, query_vec], dim=1)
        ))                                           # [B, 128]
        fusion = gate * user_vec + (1.0 - gate) * query_vec
        return F.normalize(fusion, p=2, dim=1)

    def encode_item(self, item_cf_bpr: torch.Tensor) -> torch.Tensor:
        """Returns item_vec [B, out]."""
        return F.normalize(self.item_tower(item_cf_bpr), p=2, dim=1)

    def forward(
        self,
        user_cf_bpr:  torch.Tensor,  # [B, 128]
        conv_emb:     torch.Tensor,  # [B, 1024]
        item_cf_bpr:  torch.Tensor,  # [B, 128]
    ) -> torch.Tensor:
        """Returns cosine logit scores [B]."""
        fusion_vec = self.encode_fusion(user_cf_bpr, conv_emb)
        item_vec   = self.encode_item(item_cf_bpr)
        cosine     = (fusion_vec * item_vec).sum(dim=1)   # [B]
        return cosine * self.temperature


# ─────────────────────────────────────────────────────────────────────────────
#  Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def build_training_samples(
    dataset_name: str,
    split: str,
    conv_emb_store: Dict[str, torch.Tensor],
) -> List[Dict]:
    """Build one sample per (session, music-turn) with label from goal_progress_assessments.

    Label assignment (turn offset fix):
      - system recommends track at turn N
      - assessment at turn N+1 gives feedback
      - MOVES_TOWARD_GOAL         → label=1.0, weight=1.5
      - DOES_NOT_MOVE_TOWARD_GOAL → label=0.0, weight=1.0
      - no assessment → skip
    """
    logger.info("Loading '%s' split='%s' …", dataset_name, split)
    ds = load_dataset(dataset_name, split=split)

    samples: List[Dict] = []
    skipped = 0
    label_dist = {"1.0": 0, "0.0": 0, "skipped": 0}

    for item in tqdm(ds, desc=f"Building samples [{split}]", unit="session"):
        session_id  = item["session_id"]
        user_id     = item.get("user_id", None)
        convs       = item["conversations"]
        assessments = item.get("goal_progress_assessments", [])

        asmt_map = {
            int(a["turn_number"]): a.get("goal_progress_assessment", None)
            for a in assessments
        }

        music_turns: Dict[int, str] = {}
        for c in convs:
            if c.get("role") == "music" and c.get("content"):
                music_turns[int(c["turn_number"])] = c["content"]

        if not music_turns:
            continue

        for turn_number, track_id in music_turns.items():
            feedback_turn = turn_number + 1
            if feedback_turn not in asmt_map:
                label_dist["skipped"] += 1
                skipped += 1
                continue

            gpa = asmt_map[feedback_turn]
            if gpa == "MOVES_TOWARD_GOAL":
                label  = 1.0
                weight = WEIGHT_MOVES
                label_dist["1.0"] += 1
            elif gpa == "DOES_NOT_MOVE_TOWARD_GOAL":
                label  = 0.0
                weight = WEIGHT_NO_MOVE
                label_dist["0.0"] += 1
            else:
                label_dist["skipped"] += 1
                skipped += 1
                continue

            # conv_emb key: {session_id}_{turn_number}
            emb_key = f"{session_id}_{turn_number}"
            if emb_key not in conv_emb_store:
                skipped += 1
                label_dist["skipped"] += 1
                continue

            conv_emb = conv_emb_store[emb_key].float()
            if conv_emb.shape[0] > CONV_EMB_DIM:
                conv_emb = conv_emb[:CONV_EMB_DIM]
            elif conv_emb.shape[0] < CONV_EMB_DIM:
                conv_emb = F.pad(conv_emb, (0, CONV_EMB_DIM - conv_emb.shape[0]))

            samples.append({
                "session_id": session_id,
                "turn_number": turn_number,
                "user_id":     user_id,
                "track_id":    track_id,
                "label":       label,
                "weight":      weight,
                "conv_emb":    conv_emb,
            })

    logger.info(
        "Built %d samples from '%s' | pos: %d  neg-weak: %d  skipped: %d",
        len(samples), split,
        label_dist["1.0"], label_dist["0.0"], label_dist["skipped"],
    )
    return samples


def _get_cf_bpr(embeddings: Dict, entity_id: Optional[str],
                dim: int = CF_BPR_DIM) -> torch.Tensor:
    """Retrieve CF-BPR tensor, zero if missing."""
    if entity_id and entity_id in embeddings:
        data = embeddings[entity_id]
        vec = data.get("cf-bpr", None)
        if vec is not None:
            v = vec.float()
            if v.shape[0] > dim:
                return v[:dim]
            elif v.shape[0] < dim:
                return F.pad(v, (0, dim - v.shape[0]))
            return v
    return torch.zeros(dim, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Batch collation
# ─────────────────────────────────────────────────────────────────────────────

def collate_batch(
    samples: List[Dict],
    all_track_ids: List[str],
    track_embeddings: Dict,
    user_embeddings:  Dict,
    device: str,
    num_easy_neg: int = NUM_EASY_NEG,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Build batch tensors with positive + easy negatives.

    Returns:
        model_inputs: dict with keys user_cf_bpr, conv_emb, item_cf_bpr
        labels:       float tensor [total_rows]
        weights:      float tensor [total_rows]
    """
    rows_user_bpr  = []
    rows_conv_emb  = []
    rows_item_bpr  = []
    rows_label     = []
    rows_weight    = []

    for s in samples:
        user_bpr = _get_cf_bpr(user_embeddings,  s["user_id"])
        item_bpr = _get_cf_bpr(track_embeddings, s["track_id"])

        # Positive (or weak positive) row
        rows_user_bpr.append(user_bpr)
        rows_conv_emb.append(s["conv_emb"])
        rows_item_bpr.append(item_bpr)
        rows_label.append(s["label"])
        rows_weight.append(s["weight"])

        # Easy negatives: random tracks from full track pool
        for _ in range(num_easy_neg):
            neg_tid  = random.choice(all_track_ids)
            neg_bpr  = _get_cf_bpr(track_embeddings, neg_tid)
            rows_user_bpr.append(user_bpr)
            rows_conv_emb.append(s["conv_emb"])
            rows_item_bpr.append(neg_bpr)
            rows_label.append(0.0)
            rows_weight.append(1.0)

    user_bpr_t  = torch.stack(rows_user_bpr).to(device)
    conv_emb_t  = torch.stack(rows_conv_emb).to(device)
    item_bpr_t  = torch.stack(rows_item_bpr).to(device)
    label_t     = torch.tensor(rows_label,  dtype=torch.float32, device=device)
    weight_t    = torch.tensor(rows_weight, dtype=torch.float32, device=device)

    model_inputs = dict(
        user_cf_bpr = user_bpr_t,
        conv_emb    = conv_emb_t,
        item_cf_bpr = item_bpr_t,
    )
    return model_inputs, label_t, weight_t


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop (single phase)
# ─────────────────────────────────────────────────────────────────────────────

def run_phase(
    phase_name:       str,
    samples:          List[Dict],
    model:            CFBPRThreeTower,
    optimizer,
    track_embs:       Dict,
    user_embs:        Dict,
    all_tids:         List[str],
    args,
    epochs:           int,
    device:           str,
    checkpoint_dir:   str,
):
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
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

            model_inputs, label_t, weight_t = collate_batch(
                batch, all_tids, track_embs, user_embs, device,
                num_easy_neg=NUM_EASY_NEG,
            )

            scores = model(**model_inputs)               # [B]
            loss   = (loss_fn(scores, label_t) * weight_t).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss  += loss.item()
            epoch_steps += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % args.save_every == 0:
                ckpt = os.path.join(checkpoint_dir, "model_latest.pt")
                torch.save({"model_state_dict": model.state_dict()}, ckpt)

        avg_loss = epoch_loss / max(epoch_steps, 1)
        logger.info("[%s] Epoch %d done. avg_loss=%.4f", phase_name, epoch, avg_loss)
        ep_path = os.path.join(checkpoint_dir, f"model_{phase_name}_epoch{epoch}.pt")
        torch.save({"model_state_dict": model.state_dict()}, ep_path)
        logger.info("[%s] Checkpoint → %s", phase_name, ep_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Embedding loaders (lightweight, avoids importing MultiChannelRetrieval)
# ─────────────────────────────────────────────────────────────────────────────

def _raw_to_tensor(value, dim: int) -> torch.Tensor:
    if value is None:
        return torch.zeros(dim, dtype=torch.float32)
    try:
        t = torch.tensor(value, dtype=torch.float32)
        if t.ndim == 0 or t.numel() == 0:
            return torch.zeros(dim, dtype=torch.float32)
        if t.ndim > 1:
            t = t.flatten()
        if t.shape[0] < dim:
            t = F.pad(t, (0, dim - t.shape[0]))
        elif t.shape[0] > dim:
            t = t[:dim]
        return t
    except (TypeError, ValueError):
        return torch.zeros(dim, dtype=torch.float32)


def load_track_cf_bpr(track_emb_db_name: str, split_types: List[str],
                      cache_dir: str) -> Dict[str, Dict]:
    """Load track CF-BPR embeddings from HuggingFace dataset."""
    from datasets import concatenate_datasets
    ds = load_dataset(track_emb_db_name)
    valid_splits = [s for s in split_types if s in ds.keys()] or list(ds.keys())
    concat_ds = concatenate_datasets([ds[s] for s in valid_splits])

    cf_bpr_dim = CF_BPR_DIM
    for item in concat_ds:
        v = item.get("cf-bpr")
        if v is None:
            continue
        try:
            t = torch.tensor(v, dtype=torch.float32)
            if t.ndim == 1 and t.numel() > 0:
                cf_bpr_dim = t.shape[0]
                break
        except Exception:
            pass
    logger.info("Track CF-BPR dim: %d", cf_bpr_dim)

    embeddings: Dict[str, Dict] = {}
    for item in concat_ds:
        tid = item["track_id"]
        v   = item.get("cf-bpr")
        embeddings[tid] = {"cf-bpr": _raw_to_tensor(v, cf_bpr_dim)}
    logger.info("Loaded %d track CF-BPR embeddings.", len(embeddings))
    return embeddings


def load_user_cf_bpr(user_emb_db_name: str, split_types: List[str],
                     cache_dir: str) -> Dict[str, Dict]:
    """Load user CF-BPR embeddings from HuggingFace dataset."""
    from datasets import concatenate_datasets
    ds = load_dataset(user_emb_db_name)
    valid_splits = [s for s in split_types if s in ds.keys()] or list(ds.keys())
    concat_ds = concatenate_datasets([ds[s] for s in valid_splits])

    cf_bpr_dim = CF_BPR_DIM
    for item in concat_ds:
        v = item.get("cf-bpr")
        if v is None:
            continue
        try:
            t = torch.tensor(v, dtype=torch.float32)
            if t.ndim == 1 and t.numel() > 0:
                cf_bpr_dim = t.shape[0]
                break
        except Exception:
            pass
    logger.info("User CF-BPR dim: %d", cf_bpr_dim)

    embeddings: Dict[str, Dict] = {}
    for item in concat_ds:
        uid = item["user_id"]
        v   = item.get("cf-bpr")
        embeddings[uid] = {"cf-bpr": _raw_to_tensor(v, cf_bpr_dim)}
    logger.info("Loaded %d user CF-BPR embeddings.", len(embeddings))
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
#  Save trained item tower as retrieval index
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_item_index(
    model:           CFBPRThreeTower,
    track_embeddings: Dict[str, Dict],
    save_dir:        str,
    device:          str = "cpu",
    batch_size:      int = 512,
):
    """Forward all item CF-BPR through ItemTower; save normalised matrix + ids."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    track_ids = sorted(track_embeddings.keys())

    item_vecs = []
    for start in tqdm(range(0, len(track_ids), batch_size),
                      desc="Building item index", unit="batch"):
        batch_ids  = track_ids[start: start + batch_size]
        batch_bpr  = torch.stack([
            track_embeddings[tid]["cf-bpr"] for tid in batch_ids
        ]).to(device)
        item_vec = model.encode_item(batch_bpr).cpu()
        item_vecs.append(item_vec)

    item_matrix = torch.cat(item_vecs, dim=0)   # [N, 128] already normalised
    torch.save(item_matrix, os.path.join(save_dir, "item_vectors.pt"))
    import json
    with open(os.path.join(save_dir, "track_ids.json"), "w") as f:
        json.dump(track_ids, f)
    logger.info("Item index saved: %d tracks → %s", len(track_ids), save_dir)


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    config = OmegaConf.load(args.config)
    device = args.device or config.get("device", "cuda")
    cache_dir    = config.get("cache_dir", "./cache")
    dataset_name = config.get(
        "conversation_dataset_name", "talkpl-ai/TalkPlayData-Challenge-Dataset"
    )
    track_emb_db = config.get("track_emb_db_name",
                              "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    user_emb_db  = config.get("user_emb_db_name",
                              "talkpl-ai/TalkPlayData-Challenge-User-Embeddings")
    split_types  = list(config.get("track_split_types", ["all_tracks"]))

    # ── Load conv_emb stores ─────────────────────────────────────────────────
    logger.info("Loading train conv_emb_store from %s …", args.train_conv_emb)
    train_conv_emb = torch.load(args.train_conv_emb, map_location="cpu", weights_only=True)
    logger.info("  train: %d entries.", len(train_conv_emb))

    logger.info("Loading test conv_emb_store from %s …", args.test_conv_emb)
    test_conv_emb = torch.load(args.test_conv_emb, map_location="cpu", weights_only=True)
    logger.info("  test:  %d entries.", len(test_conv_emb))

    # ── Build samples ────────────────────────────────────────────────────────
    train_samples = build_training_samples(dataset_name, "train", train_conv_emb)
    test_samples  = build_training_samples(dataset_name, "test",  test_conv_emb)
    if not train_samples:
        logger.error("No training samples built.")
        return

    # ── Load CF-BPR embeddings ────────────────────────────────────────────────
    logger.info("Loading track CF-BPR embeddings …")
    track_embs = load_track_cf_bpr(track_emb_db, split_types, cache_dir)
    logger.info("Loading user CF-BPR embeddings …")
    user_embs  = load_user_cf_bpr(user_emb_db,  split_types, cache_dir)

    all_tids   = list(track_embs.keys())

    # ── Model + optimizer ─────────────────────────────────────────────────────
    model = CFBPRThreeTower(
        cf_bpr_dim  = CF_BPR_DIM,
        conv_dim    = CONV_EMB_DIM,
        hidden      = HIDDEN_DIM,
        out         = OUTPUT_DIM,
        temperature = TEMPERATURE,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Phase 1: train split ──────────────────────────────────────────────────
    run_phase(
        phase_name     = "train",
        samples        = train_samples,
        model          = model,
        optimizer      = optimizer,
        track_embs     = track_embs,
        user_embs      = user_embs,
        all_tids       = all_tids,
        args           = args,
        epochs         = args.train_epochs,
        device         = device,
        checkpoint_dir = checkpoint_dir,
    )

    # ── Phase 2: test split (fine-tune, lower lr) ─────────────────────────────
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
            args           = args,
            epochs         = args.test_epochs,
            device         = device,
            checkpoint_dir = checkpoint_dir,
        )

    # ── Save final model ─────────────────────────────────────────────────────
    final_dir = os.path.join("qwen", "cf_bpr_retrieval")
    os.makedirs(final_dir, exist_ok=True)
    final_model_path = os.path.join(final_dir, "model.pt")
    torch.save({"model_state_dict": model.state_dict()}, final_model_path)
    logger.info("Final model saved → %s", final_model_path)

    # ── Build item index for fast retrieval ───────────────────────────────────
    item_index_dir = os.path.join("qwen", "cf_bpr_retrieval", "item_index")
    build_item_index(model, track_embs, item_index_dir, device=device)

    logger.info("Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train CF-BPR Three-Tower Retrieval Model"
    )
    p.add_argument("--config",          type=str,
                   default="config/llama1b_multi_channel_devset.yaml")
    p.add_argument("--train_conv_emb",  type=str,
                   default="qwen/hist_conversation_embeddings_train_0.6b.pt",
                   help="hist_conversation_embeddings for train split (1024d)")
    p.add_argument("--test_conv_emb",   type=str,
                   default="qwen/hist_conversation_embeddings_test_0.6b.pt",
                   help="hist_conversation_embeddings for test split (1024d)")
    p.add_argument("--train_epochs",    type=int,  default=5)
    p.add_argument("--test_epochs",     type=int,  default=2)
    p.add_argument("--batch_size",      type=int,  default=64)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--lr_finetune",     type=float, default=3e-4)
    p.add_argument("--checkpoint_dir",  type=str,
                   default="qwen/cf_bpr_retrieval_ckpt")
    p.add_argument("--save_every",      type=int,  default=500)
    p.add_argument("--device",          type=str,  default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
