"""
train_qwen_meta_tower.py — QwenMeta 双塔召回模型
==================================================

Architecture
------------
Query塔: hist_conversation_emb (1024d)
  → Linear(1024,512) → BN → ReLU → Dropout
  → Linear(512,256)  → BN → ReLU → Dropout
  → Linear(256,128)
  output: query_vec [128]  (L2 normalized)

Item塔: metadata-qwen3_embedding_0.6b (1024d)
  → Linear(1024,512) → BN → ReLU → Dropout
  → Linear(512,256)  → BN → ReLU → Dropout
  → Linear(256,128)
  output: item_vec  [128]  (L2 normalized)

Score: cosine_similarity(query_vec, item_vec) × temperature

Label / Loss
------------
- MOVES_TOWARD_GOAL         → label=1.0, weight=1.5
- DOES_NOT_MOVE_TOWARD_GOAL → label=1.0, weight=1.0
- 10 random negatives (random track),  weight=1.0, label=0.0
- Weighted BCEWithLogitsLoss

Training schedule
-----------------
Phase 1: train split, --train_epochs epochs
Phase 2: test  split, --test_epochs  epochs (fine-tune, lower lr)

Outputs
-------
  qwen/qwen_meta_tower/model.pt            — final model weights
  qwen/qwen_meta_tower/item_index/         — item_vectors.pt + track_ids.json

Usage
-----
    cd /home/lijiatong06/music-crs-baselines/music_crs_local
    nohup python train_qwen_meta_tower.py \\
        --config config/llama1b_multi_channel_devset.yaml \\
        --train_conv_emb qwen/hist_conversation_embeddings_train_0.6b.pt \\
        --test_conv_emb  qwen/hist_conversation_embeddings_test_0.6b.pt \\
        --train_epochs 5 \\
        --test_epochs  2 \\
        --batch_size 64 \\
        --lr 1e-3 \\
        --lr_finetune 3e-4 \\
        --device cuda \\
    > train_qwen_meta.log 2>&1 &
"""

import argparse
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from omegaconf import OmegaConf
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONV_EMB_DIM  = 1024
META_EMB_DIM  = 1024
HIDDEN1       = 512
HIDDEN2       = 256
OUTPUT_DIM    = 128
TEMPERATURE   = 20.0

WEIGHT_MOVES   = 1.0
WEIGHT_NO_MOVE = 1.0
NUM_EASY_NEG   = 10


# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────

class TowerMLP(nn.Module):
    """Generic 3-layer MLP tower: in_dim → 512 → 256 → 128, with BN+ReLU+Dropout."""

    def __init__(self, in_dim: int, hidden1: int = HIDDEN1, hidden2: int = HIDDEN2,
                 out: int = OUTPUT_DIM, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,  hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QwenMetaTwoTower(nn.Module):
    """
    Query塔 × Item(metadata)塔 双塔模型。
    Score = cosine(query_vec, item_vec) × temperature
    """

    def __init__(self, conv_dim: int = CONV_EMB_DIM, meta_dim: int = META_EMB_DIM,
                 hidden1: int = HIDDEN1, hidden2: int = HIDDEN2,
                 out: int = OUTPUT_DIM, dropout: float = 0.2,
                 temperature: float = TEMPERATURE):
        super().__init__()
        self.temperature  = temperature
        self.query_tower  = TowerMLP(conv_dim, hidden1, hidden2, out, dropout)
        self.item_tower   = TowerMLP(meta_dim, hidden1, hidden2, out, dropout)

    def encode_query(self, conv_emb: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalised [B, 128]."""
        return F.normalize(self.query_tower(conv_emb), p=2, dim=1)

    def encode_item(self, meta_emb: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalised [B, 128]."""
        return F.normalize(self.item_tower(meta_emb), p=2, dim=1)

    def forward(self, conv_emb: torch.Tensor,
                meta_emb: torch.Tensor) -> torch.Tensor:
        """Returns cosine logit scores [B]."""
        q = self.encode_query(conv_emb)
        v = self.encode_item(meta_emb)
        return (q * v).sum(dim=1) * self.temperature


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
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


def load_track_meta_emb(track_emb_db: str, split_types: List[str]) -> Dict[str, torch.Tensor]:
    """Load metadata-qwen3_embedding_0.6b (1024d) for every track."""
    ds = load_dataset(track_emb_db)
    valid = [s for s in split_types if s in ds.keys()] or list(ds.keys())
    concat_ds = concatenate_datasets([ds[s] for s in valid])

    result: Dict[str, torch.Tensor] = {}
    for item in tqdm(concat_ds, desc="Loading track meta-emb", unit="track"):
        tid = item["track_id"]
        v   = item.get("metadata-qwen3_embedding_0.6b")
        result[tid] = _raw_to_tensor(v, META_EMB_DIM)
    logger.info("Loaded meta-emb for %d tracks.", len(result))
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Sample builder
# ─────────────────────────────────────────────────────────────────────────────

def build_training_samples(
    dataset_name:   str,
    split:          str,
    conv_emb_store: Dict[str, torch.Tensor],
) -> List[Dict]:
    """One sample per (session, music-turn) with label from goal_progress_assessments."""
    logger.info("Loading '%s' split='%s' …", dataset_name, split)
    ds = load_dataset(dataset_name, split=split)
    samples: List[Dict] = []
    label_dist = {"strong": 0, "weak": 0, "skipped": 0}

    for item in tqdm(ds, desc=f"Building samples [{split}]", unit="session"):
        session_id  = item["session_id"]
        convs       = item["conversations"]
        assessments = item.get("goal_progress_assessments", [])
        asmt_map    = {int(a["turn_number"]): a.get("goal_progress_assessment")
                       for a in assessments}

        music_turns: Dict[int, str] = {}
        for c in convs:
            if c.get("role") == "music" and c.get("content"):
                music_turns[int(c["turn_number"])] = c["content"]

        for turn_number, track_id in music_turns.items():
            fb_turn = turn_number + 1
            if fb_turn not in asmt_map:
                label_dist["skipped"] += 1; continue

            gpa = asmt_map[fb_turn]
            if gpa == "MOVES_TOWARD_GOAL":
                label, weight = 1.0, WEIGHT_MOVES
                label_dist["strong"] += 1
            elif gpa == "DOES_NOT_MOVE_TOWARD_GOAL":
                label, weight = 1.0, WEIGHT_NO_MOVE
                label_dist["weak"] += 1
            else:
                label_dist["skipped"] += 1; continue

            emb_key = f"{session_id}_{turn_number}"
            if emb_key not in conv_emb_store:
                label_dist["skipped"] += 1; continue

            conv_emb = conv_emb_store[emb_key].float()
            if conv_emb.shape[0] > CONV_EMB_DIM:
                conv_emb = conv_emb[:CONV_EMB_DIM]
            elif conv_emb.shape[0] < CONV_EMB_DIM:
                conv_emb = F.pad(conv_emb, (0, CONV_EMB_DIM - conv_emb.shape[0]))

            samples.append({
                "track_id": track_id,
                "label":    label,
                "weight":   weight,
                "conv_emb": conv_emb,
            })

    logger.info("Built %d samples | strong=%d  weak=%d  skipped=%d",
                len(samples), label_dist["strong"], label_dist["weak"], label_dist["skipped"])
    return samples


# ─────────────────────────────────────────────────────────────────────────────
#  Batch collation
# ─────────────────────────────────────────────────────────────────────────────

def collate_batch(
    samples:       List[Dict],
    all_tids:      List[str],
    track_meta:    Dict[str, torch.Tensor],
    device:        str,
    num_easy_neg:  int = NUM_EASY_NEG,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    conv_embs, meta_embs, labels, weights = [], [], [], []

    for s in samples:
        pos_meta = track_meta.get(s["track_id"], torch.zeros(META_EMB_DIM))

        # Positive (or weak positive)
        conv_embs.append(s["conv_emb"])
        meta_embs.append(pos_meta)
        labels.append(s["label"])
        weights.append(s["weight"])

        # 10 random easy negatives
        for _ in range(num_easy_neg):
            neg_tid  = random.choice(all_tids)
            neg_meta = track_meta.get(neg_tid, torch.zeros(META_EMB_DIM))
            conv_embs.append(s["conv_emb"])
            meta_embs.append(neg_meta)
            labels.append(0.0)
            weights.append(1.0)

    return (
        torch.stack(conv_embs).to(device),
        torch.stack(meta_embs).to(device),
        torch.tensor(labels,  dtype=torch.float32, device=device),
        torch.tensor(weights, dtype=torch.float32, device=device),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Training phase
# ─────────────────────────────────────────────────────────────────────────────

def run_phase(phase_name, samples, model, optimizer,
              track_meta, all_tids, args, epochs, device, ckpt_dir):
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    logger.info("[%s] %d samples, %d epochs, batch=%d",
                phase_name, len(samples), epochs, args.batch_size)
    global_step = 0

    for epoch in range(1, epochs + 1):
        random.shuffle(samples)
        model.train()
        ep_loss, ep_steps = 0.0, 0

        pbar = tqdm(range(0, len(samples), args.batch_size),
                    desc=f"[{phase_name}] Epoch {epoch}/{epochs}", unit="batch")
        for start in pbar:
            batch = samples[start: start + args.batch_size]
            if not batch: continue

            conv_t, meta_t, label_t, weight_t = collate_batch(
                batch, all_tids, track_meta, device)

            scores = model(conv_t, meta_t)
            loss   = (loss_fn(scores, label_t) * weight_t).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item(); ep_steps += 1; global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % args.save_every == 0:
                torch.save({"model_state_dict": model.state_dict()},
                           os.path.join(ckpt_dir, "model_latest.pt"))

        avg = ep_loss / max(ep_steps, 1)
        logger.info("[%s] Epoch %d  avg_loss=%.4f", phase_name, epoch, avg)
        torch.save({"model_state_dict": model.state_dict()},
                   os.path.join(ckpt_dir, f"model_{phase_name}_epoch{epoch}.pt"))


# ─────────────────────────────────────────────────────────────────────────────
#  Item index builder
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_item_index(model: QwenMetaTwoTower,
                     track_meta: Dict[str, torch.Tensor],
                     save_dir: str, device: str = "cpu", bs: int = 512):
    import json
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    track_ids = sorted(track_meta.keys())
    vecs = []
    for start in tqdm(range(0, len(track_ids), bs), desc="Building item index"):
        batch_ids  = track_ids[start: start + bs]
        batch_meta = torch.stack([track_meta[t] for t in batch_ids]).to(device)
        vecs.append(model.encode_item(batch_meta).cpu())
    matrix = torch.cat(vecs, dim=0)
    torch.save(matrix, os.path.join(save_dir, "item_vectors.pt"))
    with open(os.path.join(save_dir, "track_ids.json"), "w") as f:
        json.dump(track_ids, f)
    logger.info("Item index: %d tracks → %s", len(track_ids), save_dir)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    config       = OmegaConf.load(args.config)
    device       = args.device or config.get("device", "cuda")
    dataset_name = config.get("conversation_dataset_name",
                               "talkpl-ai/TalkPlayData-Challenge-Dataset")
    track_emb_db = config.get("track_emb_db_name",
                               "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    split_types  = list(config.get("track_split_types", ["all_tracks"]))

    # Load conv_emb stores
    logger.info("Loading train conv_emb from %s …", args.train_conv_emb)
    train_ce = torch.load(args.train_conv_emb, map_location="cpu", weights_only=True)
    logger.info("  %d entries.", len(train_ce))
    logger.info("Loading test conv_emb from %s …", args.test_conv_emb)
    test_ce  = torch.load(args.test_conv_emb,  map_location="cpu", weights_only=True)
    logger.info("  %d entries.", len(test_ce))

    # Build samples
    train_samples = build_training_samples(dataset_name, "train", train_ce)
    test_samples  = build_training_samples(dataset_name, "test",  test_ce)
    if not train_samples:
        logger.error("No training samples."); return

    # Load track metadata-qwen3 embeddings
    logger.info("Loading track metadata-qwen3 embeddings …")
    track_meta = load_track_meta_emb(track_emb_db, split_types)
    all_tids   = list(track_meta.keys())

    # Model + optimizer
    model     = QwenMetaTwoTower().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ckpt_dir  = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    # Phase 1 — train split
    run_phase("train", train_samples, model, optimizer,
              track_meta, all_tids, args, args.train_epochs, device, ckpt_dir)

    # Phase 2 — test fine-tune
    if test_samples and args.test_epochs > 0:
        logger.info("Phase 2: fine-tune on test split (lr=%.2e) …", args.lr_finetune)
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr_finetune
        run_phase("test_ft", test_samples, model, optimizer,
                  track_meta, all_tids, args, args.test_epochs, device, ckpt_dir)

    # Save final model
    final_dir = os.path.join("qwen", "qwen_meta_tower")
    os.makedirs(final_dir, exist_ok=True)
    final_path = os.path.join(final_dir, "model.pt")
    torch.save({"model_state_dict": model.state_dict()}, final_path)
    logger.info("Final model → %s", final_path)

    # Build item index
    build_item_index(model, track_meta,
                     os.path.join(final_dir, "item_index"), device=device)
    logger.info("Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train QwenMeta Two-Tower Retrieval Model")
    p.add_argument("--config",          type=str,
                   default="config/llama1b_multi_channel_devset.yaml")
    p.add_argument("--train_conv_emb",  type=str,
                   default="qwen/hist_conversation_embeddings_train_0.6b.pt")
    p.add_argument("--test_conv_emb",   type=str,
                   default="qwen/hist_conversation_embeddings_test_0.6b.pt")
    p.add_argument("--train_epochs",    type=int,   default=5)
    p.add_argument("--test_epochs",     type=int,   default=2)
    p.add_argument("--batch_size",      type=int,   default=64)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--lr_finetune",     type=float, default=3e-4)
    p.add_argument("--checkpoint_dir",  type=str,   default="qwen/qwen_meta_tower_ckpt")
    p.add_argument("--save_every",      type=int,   default=500)
    p.add_argument("--device",          type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
