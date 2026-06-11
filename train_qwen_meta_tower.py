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
    """All music-turns are positives (no GPA filter)."""
    logger.info("Loading '%s' split='%s' …", dataset_name, split)
    ds = load_dataset(dataset_name, split=split)
    samples: List[Dict] = []
    skipped = 0

    for item in tqdm(ds, desc=f"Building samples [{split}]", unit="session"):
        session_id = item["session_id"]
        convs      = item["conversations"]

        music_turns: Dict[int, str] = {}
        for c in convs:
            if c.get("role") == "music" and c.get("content"):
                music_turns[int(c["turn_number"])] = c["content"]

        for turn_number, track_id in music_turns.items():
            emb_key = f"{session_id}_{turn_number}"
            if emb_key not in conv_emb_store:
                skipped += 1; continue

            conv_emb = conv_emb_store[emb_key].float()
            if conv_emb.shape[0] > CONV_EMB_DIM:
                conv_emb = conv_emb[:CONV_EMB_DIM]
            elif conv_emb.shape[0] < CONV_EMB_DIM:
                conv_emb = F.pad(conv_emb, (0, CONV_EMB_DIM - conv_emb.shape[0]))

            samples.append({
                "track_id": track_id,
                "label":    1.0,
                "weight":   1.0,
                "conv_emb": conv_emb,
            })

    logger.info("Built %d samples  (skipped_no_emb=%d)", len(samples), skipped)
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
              track_meta, all_tids, args, epochs, device, ckpt_dir,
              val_samples=None, eval_sets=None, item_matrix=None,
              item_ids=None, out_txt=None):
    loss_fn    = nn.BCEWithLogitsLoss(reduction="none")
    patience   = args.early_stop_patience
    min_delta  = args.early_stop_min_delta
    best_loss  = float("inf")
    no_improve = 0
    best_ckpt  = os.path.join(ckpt_dir, f"model_{phase_name}_best.pt")

    use_val = val_samples is not None and len(val_samples) > 0
    logger.info("[%s] train=%d  val=%d  max_epochs=%d  batch=%d  patience=%d",
                phase_name, len(samples), len(val_samples) if use_val else 0,
                epochs, args.batch_size, patience)
    global_step = 0

    for epoch in range(1, epochs + 1):
        # ── Train ──
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

        train_avg = ep_loss / max(ep_steps, 1)

        # ── Val loss ──
        if use_val:
            model.eval()
            val_loss, val_steps = 0.0, 0
            with torch.no_grad():
                for start in range(0, len(val_samples), args.batch_size):
                    batch = val_samples[start: start + args.batch_size]
                    if not batch: continue
                    conv_t, meta_t, label_t, weight_t = collate_batch(
                        batch, all_tids, track_meta, device)
                    scores = model(conv_t, meta_t)
                    loss   = (loss_fn(scores, label_t) * weight_t).mean()
                    val_loss += loss.item(); val_steps += 1
            monitor_loss = val_loss / max(val_steps, 1)
            logger.info("[%s] Epoch %d  train_loss=%.4f  val_loss=%.4f",
                        phase_name, epoch, train_avg, monitor_loss)
        else:
            monitor_loss = train_avg
            logger.info("[%s] Epoch %d  train_loss=%.4f", phase_name, epoch, train_avg)

        torch.save({"model_state_dict": model.state_dict()},
                   os.path.join(ckpt_dir, f"model_{phase_name}_epoch{epoch}.pt"))

        # ── Per-epoch recall eval ──
        if eval_sets and item_matrix is not None and out_txt:
            # 每 epoch 重新 encode item（因为 model 参数在变）
            cur_mat, cur_ids = build_item_matrix(model, track_meta, device)
            eval_recall_ch3(model, cur_mat, cur_ids, eval_sets, device,
                            out_txt, epoch, phase_name)

        # ── Early stopping on monitor_loss ──
        if monitor_loss < best_loss - min_delta:
            best_loss  = monitor_loss
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict()}, best_ckpt)
            logger.info("[%s] ★ New best val_loss=%.4f → %s",
                        phase_name, best_loss, best_ckpt)
        else:
            no_improve += 1
            logger.info("[%s] No improvement %d/%d (best=%.4f)",
                        phase_name, no_improve, patience, best_loss)
            if no_improve >= patience:
                logger.info("[%s] Early stopping at epoch %d.", phase_name, epoch)
                break

    # 加载最优权重
    if os.path.exists(best_ckpt):
        logger.info("[%s] Restoring best model from %s", phase_name, best_ckpt)
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    return best_loss


# ─────────────────────────────────────────────────────────────────────────────
#  Per-epoch recall evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

K_EVAL = [20, 50, 100, 200]

def _build_eval_gt(
    ds_iter,
    conv_emb_store: Dict[str, torch.Tensor],
    max_sessions: int = 100,
    non_last_only: bool = False,
) -> List[Dict]:
    """从数据集中抽取最多 max_sessions 个 session 的 (conv_emb, gt_tid) 对。
    non_last_only=True: 只取非最后一个 user turn（Blind-A 评估用）。
    non_last_only=False: 取最后一个 music turn（val/test 评估用）。
    """
    records = []
    n_sess  = 0
    for item in ds_iter:
        if n_sess >= max_sessions:
            break
        convs   = item["conversations"]
        sid     = item["session_id"]

        music_turns = {int(c["turn_number"]): c["content"]
                       for c in convs if c.get("role") == "music" and c.get("content")}
        if not music_turns:
            continue

        user_turns = [int(c["turn_number"]) for c in convs if c.get("role") == "user"]
        last_user  = max(user_turns) if user_turns else 0

        if non_last_only:
            # 非最后 user turn 的 music GT
            pairs = [(t, gt) for t, gt in music_turns.items() if t < last_user]
        else:
            # 最后一个 music turn
            last_t = max(music_turns.keys())
            pairs  = [(last_t, music_turns[last_t])]

        for t, gt in pairs:
            key = f"{sid}_{t}"
            if key not in conv_emb_store:
                # 往前扫
                for tt in range(t - 1, -1, -1):
                    k2 = f"{sid}_{tt}"
                    if k2 in conv_emb_store:
                        key = k2; break
                else:
                    continue
            emb = conv_emb_store[key].float()
            if emb.shape[0] > CONV_EMB_DIM:
                emb = emb[:CONV_EMB_DIM]
            elif emb.shape[0] < CONV_EMB_DIM:
                emb = F.pad(emb, (0, CONV_EMB_DIM - emb.shape[0]))
            records.append({"conv_emb": emb, "gt_tid": gt})

        n_sess += 1
    return records


@torch.no_grad()
def eval_recall_ch3(
    model:       QwenMetaTwoTower,
    item_matrix: torch.Tensor,       # [N, 128] L2-normed
    item_ids:    List[str],
    eval_sets:   Dict[str, List[Dict]],  # name → list of {conv_emb, gt_tid}
    device:      str,
    out_txt:     str,
    epoch:       int,
    phase:       str,
) -> None:
    """对每个 eval_set 计算 Recall@K 和 NDCG@20，追加到 out_txt。"""
    import math
    model.eval()
    item_matrix = item_matrix.to(device)
    id2idx = {tid: i for i, tid in enumerate(item_ids)}

    lines = [f"\n=== Epoch {epoch} [{phase}] ==="]
    for name, records in eval_sets.items():
        if not records:
            lines.append(f"  [{name}]  (empty)")
            continue
        hits  = {k: 0 for k in K_EVAL}
        ndcg20 = 0.0
        valid  = 0
        bs = 256
        all_embs = torch.stack([r["conv_emb"] for r in records]).to(device)
        # encode all queries in one shot
        qvecs = []
        for s in range(0, len(all_embs), bs):
            qvecs.append(model.encode_query(all_embs[s:s+bs]))
        qvecs = torch.cat(qvecs, dim=0)  # [M, 128]

        scores = qvecs @ item_matrix.T   # [M, N]

        for i, r in enumerate(records):
            gt  = r["gt_tid"]
            if gt not in id2idx:
                continue
            valid += 1
            sc   = scores[i]
            topK = torch.topk(sc, min(max(K_EVAL), sc.shape[0])).indices.tolist()
            top_ids = [item_ids[j] for j in topK]
            for k in K_EVAL:
                if gt in top_ids[:k]:
                    hits[k] += 1
            if gt in top_ids[:20]:
                rank = top_ids[:20].index(gt) + 1
                ndcg20 += 1.0 / math.log2(rank + 1)

        n = max(valid, 1)
        r_str = "  ".join(f"R@{k}={hits[k]/n*100:.2f}%" for k in K_EVAL)
        lines.append(f"  [{name}] n={valid}  {r_str}  NDCG@20={ndcg20/n:.4f}")

    block = "\n".join(lines)
    logger.info(block)
    with open(out_txt, "a", encoding="utf-8") as f:
        f.write(block + "\n")


@torch.no_grad()
def build_item_matrix(model: QwenMetaTwoTower,
                      track_meta: Dict[str, torch.Tensor],
                      device: str, bs: int = 512):
    model.eval()
    track_ids = sorted(track_meta.keys())
    vecs = []
    for s in range(0, len(track_ids), bs):
        b = track_ids[s:s+bs]
        m = torch.stack([track_meta[t] for t in b]).to(device)
        vecs.append(model.encode_item(m).cpu())
    return torch.cat(vecs, dim=0), track_ids  # [N,128], list[str]


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

    # Load conv_emb (train only)
    logger.info("Loading train conv_emb from %s …", args.train_conv_emb)
    train_ce = torch.load(args.train_conv_emb, map_location="cpu", weights_only=True)
    logger.info("  %d entries.", len(train_ce))

    # Build all samples from train split, then 8:2 split into train/val
    all_samples = build_training_samples(dataset_name, "train", train_ce)
    if not all_samples:
        logger.error("No training samples."); return

    random.shuffle(all_samples)
    val_size    = max(1, int(len(all_samples) * 0.2))
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]
    logger.info("Train/Val split: %d / %d", len(train_samples), len(val_samples))

    # Load track metadata-qwen3 embeddings
    logger.info("Loading track metadata-qwen3 embeddings …")
    track_meta = load_track_meta_emb(track_emb_db, split_types)
    all_tids   = list(track_meta.keys())

    # ── Prepare per-epoch recall eval sets ────────────────────────────────────
    blind_dataset = (config.get("blind_dataset_name")
                     or "talkpl-ai/TalkPlayData-Challenge-Blind-A")
    test_ce_path  = args.train_conv_emb.replace("train", "test")
    blind_ce_path = args.train_conv_emb.replace("train", "blinda")

    logger.info("Preparing eval GT sets …")
    # val 前100（从 train 数据按 8:2 拆出的 val_samples 里提取）
    val_gt = [{"conv_emb": s["conv_emb"], "gt_tid": s["track_id"]}
              for s in val_samples[:100]]
    # test 前100
    test_gt = []
    if os.path.exists(test_ce_path):
        test_ce = torch.load(test_ce_path, map_location="cpu", weights_only=True)
        test_ds = load_dataset(dataset_name, split="test")
        test_gt = _build_eval_gt(iter(test_ds), test_ce, max_sessions=100,
                                  non_last_only=False)
    else:
        logger.warning("test conv_emb not found at %s, skip test eval", test_ce_path)
    # blind-A 非最后轮
    blind_gt = []
    if os.path.exists(blind_ce_path):
        blind_ce = torch.load(blind_ce_path, map_location="cpu", weights_only=True)
        blind_ds = load_dataset(blind_dataset, split="test")
        blind_gt = _build_eval_gt(iter(blind_ds), blind_ce, max_sessions=200,
                                   non_last_only=True)
    else:
        logger.warning("blind conv_emb not found at %s, skip blind eval", blind_ce_path)

    eval_sets = {}
    if val_gt:   eval_sets["val_100"]   = val_gt
    if test_gt:  eval_sets["test_100"]  = test_gt
    if blind_gt: eval_sets["blinda_non_last"] = blind_gt
    logger.info("Eval sets: %s", {k: len(v) for k, v in eval_sets.items()})

    out_txt = os.path.join(args.checkpoint_dir, "recall_by_epoch.txt")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"ch3 QwenMeta Two-Tower — per-epoch recall\n")
        f.write(f"K_EVAL={K_EVAL}\n")

    # Model + optimizer
    model     = QwenMetaTwoTower().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ckpt_dir_ = args.checkpoint_dir
    os.makedirs(ckpt_dir_, exist_ok=True)

    # Train with val-based early stopping + recall eval
    run_phase("train", train_samples, model, optimizer,
              track_meta, all_tids, args, args.train_epochs, device, ckpt_dir_,
              val_samples=val_samples,
              eval_sets=eval_sets,
              item_matrix=None,   # built inside each epoch
              item_ids=None,
              out_txt=out_txt)

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
    logger.info("Recall log → %s", out_txt)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train QwenMeta Two-Tower Retrieval Model")
    p.add_argument("--config",          type=str,
                   default="config/llama1b_multi_channel_devset.yaml")
    p.add_argument("--train_conv_emb",  type=str,
                   default="qwen/hist_conversation_embeddings_train_0.6b.pt")
    p.add_argument("--train_epochs",    type=int,   default=20)
    p.add_argument("--batch_size",      type=int,   default=64)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--checkpoint_dir",  type=str,   default="qwen/qwen_meta_tower_ckpt")
    p.add_argument("--save_every",           type=int,   default=500)
    p.add_argument("--early_stop_patience",  type=int,   default=5,
                   help="连续 N 个 epoch val_loss 无改善则停止")
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    p.add_argument("--device",               type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
