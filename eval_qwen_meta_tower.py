"""
eval_qwen_meta_tower.py — 单独评估 QwenMeta 双塔模型召回效果
=============================================================

Usage:
    python eval_qwen_meta_tower.py \\
        --config config/llama1b_multi_channel_devset.yaml \\
        --conv_emb_store qwen/hist_conversation_embeddings_test_0.6b.pt \\
        --model_path     qwen/qwen_meta_tower/model.pt \\
        --max_sessions   200 \\
        --split          test
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional

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

TOPK_LIST    = [20, 50, 100, 200]
CONV_EMB_DIM = 1024
META_EMB_DIM = 1024
HIDDEN1      = 512
HIDDEN2      = 256
OUTPUT_DIM   = 128
TEMPERATURE  = 20.0


# ─────────────────────────────────────────────────────────────────────────────
#  Model (mirrors train_qwen_meta_tower.py)
# ─────────────────────────────────────────────────────────────────────────────

class TowerMLP(nn.Module):
    def __init__(self, in_dim, h1=HIDDEN1, h2=HIDDEN2, out=OUTPUT_DIM, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2),    nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, out),
        )
    def forward(self, x): return self.net(x)


class QwenMetaTwoTower(nn.Module):
    def __init__(self, conv_dim=CONV_EMB_DIM, meta_dim=META_EMB_DIM,
                 temperature=TEMPERATURE, dropout=0.2):
        super().__init__()
        self.temperature = temperature
        self.query_tower = TowerMLP(conv_dim, dropout=dropout)
        self.item_tower  = TowerMLP(meta_dim, dropout=dropout)

    def encode_query(self, x):
        return F.normalize(self.query_tower(x), p=2, dim=1)

    def encode_item(self, x):
        return F.normalize(self.item_tower(x), p=2, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _raw_to_tensor(value, dim: int) -> torch.Tensor:
    if value is None:
        return torch.zeros(dim, dtype=torch.float32)
    try:
        t = torch.tensor(value, dtype=torch.float32)
        if t.ndim == 0 or t.numel() == 0: return torch.zeros(dim)
        if t.ndim > 1: t = t.flatten()
        if t.shape[0] < dim: t = F.pad(t, (0, dim - t.shape[0]))
        elif t.shape[0] > dim: t = t[:dim]
        return t
    except Exception:
        return torch.zeros(dim, dtype=torch.float32)


def load_track_meta_emb(track_emb_db: str, split_types: List[str]) -> Dict[str, torch.Tensor]:
    ds = load_dataset(track_emb_db)
    valid = [s for s in split_types if s in ds.keys()] or list(ds.keys())
    concat_ds = concatenate_datasets([ds[s] for s in valid])
    result: Dict[str, torch.Tensor] = {}
    for item in tqdm(concat_ds, desc="Loading track meta-emb", unit="track"):
        tid = item["track_id"]
        v   = item.get("metadata-qwen3_embedding_0.6b")
        result[tid] = _raw_to_tensor(v, META_EMB_DIM)
    logger.info("Loaded %d tracks.", len(result))
    return result


def build_item_index(model: QwenMetaTwoTower,
                     track_meta: Dict[str, torch.Tensor],
                     bs: int = 512) -> tuple:
    """Returns (item_matrix [N,128], track_ids [N])."""
    model.eval()
    track_ids = sorted(track_meta.keys())
    vecs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(track_ids), bs), desc="Building item index"):
            b_ids = track_ids[start: start + bs]
            b_emb = torch.stack([track_meta[t] for t in b_ids])
            vecs.append(model.encode_item(b_emb))
    return torch.cat(vecs, dim=0), track_ids   # [N,128], List[str]


def retrieve(model: QwenMetaTwoTower,
             conv_emb: torch.Tensor,   # [1024]
             item_matrix: torch.Tensor,
             track_ids: List[str],
             topk: int) -> List[str]:
    with torch.no_grad():
        q = model.encode_query(conv_emb.unsqueeze(0)).squeeze(0)  # [128]
    scores  = (item_matrix * q.unsqueeze(0)).sum(dim=1)
    topk    = min(topk, scores.shape[0])
    top_idx = torch.topk(scores, k=topk).indices.tolist()
    return [track_ids[i] for i in top_idx]


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    config       = OmegaConf.load(args.config)
    dataset_name = config.get("conversation_dataset_name",
                               "talkpl-ai/TalkPlayData-Challenge-Dataset")
    track_emb_db = config.get("track_emb_db_name",
                               "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    split_types  = list(config.get("track_split_types", ["all_tracks"]))

    # Load conv_emb store
    logger.info("Loading conv_emb store from %s …", args.conv_emb_store)
    conv_emb_store = torch.load(args.conv_emb_store, map_location="cpu", weights_only=True)
    logger.info("  %d entries.", len(conv_emb_store))

    # Load track meta embeddings
    logger.info("Loading track meta-emb …")
    track_meta = load_track_meta_emb(track_emb_db, split_types)

    # Load model
    model = QwenMetaTwoTower()
    if args.model_path and os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Model loaded from %s", args.model_path)
    else:
        logger.warning("No model found at '%s'; using random weights.", args.model_path)
    model.eval()

    # Build item index
    logger.info("Building item index …")
    item_matrix, item_ids = build_item_index(model, track_meta)
    logger.info("  Item index: %d tracks.", len(item_ids))

    max_k = max(TOPK_LIST)

    # Load dataset split
    logger.info("Loading dataset split='%s' …", args.split)
    ds = load_dataset(dataset_name, split=args.split)

    hit_counts = {k: 0 for k in TOPK_LIST}
    total = 0
    n_sessions = 0

    for item in tqdm(ds, desc=f"Evaluating [{args.split}]", unit="session"):
        if n_sessions >= args.max_sessions:
            break
        n_sessions += 1

        session_id  = item["session_id"]
        convs       = item["conversations"]
        assessments = item.get("goal_progress_assessments", [])
        asmt_map    = {int(a["turn_number"]): a.get("goal_progress_assessment")
                       for a in assessments}

        music_turns: Dict[int, str] = {}
        for c in convs:
            if c.get("role") == "music" and c.get("content"):
                music_turns[int(c["turn_number"])] = c["content"]

        for turn_number, gt_track_id in music_turns.items():
            fb_turn = turn_number + 1
            if fb_turn not in asmt_map:
                continue
            gpa = asmt_map[fb_turn]
            if gpa not in ("MOVES_TOWARD_GOAL", "DOES_NOT_MOVE_TOWARD_GOAL"):
                continue

            # Lookup conv_emb: try turn_number, fallback turn_number - 1
            emb_key = f"{session_id}_{turn_number}"
            user_turn_number = turn_number
            if emb_key not in conv_emb_store:
                prev = turn_number - 1
                emb_key = f"{session_id}_{prev}"
                user_turn_number = prev

            if emb_key not in conv_emb_store:
                continue

            conv_emb = conv_emb_store[emb_key].float()
            if conv_emb.shape[0] > CONV_EMB_DIM:
                conv_emb = conv_emb[:CONV_EMB_DIM]
            elif conv_emb.shape[0] < CONV_EMB_DIM:
                conv_emb = F.pad(conv_emb, (0, CONV_EMB_DIM - conv_emb.shape[0]))

            retrieved = retrieve(model, conv_emb, item_matrix, item_ids, max_k)

            total += 1
            for k in TOPK_LIST:
                if gt_track_id in retrieved[:k]:
                    hit_counts[k] += 1

    # ── Print result table ──────────────────────────────────────────────────
    lines = []
    lines.append(f"\n{'='*50}")
    lines.append(f"QwenMeta Two-Tower — {args.split} split ({n_sessions} sessions)")
    lines.append(f"{'='*50}")
    header = f"{'Channel':<20}" + "".join(f"  @{k:<6}" for k in TOPK_LIST)
    lines.append(header)
    lines.append("-" * len(header))

    row = f"{'QwenMeta-TwoTower':<20}"
    for k in TOPK_LIST:
        pct = hit_counts[k] / total * 100 if total > 0 else 0.0
        row += f"  {pct:>6.2f}%"
    lines.append(row)
    lines.append(f"\nTotal evaluated pairs: {total}")
    lines.append("=" * 50)

    result_str = "\n".join(lines)
    print(result_str)

    # Save to file
    out_dir = "exp/eval"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"qwen_meta_tower_{args.split}.txt")
    with open(out_path, "w") as f:
        f.write(result_str + "\n")
    logger.info("Results saved to %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate QwenMeta Two-Tower Retrieval")
    p.add_argument("--config",          type=str,
                   default="config/llama1b_multi_channel_devset.yaml")
    p.add_argument("--conv_emb_store",  type=str,
                   default="qwen/hist_conversation_embeddings_test_0.6b.pt")
    p.add_argument("--model_path",      type=str,
                   default="qwen/qwen_meta_tower/model.pt")
    p.add_argument("--max_sessions",    type=int,  default=200)
    p.add_argument("--split",           type=str,  default="test",
                   choices=["train", "test"])
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
