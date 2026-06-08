"""
eval_retrieval_accuracy.py
===========================
在 test 数据集上评估3路召回模型的准确率（含 CF-BPR 三塔）。

统计指标:
  - 每个召回路 (ch1, ch3, ch5) + union 在 @K = 20, 50, 100, 200 的命中率
  - 只统计有 ground-truth track_id 的 (session, turn) 对

Usage:
    python eval_retrieval_accuracy.py \
        --config config/llama1b_multi_channel_devset.yaml \
        --conv_emb_store  qwen/hist_conversation_embeddings_test_0.6b.pt \
        --query_split_store qwen/query_split_test.pt \
        --cf_model_path   qwen/cf_bpr_retrieval/model.pt \
        --max_sessions    200
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set

import torch
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TOPK_LIST    = [20, 50, 100, 200]
CONV_EMB_DIM = 1024
CF_BPR_DIM   = 128
OUTPUT_DIM   = 128

CHANNEL_NAMES = ["ch1_CF-BPR", "ch3_QwenMeta", "ch5_BM25", "ALL (union)"]


# ─────────────────────────────────────────────────────────────────────────────
#  Inline model (same as train_three_tower.py)
# ─────────────────────────────────────────────────────────────────────────────

import torch.nn as nn


class UserTower(nn.Module):
    def __init__(self, cf_bpr_dim=CF_BPR_DIM, hidden=256, out=OUTPUT_DIM, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cf_bpr_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, out),
        )
    def forward(self, x): return self.net(x)


class QueryTower(nn.Module):
    def __init__(self, conv_dim=CONV_EMB_DIM, hidden=256, out=OUTPUT_DIM, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(conv_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out),
        )
    def forward(self, x): return self.net(x)


class ItemTower(nn.Module):
    def __init__(self, cf_bpr_dim=CF_BPR_DIM, hidden=256, out=OUTPUT_DIM, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cf_bpr_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, out),
        )
    def forward(self, x): return self.net(x)


class CFBPRThreeTower(nn.Module):
    def __init__(self, cf_bpr_dim=CF_BPR_DIM, conv_dim=CONV_EMB_DIM,
                 hidden=256, out=OUTPUT_DIM, dropout=0.2, temperature=20.0):
        super().__init__()
        self.temperature = temperature
        self.user_tower  = UserTower(cf_bpr_dim, hidden, out, dropout)
        self.query_tower = QueryTower(conv_dim, hidden, out, dropout)
        self.item_tower  = ItemTower(cf_bpr_dim, hidden, out, dropout)
        self.gate_linear = nn.Linear(out * 2, out)

    def encode_fusion(self, user_cf_bpr, conv_emb):
        user_vec  = self.user_tower(user_cf_bpr)
        query_vec = self.query_tower(conv_emb)
        gate      = torch.sigmoid(self.gate_linear(
            torch.cat([user_vec, query_vec], dim=1)
        ))
        fusion = gate * user_vec + (1.0 - gate) * query_vec
        return F.normalize(fusion, p=2, dim=1)

    def encode_item(self, item_cf_bpr):
        return F.normalize(self.item_tower(item_cf_bpr), p=2, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_cf_bpr(embeddings: Dict, entity_id: Optional[str],
                dim: int = CF_BPR_DIM) -> torch.Tensor:
    if entity_id and entity_id in embeddings:
        data = embeddings[entity_id]
        vec  = data.get("cf-bpr", None)
        if vec is not None:
            v = vec.float()
            if v.shape[0] > dim:
                return v[:dim]
            elif v.shape[0] < dim:
                return F.pad(v, (0, dim - v.shape[0]))
            return v
    return torch.zeros(dim, dtype=torch.float32)


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


def load_embeddings(db_name: str, split_types: List[str],
                    id_field: str = "track_id",
                    emb_field: str = "cf-bpr") -> Dict:
    from datasets import concatenate_datasets
    ds = load_dataset(db_name)
    valid_splits = [s for s in split_types if s in ds.keys()] or list(ds.keys())
    concat_ds = concatenate_datasets([ds[s] for s in valid_splits])
    dim = CF_BPR_DIM
    for item in concat_ds:
        v = item.get(emb_field)
        if v is None:
            continue
        try:
            t = torch.tensor(v, dtype=torch.float32)
            if t.ndim == 1 and t.numel() > 0:
                dim = t.shape[0]
                break
        except Exception:
            pass
    result: Dict = {}
    for item in concat_ds:
        eid = item[id_field]
        v   = item.get(emb_field)
        result[eid] = {"cf-bpr": _raw_to_tensor(v, dim)}
    logger.info("Loaded %d '%s' embeddings (dim=%d).", len(result), id_field, dim)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Channel retrievals
# ─────────────────────────────────────────────────────────────────────────────

_QUERY_SPLIT_TEXT_FIELDS = [
    "artist", "album", "genre", "decade", "language",
    "popularity", "scene", "tempo",
]


def retrieve_ch1_cf_bpr(
    model: Optional[CFBPRThreeTower],
    user_cf_bpr: torch.Tensor,
    conv_emb: torch.Tensor,
    item_matrix: torch.Tensor,   # [N, 128] normalised
    track_ids:   List[str],
    topk:        int,
) -> List[str]:
    """ch1: three-tower fusion vector × item matrix cosine."""
    if model is None:
        # Fallback: user CF-BPR × item CF-BPR
        u = F.normalize(user_cf_bpr.unsqueeze(0), p=2, dim=1).squeeze(0)
        # If item_matrix col-dim != user dim, skip
        if item_matrix.shape[1] == u.shape[0]:
            scores  = (item_matrix * u.unsqueeze(0)).sum(dim=1)
        else:
            return track_ids[:topk]
    else:
        model.eval()
        with torch.no_grad():
            fusion = model.encode_fusion(
                user_cf_bpr.unsqueeze(0),
                conv_emb.unsqueeze(0),
            ).squeeze(0)                         # [128]
        scores = (item_matrix * fusion.unsqueeze(0)).sum(dim=1)   # [N]

    topk   = min(topk, scores.shape[0])
    top_idx = torch.topk(scores, k=topk).indices.tolist()
    return [track_ids[i] for i in top_idx]


def retrieve_ch3_metadata(
    query_emb:       torch.Tensor,  # [1024]
    metadata_matrix: torch.Tensor,  # [N, 1024] normalised
    track_ids:       List[str],
    topk:            int,
) -> List[str]:
    """ch3: 1024-dim conv emb × metadata-qwen3."""
    q = F.normalize(query_emb.unsqueeze(0), p=2, dim=1).squeeze(0)
    scores  = (metadata_matrix * q.unsqueeze(0)).sum(dim=1)
    topk    = min(topk, scores.shape[0])
    top_idx = torch.topk(scores, k=topk).indices.tolist()
    return [track_ids[i] for i in top_idx]


def retrieve_ch5_bm25(
    bm25_model,
    bm25_track_ids: List[str],
    current_query:  str,
    query_split:    Optional[Dict],
    topk:           int,
) -> List[str]:
    """ch5: BM25 keyword search from query_split."""
    import bm25s as _bm25s

    keyword_strings: List[str] = []
    if query_split:
        for field in _QUERY_SPLIT_TEXT_FIELDS:
            val = query_split.get(field)
            if not val:
                continue
            if isinstance(val, list):
                kw = " ".join(str(v) for v in val if v)
            else:
                kw = str(val).strip()
            if kw:
                keyword_strings.append(kw)

    def _single(q_str: str, k: int) -> List[str]:
        if not q_str.strip():
            return []
        tokens  = _bm25s.tokenize([q_str.lower()])
        results = bm25_model.retrieve(
            tokens, k=min(k, len(bm25_track_ids)), return_as="tuple"
        )
        return [bm25_track_ids[item["id"]] for item in results.documents[0]]

    if not keyword_strings:
        return _single(current_query, topk)

    n       = len(keyword_strings)
    per_q   = topk // n
    remain  = topk - per_q * n
    seen: Set[str] = set()
    result: List[str] = []
    for i, kw in enumerate(keyword_strings):
        k_i = per_q + (remain if i == n - 1 else 0)
        for tid in _single(kw, k_i):
            if tid not in seen:
                result.append(tid)
                seen.add(tid)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    config          = OmegaConf.load(args.config)
    cache_dir       = config.get("cache_dir", "./cache")
    qwen_model_path = config.get(
        "qwen_model_path",
        "/home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B",
    )
    dataset_name = config.get(
        "conversation_dataset_name", "talkpl-ai/TalkPlayData-Challenge-Dataset"
    )
    track_emb_db = config.get("track_emb_db_name",
                               "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    user_emb_db  = config.get("user_emb_db_name",
                               "talkpl-ai/TalkPlayData-Challenge-User-Embeddings")
    split_types  = list(config.get("track_split_types", ["all_tracks"]))

    # ── Load conv_emb store ───────────────────────────────────────────────────
    logger.info("Loading conv_emb_store from %s …", args.conv_emb_store)
    conv_emb_store = torch.load(args.conv_emb_store, map_location="cpu",
                                weights_only=True)
    logger.info("  %d entries.", len(conv_emb_store))

    # ── Load query_split store ───────────────────────────────────────────────
    query_split_store: Optional[Dict] = None
    if args.query_split_store and os.path.exists(args.query_split_store):
        logger.info("Loading query_split_store from %s …", args.query_split_store)
        query_split_store = torch.load(
            args.query_split_store, map_location="cpu", weights_only=True
        )
        logger.info("  %d entries.", len(query_split_store))

    # ── Load CF-BPR embeddings ────────────────────────────────────────────────
    logger.info("Loading track CF-BPR …")
    track_embs = load_embeddings(track_emb_db, split_types,
                                 id_field="track_id", emb_field="cf-bpr")
    logger.info("Loading user CF-BPR …")
    user_embs  = load_embeddings(user_emb_db,  split_types,
                                 id_field="user_id",  emb_field="cf-bpr")

    # ── Load or build track indices ───────────────────────────────────────────
    _idx_dir       = os.path.join("qwen", "retrieval_indices", "track_indices")
    metadata_path  = os.path.join(_idx_dir, "track_metadata_1024.pt")
    track_ids_path = os.path.join(_idx_dir, "track_ids.json")
    cf_item_index_dir = os.path.join("qwen", "cf_bpr_retrieval", "item_index")
    cf_item_path  = os.path.join(cf_item_index_dir, "item_vectors.pt")
    cf_item_ids_path = os.path.join(cf_item_index_dir, "track_ids.json")

    logger.info("Loading track metadata matrix …")
    metadata_matrix = torch.load(metadata_path, map_location="cpu", weights_only=True)
    with open(track_ids_path) as f:
        meta_track_ids = json.load(f)
    logger.info("  metadata matrix: %s", list(metadata_matrix.shape))

    # CF-BPR item index (from trained three-tower) – optional
    cf_model: Optional[CFBPRThreeTower] = None
    cf_item_matrix: Optional[torch.Tensor] = None
    cf_item_ids: Optional[List[str]] = None

    if os.path.exists(cf_item_path) and os.path.exists(cf_item_ids_path):
        logger.info("Loading three-tower item index from %s …", cf_item_index_dir)
        cf_item_matrix = torch.load(cf_item_path, map_location="cpu", weights_only=True)
        with open(cf_item_ids_path) as f:
            cf_item_ids = json.load(f)
        logger.info("  item index: %d tracks", len(cf_item_ids))

    if args.cf_model_path and os.path.exists(args.cf_model_path):
        logger.info("Loading three-tower model from %s …", args.cf_model_path)
        cf_model = CFBPRThreeTower()
        ckpt = torch.load(args.cf_model_path, map_location="cpu", weights_only=True)
        cf_model.load_state_dict(ckpt["model_state_dict"])
        cf_model.eval()
        logger.info("  Model loaded.")

        # If item index not yet built, build on the fly
        if cf_item_matrix is None:
            logger.info("Building item index from model …")
            track_ids_sorted = sorted(track_embs.keys())
            item_vecs = []
            bs = 512
            with torch.no_grad():
                for start in tqdm(range(0, len(track_ids_sorted), bs),
                                  desc="Item index", unit="batch"):
                    b_ids = track_ids_sorted[start: start + bs]
                    b_bpr = torch.stack([track_embs[tid]["cf-bpr"] for tid in b_ids])
                    item_vecs.append(cf_model.encode_item(b_bpr).cpu())
            cf_item_matrix = torch.cat(item_vecs, dim=0)
            cf_item_ids    = track_ids_sorted
            logger.info("  Item index built: %d tracks", len(cf_item_ids))
    else:
        logger.warning(
            "No three-tower model found at '%s'; ch1 falls back to raw CF-BPR cosine.",
            args.cf_model_path,
        )
        # Build fallback item matrix from raw CF-BPR
        logger.info("Building fallback raw CF-BPR item matrix …")
        cf_item_ids     = sorted(track_embs.keys())
        cf_bpr_list     = [track_embs[tid]["cf-bpr"] for tid in cf_item_ids]
        cf_item_matrix  = F.normalize(torch.stack(cf_bpr_list), p=2, dim=1)
        logger.info("  Fallback matrix: %d tracks", len(cf_item_ids))

    # ── BM25 index ───────────────────────────────────────────────────────────
    import bm25s as _bm25s
    bm25_dir  = os.path.join("qwen", "retrieval_indices", "bm25_index")
    ids_path  = os.path.join(bm25_dir, "track_ids.json")
    logger.info("Loading BM25 index …")
    bm25_model = _bm25s.BM25.load(bm25_dir, load_corpus=True)
    with open(ids_path) as f:
        bm25_track_ids = json.load(f)
    logger.info("  BM25: %d tracks", len(bm25_track_ids))

    # ── Load dataset ──────────────────────────────────────────────────────────
    split_name = getattr(args, "split", "test")
    logger.info("Loading '%s' split dataset …", split_name)
    ds = load_dataset(dataset_name, split=split_name)
    total_sessions = len(ds)
    if args.max_sessions > 0:
        total_sessions = min(args.max_sessions, total_sessions)
    logger.info("Evaluating on %d sessions.", total_sessions)

    # ── Accumulate hits @K per channel ───────────────────────────────────────
    # hits[channel][k] = count
    hits: Dict[str, Dict[int, int]] = {
        name: {k: 0 for k in TOPK_LIST} for name in CHANNEL_NAMES
    }
    total_queries = 0

    for idx in tqdm(range(total_sessions), desc="Eval", unit="session"):
        item       = ds[idx]
        session_id = item["session_id"]
        user_id    = item.get("user_id", None)
        convs      = item["conversations"]

        music_turns = {
            int(c["turn_number"]): c["content"]
            for c in convs
            if c.get("role") == "music" and c.get("content")
        }
        if not music_turns:
            continue

        for turn_number, gt_track_id in music_turns.items():
            # music 在 turn N 推荐，用户 query 在 turn N-1（或同轮 user 角色）
            # 先找同 turn_number 的 user turn，找不到再找 turn_number-1
            user_query = ""
            user_turn_number = turn_number  # 用于 conv_emb / query_split key
            for c in convs:
                if int(c["turn_number"]) == turn_number and c["role"] == "user":
                    user_query = c.get("content", "")
                    user_turn_number = turn_number
                    break
            if not user_query:
                # user 在上一轮发言
                prev_turn = turn_number - 1
                for c in convs:
                    if int(c["turn_number"]) == prev_turn and c["role"] == "user":
                        user_query = c.get("content", "")
                        user_turn_number = prev_turn
                        break
            if not user_query:
                continue

            # conv_emb for ch3 — key 用 user 所在 turn
            emb_key  = f"{session_id}_{user_turn_number}"
            conv_emb = conv_emb_store.get(emb_key, None)
            if conv_emb is None:
                conv_emb = torch.zeros(CONV_EMB_DIM)
            else:
                conv_emb = conv_emb.float()
                if conv_emb.shape[0] > CONV_EMB_DIM:
                    conv_emb = conv_emb[:CONV_EMB_DIM]
                elif conv_emb.shape[0] < CONV_EMB_DIM:
                    conv_emb = F.pad(conv_emb, (0, CONV_EMB_DIM - conv_emb.shape[0]))

            user_cf_bpr = _get_cf_bpr(user_embs, user_id)

            # query_split
            qs_raw = query_split_store.get(emb_key, None) if query_split_store else None
            query_split_dict: Optional[Dict] = None
            if qs_raw:
                try:
                    query_split_dict = json.loads(qs_raw) if isinstance(qs_raw, str) else qs_raw
                except Exception:
                    pass

            # ── Retrieve max(TOPK_LIST) candidates per channel ───────────────
            max_k = max(TOPK_LIST)

            ch1_all = retrieve_ch1_cf_bpr(
                cf_model, user_cf_bpr, conv_emb,
                cf_item_matrix, cf_item_ids, max_k,
            )
            ch3_all = retrieve_ch3_metadata(
                conv_emb, metadata_matrix, meta_track_ids, max_k
            )
            ch5_all = retrieve_ch5_bm25(
                bm25_model, bm25_track_ids, user_query, query_split_dict, max_k
            )

            total_queries += 1

            for k in TOPK_LIST:
                ch1_k = ch1_all[:k]
                ch3_k = ch3_all[:k]
                ch5_k = ch5_all[:k]
                union_k: Set[str] = set(ch1_k) | set(ch3_k) | set(ch5_k)

                if gt_track_id in ch1_k:
                    hits["ch1_CF-BPR"][k] += 1
                if gt_track_id in ch3_k:
                    hits["ch3_QwenMeta"][k] += 1
                if gt_track_id in ch5_k:
                    hits["ch5_BM25"][k] += 1
                if gt_track_id in union_k:
                    hits["ALL (union)"][k] += 1

    # ── Report ────────────────────────────────────────────────────────────────
    header_ks = "   ".join(f"@{k:<6}" for k in TOPK_LIST)
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(
        f"Retrieval Accuracy Evaluation  ({getattr(args,'split','test')} split, {total_queries} queries, "
        f"max_sessions={args.max_sessions})"
    )
    lines.append(f"{'='*80}")
    lines.append(f"{'Channel':<20}   {header_ks}")
    lines.append("-" * 80)

    for name in CHANNEL_NAMES:
        row_vals = []
        for k in TOPK_LIST:
            n     = hits[name][k]
            pct   = 100.0 * n / total_queries if total_queries else 0.0
            row_vals.append(f"{pct:6.2f}%")
        lines.append(f"{name:<20}   {'   '.join(row_vals)}")

    lines.append(f"{'='*80}")
    lines.append(f"Total queries: {total_queries}")
    report = "\n".join(lines)
    print(report)

    out_dir  = os.path.join("exp", "eval")
    os.makedirs(out_dir, exist_ok=True)
    split_tag = getattr(args, "split", "test")
    out_path = os.path.join(out_dir, f"retrieval_accuracy_{split_tag}.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    logger.info("Report saved to %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate 3-channel retrieval accuracy @K"
    )
    p.add_argument("--config",            type=str,
                   default="config/llama1b_multi_channel_devset.yaml")
    p.add_argument("--conv_emb_store",    type=str,
                   default="qwen/hist_conversation_embeddings_test_0.6b.pt")
    p.add_argument("--query_split_store", type=str,
                   default="qwen/query_split_test.pt")
    p.add_argument("--cf_model_path",     type=str,
                   default="qwen/cf_bpr_retrieval/model.pt",
                   help="Trained three-tower model checkpoint (.pt)")
    p.add_argument("--max_sessions",      type=int, default=200,
                   help="Number of sessions to evaluate")
    p.add_argument("--split",             type=str, default="test",
                   help="Dataset split: train / test")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
