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

TOPK_LIST    = [20, 50, 100, 200, 300, 400, 500]
CONV_EMB_DIM = 1024
CF_BPR_DIM   = 128
OUTPUT_DIM   = 128

# must match train_three_tower.py v5
AGE_GROUP_DIM           = 8
GENDER_DIM              = 4
COUNTRY_DIM             = 32
USER_PROFILE_DIM        = AGE_GROUP_DIM + GENDER_DIM + COUNTRY_DIM   # 44
USER_INPUT_DIM          = CF_BPR_DIM + USER_PROFILE_DIM              # 172
ATTR_PROJ_DIM           = 64
META_PROJ_DIM           = 64
POP_BUCKET_DIM          = 8
RELEASE_YEAR_BUCKET_DIM = 8
DURATION_BUCKET_DIM     = 8
ITEM_META_DIM           = POP_BUCKET_DIM + RELEASE_YEAR_BUCKET_DIM + DURATION_BUCKET_DIM  # 24
ITEM_INPUT_DIM          = CF_BPR_DIM + ATTR_PROJ_DIM + META_PROJ_DIM + ITEM_META_DIM      # 280

CHANNEL_NAMES = ["ch1_CF-BPR", "ch3_QwenMeta", "ch5_BM25", "ALL (union)"]


# ─────────────────────────────────────────────────────────────────────────────
#  Inline models — mirrors train_three_tower.py v5 & train_qwen_meta_tower.py
# ─────────────────────────────────────────────────────────────────────────────

import torch.nn as nn


def _hash_bucket(value: str, n: int) -> int:
    if not value: return 0
    return abs(hash(value)) % n

def _popularity_bucket(pop) -> int:
    try: p = float(pop)
    except: return 0
    if p < 10: return 1
    if p < 20: return 2
    if p < 35: return 3
    if p < 50: return 4
    if p < 65: return 5
    if p < 80: return 6
    return 7

def _release_year_bucket(rd) -> int:
    try: year = int(str(rd)[:4])
    except: return 0
    if year < 1970: return 1
    if year < 1980: return 2
    if year < 1990: return 3
    if year < 2000: return 4
    if year < 2005: return 5
    if year < 2010: return 6
    if year < 2015: return 7
    return 7

def _duration_bucket(d) -> int:
    try: d = float(d)
    except: return 0
    if d < 60000:  return 1
    if d < 120000: return 2
    if d < 180000: return 3
    if d < 210000: return 4
    if d < 240000: return 5
    if d < 300000: return 6
    return 7


class UserTower(nn.Module):
    def __init__(self, input_dim=USER_INPUT_DIM, hidden=256, out=OUTPUT_DIM, dropout=0.2):
        super().__init__()
        self.age_emb     = nn.Embedding(32,  AGE_GROUP_DIM)
        self.gender_emb  = nn.Embedding(8,   GENDER_DIM)
        self.country_emb = nn.Embedding(512, COUNTRY_DIM)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, out),
        )
    def forward(self, cf_bpr, age_idx, gender_idx, country_idx):
        x = torch.cat([cf_bpr, self.age_emb(age_idx),
                       self.gender_emb(gender_idx), self.country_emb(country_idx)], dim=1)
        return self.net(x)


class QueryTower(nn.Module):
    def __init__(self, conv_dim=CONV_EMB_DIM, hidden=256, out=OUTPUT_DIM, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(conv_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, out),
        )
    def forward(self, x): return self.net(x)


class ItemTower(nn.Module):
    def __init__(self, input_dim=ITEM_INPUT_DIM, hidden=256, out=OUTPUT_DIM, dropout=0.2,
                 attr_in=1024, meta_in=1024):
        super().__init__()
        self.attr_proj    = nn.Linear(attr_in, ATTR_PROJ_DIM)
        self.meta_proj    = nn.Linear(meta_in, META_PROJ_DIM)
        self.pop_emb      = nn.Embedding(8, POP_BUCKET_DIM)
        self.year_emb     = nn.Embedding(8, RELEASE_YEAR_BUCKET_DIM)
        self.duration_emb = nn.Embedding(8, DURATION_BUCKET_DIM)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, out),
        )
    def forward(self, cf_bpr, attr_emb, meta_emb, pop_idx, year_idx, dur_idx):
        x = torch.cat([cf_bpr, self.attr_proj(attr_emb), self.meta_proj(meta_emb),
                       self.pop_emb(pop_idx), self.year_emb(year_idx),
                       self.duration_emb(dur_idx)], dim=1)
        return self.net(x)


class CFBPRThreeTower(nn.Module):
    def __init__(self, temperature=20.0, dropout=0.2):
        super().__init__()
        self.temperature = temperature
        self.user_tower  = UserTower(USER_INPUT_DIM,  256, OUTPUT_DIM, dropout)
        self.query_tower = QueryTower(CONV_EMB_DIM,   256, OUTPUT_DIM, dropout)
        self.item_tower  = ItemTower(ITEM_INPUT_DIM,  256, OUTPUT_DIM, dropout)
        self.gate_linear = nn.Linear(OUTPUT_DIM * 2, OUTPUT_DIM)

    def encode_fusion(self, cf_bpr_user, age_idx, gender_idx, country_idx, conv_emb):
        u = self.user_tower(cf_bpr_user, age_idx, gender_idx, country_idx)
        q = self.query_tower(conv_emb)
        gate = torch.sigmoid(self.gate_linear(torch.cat([u, q], dim=1)))
        return F.normalize(gate * u + (1 - gate) * q, p=2, dim=1)

    def encode_item(self, cf_bpr, attr_emb, meta_emb, pop_idx, year_idx, dur_idx):
        return F.normalize(
            self.item_tower(cf_bpr, attr_emb, meta_emb, pop_idx, year_idx, dur_idx),
            p=2, dim=1,
        )


# ── QwenMeta Two-Tower (mirrors train_qwen_meta_tower.py) ───────────────────

class _TowerMLP(nn.Module):
    def __init__(self, in_dim, h1=512, h2=256, out=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2),    nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, out),
        )
    def forward(self, x): return self.net(x)


class QwenMetaTwoTower(nn.Module):
    def __init__(self, conv_dim=1024, meta_dim=1024, temperature=20.0, dropout=0.2):
        super().__init__()
        self.temperature = temperature
        self.query_tower = _TowerMLP(conv_dim, dropout=dropout)
        self.item_tower  = _TowerMLP(meta_dim, dropout=dropout)

    def encode_query(self, x):
        return F.normalize(self.query_tower(x), p=2, dim=1)

    def encode_item(self, x):
        return F.normalize(self.item_tower(x), p=2, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_cf_bpr(embeddings: Dict, entity_id: Optional[str],
                dim: int = CF_BPR_DIM) -> torch.Tensor:
    if entity_id and entity_id in embeddings:
        vec = embeddings[entity_id].get("cf-bpr", None)
        if vec is not None:
            v = vec.float()
            if v.shape[0] > dim: return v[:dim]
            if v.shape[0] < dim: return F.pad(v, (0, dim - v.shape[0]))
            return v
    return torch.zeros(dim, dtype=torch.float32)


def _raw_to_tensor(value, dim: int) -> torch.Tensor:
    if value is None:
        return torch.zeros(dim, dtype=torch.float32)
    try:
        t = torch.tensor(value, dtype=torch.float32)
        if t.ndim == 0 or t.numel() == 0: return torch.zeros(dim, dtype=torch.float32)
        if t.ndim > 1: t = t.flatten()
        if t.shape[0] < dim: t = F.pad(t, (0, dim - t.shape[0]))
        elif t.shape[0] > dim: t = t[:dim]
        return t
    except (TypeError, ValueError):
        return torch.zeros(dim, dtype=torch.float32)


def load_track_data(track_emb_db: str, track_meta_db: str,
                    split_types: List[str]) -> Dict:
    """Load track CF-BPR + attr/meta embeddings + meta buckets."""
    from datasets import concatenate_datasets
    ds = load_dataset(track_emb_db)
    valid = [s for s in split_types if s in ds.keys()] or list(ds.keys())
    emb_ds = concatenate_datasets([ds[s] for s in valid])

    cf_dim = CF_BPR_DIM
    for item in emb_ds:
        v = item.get("cf-bpr")
        if v is None: continue
        try:
            t = torch.tensor(v, dtype=torch.float32)
            if t.ndim == 1: cf_dim = t.shape[0]; break
        except Exception: pass

    track_data: Dict = {}
    for item in tqdm(emb_ds, desc="Loading track embs", unit="track"):
        tid = item["track_id"]
        track_data[tid] = {
            "cf-bpr":   _raw_to_tensor(item.get("cf-bpr"), cf_dim),
            "attr_emb": _raw_to_tensor(item.get("attributes-qwen3_embedding_0.6b"), 1024),
            "meta_emb": _raw_to_tensor(item.get("metadata-qwen3_embedding_0.6b"),   1024),
        }

    ds_meta  = load_dataset(track_meta_db)
    valid_m  = [s for s in split_types if s in ds_meta.keys()] or list(ds_meta.keys())
    meta_ds  = concatenate_datasets([ds_meta[s] for s in valid_m])
    for item in meta_ds:
        tid = item["track_id"]
        if tid not in track_data:
            track_data[tid] = {"cf-bpr": torch.zeros(cf_dim),
                               "attr_emb": torch.zeros(1024), "meta_emb": torch.zeros(1024)}
        track_data[tid]["pop_bucket"]  = _popularity_bucket(item.get("popularity"))
        track_data[tid]["year_bucket"] = _release_year_bucket(item.get("release_date"))
        track_data[tid]["dur_bucket"]  = _duration_bucket(item.get("duration"))

    for d in track_data.values():
        d.setdefault("pop_bucket", 0)
        d.setdefault("year_bucket", 0)
        d.setdefault("dur_bucket", 0)

    logger.info("Loaded %d tracks.", len(track_data))
    return track_data


def load_user_data(user_emb_db: str, user_meta_db: str,
                   split_types: List[str]) -> Dict:
    """Load user CF-BPR + profile buckets."""
    from datasets import concatenate_datasets
    ds = load_dataset(user_emb_db)
    valid = [s for s in split_types if s in ds.keys()] or list(ds.keys())
    emb_ds = concatenate_datasets([ds[s] for s in valid])

    cf_dim = CF_BPR_DIM
    for item in emb_ds:
        v = item.get("cf-bpr")
        if v is None: continue
        try:
            t = torch.tensor(v, dtype=torch.float32)
            if t.ndim == 1: cf_dim = t.shape[0]; break
        except Exception: pass

    user_data: Dict = {}
    for item in emb_ds:
        uid = item["user_id"]
        user_data[uid] = {"cf-bpr": _raw_to_tensor(item.get("cf-bpr"), cf_dim)}

    ds_meta = load_dataset(user_meta_db)
    valid_m = [s for s in split_types if s in ds_meta.keys()] or list(ds_meta.keys())
    meta_ds = concatenate_datasets([ds_meta[s] for s in valid_m])
    for item in meta_ds:
        uid = item["user_id"]
        if uid not in user_data:
            user_data[uid] = {"cf-bpr": torch.zeros(cf_dim)}
        user_data[uid]["age_idx"]     = _hash_bucket(str(item.get("age_group",   "") or ""), 32)
        user_data[uid]["gender_idx"]  = _hash_bucket(str(item.get("gender",      "") or ""), 8)
        user_data[uid]["country_idx"] = _hash_bucket(str(item.get("country_name","") or ""), 512)

    for d in user_data.values():
        d.setdefault("age_idx", 0); d.setdefault("gender_idx", 0); d.setdefault("country_idx", 0)

    logger.info("Loaded %d users.", len(user_data))
    return user_data


# ─────────────────────────────────────────────────────────────────────────────
#  Channel retrievals
# ─────────────────────────────────────────────────────────────────────────────

_QUERY_SPLIT_TEXT_FIELDS = [
    "artist", "album", "genre", "decade", "language",
    "popularity", "scene", "tempo",
]


def retrieve_ch1_cf_bpr(
    model:        Optional[CFBPRThreeTower],
    user_data_d:  Dict,                  # single user's data dict
    conv_emb:     torch.Tensor,          # [1024]
    track_data:   Dict,
    item_matrix:  torch.Tensor,          # [N, 128] prebuilt (fallback or 3-tower)
    track_ids:    List[str],
    topk:         int,
) -> List[str]:
    """ch1: three-tower fusion × item matrix cosine."""
    if model is None:
        # Fallback: raw user CF-BPR × item CF-BPR matrix
        u = F.normalize(user_data_d.get("cf-bpr", torch.zeros(CF_BPR_DIM)).float().unsqueeze(0),
                        p=2, dim=1).squeeze(0)
        scores = (item_matrix * u.unsqueeze(0)).sum(dim=1)
    else:
        model.eval()
        with torch.no_grad():
            u_cf  = user_data_d.get("cf-bpr", torch.zeros(CF_BPR_DIM)).float().unsqueeze(0)
            age   = torch.tensor([user_data_d.get("age_idx",     0)], dtype=torch.long)
            gen   = torch.tensor([user_data_d.get("gender_idx",  0)], dtype=torch.long)
            cnt   = torch.tensor([user_data_d.get("country_idx", 0)], dtype=torch.long)
            fusion = model.encode_fusion(u_cf, age, gen, cnt,
                                         conv_emb.unsqueeze(0)).squeeze(0)  # [128]
        scores = (item_matrix * fusion.unsqueeze(0)).sum(dim=1)

    topk    = min(topk, scores.shape[0])
    top_idx = torch.topk(scores, k=topk).indices.tolist()
    return [track_ids[i] for i in top_idx]


def retrieve_ch3_metadata(
    ch3_model:       Optional["QwenMetaTwoTower"],
    conv_emb:        torch.Tensor,   # [1024]
    item_matrix:     torch.Tensor,   # [N, 128] (learned) or [N, 1024] (raw)
    track_ids:       List[str],
    topk:            int,
) -> List[str]:
    """ch3: QwenMeta双塔模型 query_vec × item_vec cosine（若无模型则 raw 余弦）."""
    if ch3_model is not None:
        ch3_model.eval()
        with torch.no_grad():
            q = ch3_model.encode_query(conv_emb.unsqueeze(0)).squeeze(0)  # [128]
    else:
        q = F.normalize(conv_emb.unsqueeze(0), p=2, dim=1).squeeze(0)
    scores  = (item_matrix * q.unsqueeze(0)).sum(dim=1)
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
    track_emb_db  = config.get("track_emb_db_name",
                               "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    track_meta_db = config.get("item_db_name",
                               "talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    user_emb_db   = config.get("user_emb_db_name",
                               "talkpl-ai/TalkPlayData-Challenge-User-Embeddings")
    user_meta_db  = config.get("user_db_name",
                               "talkpl-ai/TalkPlayData-Challenge-User-Metadata")
    split_types   = list(config.get("track_split_types", ["all_tracks"]))

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

    # ── Load CF-BPR + rich feature embeddings ────────────────────────────────
    logger.info("Loading track data (CF-BPR + attr/meta + buckets) …")
    track_data = load_track_data(track_emb_db, track_meta_db, split_types)
    logger.info("Loading user data (CF-BPR + profile) …")
    user_data  = load_user_data(user_emb_db, user_meta_db, split_types)

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

    # ── QwenMeta双塔 ch3 item index ────────────────────────────────────────────
    ch3_model: Optional[QwenMetaTwoTower] = None
    ch3_item_matrix: Optional[torch.Tensor] = None
    ch3_item_ids: Optional[List[str]] = None

    if args.ch3_model_path and os.path.exists(args.ch3_model_path):
        logger.info("Loading QwenMeta two-tower model from %s …", args.ch3_model_path)
        ch3_model = QwenMetaTwoTower()
        ckpt = torch.load(args.ch3_model_path, map_location="cpu", weights_only=True)
        ch3_model.load_state_dict(ckpt["model_state_dict"])
        ch3_model.eval()
        # 检查是否有预建 item index
        ch3_idx_dir   = os.path.join("qwen", "qwen_meta_tower", "item_index")
        ch3_vec_path  = os.path.join(ch3_idx_dir, "item_vectors.pt")
        ch3_ids_path  = os.path.join(ch3_idx_dir, "track_ids.json")
        if os.path.exists(ch3_vec_path) and os.path.exists(ch3_ids_path):
            ch3_item_matrix = torch.load(ch3_vec_path, map_location="cpu", weights_only=True)
            with open(ch3_ids_path) as f:
                ch3_item_ids = json.load(f)
            logger.info("  ch3 item index loaded: %d tracks", len(ch3_item_ids))
        else:
            logger.info("  Building ch3 item index on-the-fly …")
            all_tids = sorted(track_data.keys())
            vecs = []
            bs = 512
            with torch.no_grad():
                for start in tqdm(range(0, len(all_tids), bs), desc="ch3 item index"):
                    b_ids = all_tids[start: start + bs]
                    b_emb = torch.stack([track_data[b]["meta_emb"] for b in b_ids])
                    vecs.append(ch3_model.encode_item(b_emb).cpu())
            ch3_item_matrix = torch.cat(vecs, dim=0)
            ch3_item_ids    = all_tids
            logger.info("  ch3 item index built: %d tracks", len(ch3_item_ids))
    else:
        logger.info("No ch3 model found; ch3 uses raw metadata-qwen3 cosine.")
        ch3_item_ids   = meta_track_ids
        ch3_item_matrix = metadata_matrix

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

        # Build item index on-the-fly if not present
        if cf_item_matrix is None:
            logger.info("Building item index from model …")
            track_ids_sorted = sorted(track_data.keys())
            item_vecs = []
            bs = 512
            with torch.no_grad():
                for start in tqdm(range(0, len(track_ids_sorted), bs),
                                  desc="Item index", unit="batch"):
                    b_ids = track_ids_sorted[start: start + bs]
                    cf   = torch.stack([track_data[t]["cf-bpr"]   for t in b_ids])
                    attr = torch.stack([track_data[t]["attr_emb"] for t in b_ids])
                    meta = torch.stack([track_data[t]["meta_emb"] for t in b_ids])
                    pop  = torch.tensor([track_data[t]["pop_bucket"]  for t in b_ids], dtype=torch.long)
                    year = torch.tensor([track_data[t]["year_bucket"] for t in b_ids], dtype=torch.long)
                    dur  = torch.tensor([track_data[t]["dur_bucket"]  for t in b_ids], dtype=torch.long)
                    item_vecs.append(cf_model.encode_item(cf, attr, meta, pop, year, dur).cpu())
            cf_item_matrix = torch.cat(item_vecs, dim=0)
            cf_item_ids    = track_ids_sorted
            logger.info("  Item index built: %d tracks", len(cf_item_ids))
    else:
        logger.warning("No three-tower model at '%s'; ch1 falls back to raw CF-BPR.", args.cf_model_path)
        cf_item_ids    = sorted(track_data.keys())
        cf_bpr_list    = [track_data[tid]["cf-bpr"] for tid in cf_item_ids]
        cf_item_matrix = F.normalize(torch.stack(cf_bpr_list), p=2, dim=1)
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

            user_d = user_data.get(user_id, {}) if user_id else {}

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
                cf_model, user_d, conv_emb,
                track_data, cf_item_matrix, cf_item_ids, max_k,
            )
            ch3_all = retrieve_ch3_metadata(
                ch3_model, conv_emb, ch3_item_matrix, ch3_item_ids, max_k
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
    p.add_argument("--ch3_model_path",    type=str,
                   default="qwen/qwen_meta_tower/model.pt",
                   help="Trained QwenMeta two-tower model checkpoint (.pt)")
    p.add_argument("--max_sessions",      type=int, default=200,
                   help="Number of sessions to evaluate")
    p.add_argument("--split",             type=str, default="test",
                   help="Dataset split: train / test")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
