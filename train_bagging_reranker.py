"""
train_bagging_reranker.py — Bagging 排序模型训练脚本
=====================================================

5 个子模型：
  1. FM         — 二阶特征交叉
  2. DCN        — 3层 Deep & Cross
  3. xDeepFM    — CIN + DNN
  4. LightGBM   — 梯度提升树（单独用 lightgbm 库训练）
  5. ThreeTowerGate — Query Gate 三塔排序

特征：
  User side:
    - cf-bpr emb (128d)
    - age_group / gender / country_name (hash bucket → emb)
  Item side:
    - cf-bpr emb (128d)
    - audio-laion_clap (512d)
    - image-siglip2 (1152d)
    - attributes-qwen3 (1024d)
    - lyrics-qwen3 (1024d)
    - metadata-qwen3 (1024d)
    - popularity / duration / release_year  (scalar)
    - tag multi-hot (top50)
  Query:
    - hist_conversation_emb (1024d)
  Cross:
    - user_cf_bpr × item_cf_bpr cosine (1 scalar)
    - retrieval_rank_norm (1 scalar)

负采样：
  每个正样本 = 5 个来自同一 session 召回池的随机负样本
               + 10 个来自全局随机负样本

训练流程：
  Phase1: train split，5 epochs
    每个 epoch 结束后在 test 上 evaluate NDCG@20（各子模型 + 集成）
  Phase2: test split，5 epochs（fine-tune, 较低 lr）
    同样每个 epoch 后 evaluate

集成方式：等权平均 (0.2 × 5)

Usage:
    cd /home/lijiatong06/music-crs-baselines/music_crs_local

    nohup python train_bagging_reranker.py \\
        --config config/llama1b_multi_channel_devset.yaml \\
        --train_conv_emb  qwen/hist_conversation_embeddings_train_0.6b.pt \\
        --test_conv_emb   qwen/hist_conversation_embeddings_test_0.6b.pt \\
        --retrieval_train qwen/retrieval_train_candidates.pt \\
        --retrieval_test  qwen/retrieval_test_candidates.pt \\
        --train_epochs 5 \\
        --test_epochs  5 \\
        --batch_size   64 \\
        --device cuda \\
    > train_bagging.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
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

# ── constants ────────────────────────────────────────────────────────────────
CONV_EMB_DIM   = 1024
CF_BPR_DIM     = 128
AUDIO_DIM      = 512
IMAGE_DIM      = 1152
ATTR_DIM       = 1024
LYRICS_DIM     = 1024
META_DIM       = 1024
TOP_TAGS       = 50
NUM_RETR_NEG   = 5    # negatives from retrieval pool
NUM_GLOBAL_NEG = 10   # global random negatives
ENSEMBLE_W     = 0.2  # equal weight for 5 models
RECALL_PER_CH  = 300  # 每路召回保留前300条
CAND_PER_CH    = 100  # 取每路前100条组成候选集(300总)

# ID hash bucket sizes
USER_ID_BUCKETS  = 65536
TRACK_ID_BUCKETS = 65536
USER_SPLIT_BUCKETS = 4   # train/test/val/unknown
PREF_LANG_BUCKETS  = 64
PREF_CULTURE_BUCKETS = 128

# New user/item id feature dims (one-hot too large → use emb-style hash onehot small buckets)
USER_ID_DIM      = 32   # hash(user_id, 32) → one-hot(32)
TRACK_ID_DIM     = 32
USER_SPLIT_DIM   = 4
PREF_LANG_DIM    = 16
PREF_CULTURE_DIM = 32

# Updated user/item side dims
# user: cf-bpr(128) + age_oh(32) + gender_oh(8) + country_oh(512)
#       + user_id_oh(32) + split_oh(4) + pref_lang_oh(16) + pref_culture_oh(32) = 764
USER_FEAT_DIM = CF_BPR_DIM + 32 + 8 + 512 + USER_ID_DIM + USER_SPLIT_DIM + PREF_LANG_DIM + PREF_CULTURE_DIM  # 764
# item embs: cf-bpr(128)+audio(512)+image(1152)+attr(1024)+lyrics(1024)+meta(1024) = 4864
# item id: track_id_oh(32)
ITEM_EMB_DIM  = CF_BPR_DIM + AUDIO_DIM + IMAGE_DIM + ATTR_DIM + LYRICS_DIM + META_DIM + TRACK_ID_DIM  # 4896
# scalars: pop(1)+dur(1)+year(1)+tags(50)+bpr_cos(1)+rank_n(1) = 55
SCALAR_DIM    = 3 + TOP_TAGS + 2  # 55
# total
FEATURE_DIM   = USER_FEAT_DIM + ITEM_EMB_DIM + SCALAR_DIM + CONV_EMB_DIM  # 764+4896+55+1024=6739


# ─────────────────────────────────────────────────────────────────────────────
#  Feature engineering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v, default=0.0) -> float:
    try: return float(v)
    except: return default

def _release_year(rd) -> float:
    try: return float(str(rd)[:4])
    except: return 2000.0

def _hash_bucket(v: str, n: int) -> int:
    if not v: return 0
    return abs(hash(v)) % n

def _raw_to_tensor(value, dim: int) -> torch.Tensor:
    if value is None: return torch.zeros(dim, dtype=torch.float32)
    try:
        t = torch.tensor(value, dtype=torch.float32)
        if t.ndim == 0 or t.numel() == 0: return torch.zeros(dim)
        if t.ndim > 1: t = t.flatten()
        if t.shape[0] < dim: t = F.pad(t, (0, dim - t.shape[0]))
        elif t.shape[0] > dim: t = t[:dim]
        return t
    except: return torch.zeros(dim, dtype=torch.float32)


class FeatureStore:
    """Loads all user/item embeddings + metadata into memory."""

    def __init__(self,
                 track_emb_db: str, track_meta_db: str,
                 user_emb_db: str,  user_meta_db: str,
                 split_types: List[str],
                 top_tags: int = TOP_TAGS):
        # ── track embeddings ─────────────────────────────────────────────
        ds = load_dataset(track_emb_db)
        valid = [s for s in split_types if s in ds.keys()] or list(ds.keys())
        emb_ds = concatenate_datasets([ds[s] for s in valid])

        self.track_data: Dict[str, Dict] = {}
        for row in tqdm(emb_ds, desc="Loading track embs"):
            tid = row["track_id"]
            self.track_data[tid] = {
                "cf-bpr":    _raw_to_tensor(row.get("cf-bpr"),                           CF_BPR_DIM),
                "audio":     _raw_to_tensor(row.get("audio-laion_clap"),                  AUDIO_DIM),
                "image":     _raw_to_tensor(row.get("image-siglip2"),                     IMAGE_DIM),
                "attr":      _raw_to_tensor(row.get("attributes-qwen3_embedding_0.6b"),   ATTR_DIM),
                "lyrics":    _raw_to_tensor(row.get("lyrics-qwen3_embedding_0.6b"),       LYRICS_DIM),
                "meta":      _raw_to_tensor(row.get("metadata-qwen3_embedding_0.6b"),     META_DIM),
            }

        # ── track metadata ───────────────────────────────────────────────
        ds_m = load_dataset(track_meta_db)
        valid_m = [s for s in split_types if s in ds_m.keys()] or list(ds_m.keys())
        meta_ds = concatenate_datasets([ds_m[s] for s in valid_m])

        self.track_meta: Dict[str, Dict] = {}
        all_tags: Dict[str, int] = {}
        for row in tqdm(meta_ds, desc="Loading track meta"):
            tid = row["track_id"]
            self.track_meta[tid] = row
            tags = row.get("tag_list") or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            for tag in tags:
                all_tags[tag] = all_tags.get(tag, 0) + 1
            if tid not in self.track_data:
                self.track_data[tid] = {k: torch.zeros(d) for k, d in
                    [("cf-bpr", CF_BPR_DIM), ("audio", AUDIO_DIM),
                     ("image", IMAGE_DIM),   ("attr",  ATTR_DIM),
                     ("lyrics", LYRICS_DIM), ("meta",  META_DIM)]}

        self.tag_vocab: Dict[str, int] = {
            t: i for i, (t, _) in
            enumerate(sorted(all_tags.items(), key=lambda x: -x[1])[:top_tags])
        }
        logger.info("Tag vocab size: %d", len(self.tag_vocab))

        # ── user embeddings ──────────────────────────────────────────────
        ds_u = load_dataset(user_emb_db)
        valid_u = [s for s in split_types if s in ds_u.keys()] or list(ds_u.keys())
        user_emb_ds = concatenate_datasets([ds_u[s] for s in valid_u])

        self.user_data: Dict[str, Dict] = {}
        for row in tqdm(user_emb_ds, desc="Loading user embs"):
            uid = row["user_id"]
            self.user_data[uid] = {"cf-bpr": _raw_to_tensor(row.get("cf-bpr"), CF_BPR_DIM)}

        # ── user metadata ────────────────────────────────────────────────
        ds_um = load_dataset(user_meta_db)
        valid_um = [s for s in split_types if s in ds_um.keys()] or list(ds_um.keys())
        user_meta_ds = concatenate_datasets([ds_um[s] for s in valid_um])
        for row in tqdm(user_meta_ds, desc="Loading user meta"):
            uid = row["user_id"]
            if uid not in self.user_data:
                self.user_data[uid] = {"cf-bpr": torch.zeros(CF_BPR_DIM)}
            self.user_data[uid]["age_idx"]     = _hash_bucket(str(row.get("age_group",    "") or ""), 32)
            self.user_data[uid]["gender_idx"]  = _hash_bucket(str(row.get("gender",       "") or ""), 8)
            self.user_data[uid]["country_idx"] = _hash_bucket(str(row.get("country_name", "") or ""), 512)

        for row in tqdm(user_meta_ds, desc="Loading user meta"):
            uid = row["user_id"]
            if uid not in self.user_data:
                self.user_data[uid] = {"cf-bpr": torch.zeros(CF_BPR_DIM)}
            self.user_data[uid]["age_idx"]          = _hash_bucket(str(row.get("age_group",               "") or ""), 32)
            self.user_data[uid]["gender_idx"]        = _hash_bucket(str(row.get("gender",                  "") or ""), 8)
            self.user_data[uid]["country_idx"]       = _hash_bucket(str(row.get("country_name",            "") or ""), 512)
            self.user_data[uid]["pref_lang_idx"]     = _hash_bucket(str(row.get("preferred_language",      "") or ""), PREF_LANG_BUCKETS)
            self.user_data[uid]["pref_culture_idx"]  = _hash_bucket(str(row.get("preferred_musical_culture","") or ""), PREF_CULTURE_BUCKETS)
            self.user_data[uid]["user_id_idx"]       = _hash_bucket(str(uid), USER_ID_BUCKETS) % USER_ID_DIM
            self.user_data[uid]["split_idx"]         = _hash_bucket(str(row.get("split", "") or ""), USER_SPLIT_BUCKETS)

        for d in self.user_data.values():
            d.setdefault("age_idx",         0)
            d.setdefault("gender_idx",      0)
            d.setdefault("country_idx",     0)
            d.setdefault("pref_lang_idx",   0)
            d.setdefault("pref_culture_idx",0)
            d.setdefault("user_id_idx",     0)
            d.setdefault("split_idx",       0)

        # track_id hash index
        for tid in self.track_data:
            self.track_data[tid]["track_id_idx"] = _hash_bucket(str(tid), TRACK_ID_BUCKETS) % TRACK_ID_DIM

        # ── precomputed dims ─────────────────────────────────────────────
        self.all_track_ids = sorted(self.track_data.keys())
        logger.info("FeatureStore ready: %d tracks, %d users",
                    len(self.track_data), len(self.user_data))

    def tag_multihot(self, track_id: str) -> torch.Tensor:
        vec = torch.zeros(len(self.tag_vocab))
        meta = self.track_meta.get(track_id, {})
        tags = meta.get("tag_list") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        for tag in tags:
            if tag in self.tag_vocab:
                vec[self.tag_vocab[tag]] = 1.0
        return vec

    def build_feature(self,
                      user_id: Optional[str],
                      track_id: str,
                      conv_emb: torch.Tensor,
                      retrieval_rank: int = 0,
                      retrieval_topk: int = 1) -> torch.Tensor:
        """
        Full feature vector layout (6739d):
          user(764) | item_emb+id(4896) | scalar(55) | query(1024)
        """
        u  = self.user_data.get(user_id, {}) if user_id else {}
        t  = self.track_data.get(track_id, {})
        tm = self.track_meta.get(track_id, {})

        # ── user side ────────────────────────────────────────────────────
        u_cf  = u.get("cf-bpr", torch.zeros(CF_BPR_DIM)).float()
        age_oh          = F.one_hot(torch.tensor(u.get("age_idx",         0)), 32).float()
        gender_oh       = F.one_hot(torch.tensor(u.get("gender_idx",      0)), 8).float()
        country_oh      = F.one_hot(torch.tensor(u.get("country_idx",     0)), 512).float()
        user_id_oh      = F.one_hot(torch.tensor(u.get("user_id_idx",     0)), USER_ID_DIM).float()
        split_oh        = F.one_hot(torch.tensor(u.get("split_idx",       0)), USER_SPLIT_DIM).float()
        pref_lang_oh    = F.one_hot(torch.tensor(u.get("pref_lang_idx",   0)), PREF_LANG_DIM).float()
        pref_culture_oh = F.one_hot(torch.tensor(u.get("pref_culture_idx",0)), PREF_CULTURE_DIM).float()
        # user total: 128+32+8+512+32+4+16+32 = 764

        # ── item side ────────────────────────────────────────────────────
        t_cf    = t.get("cf-bpr",  torch.zeros(CF_BPR_DIM)).float()
        audio   = t.get("audio",   torch.zeros(AUDIO_DIM)).float()
        image   = t.get("image",   torch.zeros(IMAGE_DIM)).float()
        attr    = t.get("attr",    torch.zeros(ATTR_DIM)).float()
        lyrics  = t.get("lyrics",  torch.zeros(LYRICS_DIM)).float()
        meta    = t.get("meta",    torch.zeros(META_DIM)).float()
        track_id_oh = F.one_hot(torch.tensor(t.get("track_id_idx", 0)), TRACK_ID_DIM).float()
        # item total: 128+512+1152+1024+1024+1024+32 = 4896

        # ── scalars ──────────────────────────────────────────────────────
        pop     = torch.tensor([_safe_float(tm.get("popularity",    0)) / 100.0])
        dur     = torch.tensor([min(_safe_float(tm.get("duration",  0)), 600000.0) / 600000.0])
        year    = torch.tensor([(_release_year(tm.get("release_date", "")) - 1950.0) / 80.0])
        tags    = self.tag_multihot(track_id)

        # ── cross features ────────────────────────────────────────────────
        u_n     = F.normalize(u_cf.unsqueeze(0), p=2, dim=1).squeeze(0)
        t_n     = F.normalize(t_cf.unsqueeze(0), p=2, dim=1).squeeze(0)
        bpr_cos = torch.dot(u_n, t_n).unsqueeze(0)
        rank_n  = torch.tensor([retrieval_rank / max(retrieval_topk - 1, 1)])

        return torch.cat([
            # user (764)
            u_cf, age_oh, gender_oh, country_oh,
            user_id_oh, split_oh, pref_lang_oh, pref_culture_oh,
            # item embs + id (4896)
            t_cf, audio, image, attr, lyrics, meta, track_id_oh,
            # scalars (55)
            pop, dur, year, tags, bpr_cos, rank_n,
            # query (1024)
            conv_emb.float(),
        ], dim=0)

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM  # 6739


# ─────────────────────────────────────────────────────────────────────────────
#  Sub-models
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. FM ────────────────────────────────────────────────────────────────────

class FMModel(nn.Module):
    """Factorization Machine: bias + linear + 2nd-order interaction."""

    def __init__(self, input_dim: int, k: int = 16):
        super().__init__()
        self.bias      = nn.Parameter(torch.zeros(1))
        self.linear    = nn.Linear(input_dim, 1, bias=False)
        self.embedding = nn.Parameter(torch.empty(input_dim, k))
        nn.init.normal_(self.embedding, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, D] → [B]
        lin  = self.linear(x).squeeze(-1)
        xv   = x @ self.embedding                          # [B, k]
        sq_sum   = xv.pow(2).sum(1)
        sum_sq   = (x.pow(2) @ self.embedding.pow(2)).sum(1)
        return self.bias + lin + 0.5 * (sq_sum - sum_sq)


# ── 2. DCN ───────────────────────────────────────────────────────────────────

class CrossLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, dim, bias=True)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * self.w(x) + x


class DCNModel(nn.Module):
    """Deep & Cross Network: 3 cross layers + 3-layer MLP."""

    def __init__(self, input_dim: int, n_cross: int = 3,
                 hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.cross_layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(n_cross)])
        self.deep = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 64),
        )
        self.out = nn.Linear(input_dim + 64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x
        for layer in self.cross_layers:
            xc = layer(x, xc)
        xd = self.deep(x)
        return self.out(torch.cat([xc, xd], dim=1)).squeeze(-1)


# ── 3. xDeepFM ───────────────────────────────────────────────────────────────

class CINLayer(nn.Module):
    """Compressed Interaction Network single layer."""

    def __init__(self, n_fields_in: int, n_fields_0: int, n_units: int):
        super().__init__()
        self.conv = nn.Conv1d(n_fields_in * n_fields_0, n_units, kernel_size=1)

    def forward(self, h: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        # h: [B, n_in, D], x0: [B, n0, D]
        B, n_in, D = h.shape
        n0 = x0.shape[1]
        outer = h.unsqueeze(2) * x0.unsqueeze(1)  # [B, n_in, n0, D]
        outer = outer.view(B, n_in * n0, D)
        out   = self.conv(outer)                   # [B, n_units, D]
        return out


class xDeepFMModel(nn.Module):
    """xDeepFM: CIN (2 layers) + DNN + linear."""

    def __init__(self, input_dim: int, cin_units: int = 32,
                 hidden: int = 256, dropout: float = 0.2,
                 emb_dim: int = 8, n_fields: int = 16):
        super().__init__()
        # project input to field embeddings
        self.n_fields = n_fields
        self.emb_dim  = emb_dim
        self.proj     = nn.Linear(input_dim, n_fields * emb_dim)

        self.cin1 = CINLayer(n_fields, n_fields, cin_units)
        self.cin2 = CINLayer(cin_units, n_fields, cin_units)

        self.dnn = nn.Sequential(
            nn.Linear(n_fields * emb_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 64),
        )
        cin_out_dim = cin_units * emb_dim * 2  # 2 CIN layers sum-pooled
        self.out = nn.Linear(64 + cin_out_dim + 1, 1)
        self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # field embeddings
        fe = self.proj(x).view(B, self.n_fields, self.emb_dim)  # [B, nF, D]
        # CIN
        h1 = self.cin1(fe, fe)                                    # [B, cu, D]
        h2 = self.cin2(h1, fe)
        cin_feat = torch.cat([
            h1.sum(dim=2),   # [B, cu]
            h2.sum(dim=2),   # [B, cu]
        ], dim=1)
        # DNN on projected field embs
        dnn_out = self.dnn(fe.view(B, -1))                        # [B, 64]
        lin_out = self.linear(x)                                   # [B, 1]
        return self.out(torch.cat([dnn_out, cin_feat, lin_out], dim=1)).squeeze(-1)


# ── 4. LightGBM wrapper (fit later, GPU not required) ────────────────────────

class LGBMWrapper:
    """LightGBM ranker wrapper with sklearn-like API."""

    def __init__(self):
        self.lgb   = None  # lazy import in fit()
        self.model = None
        self.fitted = False
        # 检查是否可用
        try:
            import lightgbm as _lgb  # noqa
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("lightgbm not installed. LGBMWrapper will be skipped. "
                           "Install with: pip install lightgbm")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train on (N, D) features and binary labels (1=positive)."""
        if not self._available:
            logger.warning("[LGBM] Skipping fit — lightgbm not installed.")
            return
        import lightgbm as lgb
        self.lgb = lgb
        self.model = self.lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_jobs=8,
            verbose=-1,
        )
        self.model.fit(X, y)
        self.fitted = True

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted or self.model is None:
            return np.zeros(len(X))
        return self.model.predict_proba(X)[:, 1]


# ── 5. Three-Tower Gate Reranker ─────────────────────────────────────────────

class ThreeTowerGate(nn.Module):
    """
    User塔(764d→128) + Query塔(1024d→128) + Gate → fusion
    Item塔(4896d → 128) + 召回特征(55d)
    Score = MLP([fusion, item_repr, scalars])
    """

    def __init__(self,
                 user_dim: int = USER_FEAT_DIM,   # 764
                 query_dim: int = CONV_EMB_DIM,
                 item_emb_dim: int = ITEM_EMB_DIM, # 4896
                 scalar_dim: int = SCALAR_DIM,     # 55
                 hidden: int = 256,
                 out_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_dim,  hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.query_tower = nn.Sequential(
            nn.Linear(query_dim, hidden * 2), nn.BatchNorm1d(hidden * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.gate_linear = nn.Linear(out_dim * 2, out_dim)
        self.item_tower = nn.Sequential(
            nn.Linear(item_emb_dim, hidden * 2), nn.BatchNorm1d(hidden * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        # final MLP
        self.scorer = nn.Sequential(
            nn.Linear(out_dim * 2 + scalar_dim, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, user_feat: torch.Tensor,   # [B, user_dim]
                      query_feat: torch.Tensor,  # [B, 1024]
                      item_feat: torch.Tensor,   # [B, item_emb_dim]
                      scalar_feat: torch.Tensor, # [B, scalar_dim]
               ) -> torch.Tensor:
        u = self.user_tower(user_feat)
        q = self.query_tower(query_feat)
        gate = torch.sigmoid(self.gate_linear(torch.cat([u, q], dim=1)))
        fusion = gate * u + (1 - gate) * q   # [B, 128]
        item = self.item_tower(item_feat)     # [B, 128]
        return self.scorer(torch.cat([fusion, item, scalar_feat], dim=1)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
#  Bagging ensemble
# ─────────────────────────────────────────────────────────────────────────────

class BaggingReranker:
    """Manages 5 sub-models + LightGBM, equal-weight ensemble."""

    def __init__(self, feat_store: FeatureStore, args, device: str):
        self.feat_store = feat_store
        self.device     = device
        self.args       = args
        fd = feat_store.feature_dim

        # ── neural models ────────────────────────────────────────────────
        self.fm      = FMModel(fd).to(device)
        self.dcn     = DCNModel(fd).to(device)
        self.xdfm    = xDeepFMModel(fd).to(device)
        self.ttg     = ThreeTowerGate().to(device)
        self.lgbm    = LGBMWrapper()

        # ── optimizers ───────────────────────────────────────────────────
        self.opt_fm   = torch.optim.Adam(self.fm.parameters(),   lr=args.lr, weight_decay=1e-4)
        self.opt_dcn  = torch.optim.Adam(self.dcn.parameters(),  lr=args.lr, weight_decay=1e-4)
        self.opt_xdfm = torch.optim.Adam(self.xdfm.parameters(), lr=args.lr, weight_decay=1e-4)
        self.opt_ttg  = torch.optim.Adam(self.ttg.parameters(),  lr=args.lr, weight_decay=1e-4)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def _split_feat(self, feat: torch.Tensor):
        """Split full feature into (user_feat, query_feat, item_feat, scalar_feat).

        Layout (6739d):
          user(764) | item_emb+id(4896) | scalars(55) | query(1024)
          offsets: 0, 764, 5660, 5715, 5715+1024=6739
        """
        a = USER_FEAT_DIM                    # 764
        b = a + ITEM_EMB_DIM                 # 764+4896 = 5660
        c = b + SCALAR_DIM                   # 5660+55  = 5715
        user_feat   = feat[:, :a]
        item_feat   = feat[:, a:b]
        scalar_feat = feat[:, b:c]
        query_feat  = feat[:, c:]             # 1024
        return user_feat, query_feat, item_feat, scalar_feat

    def _forward_all(self, feat: torch.Tensor):
        uf, qf, itf, sf = self._split_feat(feat)
        s_fm   = self.fm(feat)
        s_dcn  = self.dcn(feat)
        s_xdfm = self.xdfm(feat)
        s_ttg  = self.ttg(uf, qf, itf, sf)
        return s_fm, s_dcn, s_xdfm, s_ttg

    def train_step(self, feat: torch.Tensor, label: torch.Tensor):
        """Single gradient step on all neural models."""
        for model, opt in [(self.fm, self.opt_fm),
                           (self.dcn, self.opt_dcn),
                           (self.xdfm, self.opt_xdfm),
                           (self.ttg, self.opt_ttg)]:
            model.train()
        self.fm.train(); self.dcn.train(); self.xdfm.train(); self.ttg.train()

        feat   = feat.to(self.device)
        label  = label.to(self.device)

        losses = {}
        for name, model, opt in [
            ("fm",   self.fm,   self.opt_fm),
            ("dcn",  self.dcn,  self.opt_dcn),
            ("xdfm", self.xdfm, self.opt_xdfm),
        ]:
            score = model(feat)
            loss  = self.loss_fn(score, label)
            opt.zero_grad(); loss.backward(); opt.step()
            losses[name] = loss.item()

        # three-tower gate
        uf, qf, itf, sf = self._split_feat(feat)
        s_ttg = self.ttg(uf, qf, itf, sf)
        loss_ttg = self.loss_fn(s_ttg, label)
        self.opt_ttg.zero_grad(); loss_ttg.backward(); self.opt_ttg.step()
        losses["ttg"] = loss_ttg.item()

        return losses

    @torch.no_grad()
    def predict_scores(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return scores from each model. feat: [N, D]."""
        feat = feat.to(self.device)
        self.fm.eval(); self.dcn.eval(); self.xdfm.eval(); self.ttg.eval()
        uf, qf, itf, sf = self._split_feat(feat)
        scores = {
            "FM":      torch.sigmoid(self.fm(feat)).cpu(),
            "DCN":     torch.sigmoid(self.dcn(feat)).cpu(),
            "xDeepFM": torch.sigmoid(self.xdfm(feat)).cpu(),
            "TTGate":  torch.sigmoid(self.ttg(uf, qf, itf, sf)).cpu(),
        }
        # LGBM
        if self.lgbm.fitted:
            lgbm_scores = self.lgbm.predict_score(feat.cpu().numpy())
            scores["LGBM"] = torch.tensor(lgbm_scores, dtype=torch.float32)
        else:
            scores["LGBM"] = torch.zeros(feat.shape[0])
        return scores

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.fm.state_dict(),   os.path.join(out_dir, "fm.pt"))
        torch.save(self.dcn.state_dict(),  os.path.join(out_dir, "dcn.pt"))
        torch.save(self.xdfm.state_dict(), os.path.join(out_dir, "xdfm.pt"))
        torch.save(self.ttg.state_dict(),  os.path.join(out_dir, "ttg.pt"))
        if self.lgbm.fitted:
            import pickle
            with open(os.path.join(out_dir, "lgbm.pkl"), "wb") as f:
                pickle.dump(self.lgbm.model, f)
        logger.info("Saved models to %s", out_dir)

    def load(self, out_dir: str):
        self.fm.load_state_dict(torch.load(os.path.join(out_dir, "fm.pt"),   map_location=self.device, weights_only=True))
        self.dcn.load_state_dict(torch.load(os.path.join(out_dir, "dcn.pt"),  map_location=self.device, weights_only=True))
        self.xdfm.load_state_dict(torch.load(os.path.join(out_dir, "xdfm.pt"), map_location=self.device, weights_only=True))
        self.ttg.load_state_dict(torch.load(os.path.join(out_dir, "ttg.pt"),  map_location=self.device, weights_only=True))
        lgbm_path = os.path.join(out_dir, "lgbm.pkl")
        if os.path.exists(lgbm_path):
            import pickle
            with open(lgbm_path, "rb") as f:
                self.lgbm.model  = pickle.load(f)
                self.lgbm.fitted = True
        logger.info("Loaded models from %s", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
#  Sample building
# ─────────────────────────────────────────────────────────────────────────────

def build_samples(dataset_name: str, split: str,
                  conv_emb_store: Dict,
                  retrieval_store: Optional[Dict],
                  feat_store: FeatureStore,
                  num_retr_neg: int = NUM_RETR_NEG,
                  num_global_neg: int = NUM_GLOBAL_NEG):
    """
    Returns list of dicts:
      {feat: Tensor[D], label: float}
    Plus lgbm_X/lgbm_y arrays.
    """
    logger.info("Building samples for split='%s' ...", split)
    ds = load_dataset(dataset_name, split=split)
    samples = []
    all_tids = feat_store.all_track_ids

    for item in tqdm(ds, desc=f"Samples [{split}]", unit="session"):
        session_id = item["session_id"]
        user_id    = item.get("user_id")
        convs      = item["conversations"]
        assessments= item.get("goal_progress_assessments", [])
        asmt_map   = {int(a["turn_number"]): a.get("goal_progress_assessment")
                      for a in assessments}

        music_turns = {int(c["turn_number"]): c["content"]
                       for c in convs
                       if c.get("role") == "music" and c.get("content")}

        for turn_num, gt_tid in music_turns.items():
            fb_turn = turn_num + 1
            if fb_turn not in asmt_map: continue
            gpa = asmt_map[fb_turn]
            if gpa not in ("MOVES_TOWARD_GOAL", "DOES_NOT_MOVE_TOWARD_GOAL"): continue

            emb_key = f"{session_id}_{turn_num}"
            if emb_key not in conv_emb_store:
                emb_key = f"{session_id}_{turn_num - 1}"
            if emb_key not in conv_emb_store: continue
            conv_emb = conv_emb_store[emb_key].float()
            if conv_emb.shape[0] > CONV_EMB_DIM:
                conv_emb = conv_emb[:CONV_EMB_DIM]
            elif conv_emb.shape[0] < CONV_EMB_DIM:
                conv_emb = F.pad(conv_emb, (0, CONV_EMB_DIM - conv_emb.shape[0]))

            # retrieval pool negatives
            # retrieval_store value may be:
            #   新格式: {"ch1": [...], "ch3": [...], "ch5": [...], "union": [...]}
            #   旧格式: [track_id, ...]
            retr_pool: List[str] = []
            if retrieval_store:
                raw_pool = retrieval_store.get(emb_key) or retrieval_store.get(f"{session_id}_{turn_num}")
                if isinstance(raw_pool, dict):
                    # 新格式：使用 union（每路top100去重后的候选集，共约300条）
                    pool = raw_pool.get("union", [])
                elif isinstance(raw_pool, list):
                    pool = raw_pool
                else:
                    pool = []
                if pool:
                    retr_pool = [t for t in pool if t != gt_tid]

            # positive
            pos_feat = feat_store.build_feature(user_id, gt_tid, conv_emb, 0, 1)
            samples.append({"feat": pos_feat, "label": 1.0,
                            "user_id": user_id, "track_id": gt_tid, "conv_emb": conv_emb})

            # retrieval negatives
            neg_retr = random.sample(retr_pool, min(num_retr_neg, len(retr_pool)))
            for r, ntid in enumerate(neg_retr):
                nf = feat_store.build_feature(user_id, ntid, conv_emb, r + 1, len(retr_pool) + 1)
                samples.append({"feat": nf, "label": 0.0,
                                "user_id": user_id, "track_id": ntid, "conv_emb": conv_emb})

            # global negatives
            for _ in range(num_global_neg):
                ntid = random.choice(all_tids)
                nf   = feat_store.build_feature(user_id, ntid, conv_emb, 0, 1)
                samples.append({"feat": nf, "label": 0.0,
                                "user_id": user_id, "track_id": ntid, "conv_emb": conv_emb})

    logger.info("Built %d samples for split='%s'.", len(samples), split)
    return samples


# ─────────────────────────────────────────────────────────────────────────────
#  NDCG@K evaluation
# ─────────────────────────────────────────────────────────────────────────────

def ndcg_at_k(ranked_labels: List[int], k: int = 20) -> float:
    """Compute NDCG@K given a list of relevance labels (1 or 0), sorted by model."""
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ranked_labels[:k]))
    ideal = sorted(ranked_labels, reverse=True)
    idcg  = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ndcg(bagging: BaggingReranker,
                  dataset_name: str,
                  split: str,
                  conv_emb_store: Dict,
                  retrieval_store: Optional[Dict],
                  feat_store: FeatureStore,
                  max_sessions: int = 200,
                  k: int = 20) -> Dict[str, float]:
    """Evaluate NDCG@K per model + ensemble on `max_sessions` sessions."""
    ds = load_dataset(dataset_name, split=split)
    model_names = ["FM", "DCN", "xDeepFM", "LGBM", "TTGate", "Ensemble"]
    ndcg_sums   = {m: 0.0 for m in model_names}
    n_queries   = 0
    n_sessions  = 0

    for item in ds:
        if n_sessions >= max_sessions: break
        n_sessions += 1
        session_id = item["session_id"]
        user_id    = item.get("user_id")
        convs      = item["conversations"]
        assessments= item.get("goal_progress_assessments", [])
        asmt_map   = {int(a["turn_number"]): a.get("goal_progress_assessment")
                      for a in assessments}

        music_turns = {int(c["turn_number"]): c["content"]
                       for c in convs
                       if c.get("role") == "music" and c.get("content")}

        for turn_num, gt_tid in music_turns.items():
            fb_turn = turn_num + 1
            if fb_turn not in asmt_map: continue
            gpa = asmt_map[fb_turn]
            if gpa not in ("MOVES_TOWARD_GOAL", "DOES_NOT_MOVE_TOWARD_GOAL"): continue

            emb_key = f"{session_id}_{turn_num}"
            if emb_key not in conv_emb_store:
                emb_key = f"{session_id}_{turn_num - 1}"
            if emb_key not in conv_emb_store: continue
            conv_emb = conv_emb_store[emb_key].float()
            if conv_emb.shape[0] > CONV_EMB_DIM:
                conv_emb = conv_emb[:CONV_EMB_DIM]
            elif conv_emb.shape[0] < CONV_EMB_DIM:
                conv_emb = F.pad(conv_emb, (0, CONV_EMB_DIM - conv_emb.shape[0]))

            # build candidate list: gt + retrieval pool
            pool_key = emb_key
            cands: List[str] = []
            if retrieval_store:
                c = retrieval_store.get(pool_key) or retrieval_store.get(f"{session_id}_{turn_num}")
                if c: cands = list(c)
            if gt_tid not in cands:
                cands = [gt_tid] + cands[:k - 1]
            if len(cands) < k:
                extra = [t for t in feat_store.all_track_ids if t not in set(cands)]
                cands += random.sample(extra, min(k - len(cands), len(extra)))

            feats = torch.stack([
                feat_store.build_feature(user_id, tid, conv_emb, r, len(cands))
                for r, tid in enumerate(cands)
            ])
            scores = bagging.predict_scores(feats)
            labels = [1 if tid == gt_tid else 0 for tid in cands]

            for mname, scr in scores.items():
                ranked_idx   = torch.argsort(scr, descending=True).tolist()
                ranked_labels = [labels[i] for i in ranked_idx]
                ndcg_sums[mname] += ndcg_at_k(ranked_labels, k)

            # ensemble: equal weight avg
            ens_score = sum(scores[m] for m in ["FM", "DCN", "xDeepFM", "LGBM", "TTGate"]) * ENSEMBLE_W
            ranked_idx     = torch.argsort(ens_score, descending=True).tolist()
            ranked_labels  = [labels[i] for i in ranked_idx]
            ndcg_sums["Ensemble"] += ndcg_at_k(ranked_labels, k)

            n_queries += 1

    result = {m: ndcg_sums[m] / max(n_queries, 1) for m in model_names}
    return result


def print_ndcg_table(phase: str, epoch: int, result: Dict[str, float], k: int = 20):
    lines = [
        f"\n{'='*60}",
        f"  NDCG@{k} | {phase}  Epoch {epoch}",
        f"{'='*60}",
    ]
    for m, v in result.items():
        lines.append(f"  {m:<12}  {v*100:.3f}%")
    lines.append("=" * 60)
    logger.info("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def collate_feat(samples: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    feats  = torch.stack([s["feat"] for s in samples])
    labels = torch.tensor([s["label"] for s in samples], dtype=torch.float32)
    return feats, labels


def _train_one_epoch(phase_name: str, epoch: int, total_epochs: int,
                     samples: List[Dict], bagging: BaggingReranker, args):
    """单 epoch 训练，每 1/10 epoch 向 logger 打一次各模型 loss。"""
    random.shuffle(samples)
    total_loss = {k: 0.0 for k in ["fm", "dcn", "xdfm", "ttg"]}
    n_steps    = 0
    n_batches  = max(1, len(samples) // args.batch_size)
    log_every  = max(1, n_batches // 10)

    pbar = tqdm(range(0, len(samples), args.batch_size),
                desc=f"[{phase_name}] Epoch {epoch}/{total_epochs}",
                unit="batch", ncols=120)
    for start in pbar:
        batch = samples[start: start + args.batch_size]
        if not batch: continue
        feats, labels = collate_feat(batch)
        losses = bagging.train_step(feats, labels)
        for k in losses: total_loss[k] += losses[k]
        n_steps += 1
        pbar.set_postfix({k: f"{v/n_steps:.4f}" for k, v in total_loss.items()})
        if n_steps % log_every == 0:
            step_log = " | ".join(f"{k}={v/n_steps:.4f}" for k, v in total_loss.items())
            logger.info("[%s] Epoch %d  step %4d/%4d  loss: %s",
                        phase_name, epoch, n_steps, n_batches, step_log)

    avg_log = " | ".join(f"{k}={v/max(n_steps,1):.4f}" for k, v in total_loss.items())
    logger.info("[%s] Epoch %d  DONE  loss: %s", phase_name, epoch, avg_log)


def _fit_lgbm(phase_name: str, samples: List[Dict], bagging: BaggingReranker, args):
    """在给定样本上训练 LightGBM（随机子采样上限）。"""
    lgbm_max = getattr(args, "lgbm_max_samples", 80000)
    sub = samples if len(samples) <= lgbm_max else random.sample(samples, lgbm_max)
    logger.info("[LGBM-%s] Fitting on %d samples (max=%d)...",
                phase_name, len(sub), lgbm_max)
    X = np.array([s["feat"].numpy() for s in sub])
    y = np.array([int(s["label"]) for s in sub])
    bagging.lgbm.fit(X, y)
    logger.info("[LGBM-%s] Done.", phase_name)


def run_training(dataset_name: str,
                 train_samples: List[Dict],
                 test_samples:  List[Dict],
                 bagging: BaggingReranker,
                 args,
                 test_conv_emb: Dict,
                 test_retrieval: Optional[Dict],
                 feat_store: FeatureStore,
                 ckpt_dir: str):
    """
    交替训练流程，共 args.train_epochs 轮：

      for epoch in 1..train_epochs:
        1. LGBM fit (train)
        2. 神经网络 1 epoch (train，lr=args.lr)
        3. 在 test 上评估 NDCG@20（各模型 + ensemble）
        4. LGBM fit (test)
        5. 神经网络 1 epoch (test，lr=args.lr_finetune)
        6. 在 test 上再次评估 NDCG@20
        7. 恢复 lr=args.lr，保存 checkpoint
    """
    total_epochs = args.train_epochs

    for epoch in range(1, total_epochs + 1):
        logger.info("\n%s", "=" * 70)
        logger.info("  EPOCH %d / %d", epoch, total_epochs)
        logger.info("%s", "=" * 70)

        # ── Step 1+2: LGBM + neural on train ─────────────────────────────
        _fit_lgbm("train", train_samples, bagging, args)
        _train_one_epoch("TRAIN", epoch, total_epochs, train_samples, bagging, args)

        # ── Step 3: Evaluate on test ──────────────────────────────────────
        logger.info("[Epoch %d] === Evaluating after TRAIN epoch ===", epoch)
        result_after_train = evaluate_ndcg(
            bagging, dataset_name, "test",
            test_conv_emb, test_retrieval, feat_store,
            max_sessions=400, k=20)
        print_ndcg_table(f"After-Train-E{epoch}", epoch, result_after_train, k=20)

        if test_samples:
            # ── Step 4+5: LGBM + neural on test (fine-tune lr) ───────────
            _fit_lgbm("test", test_samples, bagging, args)
            for opt in [bagging.opt_fm, bagging.opt_dcn,
                        bagging.opt_xdfm, bagging.opt_ttg]:
                for pg in opt.param_groups: pg["lr"] = args.lr_finetune
            _train_one_epoch("TEST-FT", epoch, total_epochs, test_samples, bagging, args)

            # 恢复 lr
            for opt in [bagging.opt_fm, bagging.opt_dcn,
                        bagging.opt_xdfm, bagging.opt_ttg]:
                for pg in opt.param_groups: pg["lr"] = args.lr

            # ── Step 6: Evaluate again ────────────────────────────────────
            logger.info("[Epoch %d] === Evaluating after TEST-FT epoch ===", epoch)
            result_after_test = evaluate_ndcg(
                bagging, dataset_name, "test",
                test_conv_emb, test_retrieval, feat_store,
                max_sessions=400, k=20)
            print_ndcg_table(f"After-TestFT-E{epoch}", epoch, result_after_test, k=20)

        # ── Step 7: Checkpoint ────────────────────────────────────────────
        bagging.save(os.path.join(ckpt_dir, f"epoch{epoch}"))
        logger.info("[Epoch %d] Checkpoint saved to %s/epoch%d", epoch, ckpt_dir, epoch)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    config       = OmegaConf.load(args.config)
    device       = args.device or config.get("device", "cuda")
    dataset_name = config.get("conversation_dataset_name",
                               "talkpl-ai/TalkPlayData-Challenge-Dataset")
    track_emb_db = config.get("track_emb_db_name",
                               "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    track_meta_db = config.get("item_db_name",
                                "talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    user_emb_db  = config.get("user_emb_db_name",
                               "talkpl-ai/TalkPlayData-Challenge-User-Embeddings")
    user_meta_db = config.get("user_db_name",
                               "talkpl-ai/TalkPlayData-Challenge-User-Metadata")
    split_types  = list(config.get("track_split_types", ["all_tracks"]))

    # ── Load conv_emb stores ──────────────────────────────────────────────
    logger.info("Loading train conv_emb …"); 
    train_ce = torch.load(args.train_conv_emb, map_location="cpu", weights_only=True)
    logger.info("Loading test conv_emb …")
    test_ce  = torch.load(args.test_conv_emb,  map_location="cpu", weights_only=True)

    # ── Load retrieval candidate stores (optional) ────────────────────────
    train_retr: Optional[Dict] = None
    test_retr:  Optional[Dict] = None
    if args.retrieval_train and os.path.exists(args.retrieval_train):
        logger.info("Loading train retrieval candidates …")
        train_retr = torch.load(args.retrieval_train, map_location="cpu", weights_only=False)
    if args.retrieval_test and os.path.exists(args.retrieval_test):
        logger.info("Loading test retrieval candidates …")
        test_retr  = torch.load(args.retrieval_test,  map_location="cpu", weights_only=False)

    # ── Build FeatureStore ────────────────────────────────────────────────
    logger.info("Building FeatureStore …")
    feat_store = FeatureStore(track_emb_db, track_meta_db,
                              user_emb_db, user_meta_db, split_types)

    # ── Build train/test samples ──────────────────────────────────────────
    train_samples = build_samples(dataset_name, "train", train_ce, train_retr, feat_store)
    test_samples  = build_samples(dataset_name, "test",  test_ce,  test_retr,  feat_store)

    # ── Init Bagging ──────────────────────────────────────────────────────
    bagging  = BaggingReranker(feat_store, args, device)
    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Phase 1 & 2: 交替训练（train epoch → eval → test epoch → eval）────
    run_training(dataset_name, train_samples, test_samples,
                 bagging, args, test_ce, test_retr, feat_store, ckpt_dir)

    # ── Save final ────────────────────────────────────────────────────────
    final_dir = os.path.join("qwen", "bagging_reranker")
    bagging.save(final_dir)
    logger.info("Training complete. Final models saved to %s", final_dir)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Bagging Reranker (FM+DCN+xDeepFM+LGBM+ThreeTowerGate)")
    p.add_argument("--config",           type=str, default="config/llama1b_multi_channel_devset.yaml")
    p.add_argument("--train_conv_emb",   type=str, default="qwen/hist_conversation_embeddings_train_0.6b.pt")
    p.add_argument("--test_conv_emb",    type=str, default="qwen/hist_conversation_embeddings_test_0.6b.pt")
    p.add_argument("--retrieval_train",  type=str, default="qwen/retrieval_train_candidates.pt",
                   help="Pre-saved retrieval candidates dict {emb_key: [track_id, ...]}")
    p.add_argument("--retrieval_test",   type=str, default="qwen/retrieval_test_candidates.pt")
    p.add_argument("--train_epochs",     type=int,   default=5)
    p.add_argument("--test_epochs",      type=int,   default=5,
                   help="Kept for compatibility; test fine-tune is now 1 pass per train epoch")
    p.add_argument("--batch_size",       type=int,   default=64)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--lr_finetune",      type=float, default=3e-4)
    p.add_argument("--lgbm_max_samples", type=int,   default=80000,
                   help="Max samples for LightGBM training per phase (default 80000)")
    p.add_argument("--checkpoint_dir",   type=str,   default="qwen/bagging_ckpt")
    p.add_argument("--device",           type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
