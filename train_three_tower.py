"""
train_three_tower.py  (v5 — CF-BPR Retrieval Three-Tower + 画像/属性特征)
=========================================================================

Model Architecture
------------------
User塔:
  输入: CF-BPR Emb (128d) + user_profile_emb (age_group/gender/country_name → lookup emb, 合计 profile_dim)
  → Linear(128 + profile_dim, 256) → BN → ReLU → Dropout
  → Linear(256, 128)
  output: user_vec [128]

Query塔:
  输入: hist_conversation_emb (1024d)
  → Linear(1024, 512) → BN → ReLU → Dropout
  → Linear(512, 256) → BN → ReLU → Dropout
  → Linear(256, 128)
  output: query_vec [128]

融合层 (Gate Fusion):
  gate = Sigmoid(Linear(concat(user_vec, query_vec), 128))
  fusion_vec = gate ⊙ user_vec + (1 - gate) ⊙ query_vec
  output: fusion_vec [128]

Item塔:
  输入: CF-BPR Emb (128d)
        + attributes-qwen3_embedding_0.6b (1024d → proj 64d)
        + metadata-qwen3_embedding_0.6b   (1024d → proj 64d)
        + item_profile_emb (popularity_bucket/release_year_bucket/duration_bucket → lookup, 合计 item_meta_dim)
  → Linear(128 + 64 + 64 + item_meta_dim, 256) → BN → ReLU → Dropout
  → Linear(256, 128)
  output: item_vec [128]

Score: cosine_similarity(fusion_vec, item_vec) × temperature

Label & Loss
------------
- MOVES_TOWARD_GOAL         → label=1.0, weight=1.5
- DOES_NOT_MOVE_TOWARD_GOAL → label=1.0, weight=1.0
- 10 random negatives per positive, weight=1.0, label=0.0

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
    > train_cf_bpr.log 2>&1 &
"""

import argparse
import logging
import math
import os
import random
from typing import Dict, List, Optional, Set, Tuple

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
CF_BPR_DIM    = 128
ATTR_PROJ_DIM = 64    # attributes-qwen3 1024d → 64d projection
META_PROJ_DIM = 64    # metadata-qwen3  1024d → 64d projection
HIDDEN_DIM    = 256
OUTPUT_DIM    = 128
TEMPERATURE   = 20.0

WEIGHT_MOVES   = 1.0
WEIGHT_NO_MOVE = 1.0
NUM_EASY_NEG   = 10

# ── User profile vocab / emb dims ───────────────────────────────────────────
# age_group: unknown + whatever values are in dataset; we use hash-buckets
AGE_GROUP_DIM    = 8
GENDER_DIM       = 4
COUNTRY_DIM      = 32
USER_PROFILE_DIM = AGE_GROUP_DIM + GENDER_DIM + COUNTRY_DIM   # 44

# ── Item meta vocab / emb dims ────────────────────────────────────────────────
POP_BUCKET_DIM          = 8
RELEASE_YEAR_BUCKET_DIM = 8
DURATION_BUCKET_DIM     = 8
ITEM_META_DIM = POP_BUCKET_DIM + RELEASE_YEAR_BUCKET_DIM + DURATION_BUCKET_DIM  # 24

# Total item input dim = 128 + 64 + 64 + 24 = 280
ITEM_INPUT_DIM = CF_BPR_DIM + ATTR_PROJ_DIM + META_PROJ_DIM + ITEM_META_DIM
# Total user input dim = 128 + 44 = 172
USER_INPUT_DIM = CF_BPR_DIM + USER_PROFILE_DIM


# ─────────────────────────────────────────────────────────────────────────────
#  Feature encoders (discrete → embedding lookup via modular hash)
# ─────────────────────────────────────────────────────────────────────────────

def _hash_bucket(value: str, n_buckets: int) -> int:
    """Simple string → bucket index via hash, range [0, n_buckets)."""
    if not value:
        return 0
    return abs(hash(value)) % n_buckets


def _popularity_bucket(pop) -> int:
    """Bucket popularity (0-100) into 8 bins."""
    try:
        p = float(pop)
    except (TypeError, ValueError):
        return 0
    if p < 0:   return 0
    if p < 10:  return 1
    if p < 20:  return 2
    if p < 35:  return 3
    if p < 50:  return 4
    if p < 65:  return 5
    if p < 80:  return 6
    return 7


def _release_year_bucket(release_date) -> int:
    try:
        year = int(str(release_date)[:4])
    except Exception:
        return 0
    if year < 1970: return 1
    if year < 1980: return 2
    if year < 1990: return 3
    if year < 2000: return 4
    if year < 2005: return 5
    if year < 2010: return 6
    if year < 2015: return 7
    return 7   # 2015+


def _duration_bucket(duration_ms) -> int:
    try:
        d = float(duration_ms)
    except (TypeError, ValueError):
        return 0
    if d < 60000:   return 1
    if d < 120000:  return 2
    if d < 180000:  return 3
    if d < 210000:  return 4
    if d < 240000:  return 5
    if d < 300000:  return 6
    return 7


# ─────────────────────────────────────────────────────────────────────────────
#  Model definition
# ─────────────────────────────────────────────────────────────────────────────

class UserTower(nn.Module):
    """User塔: CF-BPR(128) + profile emb(44) → 256 → BN → ReLU → Dropout → 128"""

    def __init__(self, input_dim: int = USER_INPUT_DIM, hidden: int = HIDDEN_DIM,
                 out: int = OUTPUT_DIM, dropout: float = 0.2):
        super().__init__()
        # Lookup tables for discrete user features
        self.age_emb     = nn.Embedding(32,  AGE_GROUP_DIM)
        self.gender_emb  = nn.Embedding(8,   GENDER_DIM)
        self.country_emb = nn.Embedding(512, COUNTRY_DIM)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out),
        )

    def forward(self,
                cf_bpr:       torch.Tensor,   # [B, 128]
                age_idx:      torch.Tensor,   # [B]
                gender_idx:   torch.Tensor,   # [B]
                country_idx:  torch.Tensor,   # [B]
               ) -> torch.Tensor:
        age_e     = self.age_emb(age_idx)        # [B, 8]
        gender_e  = self.gender_emb(gender_idx)  # [B, 4]
        country_e = self.country_emb(country_idx)# [B, 32]
        x = torch.cat([cf_bpr, age_e, gender_e, country_e], dim=1)
        return self.net(x)


class QueryTower(nn.Module):
    """Query塔: conv_emb(1024) → 512 → BN → ReLU → Dropout → 256 → BN → ReLU → Dropout → 128"""

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
        return self.net(conv_emb)


class ItemTower(nn.Module):
    """Item塔: CF-BPR(128) + attr_proj(64) + meta_proj(64) + item_meta_emb(24) → 256 → BN → ReLU → Dropout → 128"""

    def __init__(self, input_dim: int = ITEM_INPUT_DIM, hidden: int = HIDDEN_DIM,
                 out: int = OUTPUT_DIM, dropout: float = 0.2,
                 attr_in_dim: int = 1024, meta_in_dim: int = 1024):
        super().__init__()
        # Project high-dim embeddings down first
        self.attr_proj = nn.Linear(attr_in_dim, ATTR_PROJ_DIM)
        self.meta_proj = nn.Linear(meta_in_dim, META_PROJ_DIM)

        # Lookup tables for discrete item features
        self.pop_emb      = nn.Embedding(8,   POP_BUCKET_DIM)
        self.year_emb     = nn.Embedding(8,   RELEASE_YEAR_BUCKET_DIM)
        self.duration_emb = nn.Embedding(8,   DURATION_BUCKET_DIM)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out),
        )

    def forward(self,
                cf_bpr:      torch.Tensor,   # [B, 128]
                attr_emb:    torch.Tensor,   # [B, 1024]
                meta_emb:    torch.Tensor,   # [B, 1024]
                pop_idx:     torch.Tensor,   # [B]
                year_idx:    torch.Tensor,   # [B]
                dur_idx:     torch.Tensor,   # [B]
               ) -> torch.Tensor:
        attr_p = self.attr_proj(attr_emb)    # [B, 64]
        meta_p = self.meta_proj(meta_emb)    # [B, 64]
        pop_e  = self.pop_emb(pop_idx)       # [B, 8]
        year_e = self.year_emb(year_idx)     # [B, 8]
        dur_e  = self.duration_emb(dur_idx)  # [B, 8]
        x = torch.cat([cf_bpr, attr_p, meta_p, pop_e, year_e, dur_e], dim=1)
        return self.net(x)


class CFBPRThreeTower(nn.Module):
    """Three-tower retrieval model with gate fusion and rich features."""

    def __init__(
        self,
        temperature: float = TEMPERATURE,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.temperature = temperature
        self.user_tower  = UserTower(USER_INPUT_DIM,  HIDDEN_DIM, OUTPUT_DIM, dropout)
        self.query_tower = QueryTower(CONV_EMB_DIM,   HIDDEN_DIM, OUTPUT_DIM, dropout)
        self.item_tower  = ItemTower(ITEM_INPUT_DIM,  HIDDEN_DIM, OUTPUT_DIM, dropout)
        self.gate_linear = nn.Linear(OUTPUT_DIM * 2, OUTPUT_DIM)

    def encode_fusion(self,
                      cf_bpr_user: torch.Tensor,
                      age_idx:     torch.Tensor,
                      gender_idx:  torch.Tensor,
                      country_idx: torch.Tensor,
                      conv_emb:    torch.Tensor,
                     ) -> torch.Tensor:
        user_vec  = self.user_tower(cf_bpr_user, age_idx, gender_idx, country_idx)
        query_vec = self.query_tower(conv_emb)
        gate      = torch.sigmoid(self.gate_linear(
            torch.cat([user_vec, query_vec], dim=1)
        ))
        fusion = gate * user_vec + (1.0 - gate) * query_vec
        return F.normalize(fusion, p=2, dim=1)

    def encode_item(self,
                    cf_bpr:   torch.Tensor,
                    attr_emb: torch.Tensor,
                    meta_emb: torch.Tensor,
                    pop_idx:  torch.Tensor,
                    year_idx: torch.Tensor,
                    dur_idx:  torch.Tensor,
                   ) -> torch.Tensor:
        return F.normalize(
            self.item_tower(cf_bpr, attr_emb, meta_emb, pop_idx, year_idx, dur_idx),
            p=2, dim=1,
        )

    def forward(self,
                # user side
                cf_bpr_user: torch.Tensor,
                age_idx:     torch.Tensor,
                gender_idx:  torch.Tensor,
                country_idx: torch.Tensor,
                conv_emb:    torch.Tensor,
                # item side
                cf_bpr_item: torch.Tensor,
                attr_emb:    torch.Tensor,
                meta_emb:    torch.Tensor,
                pop_idx:     torch.Tensor,
                year_idx:    torch.Tensor,
                dur_idx:     torch.Tensor,
               ) -> torch.Tensor:
        fusion = self.encode_fusion(cf_bpr_user, age_idx, gender_idx, country_idx, conv_emb)
        item   = self.encode_item(cf_bpr_item, attr_emb, meta_emb, pop_idx, year_idx, dur_idx)
        return (fusion * item).sum(dim=1) * self.temperature


# ─────────────────────────────────────────────────────────────────────────────
#  Embedding / metadata loaders
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


def load_track_data(track_emb_db: str, track_meta_db: str,
                    split_types: List[str]) -> Dict[str, Dict]:
    """Load track CF-BPR + attributes + metadata embeddings + meta fields."""
    # ── embeddings ────────────────────────────────────────────────────────────
    ds_emb = load_dataset(track_emb_db)
    valid  = [s for s in split_types if s in ds_emb.keys()] or list(ds_emb.keys())
    emb_ds = concatenate_datasets([ds_emb[s] for s in valid])

    cf_dim   = CF_BPR_DIM
    attr_dim = 1024
    meta_dim = 1024
    for item in emb_ds:
        v = item.get("cf-bpr")
        if v is not None:
            try:
                t = torch.tensor(v, dtype=torch.float32)
                if t.ndim == 1: cf_dim = t.shape[0]; break
            except Exception: pass

    logger.info("Track CF-BPR dim=%d  attr_dim=%d  meta_dim=%d", cf_dim, attr_dim, meta_dim)

    track_embs: Dict[str, Dict] = {}
    for item in tqdm(emb_ds, desc="Loading track embs", unit="track"):
        tid = item["track_id"]
        track_embs[tid] = {
            "cf-bpr":    _raw_to_tensor(item.get("cf-bpr"),   cf_dim),
            "attr_emb":  _raw_to_tensor(item.get("attributes-qwen3_embedding_0.6b"), attr_dim),
            "meta_emb":  _raw_to_tensor(item.get("metadata-qwen3_embedding_0.6b"),   meta_dim),
        }

    # ── metadata ──────────────────────────────────────────────────────────────
    ds_meta  = load_dataset(track_meta_db)
    valid_m  = [s for s in split_types if s in ds_meta.keys()] or list(ds_meta.keys())
    meta_ds  = concatenate_datasets([ds_meta[s] for s in valid_m])
    for item in tqdm(meta_ds, desc="Loading track meta", unit="track"):
        tid = item["track_id"]
        if tid not in track_embs:
            track_embs[tid] = {
                "cf-bpr":   torch.zeros(cf_dim),
                "attr_emb": torch.zeros(attr_dim),
                "meta_emb": torch.zeros(meta_dim),
            }
        track_embs[tid]["pop_bucket"]  = _popularity_bucket(item.get("popularity"))
        track_embs[tid]["year_bucket"] = _release_year_bucket(item.get("release_date"))
        track_embs[tid]["dur_bucket"]  = _duration_bucket(item.get("duration"))

    # fill missing meta buckets
    for tid, d in track_embs.items():
        d.setdefault("pop_bucket",  0)
        d.setdefault("year_bucket", 0)
        d.setdefault("dur_bucket",  0)

    logger.info("Loaded %d tracks (emb + meta).", len(track_embs))
    return track_embs


def load_user_data(user_emb_db: str, user_meta_db: str,
                   split_types: List[str]) -> Dict[str, Dict]:
    """Load user CF-BPR + profile features."""
    # ── embeddings ────────────────────────────────────────────────────────────
    ds_emb = load_dataset(user_emb_db)
    valid  = [s for s in split_types if s in ds_emb.keys()] or list(ds_emb.keys())
    emb_ds = concatenate_datasets([ds_emb[s] for s in valid])

    cf_dim = CF_BPR_DIM
    for item in emb_ds:
        v = item.get("cf-bpr")
        if v is not None:
            try:
                t = torch.tensor(v, dtype=torch.float32)
                if t.ndim == 1: cf_dim = t.shape[0]; break
            except Exception: pass

    user_data: Dict[str, Dict] = {}
    for item in tqdm(emb_ds, desc="Loading user embs", unit="user"):
        uid = item["user_id"]
        user_data[uid] = {"cf-bpr": _raw_to_tensor(item.get("cf-bpr"), cf_dim)}

    # ── metadata ──────────────────────────────────────────────────────────────
    ds_meta  = load_dataset(user_meta_db)
    valid_m  = [s for s in split_types if s in ds_meta.keys()] or list(ds_meta.keys())
    meta_ds  = concatenate_datasets([ds_meta[s] for s in valid_m])
    for item in tqdm(meta_ds, desc="Loading user meta", unit="user"):
        uid = item["user_id"]
        if uid not in user_data:
            user_data[uid] = {"cf-bpr": torch.zeros(cf_dim)}
        user_data[uid]["age_idx"]     = _hash_bucket(str(item.get("age_group",  "") or ""), 32)
        user_data[uid]["gender_idx"]  = _hash_bucket(str(item.get("gender",     "") or ""), 8)
        user_data[uid]["country_idx"] = _hash_bucket(str(item.get("country_name","") or ""), 512)

    # fill missing profile
    for uid, d in user_data.items():
        d.setdefault("age_idx",     0)
        d.setdefault("gender_idx",  0)
        d.setdefault("country_idx", 0)

    logger.info("Loaded %d users (emb + profile).", len(user_data))
    return user_data


# ─────────────────────────────────────────────────────────────────────────────
#  Sample builder
# ─────────────────────────────────────────────────────────────────────────────

def build_training_samples(
    dataset_name: str,
    split: str,
    conv_emb_store: Dict[str, torch.Tensor],
) -> List[Dict]:
    logger.info("Loading '%s' split='%s' …", dataset_name, split)
    ds = load_dataset(dataset_name, split=split)
    samples: List[Dict] = []
    label_dist = {"1.0_strong": 0, "1.0_weak": 0, "skipped": 0}

    for item in tqdm(ds, desc=f"Building samples [{split}]", unit="session"):
        session_id  = item["session_id"]
        user_id     = item.get("user_id", None)
        convs       = item["conversations"]
        assessments = item.get("goal_progress_assessments", [])
        asmt_map = {int(a["turn_number"]): a.get("goal_progress_assessment") for a in assessments}

        music_turns: Dict[int, str] = {}
        for c in convs:
            if c.get("role") == "music" and c.get("content"):
                music_turns[int(c["turn_number"])] = c["content"]

        for turn_number, track_id in music_turns.items():
            feedback_turn = turn_number + 1
            if feedback_turn not in asmt_map:
                label_dist["skipped"] += 1; continue

            gpa = asmt_map[feedback_turn]
            if gpa == "MOVES_TOWARD_GOAL":
                label, weight = 1.0, WEIGHT_MOVES
                label_dist["1.0_strong"] += 1
            elif gpa == "DOES_NOT_MOVE_TOWARD_GOAL":
                label, weight = 1.0, WEIGHT_NO_MOVE   # 正样本，权重低
                label_dist["1.0_weak"] += 1
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
                "session_id": session_id,
                "user_id":    user_id,
                "track_id":   track_id,
                "label":      label,
                "weight":     weight,
                "conv_emb":   conv_emb,
            })

    logger.info("Built %d samples | strong_pos=%d  weak_pos=%d  skipped=%d",
                len(samples), label_dist["1.0_strong"], label_dist["1.0_weak"],
                label_dist["skipped"])
    return samples


# ─────────────────────────────────────────────────────────────────────────────
#  Batch collation
# ─────────────────────────────────────────────────────────────────────────────

def _get_user_tensors(user_data: Dict, user_id: Optional[str]):
    """Return (cf_bpr, age_idx, gender_idx, country_idx)."""
    d = user_data.get(user_id, {}) if user_id else {}
    cf  = d.get("cf-bpr",      torch.zeros(CF_BPR_DIM))
    age = d.get("age_idx",     0)
    gen = d.get("gender_idx",  0)
    cnt = d.get("country_idx", 0)
    return cf.float(), age, gen, cnt


def _get_item_tensors(track_data: Dict, track_id: str):
    """Return (cf_bpr, attr_emb, meta_emb, pop_idx, year_idx, dur_idx)."""
    d = track_data.get(track_id, {})
    cf   = d.get("cf-bpr",    torch.zeros(CF_BPR_DIM)).float()
    attr = d.get("attr_emb",  torch.zeros(1024)).float()
    meta = d.get("meta_emb",  torch.zeros(1024)).float()
    pop  = d.get("pop_bucket",  0)
    year = d.get("year_bucket", 0)
    dur  = d.get("dur_bucket",  0)
    return cf, attr, meta, pop, year, dur


def collate_batch(
    samples:    List[Dict],
    all_tids:   List[str],
    track_data: Dict,
    user_data:  Dict,
    device:     str,
    num_easy_neg: int = NUM_EASY_NEG,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    rows = {k: [] for k in [
        "cf_bpr_user", "age_idx", "gender_idx", "country_idx", "conv_emb",
        "cf_bpr_item", "attr_emb", "meta_emb", "pop_idx", "year_idx", "dur_idx",
    ]}
    labels, weights = [], []

    def _add(s: Dict, tid: str, lbl: float, wt: float):
        u_cf, age, gen, cnt = _get_user_tensors(user_data, s["user_id"])
        i_cf, attr, meta, pop, yr, dur = _get_item_tensors(track_data, tid)
        rows["cf_bpr_user"].append(u_cf)
        rows["age_idx"].append(age)
        rows["gender_idx"].append(gen)
        rows["country_idx"].append(cnt)
        rows["conv_emb"].append(s["conv_emb"])
        rows["cf_bpr_item"].append(i_cf)
        rows["attr_emb"].append(attr)
        rows["meta_emb"].append(meta)
        rows["pop_idx"].append(pop)
        rows["year_idx"].append(yr)
        rows["dur_idx"].append(dur)
        labels.append(lbl)
        weights.append(wt)

    for s in samples:
        _add(s, s["track_id"], s["label"], s["weight"])
        for _ in range(num_easy_neg):
            _add(s, random.choice(all_tids), 0.0, 1.0)

    def _t(lst, dtype=torch.float32):
        return torch.stack(lst).to(device) if isinstance(lst[0], torch.Tensor) \
               else torch.tensor(lst, dtype=dtype, device=device)

    inputs = {
        "cf_bpr_user": _t(rows["cf_bpr_user"]),
        "age_idx":     _t(rows["age_idx"],     torch.long),
        "gender_idx":  _t(rows["gender_idx"],  torch.long),
        "country_idx": _t(rows["country_idx"], torch.long),
        "conv_emb":    _t(rows["conv_emb"]),
        "cf_bpr_item": _t(rows["cf_bpr_item"]),
        "attr_emb":    _t(rows["attr_emb"]),
        "meta_emb":    _t(rows["meta_emb"]),
        "pop_idx":     _t(rows["pop_idx"],      torch.long),
        "year_idx":    _t(rows["year_idx"],     torch.long),
        "dur_idx":     _t(rows["dur_idx"],      torch.long),
    }
    return inputs, _t(labels), _t(weights)


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_phase(phase_name, samples, model, optimizer,
              track_data, user_data, all_tids, args, epochs, device, ckpt_dir):
    loss_fn    = nn.BCEWithLogitsLoss(reduction="none")
    patience   = args.early_stop_patience
    min_delta  = args.early_stop_min_delta
    best_loss  = float("inf")
    no_improve = 0
    best_ckpt  = os.path.join(ckpt_dir, f"model_{phase_name}_best.pt")

    logger.info("[%s] %d samples, max_epochs=%d, batch=%d, patience=%d",
                phase_name, len(samples), epochs, args.batch_size, patience)
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
            inputs, label_t, weight_t = collate_batch(
                batch, all_tids, track_data, user_data, device)
            scores = model(**inputs)
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
        logger.info("[%s] Epoch %d avg_loss=%.4f", phase_name, epoch, avg)
        torch.save({"model_state_dict": model.state_dict()},
                   os.path.join(ckpt_dir, f"model_{phase_name}_epoch{epoch}.pt"))

        # ── Early stopping ──
        if avg < best_loss - min_delta:
            best_loss  = avg
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict()}, best_ckpt)
            logger.info("[%s] ★ New best loss=%.4f, saved to %s",
                        phase_name, best_loss, best_ckpt)
        else:
            no_improve += 1
            logger.info("[%s] No improvement %d/%d (best=%.4f)",
                        phase_name, no_improve, patience, best_loss)
            if no_improve >= patience:
                logger.info("[%s] Early stopping at epoch %d.", phase_name, epoch)
                break

    # 训练结束后恢复最优权重
    if os.path.exists(best_ckpt):
        logger.info("[%s] Restoring best model from %s", phase_name, best_ckpt)
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    return best_loss


# ─────────────────────────────────────────────────────────────────────────────
#  Item index builder
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_item_index(model: CFBPRThreeTower, track_data: Dict,
                     save_dir: str, device: str = "cpu", bs: int = 512):
    import json
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    track_ids = sorted(track_data.keys())
    vecs = []
    for start in tqdm(range(0, len(track_ids), bs), desc="Building item index"):
        batch_ids = track_ids[start: start + bs]
        cf   = torch.stack([track_data[t]["cf-bpr"]   for t in batch_ids]).to(device)
        attr = torch.stack([track_data[t]["attr_emb"] for t in batch_ids]).to(device)
        meta = torch.stack([track_data[t]["meta_emb"] for t in batch_ids]).to(device)
        pop  = torch.tensor([track_data[t]["pop_bucket"]  for t in batch_ids], dtype=torch.long, device=device)
        year = torch.tensor([track_data[t]["year_bucket"] for t in batch_ids], dtype=torch.long, device=device)
        dur  = torch.tensor([track_data[t]["dur_bucket"]  for t in batch_ids], dtype=torch.long, device=device)
        vecs.append(model.encode_item(cf, attr, meta, pop, year, dur).cpu())
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
    cache_dir    = config.get("cache_dir", "./cache")
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

    logger.info("Loading train conv_emb from %s …", args.train_conv_emb)
    train_ce = torch.load(args.train_conv_emb, map_location="cpu", weights_only=True)
    logger.info("  train: %d entries.", len(train_ce))
    logger.info("Loading test conv_emb from %s …", args.test_conv_emb)
    test_ce  = torch.load(args.test_conv_emb,  map_location="cpu", weights_only=True)
    logger.info("  test:  %d entries.", len(test_ce))

    train_samples = build_training_samples(dataset_name, "train", train_ce)
    test_samples  = build_training_samples(dataset_name, "test",  test_ce)
    if not train_samples:
        logger.error("No training samples."); return

    logger.info("Loading track data …")
    track_data = load_track_data(track_emb_db, track_meta_db, split_types)
    logger.info("Loading user data …")
    user_data  = load_user_data(user_emb_db, user_meta_db, split_types)

    all_tids = list(track_data.keys())

    model     = CFBPRThreeTower(temperature=TEMPERATURE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ckpt_dir  = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    # Phase 1 — train
    run_phase("train", train_samples, model, optimizer,
              track_data, user_data, all_tids, args,
              args.train_epochs, device, ckpt_dir)

    # Phase 2 — test fine-tune
    if test_samples and args.test_epochs > 0:
        logger.info("Phase 2: fine-tune on test split (lr=%.2e)", args.lr_finetune)
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr_finetune
        run_phase("test_ft", test_samples, model, optimizer,
                  track_data, user_data, all_tids, args,
                  args.test_epochs, device, ckpt_dir)

    final_dir = os.path.join("qwen", "cf_bpr_retrieval")
    os.makedirs(final_dir, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()},
               os.path.join(final_dir, "model.pt"))
    logger.info("Final model → %s", os.path.join(final_dir, "model.pt"))

    build_item_index(model, track_data,
                     os.path.join(final_dir, "item_index"), device=device)
    logger.info("Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train CF-BPR Three-Tower (v5 rich features)")
    p.add_argument("--config",          type=str,
                   default="config/llama1b_multi_channel_devset.yaml")
    p.add_argument("--train_conv_emb",  type=str,
                   default="qwen/hist_conversation_embeddings_train_0.6b.pt")
    p.add_argument("--test_conv_emb",   type=str,
                   default="qwen/hist_conversation_embeddings_test_0.6b.pt")
    p.add_argument("--train_epochs",    type=int,  default=5)
    p.add_argument("--test_epochs",     type=int,  default=2)
    p.add_argument("--batch_size",      type=int,  default=64)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--lr_finetune",     type=float, default=3e-4)
    p.add_argument("--checkpoint_dir",  type=str,  default="qwen/cf_bpr_retrieval_ckpt")
    p.add_argument("--save_every",           type=int,  default=500)
    p.add_argument("--early_stop_patience",  type=int,  default=5,
                   help="连续 N 个 epoch loss 无改善则提前停止")
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                   help="认为改善的最小 loss 下降量")
    p.add_argument("--device",              type=str,  default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
