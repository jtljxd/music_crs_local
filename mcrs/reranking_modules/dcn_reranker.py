"""
mcrs/reranking_modules/dcn_reranker.py
=======================================
DCN-V2 Reranker for music CRS.

Feature towers (each → 128-dim):
  User side:
    user_profile  : age_log_bucket + country16 + gender2 + language4 + culture32
                    → MLP(86 → 128)
    user_cf       : raw CF-BPR 128-dim (passed through)
    conv_goal     : category8 + listener_goal_emb1024 + specificity4
                    → MLP(1036 → 128)
    query         : current-turn query emb 1024 → MLP(1024 → 128)

  Track side:
    track_semantic: CLAP512 + SigLIP768 + attr1024 + lyrics1024 + meta1024
                    → MLP(4352 → 128)
    track_context : ISRC32 + tag32 + artist32 + album32 + log_pop1 + dur_bucket8
                    → MLP(137 → 128)   # see NOTE below
    track_cf      : raw CF-BPR 128-dim (passed through)

NOTE on track_context dims:
  ISRC32 + tag32 + artist32 + album32 = 128
  log(pop) scalar = 1  →  total scalar features alongside buckets
  duration_bucket8 = 8
  → concat dim = 128 + 1 + 8 = 137   (the spec says 200→128, we follow exact dims)

All 7 towers concatenated → 896-dim input to DCN-V2.

DCN-V2 architecture (3 cross layers + 3 deep layers in parallel):
  Cross branch: x0 ⊙ (W_i x_i + b_i) + x_i   (bilinear cross)
  Deep branch:  3-layer MLP with ReLU + LayerNorm + Dropout
  Merge: concat(cross_out, deep_out) → Linear(896+896, 256) → Linear(256, 1)
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden_dims, out_dim: int,
         dropout: float = 0.2) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _bucket_emb(val, n_buckets: int, lo: float, hi: float) -> torch.Tensor:
    """Soft one-hot bucket embedding (1-D, no batch)."""
    t = torch.zeros(n_buckets)
    if val is None:
        return t
    v = float(val)
    idx = int((v - lo) / (hi - lo + 1e-9) * n_buckets)
    idx = max(0, min(n_buckets - 1, idx))
    t[idx] = 1.0
    return t


# ── Feature towers ────────────────────────────────────────────────────────────

class UserProfileTower(nn.Module):
    """age_log_bucket(16) + country(16) + gender(2) + language(4) + culture(32)
    + log_age_scalar(1) + listen_count_bucket(8) + session_count_bucket(7)
    Total raw = 86  → MLP(86 → 128)
    """
    IN_DIM  = 86
    OUT_DIM = 128

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.net = _mlp(self.IN_DIM, [256, 192], self.OUT_DIM, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, 86]
        return F.normalize(self.net(x), p=2, dim=1)


class UserCFTower(nn.Module):
    """Pass-through for raw 128-dim CF-BPR user embedding."""
    OUT_DIM = 128

    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, 128]
        return F.normalize(self.proj(x), p=2, dim=1)


class ConvGoalTower(nn.Module):
    """category(8) + listener_goal_emb(1024) + specificity(4)
    → MLP(1036 → 128)
    """
    IN_DIM  = 1036
    OUT_DIM = 128

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.net = _mlp(self.IN_DIM, [512, 256], self.OUT_DIM, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, 1036]
        return F.normalize(self.net(x), p=2, dim=1)


class QueryTower(nn.Module):
    """Current-turn query embedding 1024 → MLP(1024 → 128)."""
    IN_DIM  = 1024
    OUT_DIM = 128

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.net = _mlp(self.IN_DIM, [512, 256], self.OUT_DIM, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, 1024]
        return F.normalize(self.net(x), p=2, dim=1)


class TrackSemanticTower(nn.Module):
    """CLAP(512) + SigLIP(768) + attr(1024) + lyrics(1024) + meta(1024)
    → MLP(4352 → 128)
    """
    IN_DIM  = 4352
    OUT_DIM = 128

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.net = _mlp(self.IN_DIM, [1024, 512, 256], self.OUT_DIM, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, 4352]
        return F.normalize(self.net(x), p=2, dim=1)


class TrackContextTower(nn.Module):
    """ISRC(32) + tag(32) + artist(32) + album(32) + log_pop(1) + dur_bucket(8)
    = 137 → MLP(137 → 128)
    """
    IN_DIM  = 137
    OUT_DIM = 128

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.net = _mlp(self.IN_DIM, [256, 192], self.OUT_DIM, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, 137]
        return F.normalize(self.net(x), p=2, dim=1)


class TrackCFTower(nn.Module):
    """Pass-through for raw 128-dim CF-BPR track embedding."""
    OUT_DIM = 128

    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, 128]
        return F.normalize(self.proj(x), p=2, dim=1)


# ── DCN-V2 ────────────────────────────────────────────────────────────────────

class CrossLayer(nn.Module):
    """One DCN-V2 cross layer (bilinear interaction):
        x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l
    """
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=True)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * self.W(xl) + xl


class DCNV2(nn.Module):
    """DCN-V2 with parallel cross + deep stack.

    Args:
        in_dim      : input feature dimension (sum of all tower outputs)
        cross_layers: number of cross layers
        deep_dims   : hidden sizes for the deep MLP branch
        dropout     : dropout rate in deep branch
    """
    def __init__(
        self,
        in_dim:       int   = 896,
        cross_layers: int   = 3,
        deep_dims           = (512, 256, 128),
        dropout:      float = 0.2,
    ):
        super().__init__()
        # Cross branch
        self.cross = nn.ModuleList([CrossLayer(in_dim) for _ in range(cross_layers)])

        # Deep branch
        deep_layers = []
        prev = in_dim
        for h in deep_dims:
            deep_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.deep = nn.Sequential(*deep_layers)
        self.deep_out_dim = prev

        # Final projection: concat(cross_out, deep_out) → MLP(128→32→1)
        self.output = nn.Sequential(
            nn.Linear(in_dim + self.deep_out_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, in_dim] → [B, 1]
        # Cross branch
        x0 = x
        xc = x
        for layer in self.cross:
            xc = layer(x0, xc)              # [B, in_dim]

        # Deep branch
        xd = self.deep(x)                   # [B, deep_out_dim]

        # Merge and score
        merged = torch.cat([xc, xd], dim=1)  # [B, in_dim + deep_out_dim]
        return self.output(merged)            # [B, 1]


# ── Full reranker model ───────────────────────────────────────────────────────

TOWER_DIM = 128
N_TOWERS  = 7
CONCAT_DIM = TOWER_DIM * N_TOWERS  # 896


class DCNReranker(nn.Module):
    """Full DCN-V2 reranker.

    Input: raw feature dicts (see forward signature).
    Output: scalar relevance score per (user, track) pair.
    """

    def __init__(
        self,
        cross_layers: int   = 3,
        deep_dims           = (512, 256, 128),
        dropout:      float = 0.2,
    ):
        super().__init__()
        # Feature towers
        self.user_profile_tower  = UserProfileTower(dropout)
        self.user_cf_tower       = UserCFTower()
        self.conv_goal_tower     = ConvGoalTower(dropout)
        self.query_tower         = QueryTower(dropout)
        self.track_semantic_tower = TrackSemanticTower(dropout)
        self.track_context_tower  = TrackContextTower(dropout)
        self.track_cf_tower       = TrackCFTower()

        # DCN-V2
        self.dcn = DCNV2(
            in_dim=CONCAT_DIM,
            cross_layers=cross_layers,
            deep_dims=deep_dims,
            dropout=dropout,
        )

    def encode(
        self,
        user_profile:   torch.Tensor,   # [B, 86]
        user_cf:        torch.Tensor,   # [B, 128]
        conv_goal:      torch.Tensor,   # [B, 1036]
        query_emb:      torch.Tensor,   # [B, 1024]
        track_semantic: torch.Tensor,   # [B, 4352]
        track_context:  torch.Tensor,   # [B, 137]
        track_cf:       torch.Tensor,   # [B, 128]
    ) -> torch.Tensor:                  # [B, 1]  raw score (logit)
        """Encode all features and return raw relevance logit."""
        u_prof  = self.user_profile_tower(user_profile.float())
        u_cf    = self.user_cf_tower(user_cf.float())
        u_goal  = self.conv_goal_tower(conv_goal.float())
        u_query = self.query_tower(query_emb.float())
        t_sem   = self.track_semantic_tower(track_semantic.float())
        t_ctx   = self.track_context_tower(track_context.float())
        t_cf    = self.track_cf_tower(track_cf.float())
        feat = torch.cat([u_prof, u_cf, u_goal, u_query,
                          t_sem, t_ctx, t_cf], dim=1)   # [B, 896]
        return self.dcn(feat)                             # [B, 1]

    def forward(
        self,
        # ── User features (shared across all tracks in batch) ─────────────
        user_profile:   torch.Tensor,   # [B, 86]
        user_cf:        torch.Tensor,   # [B, 128]
        conv_goal:      torch.Tensor,   # [B, 1036]
        query_emb:      torch.Tensor,   # [B, 1024]
        # ── Track features (positive sample) ─────────────────────────────
        track_semantic: torch.Tensor,   # [B, 4352]
        track_context:  torch.Tensor,   # [B, 137]
        track_cf:       torch.Tensor,   # [B, 128]
    ) -> dict:
        """Training forward with in-batch softmax loss.

        Each sample (query_i, track_i) is a positive pair.
        The B-1 other tracks in the batch serve as negatives.

        Computes:
          score_matrix[i, j] = score(user_i, track_j)   shape [B, B]
          loss = mean CrossEntropy(score_matrix, diag_labels)
        Returns dict with 'loss' and 'scores'.
        """
        B = user_profile.size(0)

        # ── Encode user context (same for all tracks of this query) ───────
        u_prof  = self.user_profile_tower(user_profile.float())   # [B, 128]
        u_cf    = self.user_cf_tower(user_cf.float())
        u_goal  = self.conv_goal_tower(conv_goal.float())
        u_query = self.query_tower(query_emb.float())
        user_feat = torch.cat([u_prof, u_cf, u_goal, u_query], dim=1)  # [B, 512]

        # ── Encode all tracks in batch ────────────────────────────────────
        t_sem = self.track_semantic_tower(track_semantic.float())   # [B, 128]
        t_ctx = self.track_context_tower(track_context.float())
        t_cf  = self.track_cf_tower(track_cf.float())
        track_feat = torch.cat([t_sem, t_ctx, t_cf], dim=1)         # [B, 384]

        # ── Build [B, B] score matrix via in-batch pairing ────────────────
        # For user i paired with track j:
        #   feat_ij = concat(user_feat[i], track_feat[j])  [896]
        user_exp  = user_feat.unsqueeze(1).expand(B, B, -1)   # [B, B, 512]
        track_exp = track_feat.unsqueeze(0).expand(B, B, -1)  # [B, B, 384]
        pair_feat = torch.cat([user_exp, track_exp], dim=2)   # [B, B, 896]
        pair_feat_flat = pair_feat.reshape(B * B, 896)        # [B*B, 896]

        scores_flat = self.dcn(pair_feat_flat)                 # [B*B, 1]
        score_matrix = scores_flat.reshape(B, B)               # [B, B]

        # ── In-batch softmax cross-entropy ───────────────────────────────
        # label[i] = i  (diagonal = positive)
        labels = torch.arange(B, device=score_matrix.device)
        loss = F.cross_entropy(score_matrix, labels)

        # Softmax probabilities (for monitoring)
        probs = torch.softmax(score_matrix, dim=1)             # [B, B]

        return {
            "loss":         loss,
            "score_matrix": score_matrix,
            "probs":        probs,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state_dict": self.state_dict()}, path)
        
    @classmethod
    def load(cls, path: str, device: str = "cpu",
             cross_layers: int = 3,
             deep_dims = (512, 256, 128),
             dropout: float = 0.2) -> "DCNReranker":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = cls(cross_layers=cross_layers, deep_dims=deep_dims, dropout=dropout)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model.to(device)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Feature builder helpers (for inference/training) ─────────────────────────

def build_user_profile_vec(user_meta: dict) -> torch.Tensor:
    """Build [86] user profile feature vector from metadata dict."""
    # age: log-bucket into 16 bins (ages 0-100, log scale)
    age = user_meta.get("age")
    if age is not None:
        log_age = math.log1p(float(age))
        age_bucket = _bucket_emb(log_age, 16, 0, math.log1p(100))
    else:
        age_bucket = torch.zeros(16)

    country_emb = _bucket_emb(user_meta.get("country_code_hash"), 16, 0, 1)

    gender_emb = torch.zeros(2)
    g = user_meta.get("gender", "")
    if g == "male":   gender_emb[0] = 1.0
    elif g == "female": gender_emb[1] = 1.0

    lang_emb    = _bucket_emb(user_meta.get("preferred_language_hash"), 4, 0, 1)
    culture_emb = _bucket_emb(user_meta.get("preferred_musical_culture_hash"), 32, 0, 1)

    listen_count = _bucket_emb(
        math.log1p(float(user_meta.get("listen_count", 0) or 0)), 8, 0, math.log1p(10000)
    )
    session_count = _bucket_emb(
        math.log1p(float(user_meta.get("session_count", 0) or 0)), 8, 0, math.log1p(1000)
    )

    return torch.cat([age_bucket, country_emb, gender_emb,
                      lang_emb, culture_emb,
                      listen_count, session_count])  # 16+16+2+4+32+8+8 = 86
