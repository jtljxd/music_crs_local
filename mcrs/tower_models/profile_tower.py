"""
mcrs/tower_models/profile_tower.py
====================================
User Profile Tower — demographic-only two-tower retrieval model.

User side (ProfileEncoder):
    input:  age_bucket_emb_8 + country_code_emb_16 + gender_emb_2
            + preferred_language_emb_4 + preferred_musical_culture_emb_32
            + session_year_emb_8 + session_month_emb_4 + weekday_emb_2
    → concat (76-dim) → MLP → profile_vec_32

Track side (TrackEncoderProfile):
    input:  tag_avg_32 + artist_avg_32 + album_avg_32
            + popularity_bucket_8 + release_year_bucket_8 + duration_bucket_8
    → concat (112-dim) → MLP → track_profile_vec_32

Training: in-batch negatives cross-entropy, L2 reg, early stopping.
"""

from __future__ import annotations

import math
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden_dims, out_dim: int, dropout: float = 0.3) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ProfileEncoder(nn.Module):
    """User profile encoder → 32-dim profile vector."""

    # Dimension breakdown
    AGE_DIM      = 8
    COUNTRY_DIM  = 16
    GENDER_DIM   = 2
    LANG_DIM     = 4
    CULTURE_DIM  = 32
    YEAR_DIM     = 8
    MONTH_DIM    = 4
    WEEKDAY_DIM  = 2
    OUT_DIM      = 32

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        in_dim = (self.AGE_DIM + self.COUNTRY_DIM + self.GENDER_DIM
                  + self.LANG_DIM + self.CULTURE_DIM
                  + self.YEAR_DIM + self.MONTH_DIM + self.WEEKDAY_DIM)
        self.net = _mlp(in_dim, [64], self.OUT_DIM, dropout)

    def forward(
        self,
        age_emb:     torch.Tensor,   # [B, 8]
        country_emb: torch.Tensor,   # [B, 16]
        gender_emb:  torch.Tensor,   # [B, 2]
        lang_emb:    torch.Tensor,   # [B, 4]
        culture_emb: torch.Tensor,   # [B, 32]
        year_emb:    torch.Tensor,   # [B, 8]
        month_emb:   torch.Tensor,   # [B, 4]
        weekday_emb: torch.Tensor,   # [B, 2]
    ) -> torch.Tensor:               # [B, 32]
        x = torch.cat([age_emb, country_emb, gender_emb, lang_emb,
                        culture_emb, year_emb, month_emb, weekday_emb], dim=1)
        return F.normalize(self.net(x), p=2, dim=1)


class TrackEncoderProfile(nn.Module):
    """Track profile encoder → 32-dim track profile vector."""

    TAG_DIM     = 32
    ARTIST_DIM  = 32
    ALBUM_DIM   = 32
    POP_DIM     = 8
    YEAR_DIM    = 8
    DUR_DIM     = 8
    OUT_DIM     = 32

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        in_dim = (self.TAG_DIM + self.ARTIST_DIM + self.ALBUM_DIM
                  + self.POP_DIM + self.YEAR_DIM + self.DUR_DIM)
        self.net = _mlp(in_dim, [64], self.OUT_DIM, dropout)

    def forward(
        self,
        tag_avg:    torch.Tensor,   # [B, 32]
        artist_avg: torch.Tensor,   # [B, 32]
        album_avg:  torch.Tensor,   # [B, 32]
        pop_bucket: torch.Tensor,   # [B, 8]
        year_bucket:torch.Tensor,   # [B, 8]
        dur_bucket: torch.Tensor,   # [B, 8]
    ) -> torch.Tensor:              # [B, 32]
        x = torch.cat([tag_avg, artist_avg, album_avg,
                        pop_bucket, year_bucket, dur_bucket], dim=1)
        return F.normalize(self.net(x), p=2, dim=1)


class ProfileTower(nn.Module):
    """Two-tower profile model."""

    def __init__(self, dropout: float = 0.3, temperature: float = 0.05):
        super().__init__()
        self.profile_encoder = ProfileEncoder(dropout)
        self.track_encoder   = TrackEncoderProfile(dropout)
        self.temperature     = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def forward(
        self,
        # User profile side
        age_emb: torch.Tensor, country_emb: torch.Tensor, gender_emb: torch.Tensor,
        lang_emb: torch.Tensor, culture_emb: torch.Tensor,
        year_emb: torch.Tensor, month_emb: torch.Tensor, weekday_emb: torch.Tensor,
        # Track side
        tag_avg: torch.Tensor, artist_avg: torch.Tensor, album_avg: torch.Tensor,
        pop_bucket: torch.Tensor, year_bucket: torch.Tensor, dur_bucket: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        profile_vec = self.profile_encoder(
            age_emb, country_emb, gender_emb, lang_emb, culture_emb,
            year_emb, month_emb, weekday_emb,
        )
        track_vec = self.track_encoder(
            tag_avg, artist_avg, album_avg, pop_bucket, year_bucket, dur_bucket
        )
        temp   = self.temperature.exp().clamp(max=100)
        logits = torch.matmul(profile_vec, track_vec.T) * temp
        labels = torch.arange(logits.size(0), device=logits.device)
        loss   = F.cross_entropy(logits, labels)
        return {"loss": loss, "profile_vec": profile_vec, "track_vec": track_vec}

    def encode_profile(
        self,
        age_emb: torch.Tensor, country_emb: torch.Tensor, gender_emb: torch.Tensor,
        lang_emb: torch.Tensor, culture_emb: torch.Tensor,
        year_emb: torch.Tensor, month_emb: torch.Tensor, weekday_emb: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.profile_encoder(
                age_emb, country_emb, gender_emb, lang_emb, culture_emb,
                year_emb, month_emb, weekday_emb,
            )

    def encode_query(self, ctx) -> torch.Tensor:
        """Inference from RetrievalContext (used by multi_channel_v2).
        User metadata not available in ctx at inference time, use zeros.
        """
        dev = next(self.parameters()).device
        def _z(*dims): return torch.zeros(1, *dims, device=dev)
        age_emb     = _z(8)
        country_emb = _z(16)
        gender_emb  = _z(2)
        lang_emb    = _z(4)
        culture_emb = _z(32)
        year_emb    = _z(8)
        month_emb   = _z(4)
        weekday_emb = _z(2)
        return self.encode_profile(
            age_emb, country_emb, gender_emb,
            lang_emb, culture_emb,
            year_emb, month_emb, weekday_emb,
        ).squeeze(0)

    def encode_track(
        self,
        tag_avg: torch.Tensor, artist_avg: torch.Tensor, album_avg: torch.Tensor,
        pop_bucket: torch.Tensor, year_bucket: torch.Tensor, dur_bucket: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.track_encoder(
                tag_avg, artist_avg, album_avg, pop_bucket, year_bucket, dur_bucket
            )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ProfileTower":
        ckpt  = torch.load(path, map_location=device, weights_only=True)
        model = cls()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model.to(device)
