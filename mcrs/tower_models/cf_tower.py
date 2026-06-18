"""
mcrs/tower_models/cf_tower.py
================================
CF-BPR Two-Tower model.

Maps pre-trained CF-BPR embeddings (128-dim) through a small MLP to a
shared 32-dim space, then does cosine retrieval.

User side:  user_cf_bpr_128 → MLP → user_vec_32
Track side: track_cf_bpr_128 → MLP → track_vec_32

Training: BPR loss (positive > negative margin), L2 reg on embeddings.
Only active for "old users" who appear in the user metadata dataset.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _tower(in_dim: int, hidden_dim: int, out_dim: int,
           dropout: float = 0.2) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class CFTower(nn.Module):
    """Lightweight CF-BPR two-tower: 128-dim → 32-dim for both user and track."""

    CF_DIM  = 128
    HID_DIM = 64
    OUT_DIM = 32

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.user_tower  = _tower(self.CF_DIM, self.HID_DIM, self.OUT_DIM, dropout)
        self.track_tower = _tower(self.CF_DIM, self.HID_DIM, self.OUT_DIM, dropout)

    def forward(
        self,
        user_cf:  torch.Tensor,   # [B, 128]
        pos_cf:   torch.Tensor,   # [B, 128]  positive track
        neg_cf:   torch.Tensor,   # [B, 128]  negative track (BPR)
    ) -> Dict[str, torch.Tensor]:
        u_vec   = F.normalize(self.user_tower(user_cf),  p=2, dim=1)
        p_vec   = F.normalize(self.track_tower(pos_cf),  p=2, dim=1)
        n_vec   = F.normalize(self.track_tower(neg_cf),  p=2, dim=1)

        pos_score = (u_vec * p_vec).sum(dim=1)   # [B]
        neg_score = (u_vec * n_vec).sum(dim=1)   # [B]

        # BPR loss: -log σ(pos - neg)
        bpr_loss  = -F.logsigmoid(pos_score - neg_score).mean()

        # L2 regularisation on output vectors
        l2_loss   = (u_vec.pow(2).sum() + p_vec.pow(2).sum() + n_vec.pow(2).sum()) / user_cf.size(0)
        loss      = bpr_loss + 0.01 * l2_loss

        return {
            "loss":      loss,
            "bpr_loss":  bpr_loss,
            "l2_loss":   l2_loss,
            "u_vec":     u_vec,
            "p_vec":     p_vec,
        }

    def encode_user(self, user_cf: torch.Tensor) -> torch.Tensor:
        """Inference: user CF-BPR → 32-dim vector."""
        with torch.no_grad():
            return F.normalize(self.user_tower(user_cf.float()), p=2, dim=1)

    def encode_track(self, track_cf: torch.Tensor) -> torch.Tensor:
        """Inference: track CF-BPR → 32-dim vector."""
        with torch.no_grad():
            return F.normalize(self.track_tower(track_cf.float()), p=2, dim=1)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CFTower":
        ckpt  = torch.load(path, map_location=device, weights_only=True)
        model = cls()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model.to(device)
