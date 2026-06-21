"""
mcrs/tower_models/intent_tower.py
===================================
Query + Goal Intent Tower — lightweight two-tower model for retrieval.

User side (IntentEncoder):
    input:  [query_emb_1024, goal_emb_1024, category_emb_8,
             specificity_emb_16, session_date_emb_14, user_profile_emb_62]
    → concat → MLP → intent_vec_128

Track side (TrackEncoder):
    input:  [metadata_emb_1024, lyrics_emb_1024, attributes_emb_1024,
             audio_emb_512, image_emb_1152, cf_bpr_128,
             popularity_bucket_8, release_year_bucket_8, duration_bucket_8]
    → concat → MLP → track_vec_128

Training: in-batch negatives BPR loss, dropout + L2 reg, early stopping.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helper ────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden_dims, out_dim: int, dropout: float = 0.3) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ── User / Query encoder ──────────────────────────────────────────────────────

class IntentEncoder(nn.Module):
    """Encode per-turn user context into a 128-dim intent vector."""

    # Input dimension breakdown
    QUERY_DIM      = 1024
    GOAL_DIM       = 1024
    CATEGORY_DIM   = 8
    SPEC_DIM       = 16
    DATE_DIM       = 14
    PROFILE_DIM    = 62
    OUT_DIM        = 128

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        in_dim = (self.QUERY_DIM + self.GOAL_DIM + self.CATEGORY_DIM
                  + self.SPEC_DIM + self.DATE_DIM + self.PROFILE_DIM)
        self.net = _mlp(in_dim, [512, 256], self.OUT_DIM, dropout)

    def forward(
        self,
        query_emb:    torch.Tensor,   # [B, 1024]
        goal_emb:     torch.Tensor,   # [B, 1024]
        category_emb: torch.Tensor,   # [B, 8]
        spec_emb:     torch.Tensor,   # [B, 16]
        date_emb:     torch.Tensor,   # [B, 14]
        profile_emb:  torch.Tensor,   # [B, 62]
    ) -> torch.Tensor:                # [B, 128]
        x = torch.cat([query_emb, goal_emb, category_emb,
                        spec_emb, date_emb, profile_emb], dim=1)
        return F.normalize(self.net(x), p=2, dim=1)


# ── Track encoder ─────────────────────────────────────────────────────────────

class TrackEncoderIntent(nn.Module):
    """Encode track multi-modal features into a 128-dim track vector."""

    META_DIM    = 1024
    LYRICS_DIM  = 1024
    ATTR_DIM    = 1024
    AUDIO_DIM   = 512
    IMAGE_DIM   = 768
    CF_DIM      = 128
    BUCKET_DIM  = 8    # each of: popularity, release_year, duration
    OUT_DIM     = 128

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        in_dim = (self.META_DIM + self.LYRICS_DIM + self.ATTR_DIM
                  + self.AUDIO_DIM + self.IMAGE_DIM + self.CF_DIM
                  + self.BUCKET_DIM * 3)
        self.net = _mlp(in_dim, [1024, 512, 256], self.OUT_DIM, dropout)

    def forward(
        self,
        metadata_emb:  torch.Tensor,   # [B, 1024]
        lyrics_emb:    torch.Tensor,   # [B, 1024]
        attr_emb:      torch.Tensor,   # [B, 1024]
        audio_emb:     torch.Tensor,   # [B, 512]
        image_emb:     torch.Tensor,   # [B, 1152]
        cf_bpr:        torch.Tensor,   # [B, 128]
        pop_bucket:    torch.Tensor,   # [B, 8]
        year_bucket:   torch.Tensor,   # [B, 8]
        dur_bucket:    torch.Tensor,   # [B, 8]
    ) -> torch.Tensor:                 # [B, 128]
        x = torch.cat([metadata_emb, lyrics_emb, attr_emb,
                        audio_emb, image_emb, cf_bpr,
                        pop_bucket, year_bucket, dur_bucket], dim=1)
        return F.normalize(self.net(x), p=2, dim=1)


# ── Full model ─────────────────────────────────────────────────────────────────

class IntentTower(nn.Module):
    """Two-tower intent model for retrieval training."""

    def __init__(self, dropout: float = 0.3, temperature: float = 0.05):
        super().__init__()
        self.intent_encoder = IntentEncoder(dropout)
        self.track_encoder  = TrackEncoderIntent(dropout)
        self.temperature    = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def forward(
        self,
        # User side
        query_emb:    torch.Tensor,
        goal_emb:     torch.Tensor,
        category_emb: torch.Tensor,
        spec_emb:     torch.Tensor,
        date_emb:     torch.Tensor,
        profile_emb:  torch.Tensor,
        # Track side (positive)
        metadata_emb:  torch.Tensor,
        lyrics_emb:    torch.Tensor,
        attr_emb:      torch.Tensor,
        audio_emb:     torch.Tensor,
        image_emb:     torch.Tensor,
        cf_bpr:        torch.Tensor,
        pop_bucket:    torch.Tensor,
        year_bucket:   torch.Tensor,
        dur_bucket:    torch.Tensor,
        # Negatives (optional): each [B, K, dim]
        neg_meta:    Optional[torch.Tensor] = None,
        neg_lyrics:  Optional[torch.Tensor] = None,
        neg_attr:    Optional[torch.Tensor] = None,
        neg_audio:   Optional[torch.Tensor] = None,
        neg_image:   Optional[torch.Tensor] = None,
        neg_cf:      Optional[torch.Tensor] = None,
        neg_pop:     Optional[torch.Tensor] = None,
        neg_year:    Optional[torch.Tensor] = None,
        neg_dur:     Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return loss dict for training.

        When neg_* tensors are provided (shape [B, K, dim]), uses
        1-pos + K-neg cross-entropy (label=0 for all samples).
        Otherwise falls back to in-batch NT-Xent.
        """
        intent_vec = self.intent_encoder(
            query_emb, goal_emb, category_emb, spec_emb, date_emb, profile_emb
        )
        track_vec = self.track_encoder(
            metadata_emb, lyrics_emb, attr_emb, audio_emb, image_emb,
            cf_bpr, pop_bucket, year_bucket, dur_bucket
        )
        temp = self.temperature.exp().clamp(max=100)
        if neg_meta is not None:
            # Encode negatives: flatten [B, K, dim] -> [B*K, dim] -> encode -> [B, K, 128]
            B, K = neg_meta.shape[:2]
            def _flat(t): return t.reshape(B * K, -1)
            neg_enc = self.track_encoder(
                _flat(neg_meta),   _flat(neg_lyrics), _flat(neg_attr),
                _flat(neg_audio),  _flat(neg_image),  _flat(neg_cf),
                _flat(neg_pop),    _flat(neg_year),   _flat(neg_dur),
            )  # [B*K, 128]
            neg_vecs = neg_enc.reshape(B, K, -1)  # [B, K, 128]
            # pos score: [B, 1];  neg scores: [B, K]  ->  logits: [B, 1+K]
            pos_score = (intent_vec * track_vec).sum(dim=1, keepdim=True) * temp
            neg_score = torch.bmm(neg_vecs, intent_vec.unsqueeze(2)).squeeze(2) * temp
            logits = torch.cat([pos_score, neg_score], dim=1)  # [B, 1+K]
            labels = torch.zeros(B, dtype=torch.long, device=logits.device)
        else:
            # In-batch NT-Xent fallback
            logits = torch.matmul(intent_vec, track_vec.T) * temp  # [B, B]
            labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "intent_vec": intent_vec, "track_vec": track_vec}

    def encode_intent(
        self,
        query_emb:    torch.Tensor,
        goal_emb:     torch.Tensor,
        category_emb: torch.Tensor,
        spec_emb:     torch.Tensor,
        date_emb:     torch.Tensor,
        profile_emb:  torch.Tensor,
    ) -> torch.Tensor:
        """Inference: encode user context to 128-dim vector."""
        with torch.no_grad():
            return self.intent_encoder(
                query_emb, goal_emb, category_emb, spec_emb, date_emb, profile_emb
            )

    def encode_query(self, ctx) -> torch.Tensor:
        """Inference from RetrievalContext (used by multi_channel_v2)."""
        dev = next(self.parameters()).device
        def _t(v, dim): return (v.float().unsqueeze(0).to(dev) if v is not None else torch.zeros(1, dim, device=dev))
        query_emb   = _t(ctx.query_emb, 1024)
        goal_emb    = _t(ctx.goal_emb,  1024)
        # category / specificity / date / profile: not available in ctx, use zeros
        category_emb = torch.zeros(1,  8, device=dev)
        spec_emb     = torch.zeros(1, 16, device=dev)
        date_emb     = torch.zeros(1, 14, device=dev)
        profile_emb  = torch.zeros(1, 62, device=dev)
        return self.encode_intent(query_emb, goal_emb, category_emb, spec_emb, date_emb, profile_emb).squeeze(0)

    def encode_track(
        self,
        metadata_emb:  torch.Tensor,
        lyrics_emb:    torch.Tensor,
        attr_emb:      torch.Tensor,
        audio_emb:     torch.Tensor,
        image_emb:     torch.Tensor,
        cf_bpr:        torch.Tensor,
        pop_bucket:    torch.Tensor,
        year_bucket:   torch.Tensor,
        dur_bucket:    torch.Tensor,
    ) -> torch.Tensor:
        """Inference: encode track features to 128-dim vector."""
        with torch.no_grad():
            return self.track_encoder(
                metadata_emb, lyrics_emb, attr_emb, audio_emb, image_emb,
                cf_bpr, pop_bucket, year_bucket, dur_bucket
            )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "state_dict":  self.state_dict(),
            "temperature": self.temperature.item(),
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "IntentTower":
        ckpt  = torch.load(path, map_location=device, weights_only=True)
        model = cls()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model.to(device)
