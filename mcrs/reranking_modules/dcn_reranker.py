"""
mcrs/reranking_modules/dcn_reranker.py
=======================================
DCN-V2 Reranker – v5 (clean design)

Fixes applied vs v4:
  [P1] Loss: true listwise BPR / pairwise margin instead of pseudo-pointwise
  [P2] Hard-negative mining: nearest-neighbor negatives in query space
  [P3] Retrieval signal removed — leakage / noisy at train time
  [P4] conv_goal fixed: pure goal_emb [1024] truncated to 128, no spurious zeros
  [P5] Query embedding: strict per-turn key only ({sid}_{turn}_query), no session fallback
  [P6] Feature engineering identical at train / eval

Feature schema (all dims fixed, no dynamic fallback confusion):
  user_profile   [86]   : age(16)+country(16)+gender(2)+lang(4)+culture(32)+listen(8)+session(8)
  user_cf        [128]  : CF-BPR user emb, trunc/pad to 128
  goal_emb       [128]  : session-level listener goal emb[:128]
  query_emb      [128]  : per-turn query emb[:128], ZERO if key missing (explicit)

  track_audio    [128]  : CLAP[:128]
  track_image    [128]  : SigLIP[:128]
  track_text     [128]  : mean(attr[:128], lyrics[:128], meta[:128])
  track_context  [128]  : MLP(137→128)
  track_cf       [128]  : CF-BPR item emb[:128]

  Total = 9×128 = 1152 → DCN-V2(3 cross, 256-256-128 deep) → head(128→32→1)

Loss: ListMLE / listwise softmax CE, hard negatives from retrieval candidates.
  If no retrieval store: fallback to random negatives (logged with a warning).
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


EMB_DIM    = 128
CONCAT_DIM = 9 * EMB_DIM   # 1152


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden_dims, out_dim: int,
         dropout: float = 0.1) -> nn.Sequential:
    layers, prev = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _trunc_pad(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Truncate or zero-pad the last dimension to exactly `dim`."""
    d = t.shape[-1]
    if d >= dim:
        return t[..., :dim].contiguous()
    return F.pad(t, (0, dim - d))


def _bucket_emb(value, n_bins: int, lo: float, hi: float) -> torch.Tensor:
    t = torch.zeros(n_bins)
    if value is None:
        return t
    idx = int((float(value) - lo) / (hi - lo + 1e-9) * n_bins)
    t[max(0, min(n_bins - 1, idx))] = 1.0
    return t


# ── Small structural encoder ───────────────────────────────────────────────────

class UserProfileEncoder(nn.Module):
    """86-dim bucket features → 128."""
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.net = _mlp(86, [128], EMB_DIM, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x.float()), p=2, dim=-1)


class TrackContextEncoder(nn.Module):
    """137-dim ISRC/tag/pop/dur features → 128."""
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.net = _mlp(137, [128], EMB_DIM, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x.float()), p=2, dim=-1)


# ── DCN-V2 ─────────────────────────────────────────────────────────────────────

class CrossLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=True)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * self.W(xl) + xl


class DCNV2(nn.Module):
    def __init__(self, in_dim: int = CONCAT_DIM,
                 cross_layers: int = 3,
                 deep_dims: Tuple = (256, 256, 128),
                 dropout: float = 0.1):
        super().__init__()
        self.cross = nn.ModuleList([CrossLayer(in_dim) for _ in range(cross_layers)])
        deep_list, prev = [], in_dim
        for h in deep_dims:
            deep_list += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.deep = nn.Sequential(*deep_list)
        self.deep_out_dim = prev
        merge = in_dim + self.deep_out_dim
        self.head = nn.Sequential(
            nn.Linear(merge, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, xc = x, x
        for layer in self.cross:
            xc = layer(x0, xc)
        return self.head(torch.cat([xc, self.deep(x)], dim=1))


# ── Full reranker ──────────────────────────────────────────────────────────────

class DCNReranker(nn.Module):
    """
    Inputs (all batched [B, dim]):
      user_profile  [B, 86]
      user_cf       [B, ≥1]   → trunc/pad to 128
      goal_emb      [B, 1024] → trunc to 128
      query_emb     [B, 1024] → trunc to 128   (MUST be per-turn, zero if absent)
      track_audio   [B, 512]  → trunc to 128   (CLAP)
      track_image   [B, 768]  → trunc to 128   (SigLIP)
      track_attr    [B, 1024] → trunc to 128
      track_lyrics  [B, 1024] → trunc to 128
      track_meta    [B, 1024] → trunc to 128
      track_context [B, 137]  → MLP  → 128
      track_cf      [B, ≥1]  → trunc/pad to 128
    """

    def __init__(self, cross_layers: int = 3,
                 deep_dims: Tuple = (256, 256, 128),
                 dropout: float = 0.1):
        super().__init__()
        self.user_profile_enc  = UserProfileEncoder(dropout)
        self.track_context_enc = TrackContextEncoder(dropout)
        self.dcn = DCNV2(CONCAT_DIM, cross_layers, deep_dims, dropout)

    # ── Internal feature builder ───────────────────────────────────────────────

    def _build_feat(
        self,
        user_profile:  torch.Tensor,   # [B, 86]
        user_cf:       torch.Tensor,   # [B, ≥1]
        goal_emb:      torch.Tensor,   # [B, 1024]
        query_emb:     torch.Tensor,   # [B, 1024]
        track_audio:   torch.Tensor,   # [B, 512]
        track_image:   torch.Tensor,   # [B, 768]
        track_attr:    torch.Tensor,   # [B, 1024]
        track_lyrics:  torch.Tensor,   # [B, 1024]
        track_meta:    torch.Tensor,   # [B, 1024]
        track_context: torch.Tensor,   # [B, 137]
        track_cf:      torch.Tensor,   # [B, ≥1]
    ) -> torch.Tensor:                 # [B, CONCAT_DIM]

        u_prof  = self.user_profile_enc(user_profile)                   # [B,128]
        u_cf    = F.normalize(_trunc_pad(user_cf.float(), 128), p=2, dim=-1)
        u_goal  = F.normalize(_trunc_pad(goal_emb.float(), 128), p=2, dim=-1)
        u_query = F.normalize(_trunc_pad(query_emb.float(), 128), p=2, dim=-1)

        t_audio  = F.normalize(_trunc_pad(track_audio.float(),  128), p=2, dim=-1)
        t_image  = F.normalize(_trunc_pad(track_image.float(),  128), p=2, dim=-1)
        t_text   = F.normalize(
            (_trunc_pad(track_attr.float(),   128)
             + _trunc_pad(track_lyrics.float(), 128)
             + _trunc_pad(track_meta.float(),   128)) / 3.0,
            p=2, dim=-1,
        )
        t_ctx    = self.track_context_enc(track_context)                # [B,128]
        t_cf     = F.normalize(_trunc_pad(track_cf.float(), 128), p=2, dim=-1)

        return torch.cat([
            u_prof, u_cf, u_goal, u_query,
            t_audio, t_image, t_text, t_ctx, t_cf,
        ], dim=1)  # [B, 1152]

    # ── Inference ─────────────────────────────────────────────────────────────

    def encode(self, *args) -> torch.Tensor:
        """Returns [B, 1] raw logit (higher = more relevant)."""
        return self.dcn(self._build_feat(*args))

    # ── Training forward: listwise softmax CE ─────────────────────────────────

    def forward(
        self,
        # User (identical for pos + negs; expand inside)
        user_profile:  torch.Tensor,   # [B, 86]
        user_cf:       torch.Tensor,   # [B, ≥1]
        goal_emb:      torch.Tensor,   # [B, 1024]
        query_emb:     torch.Tensor,   # [B, 1024]
        # Positive track
        pos_audio:     torch.Tensor,   # [B, 512]
        pos_image:     torch.Tensor,   # [B, 768]
        pos_attr:      torch.Tensor,   # [B, 1024]
        pos_lyrics:    torch.Tensor,   # [B, 1024]
        pos_meta:      torch.Tensor,   # [B, 1024]
        pos_context:   torch.Tensor,   # [B, 137]
        pos_cf:        torch.Tensor,   # [B, ≥1]
        # Negative tracks [B, K, dim]
        neg_audio:     torch.Tensor,
        neg_image:     torch.Tensor,
        neg_attr:      torch.Tensor,
        neg_lyrics:    torch.Tensor,
        neg_meta:      torch.Tensor,
        neg_context:   torch.Tensor,
        neg_cf:        torch.Tensor,
    ) -> torch.Tensor:
        """
        Listwise loss: cross_entropy(scores[B, 1+K], label=0).
        Positive always at position 0.
        """
        B = user_profile.size(0)
        K = neg_audio.size(1)

        pos_score = self.dcn(self._build_feat(
            user_profile, user_cf, goal_emb, query_emb,
            pos_audio, pos_image, pos_attr, pos_lyrics, pos_meta, pos_context, pos_cf,
        ))  # [B, 1]

        def _fl(t): return t.reshape(B * K, -1)
        def _ex(t): return t.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)

        neg_score = self.dcn(self._build_feat(
            _ex(user_profile), _ex(user_cf), _ex(goal_emb), _ex(query_emb),
            _fl(neg_audio), _fl(neg_image), _fl(neg_attr),
            _fl(neg_lyrics), _fl(neg_meta),
            _fl(neg_context), _fl(neg_cf),
        )).reshape(B, K)  # [B, K]

        logits = torch.cat([pos_score, neg_score], dim=1)   # [B, 1+K]
        labels = torch.zeros(B, dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu",
             cross_layers: int = 3,
             deep_dims: Tuple = (256, 256, 128),
             dropout: float = 0.0) -> "DCNReranker":
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = cls(cross_layers=cross_layers, deep_dims=deep_dims, dropout=dropout)
        model.load_state_dict(ckpt.get("state_dict", ckpt))
        return model.eval().to(device)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Canonical feature extraction (shared by train & eval) ─────────────────────
# All logic here — import this in both scripts to guarantee consistency [P6]

def extract_query_emb(
    query_store: Dict,
    session_id: str,
    turn_number: int,
) -> torch.Tensor:
    """
    [P5] Strict per-turn key: {sid}_{turn}_query ONLY.
    Returns zero vector if key absent (explicit, logged externally).
    NO session-level fallback.
    """
    k = f"{session_id}_{turn_number}_query"
    v = query_store.get(k)
    return v.float() if v is not None else torch.zeros(1024)


def extract_goal_emb(
    goal_store: Dict,
    session_id: str,
) -> torch.Tensor:
    """
    [P4] Pure goal embedding: session-level 1024-dim.
    No category / specificity padding — those were meaningless zeros.
    """
    v = goal_store.get(session_id)
    return v.float() if v is not None else torch.zeros(1024)


def extract_user_profile(user_id: str, user_meta: Dict) -> torch.Tensor:
    """86-dim user profile features."""
    um = user_meta.get(user_id, {})
    age = um.get("age")
    age_b = (_bucket_emb(math.log1p(float(age)), 16, 0, math.log1p(100))
             if age is not None else torch.zeros(16))
    gender = torch.zeros(2)
    if um.get("gender") == "male":     gender[0] = 1.0
    elif um.get("gender") == "female": gender[1] = 1.0
    return torch.cat([
        age_b,
        _bucket_emb(um.get("country_code_hash"),               16, 0, 1),
        gender,
        _bucket_emb(um.get("preferred_language_hash"),          4, 0, 1),
        _bucket_emb(um.get("preferred_musical_culture_hash"),  32, 0, 1),
        _bucket_emb(math.log1p(float(um.get("listen_count",  0) or 0)),
                    8, 0, math.log1p(10000)),
        _bucket_emb(math.log1p(float(um.get("session_count", 0) or 0)),
                    8, 0, math.log1p(1000)),
    ])  # [86]


def extract_track_features(
    track_id: str,
    index,
    track_meta: Dict,
    bge_tag_store: Dict,
    tag_proj: nn.Linear,
    proj_device,
) -> Tuple[torch.Tensor, ...]:
    """
    Returns (audio[512], image[768], attr[1024], lyrics[1024], meta[1024],
             context[137], cf[128])
    All shapes are fixed — zero vectors if embedding absent.
    """
    def _gv(mod, dim):
        v = index.get_vec(mod, track_id)
        return v.float() if v is not None else torch.zeros(dim)

    audio    = _gv("audio",      512)
    image    = _gv("image",      768)
    attr     = _gv("attributes", 1024)
    lyrics   = _gv("lyrics",     1024)
    meta_emb = _gv("metadata",   1024)

    bge_v = bge_tag_store.get(track_id, torch.zeros(384)).float()
    with torch.no_grad():
        tag32 = F.normalize(
            tag_proj(bge_v.unsqueeze(0).to(proj_device)).squeeze(0), p=2, dim=0
        ).cpu()
    tm       = track_meta.get(track_id, {})
    log_pop  = torch.tensor([math.log1p(float(tm.get("popularity", 0) or 0))])
    dur_bkt  = _bucket_emb(tm.get("duration_ms"), 8, 30000, 600000)
    # context: 4×tag32 + log_pop + dur → [4×32+1+8] = [137]
    context  = torch.cat([tag32, tag32, tag32, tag32, log_pop, dur_bkt])

    cf = _gv("cf_bpr", 128)
    return audio, image, attr, lyrics, meta_emb, context, cf
