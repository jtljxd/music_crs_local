"""
mcrs/reranking_modules/dcn_reranker.py
=======================================
DCN-V2 Reranker for music CRS.

Feature design:
  All embedding features are truncated to 128-dim directly (no large MLP encoders).
  Scalar/categorical features are projected to 128-dim via a small MLP.

  User side:
    user_profile  [86→128]   : age_bucket+country+gender+lang+culture+listen+session → MLP
    user_cf       [128]      : raw CF-BPR (truncated/padded to 128)
    conv_goal     [1036→128] : category8 + goal_emb[:120] + specificity4 → truncate & MLP
    query_emb     [128]      : current-turn dialogue emb[:128]

  Track side:
    track_audio   [128]      : CLAP[:128]
    track_image   [128]      : SigLIP[:128]
    track_text    [128]      : mean(attr[:128], lyrics[:128], meta[:128])
    track_context [128]      : MLP(ISRC32+tag32+artist32+album32+logpop1+dur8 → 128)
    track_cf      [128]      : raw CF-BPR[:128]

  Retrieval signals:
    retrieval_feat [N_CH*2]  : for each channel: (hit_i, norm_rank_i)
                               hit=1 if track in channel, norm_rank = 1-rank/topk (0 if not hit)
                               N_CH = 23 → 46-dim

  Total = 9×128 + 46 = 1198 → DCN-V2(3 cross + 3 deep) → MLP(128→32→1)
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Channel registry (must match MultiChannelRetrievalV2 order) ───────────────
# Keep in sync with direct_channels.py DIRECT_CHANNELS + CH23_BM25
ALL_CHANNELS: List[str] = [
    "CH01_CF_BPR",
    "CH02_Query_Meta",
    "CH03_Query_Lyrics",
    "CH04_Query_Attributes",
    "CH05_Goal_Meta",
    "CH06_Goal_Lyrics",
    "CH07_Goal_Attributes",
    "CH08_Pos_Sem",
    "CH09_Neg_Correct",
    "CH10_Query_Delta",
    "CH11_Last_Audio",
    "CH12_Last_Image",
    "CH13_Last_Meta",
    "CH14_Last_Lyrics",
    "CH15_Pos_Audio",
    "CH16_Pos_Image",
    "CH17_Pos_Meta",
    "CH18_Pos_Lyrics",
    "CH19_Pos_Attributes",
    "CH20_Artist_Expand",
    "CH21_Album_Expand",
    "CH22_BGE_Genre_Decade",
    "CH23_BM25",
]
N_CHANNELS = len(ALL_CHANNELS)   # 23
CH_IDX     = {ch: i for i, ch in enumerate(ALL_CHANNELS)}

EMB_DIM    = 128
N_USER     = 4   # user_profile, user_cf, conv_goal, query_emb
N_TRACK    = 5   # track_audio, track_image, track_text, track_context, track_cf
CONCAT_DIM = (N_USER + N_TRACK) * EMB_DIM + N_CHANNELS * 2  # 9×128 + 46 = 1198


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden_dims, out_dim: int,
         dropout: float = 0.1) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _trunc_pad(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Truncate or zero-pad last dimension to `dim`."""
    d = t.shape[-1]
    if d >= dim:
        return t[..., :dim]
    return F.pad(t, (0, dim - d))


def _bucket_emb(value, n_bins: int, lo: float, hi: float) -> torch.Tensor:
    t = torch.zeros(n_bins)
    if value is None:
        return t
    v = float(value)
    idx = int((v - lo) / (hi - lo + 1e-9) * n_bins)
    t[max(0, min(n_bins - 1, idx))] = 1.0
    return t


# ── Small feature encoders ─────────────────────────────────────────────────────

class UserProfileEncoder(nn.Module):
    """86-dim categorical features → 128."""
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.net = _mlp(86, [128], EMB_DIM, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x.float()), p=2, dim=-1)


class TrackContextEncoder(nn.Module):
    """ISRC32+tag32+artist32+album32+logpop1+dur8 = 137 → 128."""
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.net = _mlp(137, [128], EMB_DIM, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x.float()), p=2, dim=-1)


# ── DCN-V2 ────────────────────────────────────────────────────────────────────

class CrossLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=True)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * self.W(xl) + xl


class DCNV2(nn.Module):
    def __init__(self, in_dim: int = CONCAT_DIM, cross_layers: int = 3,
                 deep_dims=(512, 256, 128), dropout: float = 0.1):
        super().__init__()
        self.cross = nn.ModuleList([CrossLayer(in_dim) for _ in range(cross_layers)])
        deep_layers = []
        prev = in_dim
        for h in deep_dims:
            deep_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.deep = nn.Sequential(*deep_layers)
        self.deep_out_dim = prev
        # Final head: concat(cross, deep) → MLP(128→32→1)
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
        xd = self.deep(x)
        return self.head(torch.cat([xc, xd], dim=1))


# ── Full reranker ──────────────────────────────────────────────────────────────

class DCNReranker(nn.Module):
    """
    Inputs
    ------
    user_profile    : [B, 86]
    user_cf         : [B, 128+]  (will be truncated to 128)
    conv_goal       : [B, 1036]  (goal_emb will be truncated)
    query_emb       : [B, 1024+] (will be truncated to 128)
    track_audio     : [B, 512+]  CLAP (truncated to 128)
    track_image     : [B, 768+]  SigLIP (truncated to 128)
    track_attr      : [B, 1024+] attributes (truncated to 128)
    track_lyrics    : [B, 1024+] (truncated to 128)
    track_meta_emb  : [B, 1024+] metadata emb (truncated to 128)
    track_context   : [B, 137]   ISRC+tag+artist+album+pop+dur → MLP → 128
    track_cf        : [B, 128+]  (truncated to 128)
    retrieval_feat  : [B, N_CH*2]  hit + norm_rank per channel
    """

    def __init__(self, cross_layers: int = 3,
                 deep_dims=(512, 256, 128), dropout: float = 0.1):
        super().__init__()
        self.user_profile_enc = UserProfileEncoder(dropout)
        self.track_context_enc = TrackContextEncoder(dropout)
        self.dcn = DCNV2(CONCAT_DIM, cross_layers, deep_dims, dropout)

    def _encode_features(
        self,
        user_profile:   torch.Tensor,   # [B, 86]
        user_cf:        torch.Tensor,   # [B, ≥128]
        conv_goal:      torch.Tensor,   # [B, 1036]
        query_emb:      torch.Tensor,   # [B, ≥128]
        track_audio:    torch.Tensor,   # [B, ≥128]  CLAP
        track_image:    torch.Tensor,   # [B, ≥128]  SigLIP
        track_attr:     torch.Tensor,   # [B, ≥128]
        track_lyrics:   torch.Tensor,   # [B, ≥128]
        track_meta_emb: torch.Tensor,   # [B, ≥128]
        track_context:  torch.Tensor,   # [B, 137]
        track_cf:       torch.Tensor,   # [B, ≥128]
        retrieval_feat: torch.Tensor,   # [B, N_CH*2]
    ) -> torch.Tensor:                  # [B, CONCAT_DIM]

        # User features
        u_prof  = self.user_profile_enc(user_profile)               # [B, 128]
        u_cf    = F.normalize(_trunc_pad(user_cf.float(),  128), p=2, dim=-1)
        # conv_goal: use category8 + goal_emb[:112] + specificity4 = 8+112+4=124→pad to 128
        cg_cat  = conv_goal[:, :8]
        cg_goal = conv_goal[:, 8:8+112]    # first 112 dims of goal_emb
        cg_spec = conv_goal[:, 1032:1036]
        u_goal  = F.normalize(_trunc_pad(
            torch.cat([cg_cat, cg_goal, cg_spec], dim=1).float(), 128
        ), p=2, dim=-1)
        u_query = F.normalize(_trunc_pad(query_emb.float(), 128), p=2, dim=-1)

        # Track features
        t_audio  = F.normalize(_trunc_pad(track_audio.float(),    128), p=2, dim=-1)
        t_image  = F.normalize(_trunc_pad(track_image.float(),    128), p=2, dim=-1)
        # text: mean of attr, lyrics, meta (each truncated to 128)
        t_text   = F.normalize(
            (_trunc_pad(track_attr.float(),    128)
             + _trunc_pad(track_lyrics.float(), 128)
             + _trunc_pad(track_meta_emb.float(), 128)) / 3.0,
            p=2, dim=-1
        )
        t_ctx    = self.track_context_enc(track_context)            # [B, 128]
        t_cf     = F.normalize(_trunc_pad(track_cf.float(),    128), p=2, dim=-1)

        # Retrieval signals (already float, normalized externally)
        ret      = retrieval_feat.float()                           # [B, 46]

        return torch.cat([
            u_prof, u_cf, u_goal, u_query,
            t_audio, t_image, t_text, t_ctx, t_cf,
            ret,
        ], dim=1)  # [B, CONCAT_DIM]

    def encode(self, *args) -> torch.Tensor:
        """Inference: single forward → [B, 1] score."""
        feat = self._encode_features(*args)
        return self.dcn(feat)

    def forward(
        self,
        # User features
        user_profile:   torch.Tensor,
        user_cf:        torch.Tensor,
        conv_goal:      torch.Tensor,
        query_emb:      torch.Tensor,
        # Positive track features
        pos_audio:      torch.Tensor,
        pos_image:      torch.Tensor,
        pos_attr:       torch.Tensor,
        pos_lyrics:     torch.Tensor,
        pos_meta_emb:   torch.Tensor,
        pos_context:    torch.Tensor,
        pos_cf:         torch.Tensor,
        pos_ret_feat:   torch.Tensor,   # [B, N_CH*2]
        # Negative track features [B, K, dim]
        neg_audio:      torch.Tensor,
        neg_image:      torch.Tensor,
        neg_attr:       torch.Tensor,
        neg_lyrics:     torch.Tensor,
        neg_meta_emb:   torch.Tensor,
        neg_context:    torch.Tensor,
        neg_cf:         torch.Tensor,
        neg_ret_feat:   torch.Tensor,   # [B, K, N_CH*2]
    ) -> torch.Tensor:
        """Training forward: listwise softmax CE, label=0 (positive first).
        Returns scalar loss.
        """
        B = user_profile.size(0)
        K = neg_audio.size(1)

        pos_feat = self._encode_features(
            user_profile, user_cf, conv_goal, query_emb,
            pos_audio, pos_image, pos_attr, pos_lyrics, pos_meta_emb,
            pos_context, pos_cf, pos_ret_feat,
        )
        pos_score = self.dcn(pos_feat)   # [B, 1]

        # Flatten negatives
        def _fl(t): return t.reshape(B * K, -1)
        def _ex(t): return t.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)

        neg_feat = self._encode_features(
            _ex(user_profile), _ex(user_cf), _ex(conv_goal), _ex(query_emb),
            _fl(neg_audio), _fl(neg_image), _fl(neg_attr),
            _fl(neg_lyrics), _fl(neg_meta_emb),
            _fl(neg_context), _fl(neg_cf), _fl(neg_ret_feat),
        )
        neg_score = self.dcn(neg_feat).reshape(B, K)   # [B, K]

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
             deep_dims=(512, 256, 128),
             dropout: float = 0.1) -> "DCNReranker":
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = cls(cross_layers=cross_layers, deep_dims=deep_dims, dropout=dropout)
        model.load_state_dict(ckpt.get("state_dict", ckpt))
        return model.eval().to(device)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Retrieval feature builder ──────────────────────────────────────────────────

def build_retrieval_feat(
    track_id: str,
    per_channel_results: Dict[str, List[str]],
    topk_per_channel: int = 200,
) -> torch.Tensor:
    """
    Build a [N_CH * 2] tensor for one track:
      [hit_0, rank_0, hit_1, rank_1, ..., hit_22, rank_22]
    hit_i    = 1.0 if track_id appears in channel i's result list, else 0.0
    rank_i   = 1 - (rank / topk_per_channel) if hit, else 0.0
               (higher = better rank, 1.0 = rank-1, 0.0 = not retrieved)
    """
    feat = torch.zeros(N_CHANNELS * 2)
    for ch_name, cands in per_channel_results.items():
        if ch_name == "merged":
            continue
        idx = CH_IDX.get(ch_name)
        if idx is None:
            continue
        try:
            rank = cands.index(track_id)   # 0-based
            feat[idx * 2]     = 1.0
            feat[idx * 2 + 1] = 1.0 - rank / max(len(cands), topk_per_channel)
        except ValueError:
            pass   # not in this channel
    return feat
