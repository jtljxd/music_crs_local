"""
mcrs/retrieval_modules/direct_channels.py
==========================================
All "direct" (no-training-needed) retrieval channels.

Each channel is a function with signature:

    channel(ctx: RetrievalContext, store: IndexStore, k: int) -> List[str]

where RetrievalContext carries all per-request state (embeddings, history, …).

Channels
--------
 CH01  CF-BPR embedding                  (old-user only, k=30)
 CH02  Query × Metadata (Qwen)           k=55
 CH03  Query × Lyrics (Qwen)             k=35
 CH04  Query × Attributes (Qwen)         k=35
 CH05  Listener-Goal × Metadata          k=45
 CH06  Listener-Goal × Lyrics            k=25
 CH07  Listener-Goal × Attributes        k=25
 CH08  Session pos-feedback × t_sem      k=35
 CH09  Session neg-feedback correction   k=25
 CH10  Query delta (curr - last)         k=20
 CH11  Last-track × Audio (CLAP)         k=15
 CH12  Last-track × Image (SigLIP)       k=10
 CH13  Last-track × Metadata Qwen        k=15
 CH14  Last-track × Lyrics Qwen          k=10
 CH15  Pos-history × Audio (CLAP)        k=15
 CH16  Pos-history × Image (SigLIP)      k=10
 CH17  Pos-history × Metadata Qwen       k=15
 CH18  Pos-history × Lyrics Qwen         k=10
 CH19  Pos-history × Attributes Qwen     k=10
 CH20  Artist expand                     k=20
 CH21  Album / ISRC expand               k=10
 CH22  BGE genre + decade similarity     k=25
 CH23  BM25 Artist/Album/duration/pop    k=15
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .index_store import IndexStore

logger = logging.getLogger(__name__)

# ── Context ────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalContext:
    """All per-request information needed by the retrieval channels.

    Attributes:
        session_id      : str
        turn_number     : int           # 1-indexed current turn
        # ── embeddings (pre-computed, may be None if unavailable) ──
        query_emb       : Optional[Tensor[1024]]   # current turn Qwen query emb
        prev_query_emb  : Optional[Tensor[1024]]   # previous turn Qwen query emb
        goal_emb        : Optional[Tensor[1024]]   # listener_goal Qwen emb
        genre_emb       : Optional[Tensor[384]]    # current turn genre BGE emb
        decade_emb      : Optional[Tensor[384]]    # current turn decade BGE emb
        user_cf_emb     : Optional[Tensor[128]]    # user CF-BPR emb (old users)
        has_user_cf     : bool
        # ── history tracks ──────────────────────────────────────────
        last_track_id   : Optional[str]            # previous recommended track
        pos_track_ids   : List[str]   # MOVES_TOWARD_GOAL labelled tracks so far
        neg_track_ids   : List[str]   # other-label tracks (negative feedback)
        # ── metadata look-ups ───────────────────────────────────────
        artist_ids      : List[str]   # artist_ids of pos feedback tracks
        album_ids       : List[str]   # album_ids of pos feedback tracks
        # ── track_id → artist_id / album_id maps (for expand channels) ─
        track_to_artist : Dict[str, str]   # injected from item DB
        track_to_album  : Dict[str, str]
        artist_to_tracks: Dict[str, List[str]]
        album_to_tracks : Dict[str, List[str]]
        # ── BM25 query string ───────────────────────────────────────
        bm25_query      : str              # artist/album/duration/pop text query
        # ── lambda for neg-feedback correction ─────────────────────
        neg_lambda      : float = 0.5
    """
    session_id:        str                         = ""
    turn_number:       int                         = 1
    query_emb:         Optional[torch.Tensor]      = None
    prev_query_emb:    Optional[torch.Tensor]      = None
    goal_emb:          Optional[torch.Tensor]      = None
    genre_emb:         Optional[torch.Tensor]      = None
    decade_emb:        Optional[torch.Tensor]      = None
    user_cf_emb:       Optional[torch.Tensor]      = None
    has_user_cf:       bool                        = False
    last_track_id:     Optional[str]               = None
    pos_track_ids:     List[str]                   = field(default_factory=list)
    neg_track_ids:     List[str]                   = field(default_factory=list)
    artist_ids:        List[str]                   = field(default_factory=list)
    album_ids:         List[str]                   = field(default_factory=list)
    track_to_artist:   Dict[str, str]              = field(default_factory=dict)
    track_to_album:    Dict[str, str]              = field(default_factory=dict)
    artist_to_tracks:  Dict[str, List[str]]        = field(default_factory=dict)
    album_to_tracks:   Dict[str, List[str]]        = field(default_factory=dict)
    bm25_query:        str                         = ""
    neg_lambda:        float                       = 0.5

    # surplus quota (if CF-BPR channel skipped) redistributed here
    extra_quota:       Dict[str, int]              = field(default_factory=dict)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_topk(store: IndexStore, modality: str,
               vec: torch.Tensor, k: int) -> List[str]:
    """topK with guard: returns [] if modality absent or vec is all-zero."""
    if not store.has_modality(modality):
        return []
    if vec is None or vec.abs().sum().item() == 0:
        return []
    return store.topk(modality, vec, k)


def _mean_vec(store: IndexStore, modality: str,
              track_ids: List[str]) -> Optional[torch.Tensor]:
    """Mean of embedding vectors for a list of track IDs."""
    vecs = [store.get_vec(modality, tid) for tid in track_ids]
    vecs = [v for v in vecs if v is not None]
    if not vecs:
        return None
    stacked = torch.stack(vecs)          # [M, D]
    return stacked.mean(0)               # [D]


# ── Channel implementations ───────────────────────────────────────────────────

def ch01_cf_bpr(ctx: RetrievalContext, store: IndexStore, k: int = 30) -> List[str]:
    """CH01 — Old-user CF-BPR embedding recall."""
    if not ctx.has_user_cf or ctx.user_cf_emb is None:
        return []
    return _safe_topk(store, "cf_bpr", ctx.user_cf_emb, k)


def ch02_query_metadata(ctx: RetrievalContext, store: IndexStore, k: int = 55) -> List[str]:
    """CH02 — Current-turn query × metadata-qwen3 cosine topK."""
    return _safe_topk(store, "metadata", ctx.query_emb, k)


def ch03_query_lyrics(ctx: RetrievalContext, store: IndexStore, k: int = 35) -> List[str]:
    """CH03 — Current-turn query × lyrics-qwen3 cosine topK."""
    return _safe_topk(store, "lyrics", ctx.query_emb, k)


def ch04_query_attributes(ctx: RetrievalContext, store: IndexStore, k: int = 35) -> List[str]:
    """CH04 — Current-turn query × attributes-qwen3 cosine topK."""
    return _safe_topk(store, "attributes", ctx.query_emb, k)


def ch05_goal_metadata(ctx: RetrievalContext, store: IndexStore, k: int = 45) -> List[str]:
    """CH05 — Listener-goal × metadata-qwen3 cosine topK."""
    return _safe_topk(store, "metadata", ctx.goal_emb, k)


def ch06_goal_lyrics(ctx: RetrievalContext, store: IndexStore, k: int = 25) -> List[str]:
    """CH06 — Listener-goal × lyrics-qwen3 cosine topK."""
    return _safe_topk(store, "lyrics", ctx.goal_emb, k)


def ch07_goal_attributes(ctx: RetrievalContext, store: IndexStore, k: int = 25) -> List[str]:
    """CH07 — Listener-goal × attributes-qwen3 cosine topK."""
    return _safe_topk(store, "attributes", ctx.goal_emb, k)


def ch08_session_pos_feedback(ctx: RetrievalContext, store: IndexStore, k: int = 35) -> List[str]:
    """CH08 — Pos-feedback steered query recall on metadata.

    Steers the current query (or goal) vector toward the positive-history
    centroid: steered = normalize(intent + λ * pos_mean), then retrieves
    on metadata. Complements CH17 (pure pos mean) by injecting current intent.
    λ = 0.5 (equal weight between intent and pos history).
    """
    if not ctx.pos_track_ids:
        return []
    intent_vec = ctx.query_emb if ctx.query_emb is not None else ctx.goal_emb
    if intent_vec is None:
        # Fall back to pure pos-mean on audio (distinct from CH17/CH19)
        mean_v = _mean_vec(store, "audio", ctx.pos_track_ids)
        return _safe_topk(store, "audio", mean_v, k)
    pos_mean = _mean_vec(store, "metadata", ctx.pos_track_ids)
    if pos_mean is None:
        return _safe_topk(store, "metadata", intent_vec, k)
    lam = 0.5
    steered = F.normalize(
        (intent_vec.float() + lam * pos_mean.float()).unsqueeze(0),
        p=2, dim=1
    ).squeeze(0)
    return _safe_topk(store, "metadata", steered, k)


def ch09_session_neg_correction(ctx: RetrievalContext, store: IndexStore, k: int = 25) -> List[str]:
    """CH09 — Negative-feedback corrected intent vector recall.

    corrected = normalize(intent_vec - λ * hist_neg_vec)
    intent_vec = query_emb if present, else goal_emb
    """
    if not ctx.neg_track_ids:
        return []
    intent_vec = ctx.query_emb if ctx.query_emb is not None else ctx.goal_emb
    if intent_vec is None:
        return []
    neg_mean = _mean_vec(store, "metadata", ctx.neg_track_ids)
    if neg_mean is None:
        return []
    lam = ctx.neg_lambda
    corrected = F.normalize(
        (intent_vec.float() - lam * neg_mean.float()).unsqueeze(0),
        p=2, dim=1
    ).squeeze(0)
    return _safe_topk(store, "metadata", corrected, k)


def ch10_query_delta(ctx: RetrievalContext, store: IndexStore, k: int = 20) -> List[str]:
    """CH10 — Query delta (curr - last) recall on metadata.

    delta_q = normalize(q_current - q_last)
    If no prev query, falls back to plain ch02 with quota k.
    """
    if ctx.prev_query_emb is None or ctx.query_emb is None:
        return _safe_topk(store, "metadata", ctx.query_emb, k)
    delta = F.normalize(
        (ctx.query_emb.float() - ctx.prev_query_emb.float()).unsqueeze(0),
        p=2, dim=1
    ).squeeze(0)
    return _safe_topk(store, "metadata", delta, k)


def ch11_last_track_audio(ctx: RetrievalContext, store: IndexStore, k: int = 15) -> List[str]:
    """CH11 — Last-track audio (LAION-CLAP) neighbour recall."""
    if not ctx.last_track_id:
        return []
    return store.topk_from_id("audio", ctx.last_track_id, k)


def ch12_last_track_image(ctx: RetrievalContext, store: IndexStore, k: int = 10) -> List[str]:
    """CH12 — Last-track image (SigLIP2) neighbour recall."""
    if not ctx.last_track_id:
        return []
    return store.topk_from_id("image", ctx.last_track_id, k)


def ch13_last_track_metadata(ctx: RetrievalContext, store: IndexStore, k: int = 15) -> List[str]:
    """CH13 — Last-track metadata-qwen neighbour recall."""
    if not ctx.last_track_id:
        return []
    return store.topk_from_id("metadata", ctx.last_track_id, k)


def ch14_last_track_lyrics(ctx: RetrievalContext, store: IndexStore, k: int = 10) -> List[str]:
    """CH14 — Last-track lyrics-qwen neighbour recall."""
    if not ctx.last_track_id:
        return []
    return store.topk_from_id("lyrics", ctx.last_track_id, k)


def ch15_pos_audio(ctx: RetrievalContext, store: IndexStore, k: int = 15) -> List[str]:
    """CH15 — Pos-history mean audio (CLAP) recall."""
    if not ctx.pos_track_ids:
        return []
    mean_v = _mean_vec(store, "audio", ctx.pos_track_ids)
    return _safe_topk(store, "audio", mean_v, k)


def ch16_pos_image(ctx: RetrievalContext, store: IndexStore, k: int = 10) -> List[str]:
    """CH16 — Pos-history mean image (SigLIP2) recall."""
    if not ctx.pos_track_ids:
        return []
    mean_v = _mean_vec(store, "image", ctx.pos_track_ids)
    return _safe_topk(store, "image", mean_v, k)


def ch17_pos_metadata(ctx: RetrievalContext, store: IndexStore, k: int = 15) -> List[str]:
    """CH17 — Pos-history mean metadata-qwen recall."""
    if not ctx.pos_track_ids:
        return []
    mean_v = _mean_vec(store, "metadata", ctx.pos_track_ids)
    return _safe_topk(store, "metadata", mean_v, k)


def ch18_pos_lyrics(ctx: RetrievalContext, store: IndexStore, k: int = 10) -> List[str]:
    """CH18 — Pos-history mean lyrics-qwen recall."""
    if not ctx.pos_track_ids:
        return []
    mean_v = _mean_vec(store, "lyrics", ctx.pos_track_ids)
    return _safe_topk(store, "lyrics", mean_v, k)


def ch19_pos_attributes(ctx: RetrievalContext, store: IndexStore, k: int = 10) -> List[str]:
    """CH19 — Pos-history mean attributes-qwen recall."""
    if not ctx.pos_track_ids:
        return []
    mean_v = _mean_vec(store, "attributes", ctx.pos_track_ids)
    return _safe_topk(store, "attributes", mean_v, k)


def ch20_artist_expand(ctx: RetrievalContext, store: IndexStore, k: int = 20) -> List[str]:
    """CH20 — Artist-expand from pos-history artist_ids."""
    if not ctx.artist_ids:
        return []
    candidates: List[str] = []
    seen = set(ctx.pos_track_ids)
    for aid in ctx.artist_ids:
        for tid in ctx.artist_to_tracks.get(aid, []):
            if tid not in seen:
                candidates.append(tid)
                seen.add(tid)
    return candidates[:k]


def ch21_album_expand(ctx: RetrievalContext, store: IndexStore, k: int = 10) -> List[str]:
    """CH21 — Album-expand from pos-history album_ids."""
    if not ctx.album_ids:
        return []
    candidates: List[str] = []
    seen = set(ctx.pos_track_ids)
    for alid in ctx.album_ids:
        for tid in ctx.album_to_tracks.get(alid, []):
            if tid not in seen:
                candidates.append(tid)
                seen.add(tid)
    return candidates[:k]


def ch22_bge_genre_decade(ctx: RetrievalContext, store: IndexStore, k: int = 25) -> List[str]:
    """CH22 — BGE turn-query embedding recall on track rich-info (bge_rich).

    The query embedding is the BGE encoding of the current user turn text
    (from turn_query_embeddings_{split}.pt, key={session_id}_{turn}).
    The track index uses bge_rich: BGE encoding of
    track_name + artist_name + album_name + tag_list + release_date + duration + popularity.

    Falls back to tag_bge if bge_rich is not available.
    """
    # Prefer bge_rich; fall back to tag_bge for backward compatibility
    modality = "bge_rich" if store.has_modality("bge_rich") else (
               "tag_bge"  if store.has_modality("tag_bge")  else None)
    if modality is None:
        return []

    # Use turn-level BGE query emb (genre_emb stores the turn_query emb in new setup)
    # genre_emb / decade_emb may still hold old genre/decade vecs if old pipeline used
    query_emb = ctx.genre_emb  # repurposed as turn_query_emb in new pipeline

    if query_emb is None or query_emb.abs().sum().item() == 0:
        return []

    return store.topk(modality, query_emb, k)

    return results[:k]


def ch23_bm25(ctx: RetrievalContext, bm25_retrieval, k: int = 15) -> List[str]:
    """CH23 — BM25 recall on Artist/Album/duration/popularity text query.

    bm25_retrieval must implement .text_to_item_retrieval(query, topk) -> List[str]
    """
    if not ctx.bm25_query or bm25_retrieval is None:
        return []
    try:
        return bm25_retrieval.text_to_item_retrieval(ctx.bm25_query, topk=k)
    except Exception as e:
        logger.warning("CH23 BM25 failed: %s", e)
        return []


# ── Channel registry ───────────────────────────────────────────────────────────

#: Ordered list of (channel_id, fn, default_k)
#: Channels that need bm25_retrieval are handled separately in MultiChannelV2.
DIRECT_CHANNELS: List[Tuple[str, object, int]] = [
    ("CH01_CF_BPR",            ch01_cf_bpr,               200),
    ("CH02_Query_Meta",        ch02_query_metadata,        200),
    ("CH03_Query_Lyrics",      ch03_query_lyrics,          200),
    ("CH04_Query_Attributes",  ch04_query_attributes,      200),
    ("CH05_Goal_Meta",         ch05_goal_metadata,         200),
    ("CH06_Goal_Lyrics",       ch06_goal_lyrics,           200),
    ("CH07_Goal_Attributes",   ch07_goal_attributes,       200),
    ("CH08_Pos_Sem",           ch08_session_pos_feedback,  200),
    ("CH09_Neg_Correct",       ch09_session_neg_correction,200),
    ("CH10_Query_Delta",       ch10_query_delta,           200),
    ("CH11_Last_Audio",        ch11_last_track_audio,      200),
    ("CH12_Last_Image",        ch12_last_track_image,      200),
    ("CH13_Last_Meta",         ch13_last_track_metadata,   200),
    ("CH14_Last_Lyrics",       ch14_last_track_lyrics,     200),
    ("CH15_Pos_Audio",         ch15_pos_audio,             200),
    ("CH16_Pos_Image",         ch16_pos_image,             200),
    ("CH17_Pos_Meta",          ch17_pos_metadata,          200),
    ("CH18_Pos_Lyrics",        ch18_pos_lyrics,            200),
    ("CH19_Pos_Attributes",    ch19_pos_attributes,        200),
    ("CH20_Artist_Expand",     ch20_artist_expand,         200),
    ("CH21_Album_Expand",      ch21_album_expand,          200),
    ("CH22_BGE_Genre_Decade",  ch22_bge_genre_decade,      200),
    # CH23 handled in MultiChannelV2 (needs bm25 retrieval object)
]
