"""
mcrs/retrieval_modules/multi_channel_v2.py
===========================================
MultiChannelRetrievalV2 — assembles all direct embedding channels
(CH01-CH23) plus three learnable model channels (CH-Intent, CH-Profile,
CH-CF-Tower) into a single retrieve() interface.

Usage
-----
    from mcrs.retrieval_modules.multi_channel_v2 import MultiChannelRetrievalV2
    from mcrs.retrieval_modules.direct_channels import RetrievalContext

    retrieval = MultiChannelRetrievalV2.build(cfg)
    ctx = RetrievalContext(...)
    results = retrieval.retrieve(ctx, topk=200)
    # → {"CH02_Query_Meta": [...], "CH05_Goal_Meta": [...], ..., "merged": [...]}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .bm25 import BM25_MODEL
from .direct_channels import (
    DIRECT_CHANNELS,
    RetrievalContext,
    ch01_cf_bpr,
    ch23_bm25,
)
from .index_store import IndexStore

logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class MultiChannelConfig:
    """Configuration for MultiChannelRetrievalV2.

    Parameters
    ----------
    track_emb_dataset   : HuggingFace dataset with track embeddings
    track_metadata_name : HuggingFace dataset with track metadata (for BM25)
    split_types         : splits to include in the index
    cache_dir           : where to persist built indices
    bge_tag_path        : path to bge/track_tag_embeddings.pt
    goal_emb_path       : path to qwen/goal_embeddings_{split}_0.6b.pt
    query_emb_path      : path to qwen/dialogue_embeddings_{split}_0.6b.pt
    genre_emb_path      : path to bge/query_genre_embeddings_{split}.pt
    decade_emb_path     : path to bge/query_decade_embeddings_{split}.pt
    device              : "cuda" or "cpu"
    topk_per_channel    : override per-channel k (dict channel_name → k)
    cf_quota_redistrib  : if True, redistribute CH01 quota when user has no CF
    intent_model_path   : path to trained intent-tower checkpoint (optional)
    profile_model_path  : path to trained profile-tower checkpoint (optional)
    cf_tower_model_path : path to trained CF-tower checkpoint (optional)
    """
    track_emb_dataset:    str              = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
    track_metadata_name:  str              = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
    user_metadata_name:   str              = "talkpl-ai/TalkPlayData-Challenge-User-Metadata"
    # User-Embeddings dataset (contains cf-bpr vectors, separate from User-Metadata)
    user_emb_dataset:     str              = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings"
    split_types:          List[str]        = field(default_factory=lambda: ["all_tracks"])
    cache_dir:            str              = "qwen/retrieval_indices"
    bge_tag_path:         Optional[str]    = "bge/track_tag_embeddings.pt"
    goal_emb_path:        Optional[str]    = None
    query_emb_path:       Optional[str]    = None
    genre_emb_path:       Optional[str]    = None  # legacy: old genre emb; or new turn_query BGE emb path
    decade_emb_path:      Optional[str]    = None
    bge_rich_path:        Optional[str]    = "bge/track_rich_embeddings.pt"
    device:               str              = "cpu"
    topk_per_channel:     Dict[str, int]   = field(default_factory=dict)
    cf_quota_redistrib:   bool             = True
    intent_model_path:    Optional[str]    = None
    profile_model_path:   Optional[str]    = None
    cf_tower_model_path:  Optional[str]    = None


# ── Main class ─────────────────────────────────────────────────────────────────

class MultiChannelRetrievalV2:
    """Multi-channel retrieval system (v2).

    Manages the IndexStore, pre-loaded embedding stores, BM25 retrieval,
    and (optionally) trained model towers.
    """

    def __init__(
        self,
        index:              IndexStore,
        bm25:               Optional[BM25_MODEL],
        cfg:                MultiChannelConfig,
        goal_store:         Optional[Dict[str, torch.Tensor]] = None,
        query_store:        Optional[Dict[str, torch.Tensor]] = None,
        genre_store:        Optional[Dict[str, torch.Tensor]] = None,
        decade_store:       Optional[Dict[str, torch.Tensor]] = None,
        # track meta lookup tables
        track_to_artist:    Optional[Dict[str, str]] = None,
        track_to_album:     Optional[Dict[str, str]] = None,
        artist_to_tracks:   Optional[Dict[str, List[str]]] = None,
        album_to_tracks:    Optional[Dict[str, List[str]]] = None,
        # user CF-BPR embeddings store: user_id → Tensor[128]
        user_cf_store:      Optional[Dict[str, torch.Tensor]] = None,
        intent_model        = None,
        profile_model       = None,
        cf_tower_model      = None,
    ):
        self.index           = index
        self.bm25            = bm25
        self.cfg             = cfg
        self.goal_store      = goal_store   or {}
        self.query_store     = query_store  or {}
        self.genre_store     = genre_store  or {}
        self.decade_store    = decade_store or {}
        self.track_to_artist = track_to_artist  or {}
        self.track_to_album  = track_to_album   or {}
        self.artist_to_tracks = artist_to_tracks or {}
        self.album_to_tracks  = album_to_tracks  or {}
        self.user_cf_store   = user_cf_store or {}
        self.intent_model    = intent_model
        self.profile_model   = profile_model
        self.cf_tower_model  = cf_tower_model

    # ── factory ────────────────────────────────────────────────────────────────

    @classmethod
    def build(cls, cfg: MultiChannelConfig) -> "MultiChannelRetrievalV2":
        """Build the retrieval system from configuration."""
        # Index
        logger.info("Building / loading IndexStore …")
        index = IndexStore.build(
            track_emb_dataset=cfg.track_emb_dataset,
            split_types=cfg.split_types,
            cache_dir=cfg.cache_dir,
            bge_tag_path=cfg.bge_tag_path,
            bge_rich_path=cfg.bge_rich_path,
            device=cfg.device,
        )

        # BM25
        logger.info("Loading BM25 …")
        try:
            bm25 = BM25_MODEL(
                dataset_name=cfg.track_metadata_name,
                split_types=cfg.split_types,
                corpus_types=["track_name", "artist_name", "album_name"],
                cache_dir=cfg.cache_dir,
            )
        except Exception as e:
            logger.warning("BM25 init failed (%s) — CH23 disabled.", e)
            bm25 = None

        # Pre-computed embedding stores
        def _load_pt(path: Optional[str], name: str):
            if not path:
                return {}
            try:
                import os
                if not os.path.exists(path):
                    logger.warning("%s not found at %s", name, path)
                    return {}
                store = torch.load(path, map_location="cpu", weights_only=True)
                logger.info("Loaded %s: %d entries from %s", name, len(store), path)
                return store
            except Exception as e:
                logger.warning("Failed to load %s (%s)", name, e)
                return {}

        goal_store   = _load_pt(cfg.goal_emb_path,   "goal_store")
        query_store  = _load_pt(cfg.query_emb_path,  "query_store")
        genre_store  = _load_pt(cfg.genre_emb_path,  "genre_store")
        decade_store = _load_pt(cfg.decade_emb_path, "decade_store")

        # Track metadata look-up tables (for artist/album expand channels)
        track_to_artist:  Dict[str, str]         = {}
        track_to_album:   Dict[str, str]         = {}
        artist_to_tracks: Dict[str, List[str]]   = {}
        album_to_tracks:  Dict[str, List[str]]   = {}
        try:
            from datasets import load_dataset as _load_ds
            logger.info("Building track→artist/album lookup tables …")
            tm_ds = _load_ds(cfg.track_metadata_name)
            tm_splits = [s for s in cfg.split_types if s in tm_ds] or list(tm_ds.keys())
            from datasets import concatenate_datasets as _cat
            tm_all = _cat([tm_ds[s] for s in tm_splits])
            for row in tm_all:
                tid = str(row.get("track_id", ""))
                aid = str(row.get("artist_id", "")) if row.get("artist_id") else ""
                alid = str(row.get("album_id", "")) if row.get("album_id") else ""
                if tid:
                    if aid:
                        track_to_artist[tid] = aid
                        artist_to_tracks.setdefault(aid, []).append(tid)
                    if alid:
                        track_to_album[tid] = alid
                        album_to_tracks.setdefault(alid, []).append(tid)
            logger.info("  %d tracks, %d artists, %d albums mapped.",
                        len(track_to_artist), len(artist_to_tracks), len(album_to_tracks))
        except Exception as e:
            logger.warning("Track metadata lookup build failed (%s) — CH20/21 disabled.", e)

        # User CF-BPR embeddings store (user_id → Tensor[128])
        # From talkpl-ai/TalkPlayData-Challenge-User-Embeddings, field "cf-bpr"
        user_cf_store: Dict[str, torch.Tensor] = {}
        try:
            from datasets import load_dataset as _load_ds2
            logger.info("Building user CF-BPR store from %s …", cfg.user_emb_dataset)
            u_ds = _load_ds2(cfg.user_emb_dataset)
            u_splits = list(u_ds.keys())
            from datasets import concatenate_datasets as _cat2
            u_all = _cat2([u_ds[s] for s in u_splits])
            loaded_cf = 0
            _logged_cols = False
            for row in u_all:
                if not _logged_cols:
                    _logged_cols = True
                    logger.info("  User-Embeddings columns: %s", list(row.keys())[:20])
                uid = str(row.get("user_id", ""))
                # Field is "cf-bpr" in TalkPlayData-Challenge-User-Embeddings
                cf_vec = row.get("cf-bpr")
                if uid and cf_vec is not None:
                    try:
                        t = torch.tensor(cf_vec, dtype=torch.float32)
                        if t.numel() > 0:
                            user_cf_store[uid] = t
                            loaded_cf += 1
                    except Exception:
                        pass
            logger.info("  %d users with CF-BPR embeddings.", loaded_cf)
        except Exception as e:
            logger.warning("User CF-BPR store build failed (%s) — CH01 disabled.", e)

        # Trained models (optional, lazy import to avoid mandatory deps)
        intent_model = profile_model = cf_tower_model = None
        if cfg.intent_model_path:
            try:
                from mcrs.tower_models.intent_tower import IntentTower
                intent_model = IntentTower.load(cfg.intent_model_path, cfg.device)
                logger.info("Intent tower loaded from %s", cfg.intent_model_path)
            except Exception as e:
                logger.warning("Intent tower load failed (%s)", e)

        if cfg.profile_model_path:
            try:
                from mcrs.tower_models.profile_tower import ProfileTower
                profile_model = ProfileTower.load(cfg.profile_model_path, cfg.device)
                logger.info("Profile tower loaded from %s", cfg.profile_model_path)
            except Exception as e:
                logger.warning("Profile tower load failed (%s)", e)

        if cfg.cf_tower_model_path:
            try:
                from mcrs.tower_models.cf_tower import CFTower
                cf_tower_model = CFTower.load(cfg.cf_tower_model_path, cfg.device)
                logger.info("CF tower loaded from %s", cfg.cf_tower_model_path)
            except Exception as e:
                logger.warning("CF tower load failed (%s)", e)

        return cls(
            index=index, bm25=bm25, cfg=cfg,
            goal_store=goal_store, query_store=query_store,
            genre_store=genre_store, decade_store=decade_store,
            track_to_artist=track_to_artist,
            track_to_album=track_to_album,
            artist_to_tracks=artist_to_tracks,
            album_to_tracks=album_to_tracks,
            user_cf_store=user_cf_store,
            intent_model=intent_model,
            profile_model=profile_model,
            cf_tower_model=cf_tower_model,
        )

    # ── Context builder ────────────────────────────────────────────────────────

    def build_context(
        self,
        session_id:     str,
        turn_number:    int,
        session_data:   dict,
        user_id:        Optional[str] = None,
    ) -> RetrievalContext:
        """Build a RetrievalContext from raw session data.

        Args:
            session_id    : session identifier
            turn_number   : current turn (1-indexed)
            session_data  : HF dataset row (conversations list, etc.)
            user_id       : user identifier (used to look up CF-BPR embedding)
        """
        ctx = RetrievalContext(
            session_id=session_id,
            turn_number=turn_number,
            # Inject look-up tables from the system
            track_to_artist=self.track_to_artist,
            track_to_album=self.track_to_album,
            artist_to_tracks=self.artist_to_tracks,
            album_to_tracks=self.album_to_tracks,
        )

        # ── User CF ───────────────────────────────────────────────────────────
        if user_id and user_id in self.user_cf_store:
            ctx.user_cf_emb = self.user_cf_store[user_id]
            ctx.has_user_cf = True
        else:
            ctx.has_user_cf = False
            ctx.user_cf_emb = None

        # ── Query embedding ───────────────────────────────────────────────────
        # Try key formats: "{session_id}_{turn}_query" or "{session_id}_{turn}"
        # NOTE: cannot use `or` on Tensors (ambiguous bool) — use explicit None check
        _q = self.query_store.get(f"{session_id}_{turn_number}_query")
        ctx.query_emb = _q if _q is not None else self.query_store.get(f"{session_id}_{turn_number}")

        # Previous turn query
        if turn_number > 1:
            _pq = self.query_store.get(f"{session_id}_{turn_number - 1}_query")
            ctx.prev_query_emb = _pq if _pq is not None else self.query_store.get(f"{session_id}_{turn_number - 1}")

        # ── Goal embedding ────────────────────────────────────────────────────
        ctx.goal_emb = self.goal_store.get(session_id)

        # ── BGE genre / decade ────────────────────────────────────────────────
        ctx.genre_emb  = self.genre_store.get(f"{session_id}_{turn_number}_genre")
        ctx.decade_emb = self.decade_store.get(f"{session_id}_{turn_number}_decade")

        # ── History tracks ────────────────────────────────────────────────────────
        convs = session_data.get("conversations", [])
        pos_ids:  List[str] = []
        neg_ids:  List[str] = []
        last_tid: Optional[str] = None

        # goal_progress_assessments is a top-level list (separate from conversations):
        #   [{"turn_number": 2, "goal_progress_assessment": "MOVES_TOWARD_GOAL"}, ...]
        # Convention: assessment at turn T is the feedback on the music at turn T-1.
        assessments = session_data.get("goal_progress_assessments", [])
        asmt_map: Dict[int, str] = {
            int(a["turn_number"]): (a.get("goal_progress_assessment") or "")
            for a in assessments
            if isinstance(a, dict) and a.get("turn_number") is not None
        }

        for c in sorted(convs, key=lambda x: int(x.get("turn_number", 0))):
            t = int(c.get("turn_number", 0))
            if t >= turn_number:
                continue
            if c.get("role") == "music" and c.get("content"):
                last_tid = c["content"]
                track_id = c["content"]
                # Priority 1: inline label field in the conversation turn
                inline_label = c.get("label") or c.get("goal_progress_assessment") or ""
                # Priority 2: assessment keyed at turn T+1 references music at turn T
                next_turn_label = asmt_map.get(t + 1, "")
                # Priority 3: assessment keyed directly at turn T
                same_turn_label = asmt_map.get(t, "")
                label = inline_label or next_turn_label or same_turn_label
                if "MOVES_TOWARD_GOAL" in label:
                    pos_ids.append(track_id)
                else:
                    neg_ids.append(track_id)

        ctx.last_track_id = last_tid
        ctx.pos_track_ids = pos_ids
        ctx.neg_track_ids = neg_ids

        # ── Artist / Album sets from pos history ──────────────────────────────
        ctx.artist_ids = list({
            self.track_to_artist[tid]
            for tid in pos_ids
            if tid in self.track_to_artist
        })
        ctx.album_ids = list({
            self.track_to_album[tid]
            for tid in pos_ids
            if tid in self.track_to_album
        })

        # ── BM25 query string — use current turn user message ─────────────────
        for c in convs:
            if int(c.get("turn_number", 0)) == turn_number and c.get("role") == "user":
                ctx.bm25_query = c.get("content", "")
                break

        return ctx

    # ── Main retrieve ──────────────────────────────────────────────────────────

    def retrieve(
        self,
        ctx: RetrievalContext,
        topk: int = 200,
    ) -> Dict[str, List[str]]:
        """Run all channels and return per-channel + merged results.

        Returns
        -------
        Dict with keys:
            "<channel_name>" → List[track_id]   per channel
            "merged"         → List[track_id]   de-duplicated union (order by first appearance)
        """
        cfg = self.cfg
        results: Dict[str, List[str]] = {}

        # ── CH01 CF-BPR — handle quota redistribution ─────────────────────────
        cf_k    = cfg.topk_per_channel.get("CH01_CF_BPR", 200)
        ch01_res = ch01_cf_bpr(ctx, self.index, cf_k)
        results["CH01_CF_BPR"] = ch01_res
        skipped_cf_quota = cf_k if not ch01_res else 0

        # ── Direct embedding channels (CH02-CH22) ─────────────────────────────
        for ch_name, ch_fn, default_k in DIRECT_CHANNELS:
            if ch_name == "CH01_CF_BPR":
                continue  # already done
            k = cfg.topk_per_channel.get(ch_name, default_k)
            # Redistribute skipped CF quota to high-value channels
            if skipped_cf_quota and cfg.cf_quota_redistrib and ch_name in (
                "CH02_Query_Meta", "CH05_Goal_Meta", "CH08_Pos_Sem"
            ):
                each_extra = skipped_cf_quota // 3
                k += each_extra
                skipped_cf_quota -= each_extra
            try:
                res = ch_fn(ctx, self.index, k)
            except Exception as e:
                logger.warning("Channel %s failed: %s", ch_name, e)
                res = []
            results[ch_name] = res

        # ── CH23 BM25 ─────────────────────────────────────────────────────────
        bm25_k = cfg.topk_per_channel.get("CH23_BM25", 200)
        results["CH23_BM25"] = ch23_bm25(ctx, self.bm25, bm25_k)

        # ── Trained model channels (optional) ─────────────────────────────────
        if self.intent_model is not None:
            try:
                intent_k = cfg.topk_per_channel.get("CH_Intent", 200)
                intent_vec = self.intent_model.encode_query(ctx)
                results["CH_Intent"] = self.index.topk("metadata", intent_vec, intent_k)
            except Exception as e:
                logger.warning("Intent tower channel failed: %s", e)
                results["CH_Intent"] = []

        if self.profile_model is not None:
            try:
                profile_k = cfg.topk_per_channel.get("CH_Profile", 200)
                profile_vec = self.profile_model.encode_query(ctx)
                results["CH_Profile"] = self.index.topk("metadata", profile_vec, profile_k)
            except Exception as e:
                logger.warning("Profile tower channel failed: %s", e)
                results["CH_Profile"] = []

        if self.cf_tower_model is not None:
            try:
                cf_t_k = cfg.topk_per_channel.get("CH_CF_Tower", 200)
                cf_vec = self.cf_tower_model.encode_user(ctx.user_cf_emb)
                results["CH_CF_Tower"] = self.index.topk("cf_bpr", cf_vec, cf_t_k)
            except Exception as e:
                logger.warning("CF tower channel failed: %s", e)
                results["CH_CF_Tower"] = []

        # ── Merge via Reciprocal Rank Fusion (RRF) ────────────────────────────
        # score(d) = Σ_channel  weight[channel] / (RRF_K + rank[channel])
        # Channels absent or empty get no contribution.
        # CF-BPR weight is zeroed out when user has no CF embedding.
        RRF_K = 60  # standard RRF constant

        # ── Per-channel weights ────────────────────────────────────────────────
        _W: Dict[str, float] = {
            # Query / Goal text recall — high
            "CH02_Query_Meta":       3.0,
            "CH03_Query_Lyrics":     3.0,
            "CH04_Query_Attributes": 3.0,
            "CH05_Goal_Meta":        3.0,
            "CH06_Goal_Lyrics":      3.0,
            "CH07_Goal_Attributes":  3.0,
            "CH23_BM25":             3.0,
            # Session pos/neg feedback — high
            "CH08_Pos_Sem":          3.0,
            "CH09_Neg_Correct":      3.0,
            "CH10_Query_Delta":      2.5,
            # Artist / Album / Tag expansion — high (coverage-limited)
            "CH20_Artist_Expand":    3.0,
            "CH21_Album_Expand":     3.0,
            "CH22_BGE_Genre_Decade": 2.5,
            # CF-BPR — mid-high if user has CF, else 0
            "CH01_CF_BPR":           3.0 if ctx.has_user_cf else 0.0,
            # Last-track similarity — medium
            "CH11_Last_Audio":       2.0,
            "CH12_Last_Image":       2.0,
            "CH13_Last_Meta":        2.0,
            "CH14_Last_Lyrics":      2.0,
            # Pos-history audio/image — medium-low, only meaningful with history
            "CH15_Pos_Audio":        1.5,
            "CH16_Pos_Image":        1.5,
            "CH17_Pos_Meta":         2.0,
            "CH18_Pos_Lyrics":       1.5,
            "CH19_Pos_Attributes":   1.5,
            # Trained model towers — high when available
            "CH_Intent":             3.5,
            "CH_Profile":            2.5,
            "CH_CF_Tower":           3.0 if ctx.has_user_cf else 0.0,
        }

        rrf_scores: Dict[str, float] = {}
        for ch_name, candidates in results.items():
            w = _W.get(ch_name, 1.0)
            if w == 0.0 or not candidates:
                continue
            for rank_0, tid in enumerate(candidates):  # rank_0 is 0-indexed
                rrf_scores[tid] = rrf_scores.get(tid, 0.0) + w / (RRF_K + rank_0 + 1)

        # Sort by RRF score descending; keep ALL candidates (caller slices at K)
        results["merged"] = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        logger.debug(
            "Session %s turn %d: %d channels, %d merged candidates",
            ctx.session_id, ctx.turn_number,
            sum(1 for k in results if k != "merged"),
            len(results["merged"]),
        )
        return results

    # ── Per-channel stats (for eval) ──────────────────────────────────────────

    def channel_names(self) -> List[str]:
        """Return all channel names in registry order."""
        names = [ch_name for ch_name, _, _ in DIRECT_CHANNELS]
        names.append("CH23_BM25")
        if self.intent_model:
            names.append("CH_Intent")
        if self.profile_model:
            names.append("CH_Profile")
        if self.cf_tower_model:
            names.append("CH_CF_Tower")
        return names
