"""Multi-channel retrieval module for music recommendation.

This module implements a comprehensive retrieval strategy combining:
1. User CF-BPR embedding similarity (user-music tower)
2. Collaborative filtering based on similar users' liked music
3. Query-metadata semantic similarity
4. Query-attributes semantic similarity

Missing / empty embeddings are stored as zero-vectors of the expected dimension
so that every track/user remains in the index.  The reranker layer uses
learnable UNK parameters to replace those zero-vectors during training and
inference.
"""

import os
import json
import logging
from functools import lru_cache
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict

import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


# ── retrieval hyper-parameters ──────────────────────────────────────────────
USER_HISTORY_LIKE_MUSIC_NUM  = 5
USER_MUSIC_TOWER_RECALL_NUM  = 100
QUERY_METADATA_RECALL_NUM    = 100
QUERY_ATTRIBUTES_RECALL_NUM  = 100

# canonical embedding dimensions (filled in after the first valid row is seen)
_TRACK_MODAL_COLS = [
    "cf-bpr",
    "audio-laion_clap",
    "image-siglip2",
    "attributes-qwen3_embedding_0.6b",
    "lyrics-qwen3_embedding_0.6b",
    "metadata-qwen3_embedding_0.6b",
]
_USER_MODAL_COLS = ["cf-bpr"]


class MultiChannelRetrieval:
    """Multi-channel retrieval combining CF, collaborative filtering, and semantic search."""

    def __init__(
        self,
        dataset_name: str,
        item_db_name: str,
        user_db_name: str,
        track_emb_db_name: str,
        user_emb_db_name: str,
        split_types: List[str],
        cache_dir: str = "./cache",
        qwen_model_path: str = "/home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B",
        device: str = "cuda",
        batch_size: int = 32,
        qwen_model=None,       # pre-loaded shared instance (optional)
        qwen_tokenizer=None,   # pre-loaded shared instance (optional)
    ):
        self.dataset_name    = dataset_name
        self.item_db_name    = item_db_name
        self.cache_dir       = cache_dir
        self.qwen_model_path = qwen_model_path
        self.device          = device
        self.batch_size      = batch_size
        self.split_types     = split_types

        self.index_dir = os.path.join(cache_dir, "multi_channel_retrieval")
        os.makedirs(self.index_dir, exist_ok=True)

        logger.info("Loading datasets …")
        self.track_metadata_dict = self._load_track_metadata(item_db_name, split_types)
        self.user_metadata_dict  = self._load_user_metadata(user_db_name, split_types)

        logger.info("Loading embeddings …")
        self.track_embeddings = self._load_track_embeddings(track_emb_db_name, split_types)
        self.user_embeddings  = self._load_user_embeddings(user_emb_db_name, split_types)

        logger.info("Building user history index …")
        self.user_liked_music = self._build_user_history_index(dataset_name, split_types)

        # Use shared Qwen model if provided; otherwise load from disk on CPU
        if qwen_model is not None and qwen_tokenizer is not None:
            logger.info("Using shared Qwen model (CPU).")
            self.tokenizer  = qwen_tokenizer
            self.qwen_model = qwen_model
            self._qwen_device = "cpu"
        else:
            logger.info("Loading Qwen model on CPU …")
            self.tokenizer  = AutoTokenizer.from_pretrained(qwen_model_path)
            self.qwen_model = AutoModel.from_pretrained(qwen_model_path).cpu().eval()
            self._qwen_device = "cpu"

        logger.info("Building track embedding indices …")
        self._build_track_indices()

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _raw_to_tensor(value, dim: int) -> torch.Tensor:
        """Convert a raw dataset value to a float32 tensor of length *dim*.

        * If the value is None / empty / wrong type → return a zero-vector.
        * If the resulting 1-D tensor is shorter than *dim*   → zero-pad.
        * If it is longer than *dim*                          → truncate.
        The caller decides what *dim* should be (inferred from first valid row).
        """
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

    # ── dataset loaders ─────────────────────────────────────────────────────

    def _load_track_metadata(self, dataset_name: str, split_types: List[str]) -> Dict:
        ds = load_dataset(dataset_name)
        valid_splits = [s for s in split_types if s in ds.keys()] or list(ds.keys())
        return {item["track_id"]: item
                for item in concatenate_datasets([ds[s] for s in valid_splits])}

    def _load_user_metadata(self, dataset_name: str, split_types: List[str]) -> Dict:
        ds = load_dataset(dataset_name)
        valid_splits = [s for s in split_types if s in ds.keys()] or list(ds.keys())
        return {item["user_id"]: item
                for item in concatenate_datasets([ds[s] for s in valid_splits])}

    def _load_track_embeddings(self, dataset_name: str, split_types: List[str]) -> Dict:
        """Load track embeddings.

        Every track is kept in the dictionary.  Missing / malformed columns are
        stored as zero-vectors of the canonical dimension so that downstream
        ``torch.stack`` calls are always safe.  The *missing* flag lets the
        reranker replace zero-vectors with its learnable UNK parameters.
        """
        ds = load_dataset(dataset_name)
        valid_splits = [s for s in split_types if s in ds.keys()] or list(ds.keys())
        concat_ds = concatenate_datasets([ds[s] for s in valid_splits])

        # First pass: discover canonical dimensions from first valid row per column
        col_dims: Dict[str, int] = {}
        for item in concat_ds:
            for col in _TRACK_MODAL_COLS:
                if col in col_dims:
                    continue
                v = item.get(col)
                if v is None:
                    continue
                try:
                    t = torch.tensor(v, dtype=torch.float32)
                    if t.ndim == 1 and t.numel() > 0:
                        col_dims[col] = t.shape[0]
                except Exception:
                    pass
            if len(col_dims) == len(_TRACK_MODAL_COLS):
                break

        # Fall back if a column is entirely missing from the dataset
        for col in _TRACK_MODAL_COLS:
            if col not in col_dims:
                col_dims[col] = 128
                logger.warning("Column '%s' not found in track embeddings; defaulting dim to 128.", col)

        self.track_col_dims = col_dims  # expose for reranker
        logger.info("Track embedding dims: %s", col_dims)

        # Second pass: build the dict, zero-filling missing values
        embeddings: Dict[str, Dict] = {}
        missing_counts: Dict[str, int] = defaultdict(int)
        for item in concat_ds:
            track_id = item["track_id"]
            row: Dict[str, torch.Tensor] = {}
            missing_cols: List[str] = []
            for col in _TRACK_MODAL_COLS:
                v = item.get(col)
                t = self._raw_to_tensor(v, col_dims[col])
                row[col] = t
                if v is None or (not isinstance(v, (list, tuple)) and v != v):  # noqa: comparison-with-itself detects NaN
                    missing_cols.append(col)
                    missing_counts[col] += 1
                elif isinstance(v, (list, tuple)) and len(v) == 0:
                    missing_cols.append(col)
                    missing_counts[col] += 1
            row["__missing__"] = missing_cols  # track which modalities were absent
            embeddings[track_id] = row

        for col, cnt in missing_counts.items():
            if cnt:
                logger.warning("Track col '%s': %d / %d rows were missing → zero-filled.", col, cnt, len(embeddings))
        logger.info("Loaded %d tracks (zero-fill for missing modalities).", len(embeddings))
        return embeddings

    def _load_user_embeddings(self, dataset_name: str, split_types: List[str]) -> Dict:
        """Load user embeddings; zero-fill missing cf-bpr."""
        ds = load_dataset(dataset_name)
        valid_splits = [s for s in split_types if s in ds.keys()] or list(ds.keys())
        concat_ds = concatenate_datasets([ds[s] for s in valid_splits])

        # Discover dimension
        cf_bpr_dim = 128
        for item in concat_ds:
            v = item.get("cf-bpr")
            if v is None:
                continue
            try:
                t = torch.tensor(v, dtype=torch.float32)
                if t.ndim == 1 and t.numel() > 0:
                    cf_bpr_dim = t.shape[0]
                    break
            except Exception:
                pass

        self.user_cf_bpr_dim = cf_bpr_dim  # expose for reranker
        logger.info("User cf-bpr dim: %d", cf_bpr_dim)

        embeddings: Dict[str, Dict] = {}
        missing = 0
        for item in concat_ds:
            user_id = item["user_id"]
            v = item.get("cf-bpr")
            t = self._raw_to_tensor(v, cf_bpr_dim)
            is_missing = (v is None or (isinstance(v, (list, tuple)) and len(v) == 0))
            if is_missing:
                missing += 1
            embeddings[user_id] = {
                "cf-bpr":     t,
                "__missing__": ["cf-bpr"] if is_missing else [],
            }

        if missing:
            logger.warning("User cf-bpr: %d / %d rows missing → zero-filled.", missing, len(embeddings))
        logger.info("Loaded %d users.", len(embeddings))

        # Pre-build stacked matrix for fast nearest-user search.
        # Only use rows that have real embeddings (non-zero) for reliable similarity.
        valid_ids   = [uid for uid, d in embeddings.items() if not d["__missing__"]]
        invalid_ids = [uid for uid, d in embeddings.items() if d["__missing__"]]

        if valid_ids:
            stacked = torch.stack([embeddings[uid]["cf-bpr"] for uid in valid_ids])
            stacked = F.normalize(stacked, p=2, dim=1)
            self._user_ids_ordered  = valid_ids
            self._user_emb_matrix   = stacked
        else:
            self._user_ids_ordered  = []
            self._user_emb_matrix   = None

        if invalid_ids:
            logger.warning("%d users have missing cf-bpr; excluded from nearest-user search.", len(invalid_ids))

        return embeddings

    # ── user history ─────────────────────────────────────────────────────────

    def _build_user_history_index(
        self, dataset_name: str, split_types: List[str]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """user_id -> [(track_id, like_score), …] sorted by preference."""
        cache_path = os.path.join(self.index_dir, "user_liked_music.json")
        if os.path.exists(cache_path):
            logger.info("Loading user history index from %s", cache_path)
            with open(cache_path) as f:
                data = json.load(f)
            return {k: [tuple(v) for v in vals] for k, vals in data.items()}

        logger.info("Building user history index …")
        ds = load_dataset(dataset_name)
        valid_splits = [s for s in split_types if s in ds.keys()] or list(ds.keys())

        user_history: Dict[str, List] = defaultdict(list)
        for split in valid_splits:
            for session in ds[split]:
                user_id      = session["user_id"]
                conversations = session["conversations"]
                assessments  = session.get("goal_progress_assessments", [])
                assessment_map = {a["turn_number"]: a for a in assessments}
                for conv in conversations:
                    if conv["role"] != "music":
                        continue
                    turn_num    = conv["turn_number"]
                    track_id    = conv["content"]
                    asmt        = assessment_map.get(turn_num, {})
                    gpa         = asmt.get("goal_progress_assessment", "")
                    like_score  = 5.0 if gpa == "MOVES_TOWARD_GOAL" else 1.0
                    user_history[user_id].append((track_id, like_score, turn_num))

        user_liked_music: Dict[str, List] = {}
        for uid, history in user_history.items():
            sorted_h = sorted(history, key=lambda x: (-x[1], -x[2]))
            user_liked_music[uid] = [
                (tid, score) for tid, score, _ in sorted_h[:USER_HISTORY_LIKE_MUSIC_NUM]
            ]

        with open(cache_path, "w") as f:
            json.dump(user_liked_music, f)
        logger.info("User history index saved to %s", cache_path)
        return user_liked_music

    # ── index build ──────────────────────────────────────────────────────────

    def _build_track_indices(self):
        """Build CF-BPR / metadata / attributes matrices for retrieval.

        Every track in self.track_embeddings is included (zero-filled rows are
        kept so that all tracks remain retrievable).
        """
        cf_bpr_path     = os.path.join(self.index_dir, "track_cf_bpr.pt")
        metadata_path   = os.path.join(self.index_dir, "track_metadata_256.pt")
        attributes_path = os.path.join(self.index_dir, "track_attributes_256.pt")
        track_ids_path  = os.path.join(self.index_dir, "track_ids.json")

        if all(os.path.exists(p) for p in [cf_bpr_path, metadata_path, attributes_path, track_ids_path]):
            logger.info("Loading pre-computed track indices …")
            self.track_cf_bpr_matrix     = torch.load(cf_bpr_path,     map_location="cpu")
            self.track_metadata_matrix   = torch.load(metadata_path,   map_location="cpu")
            self.track_attributes_matrix = torch.load(attributes_path, map_location="cpu")
            with open(track_ids_path) as f:
                self.track_ids_list = json.load(f)
            logger.info("Loaded index for %d tracks.", len(self.track_ids_list))
            return

        logger.info("Building track indices …")
        track_ids = sorted(self.track_embeddings.keys())
        self.track_ids_list = track_ids

        cf_bpr_list     = []
        metadata_list   = []
        attributes_list = []

        for tid in track_ids:
            embs = self.track_embeddings[tid]
            cf_bpr_list.append(embs["cf-bpr"])
            # Take first 256 dims (zero-pad if shorter)
            meta  = embs["metadata-qwen3_embedding_0.6b"]
            attr  = embs["attributes-qwen3_embedding_0.6b"]
            metadata_list.append(self._raw_to_tensor(meta.tolist(), 256))
            attributes_list.append(self._raw_to_tensor(attr.tolist(), 256))

        self.track_cf_bpr_matrix     = F.normalize(torch.stack(cf_bpr_list),     p=2, dim=1)
        self.track_metadata_matrix   = F.normalize(torch.stack(metadata_list),   p=2, dim=1)
        self.track_attributes_matrix = F.normalize(torch.stack(attributes_list), p=2, dim=1)

        torch.save(self.track_cf_bpr_matrix,     cf_bpr_path)
        torch.save(self.track_metadata_matrix,   metadata_path)
        torch.save(self.track_attributes_matrix, attributes_path)
        with open(track_ids_path, "w") as f:
            json.dump(self.track_ids_list, f)

        logger.info("Track indices saved. Total: %d tracks.", len(track_ids))

    # ── query encoding ───────────────────────────────────────────────────────

    def set_turn_store(self, store: dict):
        """Inject a pre-computed turn embedding store.

        store: dict  key='{session_id}__{turn}_{role}'  value=float16 [128]
        Once set, retrieve() can skip all Qwen calls when session_id+turn_number
        are provided.
        """
        self._turn_store = store
        logger.info("Turn store injected into MultiChannelRetrieval (%d entries).", len(store))

    # kept for fallback / standalone use
    def set_embedding_cache(self, cache):
        """Inject a pre-computed QwenEmbeddingCache (legacy; prefer set_turn_store)."""
        self._qwen_cache = cache
        logger.info("QwenEmbeddingCache injected into MultiChannelRetrieval (%d entries).", len(cache))

    def _encode_query(self, query: str) -> torch.Tensor:
        """Encode query; look up pre-computed cache first, fall back to Qwen model."""
        if not query or not query.strip():
            return torch.zeros(256)

        # 1. Pre-computed cache (fastest path)
        if hasattr(self, "_qwen_cache") and self._qwen_cache is not None:
            return self._qwen_cache.get(query, dim=256, normalize=True)

        # 2. Per-instance runtime text cache
        if not hasattr(self, "_encode_cache"):
            self._encode_cache: Dict[str, torch.Tensor] = {}
        if query in self._encode_cache:
            return self._encode_cache[query]

        self.qwen_model.eval()
        with torch.no_grad():
            inputs    = self.tokenizer(
                query, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            outputs   = self.qwen_model(**inputs)
            attn_mask = inputs["attention_mask"]
            tok_embs  = outputs.last_hidden_state
            mask      = attn_mask.unsqueeze(-1).expand(tok_embs.size()).float()
            emb       = torch.sum(tok_embs * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            emb       = F.normalize(emb[:, :256], p=2, dim=1).squeeze(0)

        if len(self._encode_cache) >= 8192:
            self._encode_cache.clear()
        self._encode_cache[query] = emb
        return emb

    # ── retrieval channels ───────────────────────────────────────────────────

    def _retrieve_cf_bpr(self, user_id: Optional[str], topk: int) -> List[str]:
        """Channel 1: user CF-BPR × track CF-BPR cosine similarity."""
        if (user_id is None
                or user_id not in self.user_embeddings
                or self.user_embeddings[user_id]["__missing__"]):
            return self.track_ids_list[:topk]

        user_bpr = self.user_embeddings[user_id]["cf-bpr"]
        user_bpr = F.normalize(user_bpr.unsqueeze(0), p=2, dim=1)  # [1, D]
        scores   = torch.matmul(self.track_cf_bpr_matrix, user_bpr.T).squeeze(1)
        topk     = min(topk, scores.shape[0])
        top_idx  = torch.topk(scores, k=topk).indices.tolist()
        return [self.track_ids_list[i] for i in top_idx]

    def _retrieve_similar_users_music(self, user_id: Optional[str]) -> List[str]:
        """Channel 2: liked music from the 10 nearest users (by cf-bpr)."""
        if (user_id is None
                or user_id not in self.user_embeddings
                or self.user_embeddings[user_id]["__missing__"]
                or self._user_emb_matrix is None):
            return []

        user_bpr = self.user_embeddings[user_id]["cf-bpr"]
        user_bpr = F.normalize(user_bpr.unsqueeze(0), p=2, dim=1)  # [1, D]
        scores   = torch.matmul(self._user_emb_matrix, user_bpr.T).squeeze(1)  # [M]

        # Zero out own score
        if user_id in self._user_ids_ordered:
            scores[self._user_ids_ordered.index(user_id)] = -1.0

        top_10  = min(10, scores.shape[0])
        top_idx = torch.topk(scores, k=top_10).indices.tolist()
        similar = [self._user_ids_ordered[i] for i in top_idx]

        liked: Set[str] = set()
        for sim_uid in similar:
            for tid, _ in self.user_liked_music.get(sim_uid, []):
                liked.add(tid)
        return list(liked)

    def _retrieve_query_metadata(self, query_emb: torch.Tensor, topk: int) -> List[str]:
        """Channel 3: query × metadata-qwen3 (256 dims)."""
        scores  = torch.matmul(self.track_metadata_matrix, query_emb.unsqueeze(1)).squeeze(1)
        topk    = min(topk, scores.shape[0])
        top_idx = torch.topk(scores, k=topk).indices.tolist()
        return [self.track_ids_list[i] for i in top_idx]

    def _retrieve_query_attributes(self, query_emb: torch.Tensor, topk: int) -> List[str]:
        """Channel 4: query × attributes-qwen3 (256 dims)."""
        scores  = torch.matmul(self.track_attributes_matrix, query_emb.unsqueeze(1)).squeeze(1)
        topk    = min(topk, scores.shape[0])
        top_idx = torch.topk(scores, k=topk).indices.tolist()
        return [self.track_ids_list[i] for i in top_idx]

    # ── public API ───────────────────────────────────────────────────────────

    def retrieve(
        self,
        user_id: Optional[str],
        current_query: str,
        history_queries: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        turn_number: Optional[int] = None,
    ) -> List[str]:
        """Multi-channel retrieval; returns deduplicated track list (~350).

        If session_id and turn_number are provided AND a turn store is loaded,
        embeddings are looked up directly (no Qwen forward pass).
        """
        if history_queries is None:
            history_queries = []

        store = getattr(self, "_turn_store", None)
        if store is not None and session_id is not None and turn_number is not None:
            user_key  = f"{session_id}__{turn_number}_user"
            hist_key  = f"{session_id}__{turn_number}_history_avg"
            current_emb = (
                store[user_key].float() if user_key in store
                else self._encode_query(current_query)
            )
            hist_emb_raw = (
                store[hist_key].float() if hist_key in store else None
            )
            current_emb = F.normalize(current_emb.unsqueeze(0)[:, :256], p=2, dim=1).squeeze(0)
            if hist_emb_raw is not None:
                hist_emb = F.normalize(hist_emb_raw.unsqueeze(0)[:, :256], p=2, dim=1).squeeze(0)
                query_emb = F.normalize(0.3 * hist_emb + 0.7 * current_emb, p=2, dim=0)
            else:
                query_emb = current_emb
        else:
            # Fallback: text-based encoding
            current_emb = self._encode_query(current_query)
            if history_queries:
                hist_embs = [self._encode_query(q) for q in history_queries]
                hist_emb  = torch.stack(hist_embs).mean(dim=0)
                query_emb = F.normalize(0.3 * hist_emb + 0.7 * current_emb, p=2, dim=0)
            else:
                query_emb = current_emb

        ch1 = self._retrieve_cf_bpr(user_id, USER_MUSIC_TOWER_RECALL_NUM)
        ch2 = self._retrieve_similar_users_music(user_id)
        ch3 = self._retrieve_query_metadata(query_emb, QUERY_METADATA_RECALL_NUM)
        ch4 = self._retrieve_query_attributes(query_emb, QUERY_ATTRIBUTES_RECALL_NUM)

        seen: Set[str] = set()
        result: List[str] = []
        for tid in ch1 + ch2 + ch3 + ch4:
            if tid not in seen:
                result.append(tid)
                seen.add(tid)

        logger.debug("Multi-channel retrieval: %d unique tracks.", len(result))
        return result

    def batch_retrieve(
        self,
        user_ids: List[Optional[str]],
        current_queries: List[str],
        history_queries_list: Optional[List[List[str]]] = None,
        session_ids: Optional[List[Optional[str]]] = None,
        turn_numbers: Optional[List[Optional[int]]] = None,
    ) -> List[List[str]]:
        if history_queries_list is None:
            history_queries_list = [[] for _ in user_ids]
        if session_ids is None:
            session_ids = [None] * len(user_ids)
        if turn_numbers is None:
            turn_numbers = [None] * len(user_ids)
        return [
            self.retrieve(uid, q, hist, sid, tn)
            for uid, q, hist, sid, tn
            in zip(user_ids, current_queries, history_queries_list, session_ids, turn_numbers)
        ]
