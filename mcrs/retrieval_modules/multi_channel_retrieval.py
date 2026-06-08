"""Multi-channel retrieval module for music recommendation (v2).

Remaining channels after refactor:
  ch1_CF-BPR   : Three-tower model (User+Query gate fusion vs Item CF-BPR)
  ch3_QwenMeta : hist_conversation_embeddings (1024d) × metadata-qwen3_embedding_0.6b, top-200
  ch5_BM25     : Keyword-based BM25 from query_split, top-200 split evenly across non-empty keys

Removed:
  ch2_SimilarUsers  (deleted)
  ch4_QwenAttr      (deleted)
  ch6_Semantic      (deleted)
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict

import bm25s
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


# ── retrieval hyper-parameters ──────────────────────────────────────────────
USER_MUSIC_TOWER_RECALL_NUM  = 100   # ch1 – retained for compat; actual recall done by 3-tower
QUERY_METADATA_RECALL_NUM    = 200   # ch3 – expanded from 100 → 200
BM25_RECALL_NUM              = 200   # ch5 – expanded from 100 → 200

# canonical embedding dimensions
_TRACK_MODAL_COLS = [
    "cf-bpr",
    "audio-laion_clap",
    "image-siglip2",
    "attributes-qwen3_embedding_0.6b",
    "lyrics-qwen3_embedding_0.6b",
    "metadata-qwen3_embedding_0.6b",
]
_USER_MODAL_COLS = ["cf-bpr"]

# Query-split fields used for BM25 keyword search
_QUERY_SPLIT_TEXT_FIELDS = [
    "artist", "album", "genre", "decade", "language",
    "popularity", "scene", "tempo",
]


class MultiChannelRetrieval:
    """3-channel retrieval: CF-BPR three-tower, QwenMeta semantic, BM25 keyword."""

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
        qwen_model=None,
        qwen_tokenizer=None,
        build_indices: bool = True,
        # Optional pre-loaded hist_conversation_embeddings for ch3
        conv_emb_store: Optional[Dict[str, torch.Tensor]] = None,
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

        # Optionally accept pre-loaded hist_conv_emb store (used by ch3)
        self._conv_emb_store: Optional[Dict[str, torch.Tensor]] = conv_emb_store

        # Turn store (query/history emb per session-turn)
        self._turn_store: Optional[Dict[str, torch.Tensor]] = None
        # Query-split store ({session_id}_{turn} → json str)
        self._query_split_store: Optional[Dict[str, str]] = None

        # Qwen model (used as fallback encoder for ch3 if no conv_emb_store)
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
        self._build_track_indices(build=build_indices)

        # ── BM25 index (Channel 5) ────────────────────────────────────────
        logger.info("%s BM25 index …", "Building" if build_indices else "Loading")
        self._build_bm25_index(build=build_indices)

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
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
        """Load track embeddings, zero-filling missing columns."""
        ds = load_dataset(dataset_name)
        valid_splits = [s for s in split_types if s in ds.keys()] or list(ds.keys())
        concat_ds = concatenate_datasets([ds[s] for s in valid_splits])

        # First pass: discover canonical dimensions
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

        for col in _TRACK_MODAL_COLS:
            if col not in col_dims:
                col_dims[col] = 128
                logger.warning("Column '%s' missing; defaulting dim to 128.", col)

        self.track_col_dims = col_dims
        logger.info("Track embedding dims: %s", col_dims)

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
                if v is None or (isinstance(v, (list, tuple)) and len(v) == 0):
                    missing_cols.append(col)
                    missing_counts[col] += 1
            row["__missing__"] = missing_cols
            embeddings[track_id] = row

        for col, cnt in missing_counts.items():
            if cnt:
                logger.warning("Track col '%s': %d / %d rows missing → zero-filled.",
                               col, cnt, len(embeddings))
        logger.info("Loaded %d tracks.", len(embeddings))
        return embeddings

    def _load_user_embeddings(self, dataset_name: str, split_types: List[str]) -> Dict:
        """Load user CF-BPR embeddings."""
        ds = load_dataset(dataset_name)
        valid_splits = [s for s in split_types if s in ds.keys()] or list(ds.keys())
        concat_ds = concatenate_datasets([ds[s] for s in valid_splits])

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

        self.user_cf_bpr_dim = cf_bpr_dim
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
                "cf-bpr":      t,
                "__missing__": ["cf-bpr"] if is_missing else [],
            }

        if missing:
            logger.warning("User cf-bpr: %d / %d rows missing → zero-filled.",
                           missing, len(embeddings))
        logger.info("Loaded %d users.", len(embeddings))
        return embeddings

    # ── index build ──────────────────────────────────────────────────────────

    def _build_track_indices(self, build: bool = True):
        """Build or load CF-BPR / metadata matrices."""
        _idx_dir      = os.path.join("qwen", "retrieval_indices", "track_indices")
        cf_bpr_path   = os.path.join(_idx_dir, "track_cf_bpr.pt")
        metadata_path = os.path.join(_idx_dir, "track_metadata_1024.pt")
        track_ids_path = os.path.join(_idx_dir, "track_ids.json")

        if all(os.path.exists(p) for p in [cf_bpr_path, metadata_path, track_ids_path]):
            logger.info("Loading pre-computed track indices …")
            self.track_cf_bpr_matrix   = torch.load(cf_bpr_path,   map_location="cpu")
            self.track_metadata_matrix = torch.load(metadata_path, map_location="cpu")
            with open(track_ids_path) as f:
                self.track_ids_list = json.load(f)
            logger.info("Loaded index for %d tracks.", len(self.track_ids_list))
            return

        if not build:
            raise FileNotFoundError(
                "Track indices not found and build=False (inference mode).\n"
                f"  Expected: {cf_bpr_path}\n"
                "  Run training first to build indices."
            )

        logger.info("Building track indices …")
        os.makedirs(_idx_dir, exist_ok=True)
        track_ids = sorted(self.track_embeddings.keys())
        self.track_ids_list = track_ids

        cf_bpr_list   = []
        metadata_list = []

        for tid in track_ids:
            embs = self.track_embeddings[tid]
            cf_bpr_list.append(embs["cf-bpr"])
            meta = embs["metadata-qwen3_embedding_0.6b"]
            metadata_list.append(self._raw_to_tensor(meta.tolist(), 1024))

        self.track_cf_bpr_matrix   = F.normalize(torch.stack(cf_bpr_list),   p=2, dim=1)
        self.track_metadata_matrix = F.normalize(torch.stack(metadata_list), p=2, dim=1)

        torch.save(self.track_cf_bpr_matrix,   cf_bpr_path)
        torch.save(self.track_metadata_matrix, metadata_path)
        with open(track_ids_path, "w") as f:
            json.dump(self.track_ids_list, f)

        logger.info("Track indices saved. Total: %d tracks.", len(track_ids))

    # ── store injection ──────────────────────────────────────────────────────

    def set_turn_store(self, store: dict):
        """Inject pre-computed turn embedding store (query/history per session-turn)."""
        self._turn_store = store
        logger.info("Turn store injected (%d entries).", len(store))

    def set_conv_emb_store(self, store: Dict[str, torch.Tensor]):
        """Inject hist_conversation_embeddings store for ch3.

        Key format: {session_id}_{turn_number}   value: 1024-dim tensor
        """
        self._conv_emb_store = store
        logger.info("Conv-emb store injected (%d entries).", len(store))

    def set_query_split_store(self, store: Dict[str, str]):
        """Inject query_split store for ch5 keyword BM25.

        Key format: {session_id}_{turn_number}   value: JSON string of query-split dict
        """
        self._query_split_store = store
        logger.info("Query-split store injected (%d entries).", len(store))

    # ── Qwen query encoding (fallback) ───────────────────────────────────────

    _QWEN_TASK = "Given a music conversation query, retrieve relevant music tracks"

    @staticmethod
    def _last_token_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        seq_len = attn_mask.sum(dim=1) - 1
        return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), seq_len]

    def _encode_query(self, query: str) -> torch.Tensor:
        """Encode a query to 1024-dim via Qwen (fallback; no cache lookup)."""
        if not query or not query.strip():
            return torch.zeros(1024)
        if not hasattr(self, "_encode_cache"):
            self._encode_cache: Dict[str, torch.Tensor] = {}
        if query in self._encode_cache:
            return self._encode_cache[query]

        text = f"Instruct: {self._QWEN_TASK}\nQuery: {query}"
        self.qwen_model.eval()
        with torch.inference_mode():
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            outputs = self.qwen_model(**inputs)
            emb = self._last_token_pool(outputs.last_hidden_state,
                                        inputs["attention_mask"])   # [1, H]
            emb = F.normalize(emb[:, :1024], p=2, dim=1).squeeze(0).float()

        if len(self._encode_cache) >= 8192:
            self._encode_cache.clear()
        self._encode_cache[query] = emb
        return emb

    # ── Channel 1: CF-BPR (user CF-BPR × track CF-BPR) ──────────────────────

    def _retrieve_cf_bpr(self, user_id: Optional[str], topk: int) -> List[str]:
        """Channel 1: user CF-BPR × track CF-BPR cosine similarity.

        During inference (before three-tower model is used), falls back to
        simple user-CF vs item-CF cosine similarity.
        """
        if (user_id is None
                or user_id not in self.user_embeddings
                or self.user_embeddings[user_id]["__missing__"]):
            return self.track_ids_list[:topk]

        user_bpr = self.user_embeddings[user_id]["cf-bpr"]
        user_bpr = F.normalize(user_bpr.unsqueeze(0), p=2, dim=1)
        scores   = torch.matmul(self.track_cf_bpr_matrix, user_bpr.T).squeeze(1)
        topk     = min(topk, scores.shape[0])
        top_idx  = torch.topk(scores, k=topk).indices.tolist()
        return [self.track_ids_list[i] for i in top_idx]

    # ── Channel 3: QwenMeta (hist_conv_emb 1024d × metadata-qwen3) ──────────

    def _retrieve_query_metadata(
        self,
        query_emb: torch.Tensor,
        topk: int,
    ) -> List[str]:
        """Channel 3: 1024-dim conversation/query emb × metadata-qwen3_embedding_0.6b."""
        if query_emb.shape[0] != self.track_metadata_matrix.shape[1]:
            # Pad or truncate to match index dimension
            dim = self.track_metadata_matrix.shape[1]
            if query_emb.shape[0] > dim:
                query_emb = query_emb[:dim]
            else:
                query_emb = F.pad(query_emb, (0, dim - query_emb.shape[0]))

        q = F.normalize(query_emb.unsqueeze(0), p=2, dim=1).squeeze(0)
        scores  = torch.matmul(self.track_metadata_matrix, q.unsqueeze(1)).squeeze(1)
        topk    = min(topk, scores.shape[0])
        top_idx = torch.topk(scores, k=topk).indices.tolist()
        return [self.track_ids_list[i] for i in top_idx]

    # ── Channel 5: BM25 keyword search ───────────────────────────────────────

    def _stringify_metadata(self, track_id: str) -> str:
        """Convert track metadata to BM25 index text."""
        meta = self.track_metadata_dict.get(track_id, {})
        parts = []
        for field in ["track_name", "artist_name", "album_name", "tag_list", "release_date"]:
            val = meta.get(field, "")
            if isinstance(val, list):
                val = " ".join(str(v) for v in val)
            if val:
                parts.append(str(val))
        return " ".join(parts)

    def _build_bm25_index(self, build: bool = True):
        bm25_dir = os.path.join("qwen", "retrieval_indices", "bm25_index")
        ids_path = os.path.join(bm25_dir, "track_ids.json")
        if os.path.exists(ids_path):
            logger.info("Loading cached BM25 index …")
            self._bm25_model     = bm25s.BM25.load(bm25_dir, load_corpus=True)
            with open(ids_path) as f:
                self._bm25_track_ids = json.load(f)
            logger.info("  BM25 index loaded (%d tracks).", len(self._bm25_track_ids))
            return
        if not build:
            raise FileNotFoundError(
                "BM25 index not found and build=False (inference mode).\n"
                f"  Expected: {ids_path}\n"
                "  Run training first to build the index."
            )
        os.makedirs(bm25_dir, exist_ok=True)
        track_ids = list(self.track_metadata_dict.keys())
        corpus    = [self._stringify_metadata(tid) for tid in track_ids]
        tokens    = bm25s.tokenize(corpus)
        model     = bm25s.BM25()
        model.index(tokens)
        model.save(bm25_dir, corpus=corpus)
        with open(ids_path, "w") as f:
            json.dump(track_ids, f)
        self._bm25_model     = bm25s.BM25.load(bm25_dir, load_corpus=True)
        self._bm25_track_ids = track_ids
        logger.info("BM25 index built (%d tracks).", len(track_ids))

    def _retrieve_bm25_single(self, query: str, topk: int) -> List[str]:
        """BM25 retrieval for a single query string."""
        if not query or not query.strip():
            return []
        tokens  = bm25s.tokenize([query.lower()])
        results = self._bm25_model.retrieve(
            tokens,
            k=min(topk, len(self._bm25_track_ids)),
            return_as="tuple",
        )
        return [self._bm25_track_ids[item["id"]] for item in results.documents[0]]

    def _retrieve_bm25(
        self,
        current_query: str,
        topk: int,
        session_id: Optional[str] = None,
        turn_number: Optional[int] = None,
    ) -> List[str]:
        """Channel 5: keyword-based BM25.

        Strategy:
        1. Try to get query-split keywords from the injected _query_split_store
           (key: {session_id}_{turn_number}).
        2. Extract non-empty fields from the parsed JSON.
        3. Build one keyword string per non-empty field.
        4. Divide `topk` evenly among the non-empty fields.
        5. Merge results (preserve order, deduplicate).
        6. Fall back to raw current_query if no query-split available.
        """
        keyword_strings: List[str] = []

        if (self._query_split_store is not None
                and session_id is not None
                and turn_number is not None):
            key = f"{session_id}_{turn_number}"
            raw = self._query_split_store.get(key, None)
            if raw:
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                    for field in _QUERY_SPLIT_TEXT_FIELDS:
                        val = parsed.get(field)
                        if not val:
                            continue
                        if isinstance(val, list):
                            kw = " ".join(str(v) for v in val if v)
                        else:
                            kw = str(val).strip()
                        if kw:
                            keyword_strings.append(kw)
                except (json.JSONDecodeError, AttributeError):
                    pass

        if not keyword_strings:
            # Fallback: use raw query
            return self._retrieve_bm25_single(current_query, topk)

        # Divide topk evenly; last sub-query gets the remainder
        n = len(keyword_strings)
        per_query = topk // n
        remainder = topk - per_query * n

        seen: Set[str] = set()
        result: List[str] = []
        for i, kw in enumerate(keyword_strings):
            k_i = per_query + (remainder if i == n - 1 else 0)
            for tid in self._retrieve_bm25_single(kw, k_i):
                if tid not in seen:
                    result.append(tid)
                    seen.add(tid)

        return result

    # ── public API ───────────────────────────────────────────────────────────

    def retrieve(
        self,
        user_id: Optional[str],
        current_query: str,
        history_queries: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        turn_number: Optional[int] = None,
    ) -> List[str]:
        """3-channel retrieval; returns deduplicated track list.

        ch1: CF-BPR cosine similarity (100 candidates)
        ch3: hist_conversation_embeddings × metadata-qwen3 (200 candidates)
        ch5: BM25 keyword search from query_split (200 candidates)

        Query embedding for ch3 priority order:
          1. hist_conversation_embeddings (conv_emb_store, key={session_id}_{turn_number})
          2. turn_store query embedding (1024d)
          3. Qwen model fallback
        """
        if history_queries is None:
            history_queries = []

        # ── Resolve 1024-dim query embedding for ch3 ────────────────────────
        query_emb: Optional[torch.Tensor] = None

        # 1. hist_conversation_embeddings store (preferred)
        if (self._conv_emb_store is not None
                and session_id is not None
                and turn_number is not None):
            key = f"{session_id}_{turn_number}"
            if key in self._conv_emb_store:
                raw = self._conv_emb_store[key].float()
                if raw.shape[0] >= 1024:
                    query_emb = raw[:1024]
                else:
                    query_emb = F.pad(raw, (0, 1024 - raw.shape[0]))

        # 2. turn_store fallback
        if query_emb is None:
            store = getattr(self, "_turn_store", None)
            if store is not None and session_id is not None and turn_number is not None:
                q_key  = f"{session_id}_{turn_number}_query"
                h_key  = f"{session_id}_{turn_number}_history"
                if q_key in store:
                    cur = store[q_key].float()[:1024]
                    cur = F.normalize(cur.unsqueeze(0), p=2, dim=1).squeeze(0)
                    if h_key in store:
                        hist = store[h_key].float()[:1024]
                        hist = F.normalize(hist.unsqueeze(0), p=2, dim=1).squeeze(0)
                        query_emb = F.normalize(0.3 * hist + 0.7 * cur, p=2, dim=0)
                    else:
                        query_emb = cur

        # 3. Qwen model fallback
        if query_emb is None:
            if current_query and current_query.strip():
                query_emb = self._encode_query(current_query)
            else:
                query_emb = torch.zeros(1024)

        # ── Run 3 channels ──────────────────────────────────────────────────
        ch1 = self._retrieve_cf_bpr(user_id, USER_MUSIC_TOWER_RECALL_NUM)
        ch3 = self._retrieve_query_metadata(query_emb, QUERY_METADATA_RECALL_NUM)
        ch5 = self._retrieve_bm25(
            current_query,
            BM25_RECALL_NUM,
            session_id=session_id,
            turn_number=turn_number,
        )

        seen: Set[str] = set()
        result: List[str] = []
        for tid in ch1 + ch3 + ch5:
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
