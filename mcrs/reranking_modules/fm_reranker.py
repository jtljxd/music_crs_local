"""Factorization Machine (FM) based reranker for music track candidates.

This module implements a second-stage reranker that takes Top-K candidates from
the retrieval stage and reranks them using a Factorization Machine model.

Track embedding columns (TalkPlayData-Challenge-Track-Embeddings):
  - audio-laion_clap          : audio embedding from LAION-CLAP
  - image-siglip2             : image/album-art embedding from SigLIP-2
  - cf-bpr                    : collaborative-filtering BPR embedding
  - attributes-qwen3_embedding_0.6b : attribute text embedding
  - lyrics-qwen3_embedding_0.6b     : lyrics text embedding
  - metadata-qwen3_embedding_0.6b   : metadata text embedding

User embedding columns (TalkPlayData-Challenge-User-Embeddings):
  - cf-bpr                    : collaborative-filtering BPR embedding

All available embedding columns are concatenated to form the FM feature vector,
together with metadata scalar features and cross features.

The FM model is trained using BPR loss. In inference mode the FM scores
are used to rerank the Top-K retrieval candidates and the top-1 is selected.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column definitions (matches exact HuggingFace dataset column names)
# ---------------------------------------------------------------------------

# All embedding columns present in the Track-Embeddings dataset
TRACK_EMB_COLS: List[str] = [
    "audio-laion_clap",
    "image-siglip2",
    "cf-bpr",
    "attributes-qwen3_embedding_0.6b",
    "lyrics-qwen3_embedding_0.6b",
    "metadata-qwen3_embedding_0.6b",
]

# All embedding columns present in the User-Embeddings dataset
USER_EMB_COLS: List[str] = [
    "cf-bpr",
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_float(value, default: float = 0.0) -> float:
    """Convert an arbitrary value to float, returning ``default`` on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _release_year(release_date) -> float:
    """Extract the year from a date string like ``'2021-03-15'`` or ``'2021'``."""
    if not release_date:
        return 2000.0
    try:
        return float(str(release_date)[:4])
    except (ValueError, TypeError):
        return 2000.0


def _col_to_tensor(row: dict, col: str) -> Optional[torch.Tensor]:
    """Safely convert a dataset column value to a float32 tensor.

    Returns ``None`` if the column is absent or the value is ``None``/empty.
    """
    value = row.get(col)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.float()
    try:
        t = torch.tensor(value, dtype=torch.float32)
        if t.numel() == 0:
            return None
        return t
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# FM model
# ---------------------------------------------------------------------------

class FactorizationMachine(nn.Module):
    """A simple Factorization Machine operating on a concatenated feature vector.

    Architecture::

        score = bias + <w, x>
                + 0.5 * (||Σ_i x_i v_i||² - Σ_i x_i² ||v_i||²)

    Input ``x`` has shape ``[batch, input_dim]``; output is ``[batch]``.
    """

    def __init__(self, input_dim: int, k: int = 16) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Linear(input_dim, 1, bias=False)
        self.embedding = nn.Parameter(torch.empty(input_dim, k))
        nn.init.normal_(self.embedding, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear(x).squeeze(-1)          # [batch]
        xv = torch.matmul(x, self.embedding)              # [batch, k]
        square_of_sum = xv.pow(2).sum(dim=1)              # [batch]
        sum_of_square = torch.matmul(x.pow(2), self.embedding.pow(2)).sum(dim=1)
        interaction = 0.5 * (square_of_sum - sum_of_square)
        return self.bias + linear_out + interaction       # [batch]


# ---------------------------------------------------------------------------
# Multi-embedding store
# ---------------------------------------------------------------------------

class _EmbeddingStore:
    """Holds per-column embeddings loaded from a HuggingFace dataset.

    Attributes:
        data: ``{id_str: {col: tensor}}``
        col_dims: ``{col: dim}`` – dimension of each present column.
        cols: ordered list of columns that were actually found in the dataset.
    """

    def __init__(
        self,
        dataset_name: str,
        split_types: List[str],
        id_col: str,
        emb_cols: List[str],
    ) -> None:
        self.id_col = id_col
        self.requested_cols = emb_cols
        self.data: Dict[str, Dict[str, torch.Tensor]] = {}
        self.col_dims: Dict[str, int] = {}
        self.cols: List[str] = []      # columns that exist in the dataset

        ds = load_dataset(dataset_name)
        avail_splits = list(ds.keys())

        # Resolve which splits to actually load:
        # use requested split_types that exist; fall back to ALL available splits.
        valid_splits = [s for s in split_types if s in avail_splits]
        if not valid_splits:
            logger.warning(
                "None of the requested splits %s found in '%s' (available: %s). "
                "Falling back to all available splits: %s",
                split_types, dataset_name, avail_splits, avail_splits,
            )
            valid_splits = avail_splits
        elif len(valid_splits) < len(split_types):
            skipped = [s for s in split_types if s not in avail_splits]
            logger.warning(
                "Splits not found in '%s', skipping: %s (available: %s)",
                dataset_name, skipped, avail_splits,
            )

        combined = concatenate_datasets([ds[s] for s in valid_splits])
        avail_cols = combined.column_names
        logger.info(
            "Dataset '%s' columns: %s", dataset_name, avail_cols
        )

        # Only keep requested columns that actually exist
        self.cols = [c for c in emb_cols if c in avail_cols]
        missing = [c for c in emb_cols if c not in avail_cols]
        if missing:
            logger.warning(
                "Columns not found in '%s', will be skipped: %s",
                dataset_name, missing,
            )

        for row in combined:
            eid = str(row[id_col])
            row_embs: Dict[str, torch.Tensor] = {}
            for col in self.cols:
                t = _col_to_tensor(row, col)
                if t is not None:
                    row_embs[col] = t
                    if col not in self.col_dims:
                        self.col_dims[col] = t.shape[0]
            self.data[eid] = row_embs

        logger.info(
            "Loaded %d entries from '%s'; active columns: %s",
            len(self.data), dataset_name, self.cols,
        )

    def total_dim(self) -> int:
        """Total concatenated embedding dimension across all active columns."""
        return sum(self.col_dims.get(c, 0) for c in self.cols)

    def get_concat(self, eid: str) -> torch.Tensor:
        """Return the concatenated embedding for ``eid``, zero-padded if missing."""
        row_embs = self.data.get(eid, {})
        parts: List[torch.Tensor] = []
        for col in self.cols:
            dim = self.col_dims.get(col, 0)
            if dim == 0:
                continue
            parts.append(row_embs.get(col, torch.zeros(dim)))
        if not parts:
            return torch.zeros(0)
        return torch.cat(parts, dim=0)

    def get_col(self, eid: str, col: str) -> Optional[torch.Tensor]:
        """Return a single-column embedding or ``None``."""
        return self.data.get(eid, {}).get(col)


# ---------------------------------------------------------------------------
# FMReranker
# ---------------------------------------------------------------------------

class FMReranker:
    """Second-stage FM reranker using all available track and user embeddings.

    Track embedding columns loaded (``TalkPlayData-Challenge-Track-Embeddings``):
      ``audio-laion_clap``, ``image-siglip2``, ``cf-bpr``,
      ``attributes-qwen3_embedding_0.6b``, ``lyrics-qwen3_embedding_0.6b``,
      ``metadata-qwen3_embedding_0.6b``

    User embedding columns loaded (``TalkPlayData-Challenge-User-Embeddings``):
      ``cf-bpr``

    FM feature vector for one ``(user, track)`` pair::

        [
          user_cf_bpr          (user cf-bpr embedding)
          track_audio_clap     (track audio-laion_clap)
          track_image_siglip2  (track image-siglip2)
          track_cf_bpr         (track cf-bpr)
          track_attr_qwen      (track attributes-qwen3_embedding_0.6b)
          track_lyrics_qwen    (track lyrics-qwen3_embedding_0.6b)
          track_meta_qwen      (track metadata-qwen3_embedding_0.6b)
          ui_bpr_cosine        (cosine(user cf-bpr, track cf-bpr))
          retrieval_rank_norm  (0=best, normalised to [0,1])
          popularity_norm
          duration_norm
          release_year_norm
          tag_multihot         (top-K tags)
        ]

    Training uses BPR loss; checkpoints are cached to ``cache_dir``.
    """

    # Metadata fields
    TAG_FIELD = "tag_list"
    _POP_MAX  = 100.0
    _DUR_MAX  = 600.0
    _YEAR_MIN = 1950.0
    _YEAR_MAX = 2030.0

    def __init__(
        self,
        track_emb_dataset_name: str,
        user_emb_dataset_name: str,
        track_metadata_dict: Dict[str, Dict],
        track_split_types: List[str],
        user_split_types: List[str],
        fm_k: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        cache_dir: str = "./cache",
        device: Optional[str] = None,
        top_tags: int = 50,
    ) -> None:
        """
        Args:
            track_emb_dataset_name: HF dataset for track embeddings.
            user_emb_dataset_name:  HF dataset for user embeddings.
            track_metadata_dict:    Pre-loaded metadata dict (track_id -> row).
            track_split_types:      Splits to load from the track embedding dataset.
            user_split_types:       Splits to load from the user embedding dataset.
            fm_k:                   Latent factor dimension for the FM.
            lr:                     Adam learning rate.
            weight_decay:           L2 regularisation coefficient.
            cache_dir:              Directory for embedding cache and checkpoints.
            device:                 Torch device; auto-selected if ``None``.
            top_tags:               Vocabulary size for tag multi-hot feature.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.track_metadata_dict = track_metadata_dict
        self.cache_dir = cache_dir
        self.top_tags = top_tags
        os.makedirs(cache_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # Load track embeddings (all 6 columns)
        # ------------------------------------------------------------------
        logger.info("Loading track embeddings from '%s' ...", track_emb_dataset_name)
        self.track_store = _EmbeddingStore(
            track_emb_dataset_name, track_split_types,
            id_col="track_id", emb_cols=TRACK_EMB_COLS,
        )
        logger.info(
            "Track store: %d entries, total dim=%d, cols=%s",
            len(self.track_store.data),
            self.track_store.total_dim(),
            self.track_store.cols,
        )

        # ------------------------------------------------------------------
        # Load user embeddings (cf-bpr column)
        # ------------------------------------------------------------------
        logger.info("Loading user embeddings from '%s' ...", user_emb_dataset_name)
        self.user_store = _EmbeddingStore(
            user_emb_dataset_name, user_split_types,
            id_col="user_id", emb_cols=USER_EMB_COLS,
        )
        logger.info(
            "User store: %d entries, total dim=%d, cols=%s",
            len(self.user_store.data),
            self.user_store.total_dim(),
            self.user_store.cols,
        )

        # ------------------------------------------------------------------
        # Tag vocabulary
        # ------------------------------------------------------------------
        self.tag_vocab = self._build_tag_vocab(top_tags)
        logger.info("Tag vocabulary size: %d", len(self.tag_vocab))

        # ------------------------------------------------------------------
        # FM input dimension
        # ------------------------------------------------------------------
        # user concat emb
        self.user_emb_dim  = self.user_store.total_dim()
        # track concat emb
        self.track_emb_dim = self.track_store.total_dim()
        # user-item BPR cosine similarity (1 scalar, computed from cf-bpr cols)
        self._has_bpr_cosine = (
            "cf-bpr" in self.user_store.col_dims and
            "cf-bpr" in self.track_store.col_dims
        )
        input_dim = (
            self.user_emb_dim     # all user embeddings concatenated
            + self.track_emb_dim  # all track embeddings concatenated
            + (1 if self._has_bpr_cosine else 0)  # user-item BPR cosine
            + 1                   # retrieval rank normalised
            + 1                   # popularity
            + 1                   # duration
            + 1                   # release year
            + len(self.tag_vocab) # tag multi-hot
        )
        self.input_dim = input_dim
        logger.info(
            "FM input_dim=%d  (user=%d, track=%d, bpr_cosine=%s, scalars=4, tags=%d)",
            input_dim, self.user_emb_dim, self.track_emb_dim,
            self._has_bpr_cosine, len(self.tag_vocab),
        )

        # ------------------------------------------------------------------
        # FM model + optimiser
        # ------------------------------------------------------------------
        self.fm = FactorizationMachine(input_dim, k=fm_k).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.fm.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Restore checkpoint if available
        self._ckpt_path = os.path.join(cache_dir, "fm_reranker.pt")
        if os.path.exists(self._ckpt_path):
            self._load_checkpoint()

    # ------------------------------------------------------------------
    # Tag vocabulary
    # ------------------------------------------------------------------

    def _build_tag_vocab(self, top_k: int) -> Dict[str, int]:
        """Build a vocabulary from the top-K most frequent tags."""
        tag_counts: Dict[str, int] = {}
        for meta in self.track_metadata_dict.values():
            tags = meta.get(self.TAG_FIELD, []) or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        sorted_tags = sorted(tag_counts, key=lambda t: -tag_counts[t])[:top_k]
        return {tag: idx for idx, tag in enumerate(sorted_tags)}

    # ------------------------------------------------------------------
    # Feature construction helpers
    # ------------------------------------------------------------------

    def _zero_user_emb(self) -> torch.Tensor:
        return torch.zeros(self.user_emb_dim)

    def _zero_track_emb(self) -> torch.Tensor:
        return torch.zeros(self.track_emb_dim)

    def _get_user_emb(self, user_id: Optional[str]) -> torch.Tensor:
        """Concatenated user embedding across all loaded user columns."""
        if user_id:
            emb = self.user_store.get_concat(user_id)
            if emb.numel() == self.user_emb_dim:
                return emb
        return self._zero_user_emb()

    def _get_track_emb(self, track_id: str) -> torch.Tensor:
        """Concatenated track embedding across all loaded track columns."""
        emb = self.track_store.get_concat(track_id)
        if emb.numel() == self.track_emb_dim:
            return emb
        return self._zero_track_emb()

    def _bpr_cosine(self, user_id: Optional[str], track_id: str) -> torch.Tensor:
        """Cosine similarity between user cf-bpr and track cf-bpr vectors."""
        if not self._has_bpr_cosine:
            return torch.zeros(0)
        u = self.user_store.get_col(user_id or "", "cf-bpr")
        t = self.track_store.get_col(track_id, "cf-bpr")
        if u is None:
            u = torch.zeros(self.user_store.col_dims["cf-bpr"])
        if t is None:
            t = torch.zeros(self.track_store.col_dims["cf-bpr"])
        u_norm = F.normalize(u.unsqueeze(0), p=2, dim=1).squeeze(0)
        t_norm = F.normalize(t.unsqueeze(0), p=2, dim=1).squeeze(0)
        return torch.dot(u_norm, t_norm).unsqueeze(0)  # [1]

    def _tag_multihot(self, meta: Dict) -> torch.Tensor:
        vec = torch.zeros(len(self.tag_vocab))
        if not self.tag_vocab:
            return vec
        tags = meta.get(self.TAG_FIELD, []) or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        for tag in tags:
            if tag in self.tag_vocab:
                vec[self.tag_vocab[tag]] = 1.0
        return vec

    def _build_feature(
        self,
        user_id: Optional[str],
        track_id: str,
        retrieval_rank: int,
        retrieval_topk: int,
    ) -> torch.Tensor:
        """Construct the FM input feature vector for one (user, track) pair.

        Feature layout::

            [user_concat_emb | track_concat_emb | bpr_cosine (opt) |
             rank_norm | popularity | duration | year | tag_multihot]

        Args:
            user_id:          User identifier (``None`` for anonymous).
            track_id:         Track identifier.
            retrieval_rank:   0-based rank from the retrieval stage.
            retrieval_topk:   Total number of candidates (for normalisation).

        Returns:
            Float32 tensor of shape ``[input_dim]``.
        """
        user_emb  = self._get_user_emb(user_id)        # [U]
        track_emb = self._get_track_emb(track_id)      # [T]
        bpr_cos   = self._bpr_cosine(user_id, track_id) # [1] or []

        rank_norm = torch.tensor(
            [retrieval_rank / max(retrieval_topk - 1, 1)], dtype=torch.float32
        )

        meta = self.track_metadata_dict.get(track_id, {})
        popularity = torch.tensor(
            [min(_safe_float(meta.get("popularity", 0)), self._POP_MAX) / self._POP_MAX],
            dtype=torch.float32,
        )
        duration = torch.tensor(
            [min(_safe_float(meta.get("duration", 0)), self._DUR_MAX) / self._DUR_MAX],
            dtype=torch.float32,
        )
        year_norm = torch.tensor(
            [(_release_year(meta.get("release_date", "")) - self._YEAR_MIN)
             / (self._YEAR_MAX - self._YEAR_MIN)],
            dtype=torch.float32,
        )
        tag_vec = self._tag_multihot(meta)              # [|tag_vocab|]

        parts = [user_emb, track_emb]
        if bpr_cos.numel() > 0:
            parts.append(bpr_cos)
        parts += [rank_norm, popularity, duration, year_norm, tag_vec]
        return torch.cat(parts, dim=0)

    def _build_feature_batch(
        self,
        user_id: Optional[str],
        track_ids: List[str],
    ) -> torch.Tensor:
        """Build features for all candidates of a single user query.

        Returns:
            ``[len(track_ids), input_dim]`` float32 tensor.
        """
        topk = len(track_ids)
        return torch.stack(
            [self._build_feature(user_id, tid, rank, topk)
             for rank, tid in enumerate(track_ids)],
            dim=0,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def rerank(
        self,
        user_id: Optional[str],
        candidate_track_ids: List[str],
    ) -> List[str]:
        """Rerank candidates using FM scores.

        Args:
            user_id:               User identifier (may be ``None``).
            candidate_track_ids:   Ordered list of track IDs from retrieval.

        Returns:
            Reranked list (highest FM score first).
        """
        if not candidate_track_ids:
            return candidate_track_ids
        self.fm.eval()
        with torch.no_grad():
            feat = self._build_feature_batch(user_id, candidate_track_ids).to(self.device)
            scores = self.fm(feat)  # [topk]
        return [candidate_track_ids[i]
                for i in torch.argsort(scores, descending=True).cpu().tolist()]

    def batch_rerank(
        self,
        user_ids: List[Optional[str]],
        batch_candidate_track_ids: List[List[str]],
    ) -> List[List[str]]:
        """Rerank for a batch of (user, candidates) pairs."""
        return [
            self.rerank(uid, cands)
            for uid, cands in zip(user_ids, batch_candidate_track_ids)
        ]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit_on_batch(
        self,
        user_ids: List[Optional[str]],
        positive_track_ids: List[str],
        negative_track_ids_list: List[List[str]],
    ) -> float:
        """Train the FM for one batch using BPR loss.

        For each sample::

            loss = -mean( log σ(score_pos - score_neg) )

        Args:
            user_ids:                  User identifiers.
            positive_track_ids:        Ground-truth track IDs (one per sample).
            negative_track_ids_list:   Lists of negative track IDs per sample.

        Returns:
            Mean BPR loss over the batch.
        """
        self.fm.train()
        total_loss = 0.0
        count = 0

        for uid, pos_tid, neg_tids in zip(
            user_ids, positive_track_ids, negative_track_ids_list
        ):
            topk = 1 + len(neg_tids)
            pos_feat = (
                self._build_feature(uid, pos_tid, 0, topk)
                .unsqueeze(0).to(self.device)
            )
            pos_score = self.fm(pos_feat).squeeze()   # scalar

            if neg_tids:
                neg_feats = torch.stack(
                    [self._build_feature(uid, ntid, r + 1, topk)
                     for r, ntid in enumerate(neg_tids)],
                    dim=0,
                ).to(self.device)
                neg_scores = self.fm(neg_feats)        # [num_neg]
                loss = -F.logsigmoid(pos_score - neg_scores).mean()
            else:
                # No negatives – skip
                continue

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            count += 1

        return total_loss / max(count, 1)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self) -> None:
        """Persist FM weights, optimiser state, and tag vocab to disk."""
        torch.save(
            {
                "fm_state_dict": self.fm.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "tag_vocab": self.tag_vocab,
                "input_dim": self.input_dim,
            },
            self._ckpt_path,
        )
        logger.info("FM checkpoint saved to %s", self._ckpt_path)

    def _load_checkpoint(self) -> None:
        """Restore FM weights from disk if architecture matches."""
        ckpt = torch.load(self._ckpt_path, map_location=self.device)
        if ckpt.get("input_dim") == self.input_dim:
            self.fm.load_state_dict(ckpt["fm_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "tag_vocab" in ckpt:
                self.tag_vocab = ckpt["tag_vocab"]
            logger.info("FM checkpoint loaded from %s", self._ckpt_path)
        else:
            logger.warning(
                "FM checkpoint dim mismatch (saved=%s, current=%d); "
                "ignoring checkpoint and starting fresh.",
                ckpt.get("input_dim"), self.input_dim,
            )
