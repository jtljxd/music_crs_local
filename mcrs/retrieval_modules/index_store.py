"""
mcrs/retrieval_modules/index_store.py
=======================================
Unified embedding index for all track modalities.

Loads track embeddings from HuggingFace datasets (or a pre-built .pt cache),
normalises each modality matrix, and exposes fast cosine-topK retrieval via
matrix-vector multiplication on GPU/CPU.

Supported modalities:
    cf_bpr       : track CF-BPR           [N, 128]
    metadata     : metadata-qwen3_emb     [N, 1024]
    lyrics       : lyrics-qwen3_emb       [N, 1024]
    attributes   : attributes-qwen3_emb   [N, 1024]
    audio        : audio-laion_clap       [N, 512]
    image        : image-siglip2          [N, 768? / 1152?]
    tag_bge      : tag_list BGE           [N, 384]  (loaded from bge/ dir)

Usage:
    store = IndexStore.build(
        track_emb_dataset="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        split_types=["all_tracks"],
        cache_dir="qwen/retrieval_indices",
        bge_tag_path="bge/track_tag_embeddings.pt",
        device="cuda",
    )

    top_ids = store.topk("metadata", query_vec_1024, k=200)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)

# ── Column name mapping (HuggingFace dataset column → internal key) ────────────
_COL_MAP: Dict[str, str] = {
    "cf-bpr":                          "cf_bpr",
    "metadata-qwen3_embedding_0.6b":   "metadata",
    "lyrics-qwen3_embedding_0.6b":     "lyrics",
    "attributes-qwen3_embedding_0.6b": "attributes",
    "audio-laion_clap":                "audio",
    "image-siglip2":                   "image",
}

# Canonical dimensions (discovered at load time; these are fallback defaults)
_DEFAULT_DIMS: Dict[str, int] = {
    "cf_bpr":     128,
    "metadata":   1024,
    "lyrics":     1024,
    "attributes": 1024,
    "audio":      512,
    "image":      1152,
    "tag_bge":    384,
    "bge_rich":   384,
}


class IndexStore:
    """Holds normalised embedding matrices for all track modalities.

    Attributes:
        track_ids    : List[str] — ordered track IDs matching matrix rows
        matrices     : Dict[str, Tensor]  — modality → [N, D] fp32 CPU (norm'd)
        _gpu_cache   : Dict[str, Tensor]  — modality → [N, D] on self.device
        device       : str
    """

    def __init__(self, track_ids: List[str], matrices: Dict[str, torch.Tensor],
                 device: str = "cpu"):
        self.track_ids  = track_ids
        self.matrices   = matrices          # CPU fp32, L2-normalised
        self._gpu_cache: Dict[str, torch.Tensor] = {}
        self.device     = device
        self._id_to_idx: Dict[str, int] = {tid: i for i, tid in enumerate(track_ids)}
        logger.info(
            "IndexStore ready: %d tracks, modalities=%s, device=%s",
            len(track_ids), list(matrices.keys()), device,
        )

    # ── public API ─────────────────────────────────────────────────────────────

    def topk(self, modality: str, query_vec: torch.Tensor, k: int) -> List[str]:
        """Return top-k track IDs for a given query vector and modality.

        Args:
            modality  : one of the keys in self.matrices
            query_vec : 1-D float tensor of the correct dimension
            k         : number of results

        Returns:
            List of track_id strings (up to k), highest-score first.
        """
        mat = self._get_matrix(modality)   # [N, D] on device
        q   = F.normalize(query_vec.float().to(self.device).unsqueeze(0), p=2, dim=1)

        # Align dimension
        if q.shape[1] != mat.shape[1]:
            d = mat.shape[1]
            if q.shape[1] > d:
                q = q[:, :d]
            else:
                q = F.pad(q, (0, d - q.shape[1]))

        scores  = torch.matmul(mat, q.T).squeeze(1)  # [N]
        k_      = min(k, scores.shape[0])
        top_idx = torch.topk(scores, k=k_).indices.tolist()
        return [self.track_ids[i] for i in top_idx]

    def topk_from_id(self, modality: str, track_id: str, k: int) -> List[str]:
        """topK neighbours of an existing track (by track_id)."""
        idx = self._id_to_idx.get(track_id)
        if idx is None:
            return []
        mat  = self._get_matrix(modality)
        vec  = mat[idx]                          # already normalised [D]
        scores   = torch.matmul(mat, vec.unsqueeze(1)).squeeze(1)
        k_       = min(k + 1, scores.shape[0])
        top_idx  = torch.topk(scores, k=k_).indices.tolist()
        # Exclude self
        return [self.track_ids[i] for i in top_idx if self.track_ids[i] != track_id][:k]

    def topk_from_ids(self, modality: str, track_ids: List[str], k: int) -> List[str]:
        """topK by mean-pooling a list of track embeddings."""
        idxs = [self._id_to_idx[tid] for tid in track_ids if tid in self._id_to_idx]
        if not idxs:
            return []
        mat    = self._get_matrix(modality)
        vecs   = mat[idxs]                       # [M, D] normalised
        mean_v = F.normalize(vecs.mean(0).unsqueeze(0), p=2, dim=1)
        scores = torch.matmul(mat, mean_v.T).squeeze(1)
        k_     = min(k, scores.shape[0])
        top_idx = torch.topk(scores, k=k_).indices.tolist()
        return [self.track_ids[i] for i in top_idx]

    def has_modality(self, modality: str) -> bool:
        return modality in self.matrices

    def get_vec(self, modality: str, track_id: str) -> Optional[torch.Tensor]:
        """Return the raw (normalised) embedding for a specific track."""
        idx = self._id_to_idx.get(track_id)
        if idx is None:
            return None
        mat = self._get_matrix(modality)
        return mat[idx].cpu()

    # ── internal ───────────────────────────────────────────────────────────────

    def _get_matrix(self, modality: str) -> torch.Tensor:
        if self.device != "cpu":
            if modality not in self._gpu_cache:
                self._gpu_cache[modality] = self.matrices[modality].to(self.device)
            return self._gpu_cache[modality]
        return self.matrices[modality]

    # ── factory ────────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        track_emb_dataset:  str  = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        split_types:        List[str] = None,
        cache_dir:          str  = "qwen/retrieval_indices",
        bge_tag_path:       Optional[str] = "bge/track_tag_embeddings.pt",
        bge_rich_path:      Optional[str] = "bge/track_rich_embeddings.pt",
        device:             str  = "cpu",
        force_rebuild:      bool = False,
    ) -> "IndexStore":
        """Load or build the index store.

        If a cached index already exists under ``cache_dir/track_index/``,
        it is loaded directly (fast path).  Pass ``force_rebuild=True`` to
        recompute from the HuggingFace datasets.
        """
        if split_types is None:
            split_types = ["all_tracks"]

        idx_dir         = os.path.join(cache_dir, "track_index")
        ids_path        = os.path.join(idx_dir, "track_ids.json")
        matrices_path   = os.path.join(idx_dir, "matrices.pt")

        # ── fast path ─────────────────────────────────────────────────────────
        if not force_rebuild and os.path.exists(ids_path) and os.path.exists(matrices_path):
            logger.info("Loading cached index from %s …", idx_dir)
            with open(ids_path) as f:
                track_ids = json.load(f)
            matrices = torch.load(matrices_path, map_location="cpu", weights_only=True)

            # Optionally add BGE tag matrix if not yet in cache
            if (bge_tag_path and os.path.exists(bge_tag_path)
                    and "tag_bge" not in matrices):
                logger.info("Appending tag_bge from %s …", bge_tag_path)
                matrices["tag_bge"] = cls._load_bge_tag_matrix(
                    bge_tag_path, track_ids
                )
                torch.save(matrices, matrices_path)
            # Optionally add BGE rich track matrix if not yet in cache
            if (bge_rich_path and os.path.exists(bge_rich_path)
                    and "bge_rich" not in matrices):
                logger.info("Appending bge_rich from %s …", bge_rich_path)
                matrices["bge_rich"] = cls._load_bge_tag_matrix(
                    bge_rich_path, track_ids
                )
                torch.save(matrices, matrices_path)

            logger.info("Loaded index: %d tracks, modalities=%s",
                        len(track_ids), list(matrices.keys()))
            return cls(track_ids, matrices, device)

        # ── build from scratch ────────────────────────────────────────────────
        logger.info("Building track index from %s …", track_emb_dataset)
        os.makedirs(idx_dir, exist_ok=True)

        ds = load_dataset(track_emb_dataset)
        valid_splits = [s for s in split_types if s in ds] or list(ds.keys())
        concat_ds    = concatenate_datasets([ds[s] for s in valid_splits])
        logger.info("  %d tracks in dataset.", len(concat_ds))

        # Discover column dims from first few rows
        col_dims: Dict[str, int] = {}
        for item in concat_ds:
            for hf_col, key in _COL_MAP.items():
                if key in col_dims:
                    continue
                v = item.get(hf_col)
                if v is None:
                    continue
                try:
                    t = torch.tensor(v, dtype=torch.float32)
                    if t.ndim == 1 and t.numel() > 0:
                        col_dims[key] = t.shape[0]
                except Exception:
                    pass
            if len(col_dims) == len(_COL_MAP):
                break

        for key in _COL_MAP.values():
            if key not in col_dims:
                col_dims[key] = _DEFAULT_DIMS[key]
                logger.warning("Modality '%s' not found; using default dim %d.",
                               key, col_dims[key])

        logger.info("Discovered dims: %s", col_dims)

        # Accumulate lists
        track_ids: List[str] = []
        raw: Dict[str, List[torch.Tensor]] = {k: [] for k in _COL_MAP.values()}

        for item in concat_ds:
            tid = str(item["track_id"])
            track_ids.append(tid)
            for hf_col, key in _COL_MAP.items():
                v = item.get(hf_col)
                t = cls._raw_to_tensor(v, col_dims[key])
                raw[key].append(t)

        # Stack and L2-normalise
        matrices: Dict[str, torch.Tensor] = {}
        for key, vecs in raw.items():
            mat = torch.stack(vecs)                          # [N, D]
            mat = F.normalize(mat.float(), p=2, dim=1)
            matrices[key] = mat
            logger.info("  %-12s  shape=%s", key, mat.shape)

        # Add BGE tag matrix
        if bge_tag_path and os.path.exists(bge_tag_path):
            matrices["tag_bge"] = cls._load_bge_tag_matrix(bge_tag_path, track_ids)
            logger.info("  %-12s  shape=%s", "tag_bge", matrices["tag_bge"].shape)

        # Add BGE rich track matrix
        if bge_rich_path and os.path.exists(bge_rich_path):
            matrices["bge_rich"] = cls._load_bge_tag_matrix(bge_rich_path, track_ids)
            logger.info("  %-12s  shape=%s", "bge_rich", matrices["bge_rich"].shape)

        # Save
        with open(ids_path, "w") as f:
            json.dump(track_ids, f)
        torch.save(matrices, matrices_path)
        logger.info("Index saved to %s  (%d tracks).", idx_dir, len(track_ids))

        return cls(track_ids, matrices, device)

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _raw_to_tensor(value, dim: int) -> torch.Tensor:
        if value is None:
            return torch.zeros(dim, dtype=torch.float32)
        try:
            t = torch.tensor(value, dtype=torch.float32)
            if t.ndim == 0 or t.numel() == 0:
                return torch.zeros(dim, dtype=torch.float32)
            t = t.flatten()
            if t.shape[0] < dim:
                t = F.pad(t, (0, dim - t.shape[0]))
            elif t.shape[0] > dim:
                t = t[:dim]
            return t
        except (TypeError, ValueError):
            return torch.zeros(dim, dtype=torch.float32)

    @staticmethod
    def _load_bge_tag_matrix(
        bge_tag_path: str, track_ids: List[str]
    ) -> torch.Tensor:
        """Build [N, 384] BGE tag matrix aligned with track_ids."""
        bge_store: Dict[str, torch.Tensor] = torch.load(
            bge_tag_path, map_location="cpu", weights_only=True
        )
        dim = 384
        vecs = []
        missing = 0
        for tid in track_ids:
            v = bge_store.get(tid)
            if v is not None:
                t = v.float()
                if t.shape[0] != dim:
                    t = F.pad(t, (0, dim - t.shape[0])) if t.shape[0] < dim else t[:dim]
                vecs.append(t)
            else:
                vecs.append(torch.zeros(dim))
                missing += 1
        if missing:
            logger.warning("tag_bge: %d / %d tracks missing → zero-filled.", missing, len(track_ids))
        mat = torch.stack(vecs)
        return F.normalize(mat.float(), p=2, dim=1)
