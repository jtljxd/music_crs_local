"""Qwen text embedding pre-computation and lookup cache.

Usage
-----
1. Pre-compute (offline, before inference):
   cache = QwenEmbeddingCache(qwen_model_path, cache_file="cache/qwen_text_emb.pt")
   cache.precompute(texts, batch_size=64)   # texts: List[str]

2. Lookup (online, during inference):
   cache = QwenEmbeddingCache(qwen_model_path, cache_file="cache/qwen_text_emb.pt")
   emb = cache.get("some text", dim=256)    # returns torch.Tensor [dim]

The cache file maps text -> float16 tensor of shape [1024] (full Qwen output).
At lookup time, the caller can request any prefix dim (128 / 256 / etc.).
"""

import os
import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

_ZERO_CACHE: Dict[int, torch.Tensor] = {}


def _zero(dim: int) -> torch.Tensor:
    if dim not in _ZERO_CACHE:
        _ZERO_CACHE[dim] = torch.zeros(dim)
    return _ZERO_CACHE[dim]


class QwenEmbeddingCache:
    """Pre-computed text → embedding lookup table backed by a .pt file.

    * Encoding dimension: full Qwen output (typically 1024).  Stored as fp16
      to save ~50 % disk / RAM vs fp32.
    * Lookup: caller specifies a prefix dim; L2-normalisation is applied on-the-fly.
    * Any text not found in the cache is encoded on-the-fly and added to the
      cache (with a warning so you know you missed pre-computing it).
    """

    STORE_DIM = 1024   # full Qwen hidden size; we'll truncate at lookup time

    def __init__(
        self,
        qwen_model_path: str,
        cache_file: str = "cache/qwen_text_emb.pt",
        device: str = "cpu",      # always CPU for Qwen
    ):
        self.qwen_model_path = qwen_model_path
        self.cache_file      = cache_file
        self.device          = device

        # Lazy-load the Qwen model (only when actually needed for encoding)
        self._tokenizer  = None
        self._qwen_model = None

        # text -> float16 tensor [STORE_DIM]
        self._table: Dict[str, torch.Tensor] = {}
        self._load_cache()

    # ── persistence ─────────────────────────────────────────────────────────

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            logger.info("Loading Qwen embedding cache from %s …", self.cache_file)
            self._table = torch.load(self.cache_file, map_location="cpu")
            logger.info("  %d entries loaded.", len(self._table))
        else:
            logger.info("Qwen embedding cache not found at %s; will encode on-the-fly.", self.cache_file)

    def save(self):
        os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
        torch.save(self._table, self.cache_file)
        logger.info("Qwen embedding cache saved: %d entries → %s", len(self._table), self.cache_file)

    # ── model lazy-load ──────────────────────────────────────────────────────

    def _ensure_model(self):
        if self._qwen_model is None:
            logger.info("Loading Qwen model for embedding … (device=cpu)")
            self._tokenizer  = AutoTokenizer.from_pretrained(self.qwen_model_path)
            self._qwen_model = AutoModel.from_pretrained(self.qwen_model_path).cpu().eval()

    # ── core encode ─────────────────────────────────────────────────────────

    def _encode_batch_raw(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of texts; returns [N, STORE_DIM] float32 on CPU."""
        self._ensure_model()
        with torch.no_grad():
            inputs  = self._tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            outputs = self._qwen_model(**inputs)
            attn    = inputs["attention_mask"]
            tok_emb = outputs.last_hidden_state
            mask    = attn.unsqueeze(-1).expand(tok_emb.size()).float()
            emb     = torch.sum(tok_emb * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            # pad / truncate to STORE_DIM
            if emb.shape[1] < self.STORE_DIM:
                emb = F.pad(emb, (0, self.STORE_DIM - emb.shape[1]))
            else:
                emb = emb[:, :self.STORE_DIM]
        return emb.float()

    # ── public API ───────────────────────────────────────────────────────────

    def precompute(
        self,
        texts: List[str],
        batch_size: int = 64,
        save_every: int = 1000,
    ):
        """Encode all texts not yet in the cache, then save.

        Args:
            texts:       Flat list of strings (duplicates are fine; deduplicated internally)
            batch_size:  Qwen batch size (tune to your CPU RAM / speed tradeoff)
            save_every:  Save checkpoint every N newly encoded texts
        """
        # Deduplicate and filter out empty strings and already-cached texts
        unique = list(dict.fromkeys(
            t for t in texts if t and t.strip() and t not in self._table
        ))
        logger.info("Precomputing: %d / %d unique texts need encoding.", len(unique), len(texts))

        if not unique:
            logger.info("All texts already cached — nothing to encode.")
            return

        newly_encoded = 0
        for start in tqdm(range(0, len(unique), batch_size), desc="Qwen precompute"):
            batch = unique[start: start + batch_size]
            embs  = self._encode_batch_raw(batch)   # [K, STORE_DIM]
            for text, vec in zip(batch, embs):
                self._table[text] = vec.half()       # store as fp16 to save RAM
            newly_encoded += len(batch)
            if newly_encoded % save_every == 0:
                self.save()

        self.save()
        logger.info("Precompute complete. Cache now has %d entries.", len(self._table))

    def get(self, text: str, dim: int = 256, normalize: bool = True) -> torch.Tensor:
        """Look up embedding for a single text.

        * Returns zero-vector for empty / whitespace-only strings.
        * On cache miss: encodes on-the-fly, adds to cache (no auto-save).
        """
        if not text or not text.strip():
            return _zero(dim)

        if text not in self._table:
            logger.warning("Cache miss for text (len=%d); encoding on-the-fly.", len(text))
            vec = self._encode_batch_raw([text])[0]   # [STORE_DIM]
            self._table[text] = vec.half()

        vec = self._table[text].float()[:dim]          # [dim]
        if normalize:
            vec = F.normalize(vec.unsqueeze(0), p=2, dim=1).squeeze(0)
        return vec

    def get_batch(
        self, texts: List[str], dim: int = 256, normalize: bool = True
    ) -> List[torch.Tensor]:
        """Look up embeddings for a list of texts."""
        return [self.get(t, dim=dim, normalize=normalize) for t in texts]

    def __len__(self):
        return len(self._table)

    def __contains__(self, text: str):
        return text in self._table
