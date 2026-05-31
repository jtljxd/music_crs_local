"""Three-tower reranking model for music recommendation.

Architecture:
1. Intent Tower: Processes query, history context, listener goal, conversation goal
2. Item Tower: Multi-modal fusion with gating mechanism (audio, image, cf-bpr, attributes, lyrics, metadata)
3. User Tower: User profile and CF-BPR embedding

Missing embedding policy
------------------------
All missing / empty embeddings are stored as **zero vectors** of the canonical
dimension.  In addition, a boolean ``missing_mask`` is propagated into the
model at forward time:

* ``ItemTower``: ``modal_missing_mask`` [batch, num_modals]
  - For any modality marked as missing, its gating score is forced to 0
    and the remaining gates are re-normalised, so the tower learns to
    reconstruct the item representation from available modalities only.

* ``UserTower``: ``cf_bpr_missing`` [batch]
  - When the user cf-bpr is absent, the zero-vector is passed through as
    usual but a learnable scalar ``missing_scale`` (initialised to 0) is
    applied, letting the model learn what to do in that case.

Final score: MLP([intent_repr, item_repr, user_repr])
"""

import os
import logging
import math
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


# Embedding dimensions configuration
EMB_DIMS = {
    "user_id": 32,
    "track_id": 32,
    "year": 8,
    "month": 4,
    "is_workday": 2,
    "age": 8,
    "country_code": 16,
    "gender": 2,
    "preferred_language": 4,
    "preferred_musical_culture": 32,
    "category": 8,
    "specificity": 16,
    "ISRC": 32,
    "tag_list": 32,
    "artist_id": 32,
    "album_id": 32,
    "duration_bucket": 8,
    "release_year_bucket": 8,
}


def log_bucket(value: float, add_one: bool = True) -> int:
    """Log bucketing for numerical features."""
    if add_one:
        value = value + 1
    if value <= 0:
        return 0
    return int(math.log(value))


def duration_bucket(duration_ms: float) -> int:
    """Bucket duration into 8 categories."""
    if duration_ms < 30000:
        return 0
    elif duration_ms < 60000:
        return 1
    elif duration_ms < 120000:
        return 2
    elif duration_ms < 180000:
        return 3
    elif duration_ms < 210000:
        return 4
    elif duration_ms < 240000:
        return 5
    elif duration_ms < 360000:
        return 6
    else:
        return 7


def release_year_bucket(release_date: str) -> int:
    """Bucket release year into categories."""
    try:
        year = int(str(release_date)[:4])
    except Exception:
        return 10  # Unknown

    if year < 1950:
        return 0
    elif year < 1960:
        return 1
    elif year < 1970:
        return 2
    elif year < 1980:
        return 3
    elif year < 1985:
        return 4
    elif year < 1990:
        return 5
    elif year < 1995:
        return 6
    elif year < 2000:
        return 7
    elif year < 2005:
        return 8
    elif year < 2010:
        return 9
    elif year < 2015:
        return 10
    elif year < 2020:
        return 11
    elif year < 2025:
        return 12
    elif year < 2030:
        return 13
    else:
        return 14


class IntentTower(nn.Module):
    """Intent tower: processes query, history, and conversation goal."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim]
        Returns:
            [batch, output_dim]
        """
        return self.mlp(x)


class ItemTower(nn.Module):
    """Item tower: multi-modal fusion with gating + missing-mask support."""

    def __init__(
        self,
        intent_dim: int = 128,
        modal_dim: int = 128,
        num_modals: int = 6,
        hidden_dim: int = 256,
        output_dim: int = 128,
        vocab_sizes: Dict[str, int] = None,
    ):
        super().__init__()
        self.intent_dim  = intent_dim
        self.modal_dim   = modal_dim
        self.num_modals  = num_modals

        # Gating network: one per modality
        self.gate_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(intent_dim + modal_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_modals)
        ])

        # Embedding layers for categorical features
        self.vocab_sizes = vocab_sizes or {}
        self.embeddings  = nn.ModuleDict()
        for feat_name, vocab_size in self.vocab_sizes.items():
            emb_dim = EMB_DIMS.get(feat_name, 32)
            self.embeddings[feat_name] = nn.Embedding(vocab_size, emb_dim)

        total_emb_dim = sum(EMB_DIMS.get(k, 32) for k in self.vocab_sizes.keys())
        fusion_dim    = modal_dim + total_emb_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        intent_repr: torch.Tensor,
        modal_embs: List[torch.Tensor],
        categorical_features: Dict[str, torch.Tensor],
        modal_missing_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            intent_repr:         [batch, intent_dim]
            modal_embs:          List of num_modals × [batch, modal_dim]
            categorical_features: Dict[feat_name → [batch] LongTensor]
            modal_missing_mask:  [batch, num_modals] bool tensor;
                                 True  = this modality is missing for this sample
                                 None  = no modalities are missing
        Returns:
            [batch, output_dim]
        """
        # ── compute raw gates ──────────────────────────────────────────────
        raw_gates = []
        for i, modal_emb in enumerate(modal_embs):
            gate_input = torch.cat([intent_repr, modal_emb], dim=1)  # [B, I+M]
            gate = self.gate_networks[i](gate_input)                  # [B, 1]
            raw_gates.append(gate)
        gates = torch.cat(raw_gates, dim=1)  # [B, num_modals]

        # ── apply missing mask to gates ────────────────────────────────────
        # Zero-out gate scores for missing modalities so they contribute
        # nothing to the weighted sum; re-normalise over present modalities.
        if modal_missing_mask is not None:
            # mask: True → missing  →  set gate to 0
            gates = gates.masked_fill(modal_missing_mask, 0.0)

        # Normalise: add tiny epsilon to avoid div-by-zero when all missing
        gates_sum = gates.sum(dim=1, keepdim=True).clamp(min=1e-9)
        gates = gates / gates_sum                                     # [B, num_modals]

        # ── weighted fusion ────────────────────────────────────────────────
        fused_modal = sum(
            gates[:, i:i+1] * modal_embs[i]
            for i in range(self.num_modals)
        )  # [B, modal_dim]

        # ── categorical features ───────────────────────────────────────────
        cat_embs = []
        for feat_name, feat_values in categorical_features.items():
            if feat_name in self.embeddings:
                cat_embs.append(self.embeddings[feat_name](feat_values))

        if cat_embs:
            item_features = torch.cat([fused_modal] + cat_embs, dim=1)
        else:
            item_features = fused_modal

        return self.fusion_mlp(item_features)


class UserTower(nn.Module):
    """User tower: processes user profile and CF-BPR embedding.

    When ``cf_bpr_missing`` is provided and True for a sample, the zero-vector
    is kept as input but the tower learns a separate scaling factor
    ``missing_scale`` (initialised near 0) so it can decide how much to trust
    the absent embedding during training.
    """

    def __init__(
        self,
        cf_bpr_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 128,
        vocab_sizes: Dict[str, int] = None,
    ):
        super().__init__()

        self.vocab_sizes = vocab_sizes or {}
        self.embeddings  = nn.ModuleDict()
        for feat_name, vocab_size in self.vocab_sizes.items():
            emb_dim = EMB_DIMS.get(feat_name, 32)
            self.embeddings[feat_name] = nn.Embedding(vocab_size, emb_dim)

        total_emb_dim = sum(EMB_DIMS.get(k, 32) for k in self.vocab_sizes.keys())
        input_dim     = cf_bpr_dim + total_emb_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

        # Learnable scalar applied to zero-filled cf-bpr when embedding is
        # missing.  Initialised to 0 so the model starts by ignoring missing
        # inputs and gradually learns the best strategy.
        self.missing_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        cf_bpr: torch.Tensor,
        categorical_features: Dict[str, torch.Tensor],
        cf_bpr_missing: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            cf_bpr:              [batch, cf_bpr_dim]   (zeros where missing)
            categorical_features: Dict[feat_name → [batch]]
            cf_bpr_missing:      [batch] bool tensor;
                                 True = cf-bpr was absent for this sample
        Returns:
            [batch, output_dim]
        """
        if cf_bpr_missing is not None and cf_bpr_missing.any():
            # Scale the zero-vectors for missing samples by a learnable factor.
            # Present samples are unaffected (scale = 1).
            scale = torch.where(
                cf_bpr_missing.unsqueeze(1),
                self.missing_scale.expand_as(cf_bpr),
                torch.ones_like(cf_bpr),
            )
            cf_bpr = cf_bpr * scale

        cat_embs = []
        for feat_name, feat_values in categorical_features.items():
            if feat_name in self.embeddings:
                cat_embs.append(self.embeddings[feat_name](feat_values))

        if cat_embs:
            user_features = torch.cat([cf_bpr] + cat_embs, dim=1)
        else:
            user_features = cf_bpr

        return self.mlp(user_features)


class ThreeTowerReranker(nn.Module):
    """Three-tower reranking model with missing-mask support."""

    def __init__(
        self,
        intent_input_dim: int = 512,
        tower_output_dim: int = 128,
        item_vocab_sizes: Dict[str, int] = None,
        user_vocab_sizes: Dict[str, int] = None,
    ):
        super().__init__()

        self.intent_tower = IntentTower(
            input_dim=intent_input_dim,
            hidden_dim=256,
            output_dim=tower_output_dim,
        )
        self.item_tower = ItemTower(
            intent_dim=tower_output_dim,
            modal_dim=128,
            num_modals=6,
            hidden_dim=256,
            output_dim=tower_output_dim,
            vocab_sizes=item_vocab_sizes,
        )
        self.user_tower = UserTower(
            cf_bpr_dim=128,
            hidden_dim=128,
            output_dim=tower_output_dim,
            vocab_sizes=user_vocab_sizes,
        )

        final_input_dim = tower_output_dim * 3
        self.final_mlp = nn.Sequential(
            nn.Linear(final_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        intent_features: torch.Tensor,
        modal_embs: List[torch.Tensor],
        item_categorical: Dict[str, torch.Tensor],
        user_cf_bpr: torch.Tensor,
        user_categorical: Dict[str, torch.Tensor],
        modal_missing_mask: Optional[torch.Tensor] = None,
        cf_bpr_missing: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            intent_features:    [batch, intent_input_dim]
            modal_embs:         List of 6 × [batch, 128]  (zero-filled where absent)
            item_categorical:   Dict of item categorical features
            user_cf_bpr:        [batch, 128]               (zero-filled where absent)
            user_categorical:   Dict of user categorical features
            modal_missing_mask: [batch, 6] bool  – True = modality absent
            cf_bpr_missing:     [batch] bool      – True = user cf-bpr absent
        Returns:
            [batch] predicted scores
        """
        intent_repr = self.intent_tower(intent_features)
        item_repr   = self.item_tower(
            intent_repr, modal_embs, item_categorical,
            modal_missing_mask=modal_missing_mask,
        )
        user_repr   = self.user_tower(
            user_cf_bpr, user_categorical,
            cf_bpr_missing=cf_bpr_missing,
        )

        combined = torch.cat([intent_repr, item_repr, user_repr], dim=1)
        return self.final_mlp(combined).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Wrapper  (data loading + inference)
# ═══════════════════════════════════════════════════════════════════════════════

class ThreeTowerRerankerWrapper:
    """Wrapper for three-tower reranker with data loading and inference.

    Missing embedding policy
    ------------------------
    * All missing embeddings are zero-filled at *load time*.
    * A ``__missing__`` list is stored alongside each record so that
      ``rerank()`` can build the boolean masks passed to the model.
    * The model itself (``ItemTower`` / ``UserTower``) uses those masks to
      handle absent modalities gracefully during both training and inference.
    """

    # ── class-level column / dim constants ────────────────────────────────
    TRACK_MODAL_COLS = [
        "audio-laion_clap",
        "image-siglip2",
        "cf-bpr",
        "attributes-qwen3_embedding_0.6b",
        "lyrics-qwen3_embedding_0.6b",
        "metadata-qwen3_embedding_0.6b",
    ]
    TRACK_MODAL_TARGET_DIM = 128
    USER_CF_BPR_TARGET_DIM = 128

    def __init__(
        self,
        dataset_name: str,
        track_emb_db_name: str,
        user_emb_db_name: str,
        track_metadata_db_name: str,
        user_metadata_db_name: str,
        split_types: List[str],
        cache_dir: str = "./cache",
        qwen_model_path: str = "/home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B",
        device: str = "cuda",
        lr: float = 1e-3,
        qwen_model=None,       # pre-loaded shared instance (optional)
        qwen_tokenizer=None,   # pre-loaded shared instance (optional)
    ):
        self.device          = device
        self.cache_dir       = cache_dir
        self.qwen_model_path = qwen_model_path

        self.model_dir = os.path.join(cache_dir, "three_tower_reranker")
        os.makedirs(self.model_dir, exist_ok=True)

        logger.info("Loading datasets …")
        self.track_embeddings = self._load_track_embeddings(track_emb_db_name, split_types)
        self.user_embeddings  = self._load_user_embeddings(user_emb_db_name, split_types)
        self.track_metadata   = self._load_track_metadata(track_metadata_db_name, split_types)
        self.user_metadata    = self._load_user_metadata(user_metadata_db_name, split_types)

        logger.info("Building vocabularies …")
        self.item_vocabs, self.user_vocabs = self._build_vocabularies()

        # Use shared Qwen model if provided; otherwise load on CPU
        if qwen_model is not None and qwen_tokenizer is not None:
            logger.info("Using shared Qwen model (CPU).")
            self.tokenizer  = qwen_tokenizer
            self.qwen_model = qwen_model
        else:
            logger.info("Loading Qwen model on CPU …")
            self.tokenizer  = AutoTokenizer.from_pretrained(qwen_model_path)
            self.qwen_model = AutoModel.from_pretrained(qwen_model_path).cpu().eval()

        logger.info("Initializing three-tower model …")
        self.model = ThreeTowerReranker(
            intent_input_dim=512,
            tower_output_dim=128,
            # Categorical features not fed at inference time yet
            item_vocab_sizes={},
            user_vocab_sizes={},
        ).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self._load_checkpoint()

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _raw_to_tensor(value, target_dim: int) -> Tuple[torch.Tensor, bool]:
        """Convert a raw dataset column value to a [target_dim] float32 tensor.

        Returns
        -------
        (tensor, is_missing)
        """
        if value is None:
            return torch.zeros(target_dim, dtype=torch.float32), True
        try:
            t = torch.tensor(value, dtype=torch.float32)
            if t.ndim == 0 or t.numel() == 0:
                return torch.zeros(target_dim, dtype=torch.float32), True
            if t.ndim > 1:
                t = t.flatten()
            if t.shape[0] > target_dim:
                t = t[:target_dim]
            elif t.shape[0] < target_dim:
                t = F.pad(t, (0, target_dim - t.shape[0]))
            return t, False
        except (TypeError, ValueError):
            return torch.zeros(target_dim, dtype=torch.float32), True

    # ── data loaders ────────────────────────────────────────────────────────

    def _load_track_embeddings(self, dataset_name: str, split_types: List[str]) -> Dict:
        """Load all track modal embeddings.

        Each entry ``embeddings[track_id]`` contains:
          * One key per modality → float32 tensor [128]  (zero-filled if absent)
          * ``"__missing__"``    → List[str] of columns that were absent/empty
        """
        ds = load_dataset(dataset_name)
        valid_splits = [s for s in split_types if s in ds.keys()] or list(ds.keys())
        concat_ds    = concatenate_datasets([ds[s] for s in valid_splits])

        embeddings: Dict     = {}
        missing_counts: Dict = {col: 0 for col in self.TRACK_MODAL_COLS}

        for item in concat_ds:
            track_id     = item["track_id"]
            row: Dict    = {}
            missing_cols = []
            for col in self.TRACK_MODAL_COLS:
                t, is_missing = self._raw_to_tensor(
                    item.get(col), self.TRACK_MODAL_TARGET_DIM
                )
                row[col] = t
                if is_missing:
                    missing_cols.append(col)
                    missing_counts[col] += 1
            row["__missing__"]  = missing_cols
            embeddings[track_id] = row

        for col, cnt in missing_counts.items():
            if cnt:
                logger.warning(
                    "[Reranker] track '%s': %d/%d missing → zero-filled + mask.",
                    col, cnt, len(embeddings),
                )
        logger.info("[Reranker] Loaded %d tracks.", len(embeddings))
        return embeddings

    def _load_user_embeddings(self, dataset_name: str, split_types: List[str]) -> Dict:
        """Load user cf-bpr embeddings (zero-fill + missing flag)."""
        ds = load_dataset(dataset_name)
        valid_splits = [s for s in split_types if s in ds.keys()] or list(ds.keys())
        concat_ds    = concatenate_datasets([ds[s] for s in valid_splits])

        embeddings: Dict = {}
        missing = 0
        for item in concat_ds:
            user_id = item["user_id"]
            t, is_missing = self._raw_to_tensor(
                item.get("cf-bpr"), self.USER_CF_BPR_TARGET_DIM
            )
            if is_missing:
                missing += 1
            embeddings[user_id] = {
                "cf-bpr":      t,
                "__missing__": ["cf-bpr"] if is_missing else [],
            }

        if missing:
            logger.warning(
                "[Reranker] user cf-bpr: %d/%d missing → zero-filled + mask.",
                missing, len(embeddings),
            )
        logger.info("[Reranker] Loaded %d users.", len(embeddings))
        return embeddings

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

    def _build_vocabularies(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        item_vocabs = {
            "track_id":           len(self.track_metadata) + 1,
            "ISRC":               10000,
            "tag_list":           5000,
            "artist_id":          5000,
            "album_id":           10000,
            "duration_bucket":    8,
            "release_year_bucket": 15,
        }
        user_vocabs = {
            "user_id":                   len(self.user_metadata) + 1,
            "age":                       10,
            "country_code":              200,
            "gender":                    5,
            "preferred_language":        50,
            "preferred_musical_culture": 100,
            "year":                      10,
            "month":                     12,
            "is_workday":                2,
            "category":                  20,
            "specificity":               10,
        }
        return item_vocabs, user_vocabs

    # ── text encoding ────────────────────────────────────────────────────────

    def set_turn_store(self, store: dict):
        """Inject a pre-computed turn embedding store.

        store: dict  key='{session_id}__{turn}_{role}'  value=float16 [128]
        """
        self._turn_store = store
        logger.info("Turn store injected into ThreeTowerRerankerWrapper (%d entries).", len(store))

    # kept for legacy / fallback
    def set_embedding_cache(self, cache):
        """Inject a QwenEmbeddingCache (legacy; prefer set_turn_store)."""
        self._qwen_cache = cache
        logger.info("QwenEmbeddingCache injected into ThreeTowerRerankerWrapper (%d entries).", len(cache))

    def _get_turn_emb(self, session_id: Optional[str], turn_number: Optional[int],
                      role: str, fallback_text: str, dim: int = 128) -> torch.Tensor:
        """Look up turn store; fall back to text encoding if not found."""
        store = getattr(self, "_turn_store", None)
        if store is not None and session_id and turn_number is not None:
            key = f"{session_id}__{turn_number}_{role}"
            if key in store:
                vec = store[key].float()[:dim]
                return F.normalize(vec.unsqueeze(0), p=2, dim=1).squeeze(0)
        return self._encode_text(fallback_text, max_dim=dim)

    # ── inference ───────────────────────────────────────────────────────────

    def rerank(
        self,
        user_id: Optional[str],
        candidate_track_ids: List[str],
        current_query: str,
        history_context: str = "",
        conversation_goal: Dict = None,
        session_date: str = "",
        session_id: Optional[str] = None,
        turn_number: Optional[int] = None,
    ) -> List[str]:
        """Rerank candidates; returns track IDs sorted best-first.

        When session_id + turn_number are provided and a turn store is loaded,
        all Qwen calls are replaced by O(1) dict lookups.
        """
        if not candidate_track_ids:
            return candidate_track_ids

        N = len(candidate_track_ids)
        self.model.eval()
        with torch.no_grad():
            # ── intent embeddings via turn store or text encode ───────────
            query_emb        = self._get_turn_emb(session_id, turn_number, "user",
                                                  current_query, dim=128)
            history_emb      = self._get_turn_emb(session_id, turn_number, "history_avg",
                                                  history_context, dim=128)
            listener_goal    = conversation_goal.get("listener_goal", "") if conversation_goal else ""
            category         = conversation_goal.get("category",      "") if conversation_goal else ""
            listener_goal_emb = self._encode_text(listener_goal, max_dim=128)
            category_emb      = self._encode_text(category,      max_dim=128)

            intent_features = (
                torch.cat([query_emb, history_emb, listener_goal_emb, category_emb])
                .unsqueeze(0).repeat(N, 1).to(self.device)
            )  # [N, 512]

            # ── user features + cf_bpr_missing mask ──────────────────────
            if user_id and user_id in self.user_embeddings:
                user_data      = self.user_embeddings[user_id]
                raw_bpr        = user_data["cf-bpr"].to(self.device)
                is_bpr_missing = "cf-bpr" in user_data["__missing__"]
            else:
                raw_bpr        = torch.zeros(self.USER_CF_BPR_TARGET_DIM, device=self.device)
                is_bpr_missing = True

            user_cf_bpr    = raw_bpr.unsqueeze(0).repeat(N, 1)           # [N, 128]
            cf_bpr_missing = torch.tensor(
                [is_bpr_missing] * N, dtype=torch.bool, device=self.device
            )  # [N]

            user_categorical = {}

            # ── item modal embeddings + modal_missing_mask ───────────────
            modal_lists: List[List[torch.Tensor]] = [[] for _ in self.TRACK_MODAL_COLS]
            missing_flags: List[List[bool]]        = [[] for _ in self.TRACK_MODAL_COLS]

            for track_id in candidate_track_ids:
                if track_id in self.track_embeddings:
                    embs    = self.track_embeddings[track_id]
                    missing = set(embs["__missing__"])
                else:
                    embs    = {}
                    missing = set(self.TRACK_MODAL_COLS)

                for i, col in enumerate(self.TRACK_MODAL_COLS):
                    if col in missing or col not in embs:
                        modal_lists[i].append(torch.zeros(self.TRACK_MODAL_TARGET_DIM))
                        missing_flags[i].append(True)
                    else:
                        modal_lists[i].append(embs[col])
                        missing_flags[i].append(False)

            modal_embs = [torch.stack(lst).to(self.device) for lst in modal_lists]

            modal_missing_mask = torch.tensor(
                list(zip(*missing_flags)),
                dtype=torch.bool,
                device=self.device,
            )  # [N, num_modals]

            item_categorical = {}

            # ── forward ──────────────────────────────────────────────────
            scores = self.model(
                intent_features,
                modal_embs,
                item_categorical,
                user_cf_bpr,
                user_categorical,
                modal_missing_mask=modal_missing_mask,
                cf_bpr_missing=cf_bpr_missing,
            )

            sorted_indices = torch.argsort(scores, descending=True).cpu().tolist()
            return [candidate_track_ids[i] for i in sorted_indices]

    def batch_rerank(
        self,
        user_ids: List[Optional[str]],
        batch_candidate_track_ids: List[List[str]],
        current_queries: List[str],
        history_contexts: List[str] = None,
        conversation_goals: List[Dict] = None,
        session_dates: List[str] = None,
        session_ids: Optional[List[Optional[str]]] = None,
        turn_numbers: Optional[List[Optional[int]]] = None,
    ) -> List[List[str]]:
        if history_contexts   is None: history_contexts   = [""] * len(user_ids)
        if conversation_goals is None: conversation_goals = [None] * len(user_ids)
        if session_dates      is None: session_dates      = [""] * len(user_ids)
        if session_ids        is None: session_ids        = [None] * len(user_ids)
        if turn_numbers       is None: turn_numbers       = [None] * len(user_ids)

        return [
            self.rerank(uid, cands, q, hist, goal, date, sid, tn)
            for uid, cands, q, hist, goal, date, sid, tn in zip(
                user_ids, batch_candidate_track_ids,
                current_queries, history_contexts,
                conversation_goals, session_dates,
                session_ids, turn_numbers,
            )
        ]

    # ── checkpoint ──────────────────────────────────────────────────────────

    def _load_checkpoint(self):
        ckpt_path = os.path.join(self.model_dir, "model.pt")
        if os.path.exists(ckpt_path):
            logger.info("Loading checkpoint from %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            logger.info("Checkpoint loaded.")
        else:
            logger.info("No checkpoint found; starting from scratch.")

    def save_checkpoint(self):
        ckpt_path = os.path.join(self.model_dir, "model.pt")
        torch.save(
            {
                "model_state_dict":      self.model.state_dict(),
                "optimizer_state_dict":  self.optimizer.state_dict(),
            },
            ckpt_path,
        )
        logger.info("Checkpoint saved to %s", ckpt_path)
