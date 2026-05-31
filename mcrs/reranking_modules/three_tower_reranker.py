"""Three-tower reranking model for music recommendation.

Architecture:
1. Intent Tower: Processes query, history context, listener goal, conversation goal
2. Item Tower: Multi-modal fusion with gating mechanism (audio, image, cf-bpr, attributes, lyrics, metadata)
3. User Tower: User profile and CF-BPR embedding

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
    except:
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
            x: [batch, input_dim] - concatenated [query_emb, history_emb, listener_goal_emb, conversation_goal_emb]
        Returns:
            [batch, output_dim] - intent representation
        """
        return self.mlp(x)


class ItemTower(nn.Module):
    """Item tower: multi-modal fusion with gating mechanism."""
    
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
        self.intent_dim = intent_dim
        self.modal_dim = modal_dim
        self.num_modals = num_modals
        
        # Gating networks for each modality
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
        self.embeddings = nn.ModuleDict()
        for feat_name, vocab_size in self.vocab_sizes.items():
            emb_dim = EMB_DIMS.get(feat_name, 32)
            self.embeddings[feat_name] = nn.Embedding(vocab_size, emb_dim)
        
        # Calculate total feature dimension
        total_emb_dim = sum(EMB_DIMS.get(k, 32) for k in self.vocab_sizes.keys())
        fusion_dim = modal_dim + total_emb_dim
        
        # Fusion MLP
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
    ) -> torch.Tensor:
        """
        Args:
            intent_repr: [batch, intent_dim]
            modal_embs: List of [batch, modal_dim] tensors (6 modalities)
            categorical_features: Dict of feature_name -> [batch] tensor
        Returns:
            [batch, output_dim] - item representation
        """
        batch_size = intent_repr.size(0)
        
        # Compute gates for each modality
        gates = []
        for i, modal_emb in enumerate(modal_embs):
            gate_input = torch.cat([intent_repr, modal_emb], dim=1)  # [batch, intent_dim + modal_dim]
            gate = self.gate_networks[i](gate_input)  # [batch, 1]
            gates.append(gate)
        
        # Weighted fusion of modalities
        gates = torch.cat(gates, dim=1)  # [batch, num_modals]
        gates = F.softmax(gates, dim=1)  # Normalize gates
        
        # Apply gates
        weighted_modals = []
        for i, modal_emb in enumerate(modal_embs):
            weighted_modals.append(gates[:, i:i+1] * modal_emb)
        
        fused_modal = sum(weighted_modals)  # [batch, modal_dim]
        
        # Process categorical features
        cat_embs = []
        for feat_name, feat_values in categorical_features.items():
            if feat_name in self.embeddings:
                emb = self.embeddings[feat_name](feat_values)  # [batch, emb_dim]
                cat_embs.append(emb)
        
        if cat_embs:
            cat_emb = torch.cat(cat_embs, dim=1)  # [batch, total_emb_dim]
            item_features = torch.cat([fused_modal, cat_emb], dim=1)
        else:
            item_features = fused_modal
        
        # Final MLP
        return self.fusion_mlp(item_features)


class UserTower(nn.Module):
    """User tower: processes user profile and CF-BPR embedding."""
    
    def __init__(
        self,
        cf_bpr_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 128,
        vocab_sizes: Dict[str, int] = None,
    ):
        super().__init__()
        
        # Embedding layers for user features
        self.vocab_sizes = vocab_sizes or {}
        self.embeddings = nn.ModuleDict()
        for feat_name, vocab_size in self.vocab_sizes.items():
            emb_dim = EMB_DIMS.get(feat_name, 32)
            self.embeddings[feat_name] = nn.Embedding(vocab_size, emb_dim)
        
        # Calculate total dimension
        total_emb_dim = sum(EMB_DIMS.get(k, 32) for k in self.vocab_sizes.keys())
        input_dim = cf_bpr_dim + total_emb_dim
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(
        self,
        cf_bpr: torch.Tensor,
        categorical_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            cf_bpr: [batch, cf_bpr_dim]
            categorical_features: Dict of feature_name -> [batch] tensor
        Returns:
            [batch, output_dim] - user representation
        """
        # Process categorical features
        cat_embs = []
        for feat_name, feat_values in categorical_features.items():
            if feat_name in self.embeddings:
                emb = self.embeddings[feat_name](feat_values)
                cat_embs.append(emb)
        
        if cat_embs:
            cat_emb = torch.cat(cat_embs, dim=1)
            user_features = torch.cat([cf_bpr, cat_emb], dim=1)
        else:
            user_features = cf_bpr
        
        return self.mlp(user_features)


class ThreeTowerReranker(nn.Module):
    """Three-tower reranking model."""
    
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
        
        # Final scoring MLP
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
    ) -> torch.Tensor:
        """
        Args:
            intent_features: [batch, intent_input_dim]
            modal_embs: List of 6 [batch, 128] tensors
            item_categorical: Dict of item features
            user_cf_bpr: [batch, 128]
            user_categorical: Dict of user features
        Returns:
            [batch] - predicted scores
        """
        intent_repr = self.intent_tower(intent_features)
        item_repr = self.item_tower(intent_repr, modal_embs, item_categorical)
        user_repr = self.user_tower(user_cf_bpr, user_categorical)
        
        # Concatenate all representations
        combined = torch.cat([intent_repr, item_repr, user_repr], dim=1)
        
        # Final score
        scores = self.final_mlp(combined).squeeze(-1)
        return scores


class ThreeTowerRerankerWrapper:
    """Wrapper for three-tower reranker with data loading and training."""
    
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
    ):
        """Initialize the reranker wrapper.
        
        Args:
            dataset_name: Training dataset with conversations and assessments
            track_emb_db_name: Track embeddings dataset
            user_emb_db_name: User embeddings dataset
            track_metadata_db_name: Track metadata dataset
            user_metadata_db_name: User metadata dataset
            split_types: Dataset splits to load
            cache_dir: Cache directory
            qwen_model_path: Path to Qwen model
            device: Compute device
            lr: Learning rate
        """
        self.device = device
        self.cache_dir = cache_dir
        self.qwen_model_path = qwen_model_path
        
        # Create cache directory
        self.model_dir = os.path.join(cache_dir, "three_tower_reranker")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load datasets
        logger.info("Loading datasets...")
        self.track_embeddings = self._load_track_embeddings(track_emb_db_name, split_types)
        self.user_embeddings = self._load_user_embeddings(user_emb_db_name, split_types)
        self.track_metadata = self._load_track_metadata(track_metadata_db_name, split_types)
        self.user_metadata = self._load_user_metadata(user_metadata_db_name, split_types)
        
        # Build vocabularies
        logger.info("Building vocabularies...")
        self.item_vocabs, self.user_vocabs = self._build_vocabularies()
        
        # Load Qwen model for encoding
        logger.info("Loading Qwen model...")
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
        self.qwen_model = AutoModel.from_pretrained(qwen_model_path)
        self.qwen_model.to(device).eval()
        
        # Initialize model
        logger.info("Initializing three-tower model...")
        self.model = ThreeTowerReranker(
            intent_input_dim=512,  # query(128) + history(128) + listener_goal(128) + conversation_goal(128)
            tower_output_dim=128,
            item_vocab_sizes=self.item_vocabs,
            user_vocab_sizes=self.user_vocabs,
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Try to load checkpoint
        self._load_checkpoint()
    
    def _load_track_embeddings(self, dataset_name: str, split_types: List[str]) -> Dict:
        """Load track embeddings."""
        ds = load_dataset(dataset_name)
        valid_splits = [s for s in split_types if s in ds.keys()] or list(ds.keys())
        concat_ds = concatenate_datasets([ds[s] for s in valid_splits])
        
        embeddings = {}
        for item in concat_ds:
            track_id = item["track_id"]
            embeddings[track_id] = {
                "cf-bpr": torch.tensor(item["cf-bpr"], dtype=torch.float32),
                "audio-laion_clap": torch.tensor(item["audio-laion_clap"][:128], dtype=torch.float32),
                "image-siglip2": torch.tensor(item["image-siglip2"][:128], dtype=torch.float32),
                "attributes-qwen3_embedding_0.6b": torch.tensor(item["attributes-qwen3_embedding_0.6b"][:128], dtype=torch.float32),
                "lyrics-qwen3_embedding_0.6b": torch.tensor(item["lyrics-qwen3_embedding_0.6b"][:128], dtype=torch.float32),
                "metadata-qwen3_embedding_0.6b": torch.tensor(item["metadata-qwen3_embedding_0.6b"][:128], dtype=torch.float32),
            }
        return embeddings
    
    def _load_user_embeddings(self, dataset_name: str, split_types: List[str]) -> Dict:
        """Load user embeddings."""
        ds = load_dataset(dataset_name)
        valid_splits = [s for s in split_types if s in ds.keys()] or list(ds.keys())
        concat_ds = concatenate_datasets([ds[s] for s in valid_splits])
        
        embeddings = {}
        for item in concat_ds:
            user_id = item["user_id"]
            embeddings[user_id] = {
                "cf-bpr": torch.tensor(item["cf-bpr"], dtype=torch.float32),
            }
        return embeddings
    
    def _load_track_metadata(self, dataset_name: str, split_types: List[str]) -> Dict:
        """Load track metadata."""
        ds = load_dataset(dataset_name)
        valid_splits = [s for s in split_types if s in ds.keys()] or list(ds.keys())
        concat_ds = concatenate_datasets([ds[s] for s in valid_splits])
        return {item["track_id"]: item for item in concat_ds}
    
    def _load_user_metadata(self, dataset_name: str, split_types: List[str]) -> Dict:
        """Load user metadata."""
        ds = load_dataset(dataset_name)
        valid_splits = [s for s in split_types if s in ds.keys()] or list(ds.keys())
        concat_ds = concatenate_datasets([ds[s] for s in valid_splits])
        return {item["user_id"]: item for item in concat_ds}
    
    def _build_vocabularies(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Build vocabularies for categorical features."""
        # Item vocabularies
        item_vocabs = {
            "track_id": len(self.track_metadata) + 1,
            "ISRC": 10000,  # Approximate
            "tag_list": 5000,  # Approximate
            "artist_id": 5000,  # Approximate
            "album_id": 10000,  # Approximate
            "duration_bucket": 8,
            "release_year_bucket": 15,
        }
        
        # User vocabularies
        user_vocabs = {
            "user_id": len(self.user_metadata) + 1,
            "age": 10,
            "country_code": 200,
            "gender": 5,
            "preferred_language": 50,
            "preferred_musical_culture": 100,
            "year": 10,
            "month": 12,
            "is_workday": 2,
            "category": 20,
            "specificity": 10,
        }
        
        return item_vocabs, user_vocabs
    
    def _encode_text(self, text: str, max_dim: int = 128) -> torch.Tensor:
        """Encode text using Qwen model."""
        self.qwen_model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.qwen_model(**inputs)
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask  # [1, 1024]
            # Take first max_dim dims
            embeddings = embeddings[:, :max_dim]
            return embeddings.cpu().squeeze(0)  # [max_dim]
    
    def rerank(
        self,
        user_id: Optional[str],
        candidate_track_ids: List[str],
        current_query: str,
        history_context: str = "",
        conversation_goal: Dict = None,
        session_date: str = "",
    ) -> List[str]:
        """Rerank candidates using the three-tower model.
        
        Args:
            user_id: User identifier
            candidate_track_ids: List of candidate track IDs
            current_query: Current user query
            history_context: Chat history as string
            conversation_goal: Conversation goal dict
            session_date: Session date string
            
        Returns:
            Reranked list of track IDs
        """
        if not candidate_track_ids:
            return candidate_track_ids
        
        self.model.eval()
        with torch.no_grad():
            # Encode query and history
            query_emb = self._encode_text(current_query, 128)
            history_emb = self._encode_text(history_context, 128) if history_context else torch.zeros(128)
            
            # Encode conversation goal
            if conversation_goal:
                listener_goal = conversation_goal.get("listener_goal", "")
                listener_goal_emb = self._encode_text(listener_goal, 128)
                category = conversation_goal.get("category", "")
                category_emb = self._encode_text(category, 128)
            else:
                listener_goal_emb = torch.zeros(128)
                category_emb = torch.zeros(128)
            
            # Concatenate intent features
            intent_features = torch.cat([query_emb, history_emb, listener_goal_emb, category_emb], dim=0)  # [512]
            intent_features = intent_features.unsqueeze(0).repeat(len(candidate_track_ids), 1).to(self.device)  # [N, 512]
            
            # Get user features
            if user_id and user_id in self.user_embeddings:
                user_cf_bpr = self.user_embeddings[user_id]["cf-bpr"].unsqueeze(0).repeat(len(candidate_track_ids), 1).to(self.device)
            else:
                user_cf_bpr = torch.zeros(len(candidate_track_ids), 128).to(self.device)
            
            # User categorical features (simplified)
            user_categorical = {}
            
            # Get item features
            modal_embs = []
            for _ in range(6):
                modal_embs.append([])
            
            for track_id in candidate_track_ids:
                if track_id in self.track_embeddings:
                    embs = self.track_embeddings[track_id]
                    modal_embs[0].append(embs["audio-laion_clap"])
                    modal_embs[1].append(embs["image-siglip2"])
                    modal_embs[2].append(embs["cf-bpr"])
                    modal_embs[3].append(embs["attributes-qwen3_embedding_0.6b"])
                    modal_embs[4].append(embs["lyrics-qwen3_embedding_0.6b"])
                    modal_embs[5].append(embs["metadata-qwen3_embedding_0.6b"])
                else:
                    for i in range(6):
                        modal_embs[i].append(torch.zeros(128))
            
            # Stack modal embeddings
            modal_embs = [torch.stack(embs).to(self.device) for embs in modal_embs]
            
            # Item categorical features (simplified)
            item_categorical = {}
            
            # Forward pass
            scores = self.model(
                intent_features,
                modal_embs,
                item_categorical,
                user_cf_bpr,
                user_categorical,
            )
            
            # Sort by scores
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
    ) -> List[List[str]]:
        """Batch reranking."""
        if history_contexts is None:
            history_contexts = [""] * len(user_ids)
        if conversation_goals is None:
            conversation_goals = [None] * len(user_ids)
        if session_dates is None:
            session_dates = [""] * len(user_ids)
        
        results = []
        for i in range(len(user_ids)):
            reranked = self.rerank(
                user_ids[i],
                batch_candidate_track_ids[i],
                current_queries[i],
                history_contexts[i],
                conversation_goals[i],
                session_dates[i],
            )
            results.append(reranked)
        
        return results
    
    def _load_checkpoint(self):
        """Load model checkpoint if exists."""
        ckpt_path = os.path.join(self.model_dir, "model.pt")
        if os.path.exists(ckpt_path):
            logger.info(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Checkpoint loaded successfully")
        else:
            logger.info("No checkpoint found, starting from scratch")
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        ckpt_path = os.path.join(self.model_dir, "model.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, ckpt_path)
        logger.info(f"Checkpoint saved to {ckpt_path}")