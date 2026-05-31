"""Multi-channel retrieval module for music recommendation.

This module implements a comprehensive retrieval strategy combining:
1. User CF-BPR embedding similarity (user-music tower)
2. Collaborative filtering based on similar users' liked music
3. Query-metadata semantic similarity
4. Query-attributes semantic similarity

All embeddings are pre-computed and cached for efficiency.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict

import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


# Configuration constants
USER_HISTORY_LIKE_MUSIC_NUM = 5
USER_MUSIC_TOWER_RECALL_NUM = 100
QUERY_METADATA_RECALL_NUM = 100
QUERY_ATTRIBUTES_RECALL_NUM = 100


class MultiChannelRetrieval:
    """Multi-channel retrieval combining CF, collaborative filtering, and semantic search.
    
    Retrieval channels:
    1. CF-BPR user-music tower: cosine similarity between user cf-bpr and track cf-bpr
    2. Similar users' liked music: find similar users via user cf_emb, get their liked tracks
    3. Query-metadata: semantic similarity between query and track metadata
    4. Query-attributes: semantic similarity between query and track attributes
    """
    
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
    ):
        """Initialize multi-channel retrieval.
        
        Args:
            dataset_name: Dataset containing conversation history and goal assessments
            item_db_name: Track metadata dataset
            user_db_name: User metadata dataset
            track_emb_db_name: Track embeddings dataset
            user_emb_db_name: User embeddings dataset
            split_types: Dataset splits to load
            cache_dir: Cache directory for embeddings
            qwen_model_path: Path to Qwen3-Embedding-0.6B model
            device: Compute device
            batch_size: Batch size for embedding computation
        """
        self.dataset_name = dataset_name
        self.item_db_name = item_db_name
        self.cache_dir = cache_dir
        self.qwen_model_path = qwen_model_path
        self.device = device
        self.batch_size = batch_size
        self.split_types = split_types
        
        # Create cache directories
        self.index_dir = os.path.join(cache_dir, "multi_channel_retrieval")
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Load datasets
        logger.info("Loading datasets...")
        self.track_metadata_dict = self._load_track_metadata(item_db_name, split_types)
        self.user_metadata_dict = self._load_user_metadata(user_db_name, split_types)
        
        # Load embeddings
        logger.info("Loading embeddings...")
        self.track_embeddings = self._load_track_embeddings(track_emb_db_name, split_types)
        self.user_embeddings = self._load_user_embeddings(user_emb_db_name, split_types)
        
        # Build user history index (liked music per user)
        logger.info("Building user history index...")
        self.user_liked_music = self._build_user_history_index(dataset_name, split_types)
        
        # Load or build Qwen embeddings for metadata and attributes
        logger.info("Loading Qwen model...")
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
        self.qwen_model = AutoModel.from_pretrained(qwen_model_path)
        self.qwen_model.to(device).eval()
        
        # Build or load pre-computed track embeddings
        logger.info("Building track embedding indices...")
        self._build_track_indices()
        
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
    
    @staticmethod
    def _safe_tensor(value, expected_min_dim: int = 1) -> Optional[torch.Tensor]:
        """Safely convert a value to a float32 tensor.

        Returns None if the value is None/empty or not a valid sequence.
        """
        if value is None:
            return None
        try:
            t = torch.tensor(value, dtype=torch.float32)
            if t.ndim == 0 or t.numel() < expected_min_dim:
                return None
            return t
        except (TypeError, ValueError):
            return None

    def _load_track_embeddings(self, dataset_name: str, split_types: List[str]) -> Dict:
        """Load track embeddings (all modalities).

        Tracks whose required columns are None / empty are skipped so that
        downstream torch.stack calls never see tensors of inconsistent shape.
        """
        ds = load_dataset(dataset_name)
        valid_splits = [s for s in split_types if s in ds.keys()] or list(ds.keys())
        concat_ds = concatenate_datasets([ds[s] for s in valid_splits])

        # Detect cf-bpr dimension from the first valid row
        cf_bpr_dim: Optional[int] = None
        metadata_dim: Optional[int] = None
        attributes_dim: Optional[int] = None

        embeddings = {}
        skipped = 0
        for item in concat_ds:
            track_id = item["track_id"]

            cf_bpr      = self._safe_tensor(item.get("cf-bpr"))
            audio       = self._safe_tensor(item.get("audio-laion_clap"))
            image       = self._safe_tensor(item.get("image-siglip2"))
            attributes  = self._safe_tensor(item.get("attributes-qwen3_embedding_0.6b"))
            lyrics      = self._safe_tensor(item.get("lyrics-qwen3_embedding_0.6b"))
            metadata    = self._safe_tensor(item.get("metadata-qwen3_embedding_0.6b"))

            # Skip tracks that are missing any required embedding
            if any(t is None for t in [cf_bpr, audio, image, attributes, lyrics, metadata]):
                skipped += 1
                continue

            # On first valid row, record expected dimensions
            if cf_bpr_dim is None:
                cf_bpr_dim     = cf_bpr.shape[0]
                metadata_dim   = metadata.shape[0]
                attributes_dim = attributes.shape[0]

            # Skip tracks whose dimension doesn't match the expected size
            if (cf_bpr.shape[0] != cf_bpr_dim or
                    metadata.shape[0] != metadata_dim or
                    attributes.shape[0] != attributes_dim):
                skipped += 1
                continue

            embeddings[track_id] = {
                "cf-bpr":                           cf_bpr,
                "audio-laion_clap":                  audio,
                "image-siglip2":                     image,
                "attributes-qwen3_embedding_0.6b":   attributes,
                "lyrics-qwen3_embedding_0.6b":        lyrics,
                "metadata-qwen3_embedding_0.6b":     metadata,
            }

        if skipped:
            logger.warning(
                "Skipped %d tracks with missing/invalid embeddings during loading.", skipped
            )
        logger.info("Loaded %d tracks with valid embeddings.", len(embeddings))
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
    
    def _build_user_history_index(self, dataset_name: str, split_types: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """Build user history index: user_id -> [(track_id, like_score), ...]
        
        Like score: 5 if MOVES_TOWARD_GOAL, 1 otherwise.
        Sorted by time (descending) and like score, take top USER_HISTORY_LIKE_MUSIC_NUM.
        """
        cache_path = os.path.join(self.index_dir, "user_liked_music.json")
        if os.path.exists(cache_path):
            logger.info(f"Loading user history index from {cache_path}")
            with open(cache_path, "r") as f:
                data = json.load(f)
                return {k: [tuple(v) for v in vals] for k, vals in data.items()}
        
        logger.info("Building user history index from dataset...")
        ds = load_dataset(dataset_name)
        valid_splits = [s for s in split_types if s in ds.keys()] or list(ds.keys())
        
        user_history = defaultdict(list)
        
        for split in valid_splits:
            for session in ds[split]:
                user_id = session["user_id"]
                conversations = session["conversations"]
                assessments = session.get("goal_progress_assessments", [])
                
                # Build assessment map: turn_number -> assessment
                assessment_map = {a["turn_number"]: a for a in assessments}
                
                # Extract music recommendations and their assessments
                for conv in conversations:
                    if conv["role"] == "music":
                        turn_num = conv["turn_number"]
                        track_id = conv["content"]
                        
                        # Get assessment
                        assessment = assessment_map.get(turn_num, {})
                        goal_progress = assessment.get("goal_progress_assessment", "")
                        
                        # Assign like score
                        like_score = 5.0 if goal_progress == "MOVES_TOWARD_GOAL" else 1.0
                        
                        user_history[user_id].append((track_id, like_score, turn_num))
        
        # Sort by like_score (desc) then turn_num (desc), take top N
        user_liked_music = {}
        for user_id, history in user_history.items():
            # Sort: higher like_score first, then higher turn_num (more recent)
            sorted_history = sorted(history, key=lambda x: (-x[1], -x[2]))
            # Take top N, store as (track_id, like_score)
            user_liked_music[user_id] = [(tid, score) for tid, score, _ in sorted_history[:USER_HISTORY_LIKE_MUSIC_NUM]]
        
        # Save to cache
        with open(cache_path, "w") as f:
            json.dump(user_liked_music, f)
        
        logger.info(f"User history index built and saved to {cache_path}")
        return user_liked_music
    
    def _build_track_indices(self):
        """Build track embedding matrices for efficient retrieval."""
        # CF-BPR index
        cf_bpr_path = os.path.join(self.index_dir, "track_cf_bpr.pt")
        metadata_path = os.path.join(self.index_dir, "track_metadata_256.pt")
        attributes_path = os.path.join(self.index_dir, "track_attributes_256.pt")
        track_ids_path = os.path.join(self.index_dir, "track_ids.json")
        
        if all(os.path.exists(p) for p in [cf_bpr_path, metadata_path, attributes_path, track_ids_path]):
            logger.info("Loading pre-computed track indices...")
            self.track_cf_bpr_matrix = torch.load(cf_bpr_path, map_location="cpu")
            self.track_metadata_matrix = torch.load(metadata_path, map_location="cpu")
            self.track_attributes_matrix = torch.load(attributes_path, map_location="cpu")
            with open(track_ids_path, "r") as f:
                self.track_ids_list = json.load(f)
            logger.info(f"Loaded {len(self.track_ids_list)} tracks")
            return
        
        logger.info("Building track indices...")
        track_ids = sorted(self.track_embeddings.keys())

        # Build matrices – only keep tracks that have all valid embeddings
        cf_bpr_list = []
        metadata_list = []
        attributes_list = []
        valid_track_ids = []

        for track_id in track_ids:
            embs = self.track_embeddings[track_id]
            cf_bpr = embs.get("cf-bpr")
            metadata = embs.get("metadata-qwen3_embedding_0.6b")
            attributes = embs.get("attributes-qwen3_embedding_0.6b")

            if cf_bpr is None or metadata is None or attributes is None:
                continue
            if cf_bpr.ndim != 1 or metadata.ndim != 1 or attributes.ndim != 1:
                continue

            cf_bpr_list.append(cf_bpr)
            # Take first 256 dims
            metadata_list.append(metadata[:256])
            attributes_list.append(attributes[:256])
            valid_track_ids.append(track_id)

        if not cf_bpr_list:
            raise RuntimeError(
                "No valid track embeddings found – cannot build retrieval index. "
                "Check that the track embedding dataset columns are non-empty."
            )

        # track_ids_list must be aligned with the matrix rows
        self.track_ids_list = valid_track_ids
        logger.info(
            "Using %d / %d tracks for retrieval index "
            "(tracks with incomplete embeddings are skipped).",
            len(valid_track_ids), len(track_ids),
        )

        self.track_cf_bpr_matrix = torch.stack(cf_bpr_list)       # [N, cf_bpr_dim]
        self.track_metadata_matrix = torch.stack(metadata_list)    # [N, 256]
        self.track_attributes_matrix = torch.stack(attributes_list)  # [N, 256]
        
        # Normalize for cosine similarity
        self.track_cf_bpr_matrix = F.normalize(self.track_cf_bpr_matrix, p=2, dim=1)
        self.track_metadata_matrix = F.normalize(self.track_metadata_matrix, p=2, dim=1)
        self.track_attributes_matrix = F.normalize(self.track_attributes_matrix, p=2, dim=1)
        
        # Save
        torch.save(self.track_cf_bpr_matrix, cf_bpr_path)
        torch.save(self.track_metadata_matrix, metadata_path)
        torch.save(self.track_attributes_matrix, attributes_path)
        with open(track_ids_path, "w") as f:
            json.dump(self.track_ids_list, f)

        logger.info("Track indices built and saved. Total valid tracks: %d", len(self.track_ids_list))
    
    def _encode_query(self, query: str) -> torch.Tensor:
        """Encode query using Qwen model, return first 256 dims."""
        self.qwen_model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.qwen_model(**inputs)
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask  # [1, 1024]
            # Take first 256 dims and normalize
            embeddings = embeddings[:, :256]
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().squeeze(0)  # [256]
    
    def _retrieve_cf_bpr(self, user_id: Optional[str], topk: int) -> List[str]:
        """Channel 1: CF-BPR user-music tower retrieval."""
        if user_id is None or user_id not in self.user_embeddings:
            # Return random tracks
            return self.track_ids_list[:topk]
        
        user_cf_bpr = self.user_embeddings[user_id]["cf-bpr"]
        user_cf_bpr = F.normalize(user_cf_bpr.unsqueeze(0), p=2, dim=1)  # [1, 128]
        
        # Cosine similarity
        scores = torch.matmul(self.track_cf_bpr_matrix, user_cf_bpr.T).squeeze(1)  # [N]
        topk = min(topk, len(scores))
        top_indices = torch.topk(scores, k=topk).indices.tolist()
        
        return [self.track_ids_list[i] for i in top_indices]
    
    def _retrieve_similar_users_music(self, user_id: Optional[str]) -> List[str]:
        """Channel 2: Retrieve music liked by similar users."""
        if user_id is None or user_id not in self.user_embeddings:
            return []
        
        # Find similar users based on cf-bpr embedding
        user_cf_bpr = self.user_embeddings[user_id]["cf-bpr"]
        user_cf_bpr = F.normalize(user_cf_bpr.unsqueeze(0), p=2, dim=1)  # [1, 128]
        
        # Compute similarity with all users
        all_user_ids = list(self.user_embeddings.keys())
        user_embs = []
        for uid in all_user_ids:
            if uid != user_id:
                user_embs.append(self.user_embeddings[uid]["cf-bpr"])
        
        if not user_embs:
            return []
        
        user_embs = torch.stack(user_embs)  # [M, 128]
        user_embs = F.normalize(user_embs, p=2, dim=1)
        
        # Cosine similarity
        scores = torch.matmul(user_embs, user_cf_bpr.T).squeeze(1)  # [M]
        top_10 = min(10, len(scores))
        top_indices = torch.topk(scores, k=top_10).indices.tolist()
        
        # Get similar user IDs
        similar_user_ids = [all_user_ids[i] for i in top_indices if all_user_ids[i] != user_id]
        
        # Collect their liked music
        liked_tracks = set()
        for sim_uid in similar_user_ids:
            if sim_uid in self.user_liked_music:
                for track_id, _ in self.user_liked_music[sim_uid]:
                    liked_tracks.add(track_id)
        
        return list(liked_tracks)
    
    def _retrieve_query_metadata(self, query_emb: torch.Tensor, topk: int) -> List[str]:
        """Channel 3: Query-metadata semantic similarity."""
        # query_emb: [256]
        scores = torch.matmul(self.track_metadata_matrix, query_emb.unsqueeze(1)).squeeze(1)  # [N]
        topk = min(topk, len(scores))
        top_indices = torch.topk(scores, k=topk).indices.tolist()
        return [self.track_ids_list[i] for i in top_indices]
    
    def _retrieve_query_attributes(self, query_emb: torch.Tensor, topk: int) -> List[str]:
        """Channel 4: Query-attributes semantic similarity."""
        # query_emb: [256]
        scores = torch.matmul(self.track_attributes_matrix, query_emb.unsqueeze(1)).squeeze(1)  # [N]
        topk = min(topk, len(scores))
        top_indices = torch.topk(scores, k=topk).indices.tolist()
        return [self.track_ids_list[i] for i in top_indices]
    
    def retrieve(
        self,
        user_id: Optional[str],
        current_query: str,
        history_queries: List[str] = None,
    ) -> List[str]:
        """Multi-channel retrieval combining all channels.
        
        Args:
            user_id: User identifier
            current_query: Current user query
            history_queries: Historical queries from the user
            
        Returns:
            List of retrieved track IDs (deduplicated, up to ~350 tracks)
        """
        # Encode queries
        if history_queries is None:
            history_queries = []
        
        # Encode current query
        current_query_emb = self._encode_query(current_query)  # [256]
        
        # Encode history queries and pool
        if history_queries:
            history_embs = [self._encode_query(q) for q in history_queries]
            history_emb = torch.stack(history_embs).mean(dim=0)  # [256]
            # Weighted pooling: 0.3 history + 0.7 current
            query_emb = 0.3 * history_emb + 0.7 * current_query_emb
            query_emb = F.normalize(query_emb.unsqueeze(0), p=2, dim=1).squeeze(0)
        else:
            query_emb = current_query_emb
        
        # Channel 1: CF-BPR
        cf_bpr_tracks = self._retrieve_cf_bpr(user_id, USER_MUSIC_TOWER_RECALL_NUM)
        
        # Channel 2: Similar users' music
        similar_users_tracks = self._retrieve_similar_users_music(user_id)
        
        # Channel 3: Query-metadata
        metadata_tracks = self._retrieve_query_metadata(query_emb, QUERY_METADATA_RECALL_NUM)
        
        # Channel 4: Query-attributes
        attributes_tracks = self._retrieve_query_attributes(query_emb, QUERY_ATTRIBUTES_RECALL_NUM)
        
        # Combine and deduplicate (preserve order)
        all_tracks = []
        seen = set()
        for track_id in cf_bpr_tracks + similar_users_tracks + metadata_tracks + attributes_tracks:
            if track_id not in seen:
                all_tracks.append(track_id)
                seen.add(track_id)
        
        logger.info(f"Retrieved {len(all_tracks)} unique tracks from multi-channel retrieval")
        return all_tracks
    
    def batch_retrieve(
        self,
        user_ids: List[Optional[str]],
        current_queries: List[str],
        history_queries_list: List[List[str]] = None,
    ) -> List[List[str]]:
        """Batch retrieval for multiple queries."""
        if history_queries_list is None:
            history_queries_list = [[] for _ in range(len(current_queries))]
        
        results = []
        for user_id, query, history in zip(user_ids, current_queries, history_queries_list):
            results.append(self.retrieve(user_id, query, history))
        
        return results
