"""CRS baseline with multi-channel retrieval and three-tower reranking (add_all_feature_v2).

Pipeline:
    1. Multi-channel Retrieval – Combines CF-BPR, similar users, query-metadata, query-attributes
    2. Three-tower Reranking – Intent tower + Item tower + User tower
    3. Generation – LLM generates response with recommended track
"""

import os
import torch
from typing import Optional, Any, List, Dict

from mcrs.db_item import MusicCatalogDB
from mcrs.db_user import UserProfileDB
from mcrs.lm_modules import load_lm_module
from mcrs.retrieval_modules import MultiChannelRetrieval
from mcrs.reranking_modules import ThreeTowerRerankerWrapper


class CRS_BASELINE_V2:
    """CRS baseline with multi-channel retrieval and three-tower reranking.
    
    This is the add_all_feature_v2 implementation with:
    - Multi-channel retrieval (4 channels)
    - Three-tower reranking model
    - Same LLM generation as baseline
    """
    
    def __init__(
        self,
        lm_type: str = "meta-llama/Llama-3.2-1B-Instruct",
        conversation_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Dataset",
        item_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        user_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Metadata",
        track_emb_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        user_emb_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
        track_split_types: List[str] = None,
        user_split_types: List[str] = None,
        corpus_types: List[str] = None,
        cache_dir: str = "./cache",
        qwen_model_path: str = "/home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B",
        device: str = "cuda",
        attn_implementation: str = "eager",
        dtype: torch.dtype = torch.bfloat16,
        retrieval_topk: int = 350,
        rerank_topk: int = 20,
        reranker_lr: float = 1e-3,
    ) -> None:
        """Initialize CRS baseline V2.
        
        Args:
            lm_type: LLM model identifier
            conversation_dataset_name: Dataset with conversations and assessments
            item_db_name: Track metadata dataset
            user_db_name: User metadata dataset
            track_emb_db_name: Track embeddings dataset
            user_emb_db_name: User embeddings dataset
            track_split_types: Dataset splits for tracks
            user_split_types: Dataset splits for users
            corpus_types: Metadata fields for LLM display
            cache_dir: Cache directory
            qwen_model_path: Path to Qwen3-Embedding-0.6B model
            device: Compute device
            attn_implementation: Attention implementation for LLM
            dtype: Torch dtype for LLM
            retrieval_topk: Number of candidates from retrieval (not used, multi-channel returns ~350)
            rerank_topk: Number of candidates to return after reranking
            reranker_lr: Learning rate for reranker
        """
        if track_split_types is None:
            track_split_types = ["all_tracks"]
        if user_split_types is None:
            user_split_types = ["all_users"]
        if corpus_types is None:
            corpus_types = ["track_name", "artist_name", "album_name"]
        
        self.cache_dir = cache_dir
        self.lm_type = lm_type
        self.conversation_dataset_name = conversation_dataset_name
        self.item_db_name = item_db_name
        self.user_db_name = user_db_name
        self.track_emb_db_name = track_emb_db_name
        self.user_emb_db_name = user_emb_db_name
        self.track_split_types = track_split_types
        self.user_split_types = user_split_types
        self.corpus_types = corpus_types
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.retrieval_topk = retrieval_topk
        self.rerank_topk = rerank_topk
        self.qwen_model_path = qwen_model_path
        
        # Load LLM
        self.lm = load_lm_module(
            self.lm_type, self.device, self.attn_implementation, self.dtype
        )
        
        # Load multi-channel retrieval
        self.retrieval = MultiChannelRetrieval(
            dataset_name=self.conversation_dataset_name,
            item_db_name=self.item_db_name,
            user_db_name=self.user_db_name,
            track_emb_db_name=self.track_emb_db_name,
            user_emb_db_name=self.user_emb_db_name,
            split_types=self.track_split_types,
            cache_dir=self.cache_dir,
            qwen_model_path=self.qwen_model_path,
            device=self.device,
        )
        
        # Load item and user databases
        self.item_db = MusicCatalogDB(
            self.item_db_name, self.track_split_types, self.corpus_types
        )
        self.user_db = UserProfileDB(self.user_db_name, self.user_split_types)
        
        # Load three-tower reranker
        self.reranker = ThreeTowerRerankerWrapper(
            dataset_name=self.conversation_dataset_name,
            track_emb_db_name=self.track_emb_db_name,
            user_emb_db_name=self.user_emb_db_name,
            track_metadata_db_name=self.item_db_name,
            user_metadata_db_name=self.user_db_name,
            split_types=self.track_split_types,
            cache_dir=self.cache_dir,
            qwen_model_path=self.qwen_model_path,
            device=self.device,
            lr=reranker_lr,
        )
        
        # Load prompts
        self.prompts_dir = os.path.join(os.path.dirname(__file__), "system_prompts")
        self.role_prompt = {
            "role_play": open(
                f"{self.prompts_dir}/roleplay.txt", "r", encoding="utf-8"
            ).read(),
            "personalization": open(
                f"{self.prompts_dir}/personalization.txt", "r", encoding="utf-8"
            ).read(),
            "response_generation": open(
                f"{self.prompts_dir}/response_generation.txt", "r", encoding="utf-8"
            ).read(),
        }
        self.session_memory: List[Dict[str, Any]] = []
    
    def _reset_session_memory(self) -> None:
        """Clear session memory."""
        self.session_memory = []
    
    def _upload_session_memory(self, chat_history: List[Dict[str, Any]]) -> None:
        """Upload chat history to session memory."""
        self.session_memory = chat_history
    
    def _get_system_prompt(self, user_id: Optional[str] = None) -> str:
        """Build system prompt with optional personalization."""
        system_prompt = (
            self.role_prompt["role_play"] + self.role_prompt["response_generation"]
        )
        if user_id:
            user_profile_str = self.user_db.id_to_profile_str(user_id)
            system_prompt += self.role_prompt["personalization"] + "\n" + user_profile_str
        return system_prompt
    
    def _extract_history_queries(self, chat_history: List[Dict[str, Any]]) -> List[str]:
        """Extract user queries from chat history."""
        queries = []
        for msg in chat_history:
            if msg.get("role") == "user":
                queries.append(msg.get("content", ""))
        return queries
    
    def _format_history_context(self, chat_history: List[Dict[str, Any]]) -> str:
        """Format chat history as a string."""
        context = []
        for msg in chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            context.append(f"{role}: {content}")
        return "\n".join(context)
    
    def chat(
        self,
        user_query: str,
        user_id: Optional[str] = None,
        session_memory: Optional[List[Dict[str, Any]]] = None,
        conversation_goal: Optional[Dict] = None,
        session_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Single-turn chat.
        
        Args:
            user_query: User's query
            user_id: User identifier
            session_memory: Chat history
            conversation_goal: Conversation goal dict
            session_date: Session date
            
        Returns:
            Dict with retrieval_items, recommend_item, response
        """
        if session_memory is not None:
            self._upload_session_memory(session_memory)
        
        # Extract history queries
        history_queries = self._extract_history_queries(self.session_memory)
        
        # Multi-channel retrieval
        candidate_track_ids = self.retrieval.retrieve(
            user_id=user_id,
            current_query=user_query,
            history_queries=history_queries,
        )
        
        # Three-tower reranking
        history_context = self._format_history_context(self.session_memory)
        reranked_track_ids = self.reranker.rerank(
            user_id=user_id,
            candidate_track_ids=candidate_track_ids,
            current_query=user_query,
            history_context=history_context,
            conversation_goal=conversation_goal,
            session_date=session_date or "",
        )
        
        # Take top K
        final_track_ids = reranked_track_ids[:self.rerank_topk]
        recommend_track_id = final_track_ids[0] if final_track_ids else None
        
        # Generate response
        if recommend_track_id:
            track_metadata_str = self.item_db.id_to_metadata(recommend_track_id)
            self.session_memory.append({"role": "user", "content": user_query})
            
            system_prompt = self._get_system_prompt(user_id)
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self.session_memory)
            messages.append({"role": "assistant", "content": track_metadata_str})
            
            response = self.lm.generate(messages)
            self.session_memory.append({"role": "assistant", "content": response})
        else:
            response = "I couldn't find a suitable track for you."
        
        return {
            "user_id": user_id,
            "user_query": user_query,
            "retrieval_items": final_track_ids,
            "recommend_item": recommend_track_id,
            "response": response,
        }
    
    def batch_chat(
        self,
        batch_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Batch chat processing.
        
        Args:
            batch_data: List of dicts with keys: user_query, user_id, session_memory, 
                       conversation_goal (optional), session_date (optional)
        
        Returns:
            List of result dicts
        """
        # Extract batch inputs
        user_ids = [d.get("user_id") for d in batch_data]
        user_queries = [d["user_query"] for d in batch_data]
        session_memories = [d.get("session_memory", []) for d in batch_data]
        conversation_goals = [d.get("conversation_goal") for d in batch_data]
        session_dates = [d.get("session_date", "") for d in batch_data]
        
        # Extract history queries for each session
        history_queries_list = []
        for session_memory in session_memories:
            history_queries = self._extract_history_queries(session_memory)
            history_queries_list.append(history_queries)
        
        # Batch retrieval
        batch_candidate_track_ids = self.retrieval.batch_retrieve(
            user_ids=user_ids,
            current_queries=user_queries,
            history_queries_list=history_queries_list,
        )
        
        # Batch reranking
        history_contexts = [self._format_history_context(sm) for sm in session_memories]
        batch_reranked_track_ids = self.reranker.batch_rerank(
            user_ids=user_ids,
            batch_candidate_track_ids=batch_candidate_track_ids,
            current_queries=user_queries,
            history_contexts=history_contexts,
            conversation_goals=conversation_goals,
            session_dates=session_dates,
        )
        
        # Take top K for each
        batch_final_track_ids = [
            reranked[:self.rerank_topk] for reranked in batch_reranked_track_ids
        ]
        
        # Get recommended tracks
        recommend_items = [
            tracks[0] if tracks else None for tracks in batch_final_track_ids
        ]
        
        # Batch generate responses
        batch_messages = []
        for i, data in enumerate(batch_data):
            if recommend_items[i]:
                track_metadata_str = self.item_db.id_to_metadata(recommend_items[i])
                system_prompt = self._get_system_prompt(data.get("user_id"))
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(session_memories[i])
                messages.append({"role": "user", "content": user_queries[i]})
                messages.append({"role": "assistant", "content": track_metadata_str})
                batch_messages.append(messages)
            else:
                batch_messages.append(None)
        
        # Generate responses
        responses = []
        for messages in batch_messages:
            if messages:
                response = self.lm.generate(messages)
                responses.append(response)
            else:
                responses.append("I couldn't find a suitable track for you.")
        
        # Build results
        results = []
        for i, data in enumerate(batch_data):
            results.append({
                "user_id": data.get("user_id"),
                "user_query": user_queries[i],
                "retrieval_items": batch_final_track_ids[i],
                "recommend_item": recommend_items[i],
                "response": responses[i],
            })
        
        return results
    
    def save_reranker(self) -> None:
        """Save reranker checkpoint."""
        self.reranker.save_checkpoint()
