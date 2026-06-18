"""CRS baseline v2 with BM25/BERT retrieval and LLM generation.

Pipeline:
    1. Retrieval  – BM25 or BERT retrieves Top-K candidates.
    2. Generation – LLM generates a natural language response given the
                    system prompt, conversation history, and the top
                    retrieved track metadata.

Note: Multi-channel retrieval (ch1/ch3/ch5) and three-tower reranking have
been removed. Use CRS_BASELINE_V2 as a clean starting point to plug in a
custom reranker later.
"""

import os
import torch
from typing import Optional, Any, List, Dict

from mcrs.db_item import MusicCatalogDB
from mcrs.db_user import UserProfileDB
from mcrs.lm_modules import load_lm_module
from mcrs.retrieval_modules import load_retrieval_module


class CRS_BASELINE_V2:
    """CRS baseline v2 with BM25/BERT retrieval.

    This is a clean version of the pipeline without multi-channel retrieval
    or learned reranking models. Suitable as a base for future extensions.
    """

    def __init__(
        self,
        lm_type: str = "meta-llama/Llama-3.2-1B-Instruct",
        retrieval_type: str = "bm25",
        item_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        user_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Metadata",
        track_emb_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        user_emb_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
        track_split_types: List[str] = None,
        user_split_types: List[str] = None,
        corpus_types: List[str] = None,
        cache_dir: str = "./cache",
        device: str = "cuda",
        attn_implementation: str = "eager",
        dtype: torch.dtype = torch.bfloat16,
        retrieval_topk: int = 20,
    ) -> None:
        """Initialize CRS baseline V2.

        Args:
            lm_type: LLM model identifier.
            retrieval_type: Retrieval backend ("bm25" or "bert").
            item_db_name: Track metadata dataset.
            user_db_name: User metadata dataset.
            track_emb_db_name: Track embeddings dataset (unused; kept for API compat).
            user_emb_db_name: User embeddings dataset (unused; kept for API compat).
            track_split_types: Dataset splits for tracks.
            user_split_types: Dataset splits for users.
            corpus_types: Metadata fields for retrieval text corpus.
            cache_dir: Cache directory.
            device: Compute device.
            attn_implementation: Attention implementation for LLM.
            dtype: Torch dtype for LLM.
            retrieval_topk: Number of candidates to retrieve.
        """
        if track_split_types is None:
            track_split_types = ["all_tracks"]
        if user_split_types is None:
            user_split_types = ["all_users"]
        if corpus_types is None:
            corpus_types = ["track_name", "artist_name", "album_name"]

        self.cache_dir = cache_dir
        self.lm_type = lm_type
        self.retrieval_type = retrieval_type
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

        # Load LLM
        self.lm = load_lm_module(
            self.lm_type, self.device, self.attn_implementation, self.dtype
        )

        # Load retrieval module (BM25 or BERT)
        self.retrieval = load_retrieval_module(
            self.retrieval_type,
            self.item_db_name,
            self.track_split_types,
            self.corpus_types,
            self.cache_dir,
        )

        # Load item and user databases
        self.item_db = MusicCatalogDB(
            self.item_db_name, self.track_split_types, self.corpus_types
        )
        self.user_db = UserProfileDB(self.user_db_name, self.user_split_types)

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

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

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
        return [msg.get("content", "") for msg in chat_history if msg.get("role") == "user"]

    def _format_history_context(self, chat_history: List[Dict[str, Any]]) -> str:
        """Format chat history as a string."""
        return "\n".join(
            f"{msg.get('role', '')}: {msg.get('content', '')}" for msg in chat_history
        )

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def chat(
        self,
        user_query: str,
        user_id: Optional[str] = None,
        session_memory: Optional[List[Dict[str, Any]]] = None,
        conversation_goal: Optional[Dict] = None,
        session_date: Optional[str] = None,
        session_id: Optional[str] = None,
        turn_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Single-turn chat: retrieve → generate response.

        Args:
            user_query: User's query.
            user_id: User identifier (used for personalised system prompt).
            session_memory: Chat history for this session.
            conversation_goal: Unused; kept for API compatibility.
            session_date: Unused; kept for API compatibility.
            session_id: Unused; kept for API compatibility.
            turn_number: Unused; kept for API compatibility.

        Returns:
            Dict with retrieval_items, recommend_item, response.
        """
        if session_memory is not None:
            self._upload_session_memory(session_memory)

        # Build retrieval query from full conversation context
        retrieval_input = self._format_history_context(self.session_memory)
        if retrieval_input:
            retrieval_input += f"\nuser: {user_query}"
        else:
            retrieval_input = user_query

        # Retrieval
        retrieved_items = self.retrieval.text_to_item_retrieval(
            retrieval_input, topk=self.retrieval_topk
        )
        recommend_track_id = retrieved_items[0] if retrieved_items else None

        # Generate response
        if recommend_track_id:
            track_metadata_str = self.item_db.id_to_metadata(recommend_track_id)
            self.session_memory.append({"role": "user", "content": user_query})
            system_prompt = self._get_system_prompt(user_id)
            response = self.lm.response_generation(
                system_prompt,
                self.session_memory,
                track_metadata_str,
            )
            self.session_memory.append({"role": "assistant", "content": response})
        else:
            response = "I couldn't find a suitable track for you."

        return {
            "user_id": user_id,
            "user_query": user_query,
            "retrieval_items": retrieved_items,
            "recommend_item": recommend_track_id,
            "response": response,
        }

    def batch_chat(
        self,
        batch_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Batch chat processing: retrieve → generate.

        Args:
            batch_data: List of dicts with keys: user_query, user_id,
                        session_memory, conversation_goal (optional),
                        session_date (optional), session_id (optional),
                        turn_number (optional).

        Returns:
            List of result dicts.
        """
        user_ids         = [d.get("user_id") for d in batch_data]
        user_queries     = [d["user_query"] for d in batch_data]
        session_memories = [d.get("session_memory", []) for d in batch_data]

        # Build retrieval inputs from conversation context
        retrieval_inputs: List[str] = []
        for sm, q in zip(session_memories, user_queries):
            ctx = self._format_history_context(sm)
            retrieval_inputs.append(f"{ctx}\nuser: {q}" if ctx else q)

        # Batch retrieval
        if hasattr(self.retrieval, "batch_text_to_item_retrieval"):
            batch_retrieved = self.retrieval.batch_text_to_item_retrieval(
                retrieval_inputs, topk=self.retrieval_topk
            )
        else:
            batch_retrieved = [
                self.retrieval.text_to_item_retrieval(inp, topk=self.retrieval_topk)
                for inp in retrieval_inputs
            ]

        recommend_items = [items[0] if items else None for items in batch_retrieved]

        # Build per-sample generation inputs
        valid_indices   = [i for i, r in enumerate(recommend_items) if r is not None]
        invalid_indices = [i for i, r in enumerate(recommend_items) if r is None]

        responses = [None] * len(batch_data)
        for i in invalid_indices:
            responses[i] = "I couldn't find a suitable track for you."

        if valid_indices:
            sys_prompts_valid     = [self._get_system_prompt(user_ids[i]) for i in valid_indices]
            chat_histories_valid  = [
                list(session_memories[i]) + [{"role": "user", "content": user_queries[i]}]
                for i in valid_indices
            ]
            recommend_strs_valid  = [
                self.item_db.id_to_metadata(recommend_items[i]) for i in valid_indices
            ]

            if hasattr(self.lm, "batch_response_generation"):
                generated = self.lm.batch_response_generation(
                    sys_prompts_valid, chat_histories_valid, recommend_strs_valid
                )
            else:
                generated = [
                    self.lm.response_generation(sp, ch, ri)
                    for sp, ch, ri in zip(
                        sys_prompts_valid, chat_histories_valid, recommend_strs_valid
                    )
                ]
            for idx, gen in zip(valid_indices, generated):
                responses[idx] = gen

        results = []
        for i, data in enumerate(batch_data):
            results.append({
                "user_id": data.get("user_id"),
                "user_query": user_queries[i],
                "retrieval_items": batch_retrieved[i],
                "recommend_item": recommend_items[i],
                "response": responses[i],
            })
        return results
