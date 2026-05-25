"""Conversational Recommender System baseline with FM-based reranking.

Pipeline:
    1. Retrieval  – BM25 or BERT retrieves Top-K candidates from the
                    track metadata corpus.
    2. Reranking  – An FM model (FMReranker) reranks the Top-K candidates
                    using track embeddings, user embeddings, and rich
                    metadata features.  The top-1 after reranking is
                    selected as the recommendation.
    3. Generation – An LLM generates a natural language response given the
                    system prompt, conversation history, and the recommended
                    track metadata.
"""
import os
import torch
from typing import Optional, Any, List, Dict

from mcrs.db_item import MusicCatalogDB
from mcrs.db_user import UserProfileDB
from mcrs.lm_modules import load_lm_module
from mcrs.retrieval_modules import load_retrieval_module
from mcrs.reranking_modules import FMReranker


class CRS_BASELINE:
    """
    Conversational Recommender System (CRS) baseline that wires together an
    LLM module, an item retrieval module, and an FM-based reranker over a
    music catalog and user profiles.

    Attributes:
        cache_dir: Local path for caching artefacts and indices.
        lm_type: Identifier/name for the LLM backend to load.
        retrieval_type: Retrieval backend to use (e.g., "bm25", "bert").
        item_db_name: HuggingFace dataset name for item metadata.
        user_db_name: HuggingFace dataset name for user metadata.
        track_emb_db_name: HuggingFace dataset name for track embeddings.
        user_emb_db_name: HuggingFace dataset name for user embeddings.
        track_split_types: Dataset split names for tracks.
        user_split_types: Dataset split names for users.
        corpus_types: Item metadata fields used for retrieval text corpus.
        device: Compute device for the LLM and FM model.
        dtype: Torch dtype used by the LLM.
        lm: Loaded LLM module for response generation.
        retrieval: Retrieval module for fetching candidate items.
        reranker: FM-based reranker for second-stage ranking.
        item_db: Item metadata database accessor.
        user_db: User profile database accessor.
        prompts_dir: Directory containing prompt templates.
        role_prompt: Loaded prompt templates keyed by role.
        session_memory: In-memory list of message dicts for the current session.
        retrieval_topk: Number of candidates to retrieve before reranking.
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
        fm_k: int = 16,
        fm_lr: float = 1e-3,
        fm_top_tags: int = 50,
    ) -> None:
        """Initialize the CRS baseline components.

        Args:
            lm_type: LLM model identifier for response generation.
            retrieval_type: Retrieval backend name ("bm25" or "bert").
            item_db_name: Dataset name for track metadata (11 fields).
            user_db_name: Dataset name for user metadata.
            track_emb_db_name: Dataset name for pre-computed track embeddings.
            user_emb_db_name: Dataset name for pre-computed user embeddings.
            track_split_types: Dataset splits for tracks.
            user_split_types: Dataset splits for users.
            corpus_types: Track metadata fields for retrieval text corpus.
            cache_dir: Directory for caching artefacts.
            device: Compute device ("cuda" or "cpu").
            attn_implementation: Attention implementation for the LLM.
            dtype: Torch dtype for the LLM.
            retrieval_topk: Number of candidates retrieved before reranking.
            fm_k: Latent factor dimension for the FM model.
            fm_lr: Learning rate for the FM optimiser.
            fm_top_tags: Vocabulary size for the tag multi-hot feature.
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

        # ------------------------------------------------------------------
        # Core components
        # ------------------------------------------------------------------
        self.lm = load_lm_module(
            self.lm_type, self.device, self.attn_implementation, self.dtype
        )
        self.retrieval = load_retrieval_module(
            self.retrieval_type,
            self.item_db_name,
            self.track_split_types,
            self.corpus_types,
            self.cache_dir,
        )

        # Item DB now stores all 11 fields
        self.item_db = MusicCatalogDB(
            self.item_db_name, self.track_split_types, self.corpus_types
        )
        self.user_db = UserProfileDB(self.user_db_name, self.user_split_types)

        # ------------------------------------------------------------------
        # FM reranker (uses track & user embeddings + full metadata)
        # ------------------------------------------------------------------
        self.reranker = FMReranker(
            track_emb_dataset_name=self.track_emb_db_name,
            user_emb_dataset_name=self.user_emb_db_name,
            track_metadata_dict=self.item_db.metadata_dict,
            track_split_types=self.track_split_types,
            user_split_types=self.user_split_types,
            fm_k=fm_k,
            lr=fm_lr,
            cache_dir=self.cache_dir,
            device=self.device,
            top_tags=fm_top_tags,
        )

        # ------------------------------------------------------------------
        # Prompts
        # ------------------------------------------------------------------
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
        """Clear all messages stored in the current session memory."""
        self.session_memory = []

    def _upload_session_memory(self, chat_history: List[Dict[str, Any]]) -> None:
        """Replace the session memory with the supplied chat history."""
        self.session_memory = chat_history

    def _get_system_prompt(self, user_id: Optional[str] = None) -> str:
        """Build the system prompt, optionally personalised with a user profile.

        Args:
            user_id: Optional user identifier. When provided, includes a
                personalisation segment derived from the user profile.

        Returns:
            The final system prompt string for the LLM.
        """
        system_prompt = (
            self.role_prompt["role_play"] + self.role_prompt["response_generation"]
        )
        if user_id:
            user_profile_str = self.user_db.id_to_profile_str(user_id)
            system_prompt += self.role_prompt["personalization"] + "\n" + user_profile_str
        return system_prompt

    # ------------------------------------------------------------------
    # Retrieval + reranking helper
    # ------------------------------------------------------------------

    def _retrieve_and_rerank(
        self,
        retrieval_input: str,
        user_id: Optional[str],
    ) -> List[str]:
        """Run retrieval followed by FM reranking.

        Args:
            retrieval_input: Text to query the retrieval index with.
            user_id: User identifier for personalised reranking.

        Returns:
            Reranked list of track IDs (Top-1 is the recommendation).
        """
        candidates = self.retrieval.text_to_item_retrieval(
            retrieval_input, topk=self.retrieval_topk
        )
        reranked = self.reranker.rerank(user_id, candidates)
        return reranked

    def _batch_retrieve_and_rerank(
        self,
        retrieval_inputs: List[str],
        user_ids: List[Optional[str]],
    ) -> List[List[str]]:
        """Batch retrieval followed by FM reranking.

        Args:
            retrieval_inputs: List of text queries.
            user_ids: List of user identifiers.

        Returns:
            List of reranked track ID lists.
        """
        if hasattr(self.retrieval, "batch_text_to_item_retrieval"):
            batch_candidates = self.retrieval.batch_text_to_item_retrieval(
                retrieval_inputs, topk=self.retrieval_topk
            )
        else:
            batch_candidates = [
                self.retrieval.text_to_item_retrieval(inp, topk=self.retrieval_topk)
                for inp in retrieval_inputs
            ]
        reranked_batch = self.reranker.batch_rerank(user_ids, batch_candidates)
        return reranked_batch

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def chat(
        self, user_query: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run a single CRS turn: retrieve → FM rerank → generate response.

        Args:
            user_query: The user's latest message or request.
            user_id: Optional user identifier for personalisation.

        Returns:
            A dictionary with keys:
                - user_id: The user identifier (may be None).
                - user_query: Echo of the input query.
                - retrieval_items: Reranked list of track IDs.
                - recommend_item: Metadata string for the top track.
                - response: The generated assistant response.
        """
        self.session_memory.append({"role": "user", "content": user_query})

        # Stage 0: system prompt
        system_prompt = self._get_system_prompt(user_id)

        # Stage 1: retrieval + FM reranking
        retrieval_input = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in self.session_memory
        )
        reranked_items = self._retrieve_and_rerank(retrieval_input, user_id)
        recommend_item = self.item_db.id_to_metadata(reranked_items[0])

        # Stage 2: response generation
        response = self.lm.response_generation(
            system_prompt, self.session_memory, recommend_item
        )
        return {
            "user_id": user_id,
            "user_query": user_query,
            "retrieval_items": reranked_items,
            "recommend_item": recommend_item,
            "response": response,
        }

    def batch_chat(
        self, batch_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run multiple CRS turns in batch: retrieve → FM rerank → generate.

        Args:
            batch_data: List of dicts, each with keys:
                - user_query: The user's latest message.
                - user_id: Optional user identifier.
                - session_memory: List of prior chat messages.

        Returns:
            A list of result dicts, each with keys:
                - user_id, user_query, retrieval_items,
                  recommend_item, response.
        """
        sys_prompts: List[str] = []
        retrieval_inputs: List[str] = []
        session_memories: List[List[Dict[str, Any]]] = []
        user_ids: List[Optional[str]] = []

        for data in batch_data:
            user_query = data["user_query"]
            uid = data.get("user_id")
            session_memory = data["session_memory"].copy()
            session_memory.append({"role": "user", "content": user_query})

            sys_prompts.append(self._get_system_prompt(uid))
            retrieval_inputs.append(
                "\n".join(
                    f"{msg['role']}: {msg['content']}" for msg in session_memory
                )
            )
            session_memories.append(session_memory)
            user_ids.append(uid)

        # Stage 1: batch retrieval + FM reranking
        batch_reranked = self._batch_retrieve_and_rerank(retrieval_inputs, user_ids)
        recommend_items = [
            self.item_db.id_to_metadata(items[0]) for items in batch_reranked
        ]

        # Stage 2: batch response generation
        if hasattr(self.lm, "batch_response_generation"):
            responses = self.lm.batch_response_generation(
                sys_prompts, session_memories, recommend_items
            )
        else:
            responses = [
                self.lm.response_generation(
                    sys_prompts[i], session_memories[i], recommend_items[i]
                )
                for i in range(len(batch_data))
            ]

        results: List[Dict[str, Any]] = []
        for i, data in enumerate(batch_data):
            results.append(
                {
                    "user_id": data.get("user_id"),
                    "user_query": data["user_query"],
                    "retrieval_items": batch_reranked[i],
                    "recommend_item": recommend_items[i],
                    "response": responses[i],
                }
            )
        return results

    # ------------------------------------------------------------------
    # Online training helper (optional)
    # ------------------------------------------------------------------

    def train_reranker_on_batch(
        self,
        user_ids: List[Optional[str]],
        positive_track_ids: List[str],
        negative_track_ids_list: List[List[str]],
    ) -> float:
        """Fine-tune the FM reranker on a labelled batch.

        Wraps :py:meth:`FMReranker.fit_on_batch`.

        Args:
            user_ids: User identifiers.
            positive_track_ids: Ground-truth track IDs.
            negative_track_ids_list: Lists of negative track IDs per sample.

        Returns:
            Mean BPR loss for the batch.
        """
        loss = self.reranker.fit_on_batch(
            user_ids, positive_track_ids, negative_track_ids_list
        )
        return loss

    def save_reranker(self) -> None:
        """Persist the FM reranker checkpoint to disk."""
        self.reranker.save_checkpoint()
