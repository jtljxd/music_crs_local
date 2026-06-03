import torch
from .crs_baseline import CRS_BASELINE
from .crs_baseline_v2 import CRS_BASELINE_V2


def load_crs_baseline(
    lm_type: str = "meta-llama/Llama-3.2-1B-Instruct",
    retrieval_type: str = "bm25",
    item_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
    user_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Metadata",
    track_emb_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
    user_emb_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
    track_split_types: list = None,
    user_split_types: list = None,
    corpus_types: list = None,
    cache_dir: str = "./cache",
    device: str = "cuda",
    attn_implementation: str = "eager",
    dtype: torch.dtype = torch.bfloat16,
    retrieval_topk: int = 20,
    fm_k: int = 16,
    fm_lr: float = 1e-3,
    fm_top_tags: int = 50,
) -> CRS_BASELINE:
    """Factory function to create a CRS_BASELINE instance.

    Args:
        lm_type: LLM model identifier.
        retrieval_type: Retrieval backend ("bm25" or "bert").
        item_db_name: HuggingFace dataset for track metadata.
        user_db_name: HuggingFace dataset for user metadata.
        track_emb_db_name: HuggingFace dataset for track embeddings.
        user_emb_db_name: HuggingFace dataset for user embeddings.
        track_split_types: Splits to load for tracks.
        user_split_types: Splits to load for users.
        corpus_types: Track fields used for retrieval text corpus.
        cache_dir: Directory for caching artefacts.
        device: Torch device string.
        attn_implementation: Attention implementation for the LLM.
        dtype: Torch dtype for the LLM.
        retrieval_topk: Number of candidates to retrieve before FM reranking.
        fm_k: Latent factor dimension for the FM model.
        fm_lr: Learning rate for the FM optimiser.
        fm_top_tags: Tag vocabulary size for the multi-hot feature.

    Returns:
        Initialised CRS_BASELINE instance.
    """
    if track_split_types is None:
        track_split_types = ["all_tracks"]
    if user_split_types is None:
        user_split_types = ["all_users"]
    if corpus_types is None:
        corpus_types = ["track_name", "artist_name", "album_name"]

    return CRS_BASELINE(
        lm_type=lm_type,
        retrieval_type=retrieval_type,
        item_db_name=item_db_name,
        user_db_name=user_db_name,
        track_emb_db_name=track_emb_db_name,
        user_emb_db_name=user_emb_db_name,
        track_split_types=track_split_types,
        user_split_types=user_split_types,
        corpus_types=corpus_types,
        cache_dir=cache_dir,
        device=device,
        attn_implementation=attn_implementation,
        dtype=dtype,
        retrieval_topk=retrieval_topk,
        fm_k=fm_k,
        fm_lr=fm_lr,
        fm_top_tags=fm_top_tags,
    )


def load_crs_baseline_v2(
    lm_type: str = "meta-llama/Llama-3.2-1B-Instruct",
    conversation_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Dataset",
    item_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
    user_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Metadata",
    track_emb_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
    user_emb_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
    track_split_types: list = None,
    user_split_types: list = None,
    corpus_types: list = None,
    cache_dir: str = "./cache",
    qwen_model_path: str = "/home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B",
    device: str = "cuda",
    attn_implementation: str = "eager",
    dtype: torch.dtype = torch.bfloat16,
    retrieval_topk: int = 350,
    rerank_topk: int = 20,
    reranker_lr: float = 1e-3,
    build_indices: bool = False,  # True only during training
) -> CRS_BASELINE_V2:
    """Factory function to create a CRS_BASELINE_V2 instance (add_all_feature_v2).

    Args:
        lm_type: LLM model identifier.
        conversation_dataset_name: Dataset with conversations and assessments.
        item_db_name: HuggingFace dataset for track metadata.
        user_db_name: HuggingFace dataset for user metadata.
        track_emb_db_name: HuggingFace dataset for track embeddings.
        user_emb_db_name: HuggingFace dataset for user embeddings.
        track_split_types: Splits to load for tracks.
        user_split_types: Splits to load for users.
        corpus_types: Track fields used for LLM display.
        cache_dir: Directory for caching artefacts.
        qwen_model_path: Path to Qwen3-Embedding-0.6B model.
        device: Torch device string.
        attn_implementation: Attention implementation for the LLM.
        dtype: Torch dtype for the LLM.
        retrieval_topk: Not used (multi-channel returns ~350).
        rerank_topk: Number of candidates to return after reranking.
        reranker_lr: Learning rate for the three-tower reranker.

    Returns:
        Initialised CRS_BASELINE_V2 instance.
    """
    if track_split_types is None:
        track_split_types = ["all_tracks"]
    if user_split_types is None:
        user_split_types = ["all_users"]
    if corpus_types is None:
        corpus_types = ["track_name", "artist_name", "album_name"]

    return CRS_BASELINE_V2(
        lm_type=lm_type,
        conversation_dataset_name=conversation_dataset_name,
        item_db_name=item_db_name,
        user_db_name=user_db_name,
        track_emb_db_name=track_emb_db_name,
        user_emb_db_name=user_emb_db_name,
        track_split_types=track_split_types,
        user_split_types=user_split_types,
        corpus_types=corpus_types,
        cache_dir=cache_dir,
        qwen_model_path=qwen_model_path,
        device=device,
        attn_implementation=attn_implementation,
        dtype=dtype,
        retrieval_topk=retrieval_topk,
        rerank_topk=rerank_topk,
        reranker_lr=reranker_lr,
        build_indices=build_indices,
    )
