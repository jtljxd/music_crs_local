import torch
from .crs_baseline import CRS_BASELINE


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
