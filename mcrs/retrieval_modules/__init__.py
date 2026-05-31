from .bm25 import BM25_MODEL
from .bert import BERT_MODEL
from .multi_channel_retrieval import MultiChannelRetrieval

def load_retrieval_module(
        retrieval_type: str,
        dataset_name: str,
        track_split_types: list[str],
        corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
        cache_dir: str = "./cache",
        **kwargs
    ):
    if retrieval_type == "bm25":
        return BM25_MODEL(dataset_name, track_split_types, corpus_types, cache_dir)
    elif retrieval_type == "bert":
        return BERT_MODEL(dataset_name, track_split_types, corpus_types, cache_dir)
    elif retrieval_type == "multi_channel":
        return MultiChannelRetrieval(
            dataset_name=kwargs.get("conversation_dataset_name", dataset_name),
            item_db_name=dataset_name,
            user_db_name=kwargs.get("user_db_name"),
            track_emb_db_name=kwargs.get("track_emb_db_name"),
            user_emb_db_name=kwargs.get("user_emb_db_name"),
            split_types=track_split_types,
            cache_dir=cache_dir,
            qwen_model_path=kwargs.get("qwen_model_path", "/home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B"),
            device=kwargs.get("device", "cuda"),
        )
    else:
        raise ValueError(f"Unsupported retrieval type: {retrieval_type}")
