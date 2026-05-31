"""
Batch inference script for Music CRS with support for V2 pipeline.

For the V2 pipeline, all Qwen text embeddings (user queries + history contexts)
are pre-computed in a single batch pass BEFORE inference begins.  This avoids
repeated CPU-Qwen forward passes during the hot inference loop.
"""

import os
import json
import torch
import argparse
from mcrs import load_crs_baseline, load_crs_baseline_v2
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import pandas as pd
from omegaconf import OmegaConf


def chat_history_parser(conversations, music_crs, target_turn_number):
    """
    Parse conversation history up to a target turn.

    Args:
        conversations (List[Dict]): List of conversation turn dictionaries containing:
            - turn_number: Turn index (1-8)
            - role: Speaker role ('user', 'assistant', 'music')
            - content: Message content or track ID
        music_crs: CRS baseline instance (used to convert track IDs to metadata)
        target_turn_number (int): The turn to predict (history excludes this turn)

    Returns:
        Tuple[List[Dict], str]:
            - chat_history: List of previous messages formatted as [{"role": ..., "content": ...}]
            - user_query: The user query at the target turn
    """
    df_conversation = pd.DataFrame(conversations)
    df_history = df_conversation[df_conversation['turn_number'] < target_turn_number]
    chat_history = []
    for turn_data in df_history.to_dict(orient="records"):
        current_role    = turn_data['role']
        current_content = turn_data['content']
        if turn_data['role'] == "music":
            current_role    = "assistant"
            current_content = music_crs.item_db.id_to_metadata(turn_data['content'])
        chat_history.append({"role": current_role, "content": current_content})
    df_current_turn = df_conversation[df_conversation['turn_number'] == target_turn_number]
    user_query = df_current_turn.iloc[0]['content']
    return chat_history, user_query


def _format_history_context(chat_history: List[Dict]) -> str:
    """Format chat history as a string (mirrors crs_baseline_v2._format_history_context)."""
    return "\n".join(f"{m['role']}: {m['content']}" for m in chat_history)


def _collect_all_texts(batch_data: List[Dict]) -> List[str]:
    """Collect every unique text that will need a Qwen embedding during inference."""
    texts = set()
    for d in batch_data:
        query   = d.get("user_query", "")
        history = _format_history_context(d.get("session_memory", []))
        # history queries (individual user turns)
        for msg in d.get("session_memory", []):
            if msg.get("role") == "user":
                texts.add(msg["content"])
        if query:
            texts.add(query)
        if history:
            texts.add(history)
        # conversation_goal fields (if present)
        goal = d.get("conversation_goal") or {}
        for field in ("listener_goal", "category"):
            val = goal.get(field, "")
            if val:
                texts.add(val)
    return [t for t in texts if t and t.strip()]


def precompute_qwen_embeddings(
    batch_data: List[Dict],
    qwen_model_path: str,
    cache_file: str,
    batch_size: int = 128,
):
    """Pre-compute Qwen embeddings for ALL texts in batch_data.

    Skips already-cached texts so re-runs are fast.
    """
    from mcrs.qwen_cache import QwenEmbeddingCache

    print("Collecting unique texts for Qwen precompute …")
    texts = _collect_all_texts(batch_data)
    print(f"  {len(texts)} unique non-empty texts found.")

    cache = QwenEmbeddingCache(qwen_model_path, cache_file=cache_file)

    missing = [t for t in texts if t not in cache]
    print(f"  {len(missing)} texts not yet cached; encoding now …")
    if missing:
        cache.precompute(missing, batch_size=batch_size)
    else:
        print("  All texts already cached — skipping Qwen forward pass.")

    return cache


def main(args):
    """
    Run batch inference on TalkPlayData-2 test dataset.
    """
    print("Removing cache directory for preventing memory issues...")
    os.system("rm -rf cache")
    config           = OmegaConf.load(f"config/{args.tid}.yaml")
    pipeline_version = config.get("pipeline_version", "v1")
    qwen_model_path  = config.get("qwen_model_path",
                                   "/home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B")
    cache_dir        = config.cache_dir

    # ── Load the CRS pipeline ─────────────────────────────────────────────
    if pipeline_version == "v2":
        print("Loading CRS Baseline V2 (multi-channel + three-tower)...")
        music_crs = load_crs_baseline_v2(
            lm_type=config.lm_type,
            conversation_dataset_name=config.get(
                "conversation_dataset_name",
                "talkpl-ai/TalkPlayData-Challenge-Dataset",
            ),
            item_db_name=config.item_db_name,
            user_db_name=config.user_db_name,
            track_emb_db_name=config.get(
                "track_emb_db_name",
                "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
            ),
            user_emb_db_name=config.get(
                "user_emb_db_name",
                "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
            ),
            track_split_types=config.track_split_types,
            user_split_types=config.user_split_types,
            corpus_types=config.corpus_types,
            cache_dir=cache_dir,
            qwen_model_path=qwen_model_path,
            device=config.device,
            attn_implementation=config.attn_implementation,
            dtype=torch.bfloat16,
            retrieval_topk=int(config.get("retrieval_topk", 350)),
            rerank_topk=int(config.get("rerank_topk", 20)),
            reranker_lr=float(config.get("reranker_lr", 1e-3)),
        )
    else:
        print("Loading CRS Baseline V1 (original)...")
        music_crs = load_crs_baseline(
            lm_type=config.lm_type,
            retrieval_type=config.retrieval_type,
            item_db_name=config.item_db_name,
            user_db_name=config.user_db_name,
            track_emb_db_name=config.get(
                "track_emb_db_name",
                "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
            ),
            user_emb_db_name=config.get(
                "user_emb_db_name",
                "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
            ),
            track_split_types=config.track_split_types,
            user_split_types=config.user_split_types,
            corpus_types=config.corpus_types,
            cache_dir=cache_dir,
            device=config.device,
            attn_implementation=config.attn_implementation,
            dtype=torch.bfloat16,
            retrieval_topk=int(config.get("retrieval_topk", 20)),
            fm_k=int(config.get("fm_k", 16)),
            fm_lr=float(config.get("fm_lr", 1e-3)),
            fm_top_tags=int(config.get("fm_top_tags", 50)),
        )

    # ── Build all batch data ──────────────────────────────────────────────
    db = load_dataset(config.test_dataset_name, split="test")
    batch_data, metadata = [], []
    for item in db:
        user_id    = item['user_id']
        session_id = item['session_id']
        for target_turn_number in range(1, 9):
            chat_history, user_query = chat_history_parser(
                item['conversations'], music_crs, target_turn_number
            )
            batch_data.append({
                'user_query':    user_query,
                'user_id':       user_id,
                'session_memory': chat_history,
            })
            metadata.append({
                'session_id':  session_id,
                'user_id':     user_id,
                'turn_number': target_turn_number,
            })

    # ── Pre-compute Qwen embeddings (V2 only) ─────────────────────────────
    if pipeline_version == "v2":
        qwen_cache_file = os.path.join(cache_dir, "qwen_text_emb.pt")
        qwen_cache = precompute_qwen_embeddings(
            batch_data,
            qwen_model_path=qwen_model_path,
            cache_file=qwen_cache_file,
            batch_size=128,
        )
        # Inject the pre-computed cache into the pipeline modules
        music_crs.retrieval.set_embedding_cache(qwen_cache)
        music_crs.reranker.set_embedding_cache(qwen_cache)
        print(f"Qwen cache injected: {len(qwen_cache)} entries.")

    # ── Run inference ─────────────────────────────────────────────────────
    inference_results = []
    for i in tqdm(range(0, len(batch_data), args.batch_size), desc="Batch inference"):
        batch          = batch_data[i:i + args.batch_size]
        batch_metadata = metadata[i:i + args.batch_size]
        results        = music_crs.batch_chat(batch)
        for j, result in enumerate(results):
            inference_results.append({
                "session_id":         batch_metadata[j]['session_id'],
                "user_id":            batch_metadata[j]['user_id'],
                "turn_number":        batch_metadata[j]['turn_number'],
                "predicted_track_ids": result['retrieval_items'],
                "predicted_response": result["response"],
            })

    os.makedirs("exp/inference/devset", exist_ok=True)
    out_path = f"exp/inference/devset/{args.tid}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(inference_results, f, ensure_ascii=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch inference on TalkPlayData-2 test dataset for Music CRS evaluation."
    )
    parser.add_argument(
        "--tid",
        type=str,
        default="llama1b_multi_channel_devset",
        help="Task identifier matching a config file (e.g., 'llama1b_multi_channel_devset')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of queries per batch.  Reduce if OOM.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./exp/inference",
        help="(Unused) base output directory.",
    )
    args = parser.parse_args()
    main(args)
