"""
Batch inference script for Music CRS with support for V2 pipeline.

For the V2 pipeline, all Qwen text embeddings are pre-computed once via
precompute_turn_embeddings.py and stored in cache/turn_embeddings.pt.
During inference, retrieval and reranker look up embeddings by
(session_id, turn_number) with O(1) dict access — no Qwen forward passes.
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

    Returns:
        (chat_history: List[Dict], user_query: str)
    """
    df_conversation = pd.DataFrame(conversations)
    df_history      = df_conversation[df_conversation['turn_number'] < target_turn_number]
    chat_history    = []
    for turn_data in df_history.to_dict(orient="records"):
        current_role    = turn_data['role']
        current_content = turn_data['content']
        if turn_data['role'] == "music":
            current_role    = "assistant"
            current_content = music_crs.item_db.id_to_metadata(turn_data['content'])
        chat_history.append({"role": current_role, "content": current_content})
    df_current_turn = df_conversation[df_conversation['turn_number'] == target_turn_number]
    user_query      = df_current_turn.iloc[0]['content']
    return chat_history, user_query


def main(args):
    config           = OmegaConf.load(f"config/{args.tid}.yaml")
    pipeline_version = config.get("pipeline_version", "v1")
    qwen_model_path  = config.get(
        "qwen_model_path",
        "/home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B",
    )
    cache_dir = config.cache_dir

    # Remove runtime cache (indices, FM checkpoints etc.) but preserve
    # precomputed turn embeddings so they don't need to be regenerated.
    print("Cleaning runtime cache ...")
    import shutil
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"  Removed {cache_dir}.")
    os.makedirs(cache_dir, exist_ok=True)

    # Turn embedding store lives in qwen/ (outside cache/), never gets wiped
    turn_store_path = os.path.join("qwen", "turn_embeddings.pt")

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
                'user_query':     user_query,
                'user_id':        user_id,
                'session_memory': chat_history,
                'session_id':     session_id,       # ← new
                'turn_number':    target_turn_number,  # ← new
            })
            metadata.append({
                'session_id':  session_id,
                'user_id':     user_id,
                'turn_number': target_turn_number,
            })

    # ── Load / build turn-embedding store (V2 only) ───────────────────────
    if pipeline_version == "v2":
        turn_store_path = os.path.join(cache_dir, "turn_embeddings.pt")

        if not os.path.exists(turn_store_path):
            print(f"Turn embedding store not found at {turn_store_path}.")
            print("Run first:  python precompute_turn_embeddings.py "
                  f"--out {turn_store_path} --split test")
            raise FileNotFoundError(
                f"Missing pre-computed turn embeddings: {turn_store_path}\n"
                "Please run precompute_turn_embeddings.py before inference."
            )

        print(f"Loading turn embedding store from {turn_store_path} …")
        turn_store = torch.load(turn_store_path, map_location="cpu")
        print(f"  {len(turn_store)} entries loaded.")
        music_crs.set_turn_store(turn_store)

    # ── Run inference ─────────────────────────────────────────────────────
    inference_results = []
    for i in tqdm(range(0, len(batch_data), args.batch_size), desc="Batch inference"):
        batch          = batch_data[i:i + args.batch_size]
        batch_metadata = metadata[i:i + args.batch_size]
        results        = music_crs.batch_chat(batch)
        for j, result in enumerate(results):
            inference_results.append({
                "session_id":          batch_metadata[j]['session_id'],
                "user_id":             batch_metadata[j]['user_id'],
                "turn_number":         batch_metadata[j]['turn_number'],
                "predicted_track_ids": result['retrieval_items'],
                "predicted_response":  result["response"],
            })

    os.makedirs("exp/inference/devset", exist_ok=True)
    out_path = f"exp/inference/devset/{args.tid}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(inference_results, f, ensure_ascii=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch inference for Music CRS (V1 / V2 pipeline)."
    )
    parser.add_argument(
        "--tid", type=str, default="llama1b_multi_channel_devset",
        help="Config identifier (loads config/{tid}.yaml)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--save_path", type=str, default="./exp/inference",
        help="(Unused) base output directory.",
    )
    args = parser.parse_args()
    main(args)
