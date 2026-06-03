"""
Batch inference script for Music CRS V2 (Multi-channel + Three-Tower) on Blind-A.

Usage:
    HF_ENDPOINT=https://hf-mirror.com python run_inference_blindset_v2.py \
        --tid llama1b_multi_channel_blindset_A \
        --batch_size 16
"""

import argparse
import json
import os
import shutil

import pandas as pd
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from mcrs import load_crs_baseline_v2


def chat_history_parser(conversations, music_crs, target_turn_number):
    """Parse conversation history up to target_turn_number."""
    df_conversation = pd.DataFrame(conversations)
    df_history      = df_conversation[df_conversation["turn_number"] < target_turn_number]
    chat_history    = []
    for turn_data in df_history.to_dict(orient="records"):
        current_role    = turn_data["role"]
        current_content = turn_data["content"]
        if turn_data["role"] == "music":
            current_role    = "assistant"
            current_content = music_crs.item_db.id_to_metadata(turn_data["content"])
        chat_history.append({"role": current_role, "content": current_content})
    df_current_turn = df_conversation[df_conversation["turn_number"] == target_turn_number]
    user_query      = df_current_turn.iloc[0]["content"]
    return chat_history, user_query


def main(args):
    # --config 直接指定路径，否则用 --tid 拼路径
    config_path = args.config if args.config else f"config/{args.tid}.yaml"
    config          = OmegaConf.load(config_path)
    pipeline_version = config.get("pipeline_version", "v2")
    qwen_model_path  = config.get(
        "qwen_model_path",
        "/home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B",
    )
    cache_dir = config.cache_dir

    # Clean runtime cache (keep qwen/ turn store untouched)
    print("Cleaning runtime cache ...")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"  Removed {cache_dir}.")
    os.makedirs(cache_dir, exist_ok=True)

    # ── Load V2 pipeline ─────────────────────────────────────────────────────
    print("Loading CRS Baseline V2 (multi-channel + three-tower) for Blind-A ...")
    music_crs = load_crs_baseline_v2(
        lm_type                  = config.lm_type,
        conversation_dataset_name = config.get(
            "conversation_dataset_name",
            "talkpl-ai/TalkPlayData-Challenge-Dataset",
        ),
        item_db_name             = config.item_db_name,
        user_db_name             = config.user_db_name,
        track_emb_db_name        = config.get(
            "track_emb_db_name",
            "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        ),
        user_emb_db_name         = config.get(
            "user_emb_db_name",
            "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
        ),
        track_split_types        = config.track_split_types,
        user_split_types         = config.user_split_types,
        corpus_types             = config.corpus_types,
        cache_dir                = cache_dir,
        qwen_model_path          = qwen_model_path,
        device                   = config.device,
        attn_implementation      = config.attn_implementation,
        dtype                    = torch.bfloat16,
        retrieval_topk           = int(config.get("retrieval_topk", 350)),
        rerank_topk              = int(config.get("rerank_topk", 20)),
        reranker_lr              = float(config.get("reranker_lr", 1e-3)),
    )

    # ── Load Blind-A dataset ─────────────────────────────────────────────────
    test_ds_name = config.get(
        "test_dataset_name", "talkpl-ai/TalkPlayData-Challenge-Blind-A"
    )
    print(f"Loading blind-set: {test_ds_name} ...")
    db = load_dataset(test_ds_name, split="test")

    batch_data, metadata = [], []
    for item in db:
        user_id    = item["user_id"]
        session_id = item["session_id"]
        # Blind-A only has the final turn; use all conversations as history
        conversations = item["conversations"]
        # The last turn is the user query to predict
        target_turn_number = max(int(c["turn_number"]) for c in conversations
                                 if c["role"] == "user")
        chat_history, user_query = chat_history_parser(
            conversations, music_crs, target_turn_number
        )
        batch_data.append({
            "user_query":     user_query,
            "user_id":        user_id,
            "session_memory": chat_history,
            "session_id":     session_id,
            "turn_number":    target_turn_number,
        })
        metadata.append({
            "session_id":  session_id,
            "user_id":     user_id,
            "turn_number": target_turn_number,
        })

    # ── Load turn embedding store ────────────────────────────────────────────
    # --turn_store 直接指定路径，否则用 config 里默认或旧路径
    turn_store_path = (
        args.turn_store
        if args.turn_store
        else config.get("turn_store_path", "qwen/turn_embeddings_blindA.pt")
    )
    if not os.path.exists(turn_store_path):
        print(f"Turn embedding store not found at {turn_store_path}.")
        print("Run:  python precompute_turn_embeddings.py "
              f"--dataset {test_ds_name} --split test --out {turn_store_path}")
        raise FileNotFoundError(
            f"Missing pre-computed turn embeddings: {turn_store_path}\n"
            "Please run precompute_turn_embeddings.py before inference."
        )

    print(f"Loading turn store from {turn_store_path} ...")
    turn_store = torch.load(turn_store_path, map_location="cpu", weights_only=True)
    print(f"  {len(turn_store)} entries loaded.")
    music_crs.set_turn_store(turn_store)

    # ── Run inference ────────────────────────────────────────────────────────
    inference_results = []
    for i in tqdm(range(0, len(batch_data), args.batch_size), desc="Batch inference"):
        batch          = batch_data[i: i + args.batch_size]
        batch_metadata = metadata[i: i + args.batch_size]
        results        = music_crs.batch_chat(batch)
        for j, result in enumerate(results):
            inference_results.append({
                "session_id":          batch_metadata[j]["session_id"],
                "user_id":             batch_metadata[j]["user_id"],
                "turn_number":         batch_metadata[j]["turn_number"],
                "predicted_track_ids": result["retrieval_items"],
                "predicted_response":  result.get("response", ""),
            })

    # ── Save results ─────────────────────────────────────────────────────────
    out_dir  = os.path.join("exp", "inference", args.eval_dataset)
    out_path = os.path.join(out_dir, f"{args.tid}.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(inference_results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {out_path}  ({len(inference_results)} records)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V2 batch inference on Blind-A (Three-Tower + Multi-channel)"
    )
    parser.add_argument(
        "--tid", type=str, default="llama1b_multi_channel_blindset_A",
        help="Config file ID (e.g. llama1b_multi_channel_blindset_A)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Direct path to config yaml (overrides --tid based path)",
    )
    parser.add_argument(
        "--turn_store", type=str, default=None,
        help="Path to blind-A turn store .pt file",
    )
    parser.add_argument(
        "--eval_dataset", type=str, default="blindset_A",
        help="Sub-directory name under exp/inference/ for output",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)
