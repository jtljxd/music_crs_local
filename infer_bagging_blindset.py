"""
infer_bagging_blindset.py — 用训好的 BaggingReranker 对 Blind-A 做推理
=======================================================================

流程：
  1. 加载 FeatureStore（用户/歌曲特征）
  2. 加载 Blind-A 对话 embedding（需提前用 precompute_dialogue_embeddings.py 生成）
  3. 加载 Blind-A 召回候选（需提前用 precompute_retrieval_candidates.py 生成）
  4. 加载指定 checkpoint 的模型权重
  5. 对每个 session 的目标 turn，用指定模型对召回候选打分，输出 top-20
  6. 保存结果到 JSON

Usage (服务器):
    # Step 1: 预计算 blind-A 对话 embedding（如果还没有）
    python precompute_dialogue_embeddings.py \\
        --dataset talkpl-ai/TalkPlayData-Challenge-Blind-A \\
        --split test \\
        --out qwen/hist_conversation_embeddings_blindA_0.6b.pt

    # Step 2: 预计算 blind-A 召回候选（如果还没有）
    nohup python precompute_retrieval_candidates.py \\
        --config config/llama1b_multi_channel_blindset_A.yaml \\
        --conv_emb_store    qwen/hist_conversation_embeddings_blindA_0.6b.pt \\
        --query_split_store qwen/query_split_blindA.pt \\
        --split test \\
        --out qwen/retrieval_blindA_candidates.pt \\
    > precompute_retr_blindA.log 2>&1 &

    # Step 3: 推理
    nohup python infer_bagging_blindset.py \\
        --config      config/llama1b_multi_channel_devset.yaml \\
        --conv_emb    qwen/hist_conversation_embeddings_blindA_0.6b.pt \\
        --retrieval   qwen/retrieval_blindA_candidates.pt \\
        --checkpoint  qwen/bagging_ckpt/epoch3 \\
        --model       lgbm \\
        --topk        20 \\
        --out         exp/inference/blindset_A/bagging_lgbm_epoch3.json \\
    > infer_blindA.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 直接 import 训练脚本里的所有类和常量 ─────────────────────────────────────
from train_bagging_reranker import (
    FeatureStore,
    BaggingReranker,
    FEATURE_DIM,
    CONV_EMB_DIM,
    ENSEMBLE_W,
)


# ─────────────────────────────────────────────────────────────────────────────
#  核心推理函数
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    dataset_name: str,
    conv_emb_store: Dict,
    retrieval_store: Dict,
    feat_store: FeatureStore,
    bagging: BaggingReranker,
    model_name: str,
    topk: int = 20,
) -> List[Dict]:
    """
    遍历 Blind-A 的每个 session，找到最后一个 user 发言作为目标 turn，
    用 model_name 对召回候选打分，输出 top-k。

    model_name: "fm" | "dcn" | "xdfm" | "lgbm" | "ttg" | "ensemble"
    """
    ds = load_dataset(dataset_name, split="test")
    results = []

    for item in tqdm(ds, desc="Inference"):
        session_id = item["session_id"]
        user_id    = item.get("user_id")
        convs      = item["conversations"]

        # 找目标 turn：最后一个 user 发言
        user_turns = [int(c["turn_number"]) for c in convs if c.get("role") == "user"]
        if not user_turns:
            continue
        target_turn = max(user_turns)

        # 对话 embedding
        emb_key = f"{session_id}_{target_turn}"
        if emb_key not in conv_emb_store:
            emb_key = f"{session_id}_{target_turn - 1}"
        if emb_key not in conv_emb_store:
            logger.warning("No conv_emb for %s (turn %d), skipping.", session_id, target_turn)
            continue
        conv_emb = conv_emb_store[emb_key].float()
        if conv_emb.shape[0] > CONV_EMB_DIM:
            conv_emb = conv_emb[:CONV_EMB_DIM]
        elif conv_emb.shape[0] < CONV_EMB_DIM:
            conv_emb = F.pad(conv_emb, (0, CONV_EMB_DIM - conv_emb.shape[0]))

        # 召回候选
        raw = (retrieval_store.get(emb_key)
               or retrieval_store.get(f"{session_id}_{target_turn}"))
        if raw is None:
            logger.warning("No retrieval candidates for %s (turn %d), skipping.", session_id, target_turn)
            continue
        if isinstance(raw, dict):
            cands: List[str] = list(raw.get("union", []))
        elif isinstance(raw, list):
            cands = list(raw)
        else:
            cands = []
        if not cands:
            continue

        # 构建特征
        feats = torch.stack([
            feat_store.build_feature(user_id, tid, conv_emb, r, len(cands))
            for r, tid in enumerate(cands)
        ])

        # 打分
        model_name_lower = model_name.lower()
        if model_name_lower == "ensemble":
            scores_dict = bagging.predict_scores(feats)
            score = sum(scores_dict[m] for m in ["FM", "DCN", "xDeepFM", "LGBM", "TTGate"]) * ENSEMBLE_W
        else:
            name_map = {"fm": "FM", "dcn": "DCN", "xdfm": "xDeepFM",
                        "lgbm": "LGBM", "ttg": "TTGate"}
            scores_dict = bagging.predict_scores(feats)
            score = scores_dict[name_map[model_name_lower]]

        top_idx  = torch.argsort(score, descending=True)[:topk].tolist()
        top_tids = [cands[i] for i in top_idx]

        results.append({
            "session_id":          session_id,
            "user_id":             user_id,
            "turn_number":         target_turn,
            "predicted_track_ids": top_tids,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # ── 1. 加载 config ──────────────────────────────────────────────────────
    config = OmegaConf.load(args.config)
    dataset_name = config.get(
        "test_dataset_name", "talkpl-ai/TalkPlayData-Challenge-Blind-A"
    )
    logger.info("Dataset: %s", dataset_name)

    # ── 2. 加载 FeatureStore ────────────────────────────────────────────────
    logger.info("Loading FeatureStore ...")
    feat_store = FeatureStore(config, dataset_name="train")

    # ── 3. 加载对话 embedding ───────────────────────────────────────────────
    logger.info("Loading conv embeddings from %s ...", args.conv_emb)
    conv_emb_store: Dict = torch.load(args.conv_emb, map_location="cpu", weights_only=True)
    logger.info("  %d entries loaded.", len(conv_emb_store))

    # ── 4. 加载召回候选 ─────────────────────────────────────────────────────
    logger.info("Loading retrieval candidates from %s ...", args.retrieval)
    retrieval_store: Dict = torch.load(args.retrieval, map_location="cpu", weights_only=False)
    logger.info("  %d entries loaded.", len(retrieval_store))

    # ── 5. 初始化并加载 BaggingReranker ────────────────────────────────────
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # args 里需要有 lr / lgbm_max_samples，用 SimpleNamespace 补全
    from types import SimpleNamespace
    train_args = SimpleNamespace(
        lr=1e-3,
        lr_finetune=3e-4,
        lgbm_max_samples=80000,
    )
    bagging = BaggingReranker(feat_store, train_args, device)

    logger.info("Loading checkpoint from %s ...", args.checkpoint)
    bagging.load(args.checkpoint)
    logger.info("Checkpoint loaded.")

    # 设置为推理模式
    bagging.fm.eval(); bagging.dcn.eval(); bagging.xdfm.eval(); bagging.ttg.eval()

    # ── 6. 推理 ─────────────────────────────────────────────────────────────
    logger.info("Running inference with model=%s, topk=%d ...", args.model, args.topk)
    with torch.no_grad():
        results = run_inference(
            dataset_name   = dataset_name,
            conv_emb_store = conv_emb_store,
            retrieval_store= retrieval_store,
            feat_store     = feat_store,
            bagging        = bagging,
            model_name     = args.model,
            topk           = args.topk,
        )
    logger.info("Inference done. %d sessions processed.", len(results))

    # ── 7. 保存结果 ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Results saved to %s  (%d records)", args.out, len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BaggingReranker inference on Blind-A")
    parser.add_argument("--config",     type=str, required=True,
                        help="Config yaml (same as training)")
    parser.add_argument("--conv_emb",   type=str, required=True,
                        help="Blind-A conversation embeddings .pt")
    parser.add_argument("--retrieval",  type=str, required=True,
                        help="Blind-A retrieval candidates .pt")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint dir (e.g. qwen/bagging_ckpt/epoch3)")
    parser.add_argument("--model",      type=str, default="lgbm",
                        choices=["fm", "dcn", "xdfm", "lgbm", "ttg", "ensemble"],
                        help="Which model to use for scoring")
    parser.add_argument("--topk",       type=int, default=20,
                        help="Number of top tracks to return")
    parser.add_argument("--out",        type=str,
                        default="exp/inference/blindset_A/bagging_lgbm_epoch3.json",
                        help="Output JSON path")
    parser.add_argument("--device",     type=str, default=None)
    args = parser.parse_args()
    main(args)
