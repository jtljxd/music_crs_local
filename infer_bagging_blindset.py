"""
infer_bagging_blindset.py — 用训好的 BaggingReranker 对 Blind-A 做推理
=======================================================================

流程：
  1. 加载 FeatureStore（用户/歌曲特征）
  2. 加载 Blind-A 对话 embedding（提前用 precompute_dialogue_embeddings.py 生成）
  3. 加载 Blind-A 召回候选（提前用 precompute_retrieval_candidates.py 生成）
  4. 加载指定 checkpoint 的 LGBM 权重
  5. 对每个 session 用 LGBM 对召回候选打分，输出 top-20
  6. 用 top-1 track 调用 LLaMA 生成推荐理由（predicted_response）
  7. 保存结果到 JSON（格式与官方推理脚本一致）

Usage (服务器):
    nohup python infer_bagging_blindset.py \\
        --config     config/llama1b_multi_channel_devset.yaml \\
        --conv_emb   qwen/hist_conversation_embeddings_blindA_0.6b.pt \\
        --retrieval  qwen/retrieval_blinda_candidates.pt \\
        --checkpoint qwen/bagging_ckpt/epoch3 \\
        --model      lgbm \\
        --topk       20 \\
        --out        exp/inference/blindset_A/bagging_lgbm_epoch3.json \\
    > infer_blindA.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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

# ── 复用训练脚本里的类和常量 ───────────────────────────────────────────────────
from train_bagging_reranker import (
    FeatureStore,
    BaggingReranker,
    CONV_EMB_DIM,
    ENSEMBLE_W,
)

# ── 复用 mcrs pipeline 里的 LLM 和数据库 ─────────────────────────────────────
from mcrs.db_item import MusicCatalogDB
from mcrs.db_user import UserProfileDB
from mcrs.lm_modules import load_lm_module


# ─────────────────────────────────────────────────────────────────────────────
#  对话历史解析（和官方脚本保持一致）
# ─────────────────────────────────────────────────────────────────────────────

def build_chat_history(conversations: List[Dict], item_db: MusicCatalogDB,
                       target_turn: int) -> List[Dict]:
    """构造 LLM 用的对话历史（target_turn 之前的所有轮次）。"""
    history = []
    for c in sorted(conversations, key=lambda x: int(x["turn_number"])):
        tn = int(c["turn_number"])
        if tn >= target_turn:
            break
        role    = c.get("role", "")
        content = c.get("content", "")
        if role == "music":
            role    = "assistant"
            content = item_db.id_to_metadata(content)
        history.append({"role": role, "content": content})
    return history


def get_system_prompt(user_id: Optional[str], user_db: UserProfileDB,
                      prompts_dir: str) -> str:
    """构造 system prompt（带可选个性化信息）。"""
    role_play  = open(f"{prompts_dir}/roleplay.txt",            encoding="utf-8").read()
    resp_gen   = open(f"{prompts_dir}/response_generation.txt", encoding="utf-8").read()
    persona    = open(f"{prompts_dir}/personalization.txt",     encoding="utf-8").read()
    system     = role_play + resp_gen
    if user_id:
        try:
            profile = user_db.id_to_profile_str(user_id)
            system += persona + "\n" + profile
        except Exception:
            pass
    return system


# ─────────────────────────────────────────────────────────────────────────────
#  核心推理函数
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    dataset_name: str,
    conv_emb_store: Dict,
    retrieval_store: Dict,
    feat_store: FeatureStore,
    bagging: BaggingReranker,
    lm,
    item_db: MusicCatalogDB,
    user_db: UserProfileDB,
    prompts_dir: str,
    model_name: str,
    topk: int = 20,
    batch_size: int = 8,
) -> List[Dict]:
    """
    遍历 Blind-A 每个 session，用 BaggingReranker 对召回候选排序，
    取 top-1 用 LLaMA 生成推荐理由，输出与官方格式一致的结果。
    找不到 conv_emb 或召回候选的 session，用全局热门歌曲兜底，不跳过。
    """
    ds = load_dataset(dataset_name, split="test")

    # 全局兜底候选：FeatureStore 里所有 track，按热门度排序取前 topk
    fallback_tids: List[str] = list(feat_store.track_data.keys())[:topk]

    # 收集所有需要推理的样本
    pending = []
    for item in ds:
        session_id = item["session_id"]
        user_id    = item.get("user_id")
        convs      = item["conversations"]

        # Blind-A：找最后一个 user turn
        user_turns = [int(c["turn_number"]) for c in convs if c.get("role") == "user"]
        if not user_turns:
            # 完全没有 user turn，用兜底
            pending.append({
                "session_id":   session_id,
                "user_id":      user_id,
                "target_turn":  1,
                "user_query":   "",
                "conversations":convs,
                "emb_key":      None,
                "cands":        fallback_tids,
                "fallback":     True,
            })
            continue
        target_turn = max(user_turns)
        user_query  = next(
            (c["content"] for c in convs
             if int(c["turn_number"]) == target_turn and c.get("role") == "user"),
            ""
        )

        # 对话 embedding：往前找最近可用的 key
        emb_key = None
        for t in range(target_turn, -1, -1):
            k = f"{session_id}_{t}"
            if k in conv_emb_store:
                emb_key = k
                break

        # 召回候选：先精确匹配 emb_key，找不到则往前扫描最近有召回的 key
        cands: List[str] = []
        if emb_key is not None:
            # 先试精确匹配
            raw = (retrieval_store.get(emb_key)
                   or retrieval_store.get(f"{session_id}_{target_turn}"))
            # 找不到则往前找最近有召回的 key
            if raw is None:
                for t in range(target_turn, -1, -1):
                    k = f"{session_id}_{t}"
                    if k in retrieval_store:
                        raw = retrieval_store[k]
                        logger.debug("Retrieval fallback: %s → %s", emb_key, k)
                        break
            if raw is not None:
                if isinstance(raw, dict):
                    cands = list(raw.get("union", []))
                elif isinstance(raw, list):
                    cands = list(raw)

        if not cands:
            logger.warning("Fallback for %s (turn %d): no emb/retrieval.", session_id, target_turn)
            cands = fallback_tids

        pending.append({
            "session_id":   session_id,
            "user_id":      user_id,
            "target_turn":  target_turn,
            "user_query":   user_query,
            "conversations":convs,
            "emb_key":      emb_key,
            "cands":        cands,
            "fallback":     not bool(emb_key and cands != fallback_tids),
        })

    logger.info("Total sessions to infer: %d (fallback: %d)",
                len(pending), sum(1 for p in pending if p.get("fallback")))

    # ── Step 1: BaggingReranker 打分（批量）──────────────────────────────────
    name_map = {"fm": "FM", "dcn": "DCN", "xdfm": "xDeepFM",
                "lgbm": "LGBM", "ttg": "TTGate"}
    score_key = name_map.get(model_name.lower(), "LGBM")

    ranked_results = []  # [(session_id, user_id, target_turn, user_query, convs, top_tids)]

    for p in tqdm(pending, desc="Reranking"):
        emb_key = p["emb_key"]
        cands   = p["cands"]

        # fallback session：没有 conv_emb，直接用召回顺序作为排名（不打分）
        if emb_key is None:
            top_tids = cands[:topk]
            ranked_results.append((
                p["session_id"], p["user_id"], p["target_turn"],
                p["user_query"], p["conversations"], top_tids
            ))
            continue

        conv_emb = conv_emb_store[emb_key].float()
        if conv_emb.shape[0] > CONV_EMB_DIM:
            conv_emb = conv_emb[:CONV_EMB_DIM]
        elif conv_emb.shape[0] < CONV_EMB_DIM:
            conv_emb = F.pad(conv_emb, (0, CONV_EMB_DIM - conv_emb.shape[0]))

        feats = torch.stack([
            feat_store.build_feature(p["user_id"], tid, conv_emb)
            for tid in cands
        ])
        with torch.no_grad():
            scores_dict = bagging.predict_scores(feats)

        if model_name.lower() == "ensemble":
            score = sum(scores_dict[m] for m in ["FM", "DCN", "xDeepFM", "LGBM", "TTGate"]) * ENSEMBLE_W
        else:
            score = scores_dict[score_key]

        top_idx  = torch.argsort(score, descending=True)[:topk].tolist()
        top_tids = [cands[i] for i in top_idx]

        ranked_results.append((
            p["session_id"], p["user_id"], p["target_turn"],
            p["user_query"], p["conversations"], top_tids
        ))

    # ── Step 2: LLaMA 生成推荐理由（批量）──────────────────────────────────
    logger.info("Generating LLM responses (batch_size=%d) ...", batch_size)
    results = []

    for i in tqdm(range(0, len(ranked_results), batch_size), desc="LLM generation"):
        chunk = ranked_results[i: i + batch_size]

        sys_prompts, chat_histories, recommend_strs = [], [], []
        for (sid, uid, turn, query, convs, top_tids) in chunk:
            recommend_tid = top_tids[0] if top_tids else None
            history = build_chat_history(convs, item_db, turn)
            history.append({"role": "user", "content": query})
            sys_prompts.append(get_system_prompt(uid, user_db, prompts_dir))
            chat_histories.append(history)
            recommend_strs.append(
                item_db.id_to_metadata(recommend_tid) if recommend_tid
                else "No track found."
            )

        # 逐条生成，避免 OOM
        responses = []
        for sp, ch, ri in zip(sys_prompts, chat_histories, recommend_strs):
            resp = lm.response_generation(sp, ch, ri)
            responses.append(resp)

        for j, (sid, uid, turn, query, convs, top_tids) in enumerate(chunk):
            results.append({
                "session_id":          sid,
                "user_id":             uid,
                "turn_number":         turn,
                "predicted_track_ids": top_tids,
                "predicted_response":  responses[j],
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    config = OmegaConf.load(args.config)

    # 数据集名
    dataset_name  = config.get(
        "test_dataset_name", "talkpl-ai/TalkPlayData-Challenge-Blind-A"
    )
    logger.info("Dataset: %s", dataset_name)

    # HuggingFace db 名
    track_emb_db  = config.get("track_emb_db_name",
                                "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    track_meta_db = config.get("item_db_name",
                                "talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    user_emb_db   = config.get("user_emb_db_name",
                                "talkpl-ai/TalkPlayData-Challenge-User-Embeddings")
    user_meta_db  = config.get("user_db_name",
                                "talkpl-ai/TalkPlayData-Challenge-User-Metadata")
    split_types   = list(config.get("track_split_types", ["all_tracks"]))
    corpus_types  = list(config.get("corpus_types",
                                     ["track_name", "artist_name", "album_name"]))
    device        = args.device or config.get("device", "cuda")

    # ── FeatureStore ─────────────────────────────────────────────────────────
    logger.info("Loading FeatureStore ...")
    feat_store = FeatureStore(track_emb_db, track_meta_db,
                              user_emb_db, user_meta_db, split_types)

    # ── 对话 embedding ───────────────────────────────────────────────────────
    logger.info("Loading conv embeddings from %s ...", args.conv_emb)
    conv_emb_store: Dict = torch.load(
        args.conv_emb, map_location="cpu", weights_only=True
    )
    logger.info("  %d entries loaded.", len(conv_emb_store))

    # ── 召回候选 ─────────────────────────────────────────────────────────────
    logger.info("Loading retrieval candidates from %s ...", args.retrieval)
    retrieval_store: Dict = torch.load(
        args.retrieval, map_location="cpu", weights_only=False
    )
    logger.info("  %d entries loaded.", len(retrieval_store))

    # ── BaggingReranker ──────────────────────────────────────────────────────
    from types import SimpleNamespace
    train_args = SimpleNamespace(lr=1e-3, lr_finetune=3e-4, lgbm_max_samples=80000)
    bagging = BaggingReranker(feat_store, train_args, device)
    logger.info("Loading checkpoint from %s ...", args.checkpoint)
    bagging.load(args.checkpoint)
    bagging.fm.eval(); bagging.dcn.eval(); bagging.xdfm.eval(); bagging.ttg.eval()
    logger.info("Checkpoint loaded.")

    # ── LLM + MusicCatalogDB + UserProfileDB ────────────────────────────────
    lm_type = config.get("lm_type", "meta-llama/Llama-3.2-1B-Instruct")
    logger.info("Loading LLM: %s ...", lm_type)
    lm = load_lm_module(
        lm_type,
        device,
        config.get("attn_implementation", "eager"),
        torch.bfloat16,
    )
    item_db = MusicCatalogDB(track_meta_db, split_types, corpus_types)
    user_db = UserProfileDB(user_meta_db, list(config.get("user_split_types", ["all_users"])))
    prompts_dir = os.path.join(os.path.dirname(__file__), "mcrs", "system_prompts")

    # ── 推理 ─────────────────────────────────────────────────────────────────
    logger.info("Running inference (model=%s, topk=%d) ...", args.model, args.topk)
    results = run_inference(
        dataset_name    = dataset_name,
        conv_emb_store  = conv_emb_store,
        retrieval_store = retrieval_store,
        feat_store      = feat_store,
        bagging         = bagging,
        lm              = lm,
        item_db         = item_db,
        user_db         = user_db,
        prompts_dir     = prompts_dir,
        model_name      = args.model,
        topk            = args.topk,
        batch_size      = args.batch_size,
    )
    logger.info("Done. %d sessions processed.", len(results))

    # ── 保存 ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Results saved to %s  (%d records)", args.out, len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BaggingReranker inference on Blind-A with LLaMA response generation"
    )
    parser.add_argument("--config",     type=str, required=True,
                        help="Config yaml（同训练脚本）")
    parser.add_argument("--conv_emb",   type=str, required=True,
                        help="Blind-A 对话 embedding .pt")
    parser.add_argument("--retrieval",  type=str, required=True,
                        help="Blind-A 召回候选 .pt")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint 目录 (e.g. qwen/bagging_ckpt/epoch3)")
    parser.add_argument("--model",      type=str, default="lgbm",
                        choices=["fm", "dcn", "xdfm", "lgbm", "ttg", "ensemble"],
                        help="打分模型")
    parser.add_argument("--topk",       type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="LLM 生成的 batch size（过大会 OOM）")
    parser.add_argument("--out",        type=str,
                        default="exp/inference/blindset_A/bagging_lgbm_epoch3.json")
    parser.add_argument("--device",     type=str, default=None)
    args = parser.parse_args()
    main(args)
