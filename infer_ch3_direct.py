"""
infer_ch3_direct.py
────────────────────
直接用 ch3_QwenMeta 召回的 top-20 作为 predicted_track_ids，
跳过精排模型，过 LLaMA 生成推荐理由，输出 Blind-A 评测格式。

用法（服务器）:
    nohup python infer_ch3_direct.py \
        --config    config/llama1b_multi_channel_devset.yaml \
        --conv_emb  qwen/hist_conversation_embeddings_blinda_0.6b.pt \
        --retrieval qwen/retrieval_blinda_candidates.pt \
        --topk      20 \
        --out       exp/inference/blindset_A/ch3_direct_top20.json \
        > logs/infer_ch3_direct.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 复用 mcrs pipeline 里的 LLM 和数据库（和 infer_bagging_blindset.py 保持一致）─
from mcrs.db_item import MusicCatalogDB
from mcrs.db_user import UserProfileDB
from mcrs.lm_modules import load_lm_module


# ─────────────────────────────────────────────────────────────────────────────
#  工具函数（与 infer_bagging_blindset.py 完全一致）
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


def get_system_prompt(user_id, user_db: UserProfileDB, prompts_dir: str) -> str:
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
#  主推理逻辑
# ─────────────────────────────────────────────────────────────────────────────

def _pick_cands(raw, channel: str, topk: int) -> List[str]:
    """从 raw 召回结果里按 channel 策略取 topk 条候选。"""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw[:topk]
    # dict 格式
    if channel == "merged":
        c1 = list(raw.get("ch1", []))
        c3 = list(raw.get("ch3", []))
        c5 = list(raw.get("ch5", []))
        # 各取 top-K 去重合并，保序
        merged = list(dict.fromkeys(c1[:topk] + c3[:topk] + c5[:topk]))
        return merged[:topk]
    else:
        return list(raw.get(channel, []))[:topk]


    dataset_name: str,
    conv_emb_store: dict,
    retrieval_store: dict,
    lm,
    item_db,
    user_db,
    prompts_dir: str,
    topk: int = 20,
    batch_size: int = 8,
):
    ds = load_dataset(dataset_name, split="test")

    pending = []
    n_fallback = 0

    for item in tqdm(ds, desc="Collecting sessions", unit="session"):
        session_id = item["session_id"]
        user_id    = item.get("user_id")
        convs      = item["conversations"]

        # 最后一个 user turn
        user_turns = [int(c["turn_number"]) for c in convs if c.get("role") == "user"]
        target_turn = max(user_turns) if user_turns else 1
        user_query  = next(
            (c["content"] for c in convs
             if int(c["turn_number"]) == target_turn and c.get("role") == "user"),
            ""
        )

        # 找该 session 在 retrieval_store 里的最大 turn（即最后一轮召回）
        session_keys = [k for k in retrieval_store if k.startswith(f"{session_id}_")]
        if session_keys:
            # 取 turn 编号最大的 key
            def _turn(k): 
                try: return int(k.split("_")[-1])
                except: return -1
            last_retr_key = max(session_keys, key=_turn)
            raw = retrieval_store[last_retr_key]
        else:
            raw = None

        # 同时：conv_emb 也用 retrieval 同一 turn 的 key（保持一致）
        emb_key = last_retr_key if session_keys else None
        # 找最近可用的 conv_emb key（往前扫）
        if emb_key not in conv_emb_store:
            emb_key = None
            t_max = _turn(last_retr_key) if session_keys else target_turn
            for t in range(t_max, -1, -1):
                k = f"{session_id}_{t}"
                if k in conv_emb_store:
                    emb_key = k
                    break

        # 取指定 channel 的 top-K 候选
        top_tids = _pick_cands(raw, channel, topk)
        if not top_tids:
            n_fallback += 1
            logger.warning("No retrieval for session %s, will output empty track list.", session_id)

        pending.append({
            "session_id":   session_id,
            "user_id":      user_id,
            "target_turn":  target_turn,
            "user_query":   user_query,
            "conversations":convs,
            "top_tids":     top_tids,
        })

    logger.info("Sessions: %d  (no retrieval: %d)", len(pending), n_fallback)

    # ── LLaMA 生成推荐理由 ────────────────────────────────────────────────────
    logger.info("Generating LLM responses (topk=%d, batch_size=%d) ...", topk, batch_size)
    results = []

    for i in tqdm(range(0, len(pending), batch_size), desc="LLM generation"):
        chunk = pending[i: i + batch_size]
        for p in chunk:
            top_tids    = p["top_tids"]
            recommend_tid = top_tids[0] if top_tids else None

            history = build_chat_history(p["conversations"], item_db, p["target_turn"])
            history.append({"role": "user", "content": p["user_query"]})
            sys_prompt   = get_system_prompt(p["user_id"], user_db, prompts_dir)
            recommend_str = (
                item_db.id_to_metadata(recommend_tid) if recommend_tid
                else "No track found."
            )

            resp = lm.response_generation(sys_prompt, history, recommend_str)

            results.append({
                "session_id":          p["session_id"],
                "user_id":             p["user_id"],
                "turn_number":         p["target_turn"],
                "predicted_track_ids": top_tids,
                "predicted_response":  resp,
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    config       = OmegaConf.load(args.config)
    dataset_name = config.get("test_dataset_name",
                               "talkpl-ai/TalkPlayData-Challenge-Blind-A")
    logger.info("Dataset: %s", dataset_name)

    track_meta_db = config.get("item_db_name",
                                "talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    user_meta_db  = config.get("user_db_name",
                                "talkpl-ai/TalkPlayData-Challenge-User-Metadata")
    split_types   = list(config.get("track_split_types",   ["all_tracks"]))
    user_splits   = list(config.get("user_split_types",    ["all_users"]))
    corpus_types  = list(config.get("corpus_types",
                                     ["track_name", "artist_name", "album_name"]))
    device        = args.device or config.get("device", "cuda")

    # ── 对话 embedding（仅用于 fallback 日志，ch3 召回已预存）────────────────
    logger.info("Loading conv embeddings from %s ...", args.conv_emb)
    conv_emb_store = torch.load(args.conv_emb, map_location="cpu", weights_only=True)
    logger.info("  %d entries.", len(conv_emb_store))

    # ── 召回候选 ─────────────────────────────────────────────────────────────
    logger.info("Loading retrieval candidates from %s ...", args.retrieval)
    retrieval_store = torch.load(args.retrieval, map_location="cpu", weights_only=False)
    logger.info("  %d entries.", len(retrieval_store))

    # ── LLM + DB（与 infer_bagging_blindset.py 完全一致）───────────────────────
    lm_type = config.get("lm_type", "meta-llama/Llama-3.2-1B-Instruct")
    logger.info("Loading LLM: %s ...", lm_type)
    lm = load_lm_module(
        lm_type,
        device,
        config.get("attn_implementation", "eager"),
        torch.bfloat16,
    )
    item_db     = MusicCatalogDB(track_meta_db, split_types, corpus_types)
    user_db     = UserProfileDB(user_meta_db, user_splits)
    prompts_dir = os.path.join(os.path.dirname(__file__), "mcrs", "system_prompts")

    # ── 推理 ─────────────────────────────────────────────────────────────────
    results = run_inference(
        dataset_name    = dataset_name,
        conv_emb_store  = conv_emb_store,
        retrieval_store = retrieval_store,
        lm              = lm,
        item_db         = item_db,
        user_db         = user_db,
        prompts_dir     = prompts_dir,
        channel         = args.channel,
        topk            = args.topk,
        batch_size      = args.batch_size,
    )
    logger.info("Done. %d sessions.", len(results))

    # ── 保存 ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Results saved to %s", args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="ch3 top-K direct → LLaMA → Blind-A submission"
    )
    p.add_argument("--config",     type=str,
                   default="config/llama1b_multi_channel_devset.yaml")
    p.add_argument("--conv_emb",   type=str, required=True,
                   help="Blind-A 对话 embedding .pt")
    p.add_argument("--retrieval",  type=str, required=True,
                   help="Blind-A 召回候选 .pt（含 ch3 字段）")
    p.add_argument("--channel",   type=str, default="ch3",
                   choices=["ch1", "ch3", "ch5", "merged"],
                   help="召回 channel: ch1/ch3/ch5/merged(三路各取top-K去重合并)")
    p.add_argument("--topk",       type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4,
                   help="LLM 生成 batch size")
    p.add_argument("--out",        type=str,
                   default="exp/inference/blindset_A/direct_top20.json")
    p.add_argument("--device",     type=str, default=None)
    args = p.parse_args()
    main(args)
