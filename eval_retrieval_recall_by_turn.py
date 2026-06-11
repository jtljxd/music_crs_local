"""
eval_retrieval_recall_by_turn.py
─────────────────────────────────
按轮次（turn）分组统计各路召回的 Recall@K，同时输出全量汇总。

支持模式:
  --split train   → 评估 train split 全量
  --split test    → 评估 test  split 全量
  --split all     → 同时输出 train + test

输出维度:
  - 全量汇总（all turns）
  - 按 turn_num 分组（turn 1 / 2 / 3 / ...）

用法:
    python eval_retrieval_recall_by_turn.py \
        --config      config/llama1b_multi_channel_devset.yaml \
        --retrieval_train qwen/retrieval_train_candidates.pt \
        --retrieval_test  qwen/retrieval_test_candidates.pt \
        --split all \
        --output      eval_retrieval_by_turn.txt
"""

import argparse
import logging
import os
import sys
from collections import defaultdict

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

K_LIST        = [20, 50, 100, 200, 300, 400, 500]
CHANNELS      = ["ch1", "ch3", "ch5"]
# merged@K = ch1[:K] ∪ ch3[:K] ∪ ch5[:K] 去重，最多 3K 条
# 用于公平对比"各路各取 top-K 合并后"的召回上限
MERGED_K_LIST = [20, 50, 100, 200, 300]


# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dataset_name(cfg: dict) -> str:
    for key in ("conversation_dataset_name", "test_dataset_name",
                "dataset", "dataset_name", "data", "hf_dataset"):
        if key in cfg:
            return cfg[key]
    raise ValueError(f"Cannot find dataset name in config: {cfg}")


def parse_ch_cands(raw) -> dict:
    """把 retrieval_store 里的原始值统一转成 {ch: [track_id, ...]} 格式。"""
    if isinstance(raw, dict):
        return {
            "ch1": list(raw.get("ch1", [])),
            "ch3": list(raw.get("ch3", [])),
            "ch5": list(raw.get("ch5", [])),
        }
    elif isinstance(raw, list):
        return {"ch1": [], "ch3": [], "ch5": []}
    return {"ch1": [], "ch3": [], "ch5": []}


# ─────────────────────────────────────────────────────────────────────────────

def evaluate_split(dataset_name: str,
                   split: str,
                   retrieval_store: dict,
                   k_list: list = K_LIST,
                   channels: list = CHANNELS) -> dict:
    """
    Returns:
        {
          "all":     {ch: {k: recall_float}},   # 全量
          "by_turn": {turn_num: {ch: {k: recall_float}}},
          "meta":    {"n_sessions": int, "n_turns_total": int, ...}
        }
    """
    ds = load_dataset(dataset_name, split=split)

    # hits[turn_group][ch][k]  — "all" 表示全量
    def new_counter():
        return {ch: {k: 0 for k in k_list} for ch in channels}

    hits_all   = new_counter()
    total_all  = {ch: 0 for ch in channels}

    hits_by_turn  = defaultdict(new_counter)   # turn_num → counter
    total_by_turn = defaultdict(lambda: {ch: 0 for ch in channels})

    # merged@K: ch1[:K] ∪ ch3[:K] ∪ ch5[:K] 去重
    hits_merged_all     = {k: 0 for k in MERGED_K_LIST}
    total_merged_all    = 0
    hits_merged_by_turn = defaultdict(lambda: {k: 0 for k in MERGED_K_LIST})
    total_merged_by_turn = defaultdict(int)

    n_sessions    = 0
    n_turns_total = 0
    missing_retr  = 0   # 有 music turn 但找不到召回结果

    for item in tqdm(ds, desc=f"[{split}] Evaluating", unit="session"):
        session_id = item["session_id"]
        convs      = item["conversations"]

        music_turns = {
            int(c["turn_number"]): c["content"]
            for c in convs
            if c.get("role") == "music" and c.get("content")
        }
        if not music_turns:
            continue
        n_sessions += 1

        for turn_num, gt_tid in music_turns.items():
            n_turns_total += 1

            # 取召回结果
            emb_key  = f"{session_id}_{turn_num}"
            emb_key2 = f"{session_id}_{turn_num - 1}"
            raw = retrieval_store.get(emb_key) or retrieval_store.get(emb_key2)

            if raw is None:
                missing_retr += 1
                continue

            ch_cands = parse_ch_cands(raw)

            for ch in channels:
                cands = ch_cands[ch]
                if not cands:
                    continue

                # 全量
                total_all[ch] += 1
                total_by_turn[turn_num][ch] += 1

                for k in k_list:
                    hit = 1 if gt_tid in cands[:k] else 0
                    hits_all[ch][k]              += hit
                    hits_by_turn[turn_num][ch][k] += hit

            # merged@K: ch1[:K] ∪ ch3[:K] ∪ ch5[:K] 去重
            c1 = ch_cands["ch1"]
            c3 = ch_cands["ch3"]
            c5 = ch_cands["ch5"]
            if c1 or c3 or c5:
                total_merged_all += 1
                total_merged_by_turn[turn_num] += 1
                for k in MERGED_K_LIST:
                    merged = list(dict.fromkeys(c1[:k] + c3[:k] + c5[:k]))  # 保序去重
                    hit = 1 if gt_tid in merged else 0
                    hits_merged_all[k]              += hit
                    hits_merged_by_turn[turn_num][k] += hit

    logger.info("[%s] sessions=%d  music_turns=%d  missing_retr=%d",
                split, n_sessions, n_turns_total, missing_retr)

    def to_recall(hits_dict, total_dict):
        return {
            ch: {
                k: hits_dict[ch][k] / max(total_dict[ch], 1)
                for k in k_list
            }
            for ch in channels
        }

    result = {
        "all":     to_recall(hits_all, total_all),
        "by_turn": {
            t: to_recall(hits_by_turn[t], total_by_turn[t])
            for t in sorted(hits_by_turn.keys())
        },
        "merged_all": {
            k: hits_merged_all[k] / max(total_merged_all, 1)
            for k in MERGED_K_LIST
        },
        "merged_by_turn": {
            t: {k: hits_merged_by_turn[t][k] / max(total_merged_by_turn[t], 1)
                for k in MERGED_K_LIST}
            for t in sorted(hits_merged_by_turn.keys())
        },
        "meta": {
            "split":          split,
            "n_sessions":     n_sessions,
            "n_turns_total":  n_turns_total,
            "missing_retr":   missing_retr,
            "total_all":      total_all,
            "total_merged":   total_merged_all,
        }
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────

def format_table(result: dict, title: str) -> str:
    from datetime import datetime

    meta   = result["meta"]
    k_list = K_LIST
    chs    = CHANNELS

    col_w  = 9
    hdr    = "  ".join(f"@{k:<{col_w-2}}" for k in k_list)
    sep    = "=" * (14 + (col_w + 2) * len(k_list))

    lines = [
        "",
        f"{'='*60}",
        f"  {title}",
        f"  split={meta['split']}  sessions={meta['n_sessions']}"
        f"  turns={meta['n_turns_total']}  missing_retr={meta['missing_retr']}",
        f"{'='*60}",
        "",
        "── ALL TURNS (全量) " + "─" * 40,
        f"  {'Channel':<14}  {hdr}",
        "-" * len(sep),
    ]
    for ch in chs:
        row = result["all"][ch]
        vals = "  ".join(f"{row[k]*100:>{col_w-1}.2f}%" for k in k_list)
        n    = meta["total_all"].get(ch, 0)
        lines.append(f"  {ch:<14}  {vals}   (n={n})")

    # merged 汇总行
    merged_hdr = "  ".join(f"@{k:<{col_w-2}}" for k in MERGED_K_LIST)
    lines.append("  " + "-" * (len(sep) - 2))
    lines.append(f"  {'merged':<14}  {merged_hdr}   ← ch1[:K]∪ch3[:K]∪ch5[:K] 去重，最多3K条")
    mrow  = result["merged_all"]
    mvals = "  ".join(f"{mrow[k]*100:>{col_w-1}.2f}%" for k in MERGED_K_LIST)
    lines.append(f"  {'':14}  {mvals}   (n={meta['total_merged']})")
    lines.append("")

    # by turn
    lines.append("── BY TURN (分轮次) " + "─" * 40)
    for turn_num, turn_data in result["by_turn"].items():
        lines.append(f"  Turn {turn_num}:")
        lines.append(f"    {'Channel':<12}  {hdr}")
        lines.append("    " + "-" * (len(sep) - 4))
        for ch in chs:
            row  = turn_data[ch]
            vals = "  ".join(f"{row[k]*100:>{col_w-1}.2f}%" for k in k_list)
            lines.append(f"    {ch:<12}  {vals}")
        # merged for this turn
        mrow  = result["merged_by_turn"].get(turn_num, {k: 0.0 for k in MERGED_K_LIST})
        mvals = "  ".join(f"{mrow[k]*100:>{col_w-1}.2f}%" for k in MERGED_K_LIST)
        lines.append(f"    {'merged':<12}  {mvals}   (ch1∪ch3∪ch5, 各取top-K去重)")
        lines.append("")

    lines.append(f"[generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    lines.append("")
    return "\n".join(lines)


def write_output(text: str, path: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    logger.info("Results appended to %s", path)


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",          type=str,
                   default="config/llama1b_multi_channel_devset.yaml")
    p.add_argument("--retrieval_train", type=str,
                   default="qwen/retrieval_train_candidates.pt")
    p.add_argument("--retrieval_test",  type=str,
                   default="qwen/retrieval_test_candidates.pt")
    p.add_argument("--split",           type=str, default="all",
                   choices=["train", "test", "all"],
                   help="Which split(s) to evaluate")
    p.add_argument("--output",          type=str,
                   default="eval_retrieval_by_turn.txt")
    return p.parse_args()


def load_store(path: str, label: str) -> dict:
    if not os.path.exists(path):
        logger.error("%s not found: %s", label, path)
        sys.exit(1)
    logger.info("Loading %s from %s ...", label, path)
    store = torch.load(path, map_location="cpu", weights_only=False)
    logger.info("  → %d keys loaded", len(store))
    return store


def main():
    args = parse_args()

    cfg          = load_config(args.config)
    dataset_name = get_dataset_name(cfg)
    logger.info("Dataset: %s", dataset_name)

    splits_to_run = ["train", "test"] if args.split == "all" else [args.split]

    for split in splits_to_run:
        retr_path = args.retrieval_train if split == "train" else args.retrieval_test
        store     = load_store(retr_path, f"retrieval_{split}")

        result = evaluate_split(dataset_name, split, store)
        text   = format_table(result, f"Retrieval Recall — {split.upper()}")

        print(text)
        write_output(text, args.output)


if __name__ == "__main__":
    main()
