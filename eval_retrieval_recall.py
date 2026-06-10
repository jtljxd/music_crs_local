"""
eval_retrieval_recall.py
────────────────────────
评估 test 集上各路召回（ch1 / ch3 / ch5 / union）的 Recall@K。

策略：每个 session 只取「最后一个 music turn」对应的 GT track，
      然后检查该 GT 是否出现在各路召回的 top-K 结果中。

用法：
    python eval_retrieval_recall.py \
        --config      config/llama1b_multi_channel_devset.yaml \
        --retrieval   qwen/retrieval_test_candidates.pt \
        --output      eval_retrieval_recall.txt
"""

import argparse
import logging
import os
import sys
import torch
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

K_LIST = [20, 50, 100, 200, 300, 400, 500]
CHANNELS = ["ch1", "ch3", "ch5", "union"]


# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dataset_name(cfg: dict) -> str:
    """从 config 中读取 dataset 名称（兼容不同 config 字段名）。"""
    for key in ("conversation_dataset_name", "test_dataset_name",
                "dataset", "dataset_name", "data", "hf_dataset"):
        if key in cfg:
            return cfg[key]
    raise ValueError(f"Cannot find dataset name in config: {cfg}")


def last_music_turn(convs: list):
    """返回 session 最后一个 music turn 的 (turn_num, track_id)，找不到返回 None。"""
    music_turns = {
        int(c["turn_number"]): c["content"]
        for c in convs
        if c.get("role") == "music" and c.get("content")
    }
    if not music_turns:
        return None, None
    t = max(music_turns.keys())
    return t, music_turns[t]


def eval_recall(dataset_name: str,
                retrieval_store: dict,
                k_list: list = K_LIST,
                channels: list = CHANNELS) -> dict:
    """
    Returns:
        {
          "ch1":   {20: 0.082, 50: 0.15, ...},
          "ch3":   {...},
          "ch5":   {...},
          "union": {...},
        }
    还会额外输出每路的「实际候选条数」分布。
    """
    ds = load_dataset(dataset_name, split="test")

    # hits[channel][k] = count
    hits   = {ch: {k: 0 for k in k_list} for ch in channels}
    totals = {ch: 0 for ch in channels}   # session 中该 channel 有召回结果的数量
    n_sessions = 0

    cand_len_sum  = {ch: 0 for ch in channels}   # 统计平均候选条数

    for item in tqdm(ds, desc="Evaluating retrieval recall", unit="session"):
        session_id = item["session_id"]
        convs      = item["conversations"]

        turn_num, gt_tid = last_music_turn(convs)
        if gt_tid is None:
            continue
        n_sessions += 1

        emb_key = f"{session_id}_{turn_num}"
        raw = (retrieval_store.get(emb_key)
               or retrieval_store.get(f"{session_id}_{turn_num - 1}"))

        if raw is None:
            # 该 session 没有召回数据，所有 channel miss
            continue

        # ── 拆出各路候选 ─────────────────────────────────────────────────
        if isinstance(raw, dict):
            ch_cands = {
                "ch1":   list(raw.get("ch1",   [])),
                "ch3":   list(raw.get("ch3",   [])),
                "ch5":   list(raw.get("ch5",   [])),
                "union": list(raw.get("union", [])),
            }
        else:
            # 旧格式：只有 union
            ch_cands = {
                "ch1":   [],
                "ch3":   [],
                "ch5":   [],
                "union": list(raw),
            }

        for ch in channels:
            cands = ch_cands[ch]
            if not cands:
                continue
            totals[ch] += 1
            cand_len_sum[ch] += len(cands)

            for k in k_list:
                if gt_tid in cands[:k]:
                    hits[ch][k] += 1

    logger.info("Total sessions evaluated: %d", n_sessions)
    for ch in channels:
        avg_len = cand_len_sum[ch] / max(totals[ch], 1)
        logger.info("  %-6s  sessions_with_cands=%d  avg_cands=%.1f",
                    ch, totals[ch], avg_len)

    # ── 计算 recall（分母用 n_sessions，与精排评估一致）─────────────────
    result = {}
    for ch in channels:
        result[ch] = {}
        denom = n_sessions  # 统一用 n_sessions 做分母
        for k in k_list:
            result[ch][k] = hits[ch][k] / denom if denom > 0 else 0.0

    return result, n_sessions


def print_table(result: dict, n_sessions: int, k_list: list, output_path: str):
    from datetime import datetime

    header_ks = "  ".join(f"R@{k:<5}" for k in k_list)
    sep       = "=" * (12 + 9 * len(k_list))

    lines = [
        "",
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]  "
        f"Retrieval Recall  |  test split, last query per session  |  n={n_sessions}",
        sep,
        f"  {'Channel':<8}  {header_ks}",
        "-" * len(sep),
    ]
    for ch in CHANNELS:
        vals = "  ".join(f"{result[ch][k]*100:6.2f}%" for k in k_list)
        lines.append(f"  {ch:<8}  {vals}")
    lines.append(sep)
    lines.append("")

    text = "\n".join(lines)
    print(text)
    logger.info(text)

    # 写文件
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    logger.info("Results appended to %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate retrieval recall on test split")
    p.add_argument("--config",     type=str, default="config/llama1b_multi_channel_devset.yaml")
    p.add_argument("--retrieval",  type=str, default="qwen/retrieval_test_candidates.pt",
                   help="Path to retrieval candidates .pt file")
    p.add_argument("--output",     type=str, default="eval_retrieval_recall.txt")
    return p.parse_args()


def main():
    args = parse_args()

    # ── load config & dataset name ────────────────────────────────────────
    cfg          = load_config(args.config)
    dataset_name = get_dataset_name(cfg)
    logger.info("Dataset: %s", dataset_name)

    # ── load retrieval store ──────────────────────────────────────────────
    if not os.path.exists(args.retrieval):
        logger.error("Retrieval file not found: %s", args.retrieval)
        sys.exit(1)
    logger.info("Loading retrieval candidates from %s ...", args.retrieval)
    retrieval_store = torch.load(args.retrieval, map_location="cpu",
                                  weights_only=False)
    logger.info("Retrieval store loaded: %d keys", len(retrieval_store))

    # ── evaluate ─────────────────────────────────────────────────────────
    result, n_sessions = eval_recall(dataset_name, retrieval_store)

    # ── print & save ──────────────────────────────────────────────────────
    print_table(result, n_sessions, K_LIST, args.output)


if __name__ == "__main__":
    main()
