"""
Pre-compute per-turn dialogue embeddings for every conversation.

Key scheme
----------
  {session_id}_{turn}_query    : 当前轮 user query 的 embedding                [dim] fp16
  {session_id}_{turn}_history  : 前几轮 user+assistant 拼接文本的 embedding     [dim] fp16
  {session_id}_listener_goal   : conversation_goal 中 listener_goal 的 embedding [dim] fp16

History 拼接格式（前 turn-1 轮，按时间顺序）:
  "User: <turn1 user> Assistant: <turn1 assistant> User: <turn2 user> ..."

多卡并行：每张卡以独立子进程运行，各自保存临时文件，主进程合并。

Usage
-----
# 单卡
python precompute_dialogue_embeddings.py --split train

# 多卡（8卡）
python precompute_dialogue_embeddings.py --split all --num_gpus 8
"""

import os
import sys
import argparse
import logging
import subprocess
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Tuple

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

TARGET_DIM = 4096  # Qwen3-Embedding-8B hidden_size
QWEN_TASK = "Given a music conversation query, retrieve relevant music tracks"


def get_logger(tag):
    logger = logging.getLogger(str(tag))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            f"[%(asctime)s][{tag}] %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _get_instruct(text: str) -> str:
    return f"Instruct: {QWEN_TASK}\nQuery: {text}"


def load_qwen(model_path: str, device: str):
    dtype = torch.float16 if "cuda" in device else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=dtype,
    ).to(device).eval()
    return tokenizer, model


def _last_token_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    seq_len = attn_mask.sum(dim=1) - 1
    batch = last_hidden.shape[0]
    return last_hidden[torch.arange(batch, device=last_hidden.device), seq_len]


def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 16,
    add_instruct: bool = True,
) -> torch.Tensor:
    if add_instruct:
        texts = [_get_instruct(t) for t in texts]

    all_embs = []
    start = 0
    current_batch = batch_size

    while start < len(texts):
        chunk = texts[start: start + current_batch]
        try:
            inputs = tokenizer(
                chunk, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model(**inputs)
                emb = _last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
                emb = F.normalize(emb, p=2, dim=1)
                all_embs.append(emb.cpu().half())
            start += current_batch
            current_batch = min(current_batch * 2, batch_size)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current_batch <= 1:
                inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.inference_mode():
                    cpu_out = model.cpu()(**inputs)
                    emb = _last_token_pool(cpu_out.last_hidden_state, inputs["attention_mask"])
                    emb = F.normalize(emb, p=2, dim=1)
                    all_embs.append(emb.half())
                model.to(device)
                start += current_batch
            else:
                current_batch = max(1, current_batch // 2)

    return torch.cat(all_embs, dim=0)


def extract_turns(session: dict) -> List[Tuple[int, str, str]]:
    turns = []
    for conv in session["conversations"]:
        role = conv["role"]
        content = str(conv.get("content", "") or "").strip()
        if not content or role == "music":
            continue
        turns.append((int(conv["turn_number"]), role, content))
    turns.sort(key=lambda x: (x[0], x[1]))
    return turns


def build_history_text(turns: List[Tuple[int, str, str]], before_turn: int) -> str:
    user_buf: Dict[int, str] = {}
    asst_buf: Dict[int, str] = {}
    for t, role, content in turns:
        if t >= before_turn:
            continue
        if role == "user":
            user_buf[t] = content
        elif role == "assistant":
            asst_buf[t] = content
    lines = []
    for t in sorted(set(list(user_buf.keys()) + list(asst_buf.keys()))):
        if t in user_buf:
            lines.append(f"User: {user_buf[t]}")
        if t in asst_buf:
            lines.append(f"Assistant: {asst_buf[t]}")
    return " ".join(lines)


def process_sessions(
    sessions: List[dict],
    store: Dict[str, torch.Tensor],
    tokenizer,
    model,
    device: str,
    batch_size: int,
) -> int:
    text_to_keys: Dict[str, List[str]] = defaultdict(list)

    for session in sessions:
        sid = session["session_id"]
        turns = extract_turns(session)
        user_turns = [(t, r, c) for t, r, c in turns if r == "user"]

        for t, _, content in user_turns:
            query_key = f"{sid}_{t}_query"
            if query_key not in store:
                text_to_keys[content].append(query_key)

            hist_key = f"{sid}_{t}_history"
            if hist_key not in store:
                history_text = build_history_text(turns, before_turn=t)
                if history_text:
                    text_to_keys[history_text].append(hist_key)
                else:
                    store[hist_key] = torch.zeros(TARGET_DIM, dtype=torch.float16)

        goal_key = f"{sid}_listener_goal"
        if goal_key not in store:
            goal = session.get("conversation_goal") or {}
            listener_goal = str(goal.get("listener_goal") or "").strip()
            if listener_goal:
                text_to_keys[listener_goal].append(goal_key)
            else:
                store[goal_key] = torch.zeros(TARGET_DIM, dtype=torch.float16)

    unique_texts = list(text_to_keys.keys())
    if unique_texts:
        embs = encode_texts(unique_texts, tokenizer, model, device, batch_size)
        for i, text in enumerate(unique_texts):
            for key in text_to_keys[text]:
                store[key] = embs[i]

    return len(unique_texts)


# ── 单卡 worker 入口（被子进程调用）────────────────────────────────────────────

def run_worker(args):
    """单卡模式：处理 [rank, world_size) 的 session 分片。"""
    logger = get_logger(f"gpu{args.rank}")
    device = "cuda:0"  # 每个子进程通过 CUDA_VISIBLE_DEVICES 只看到一张卡，固定用 cuda:0

    logger.info("Loading dataset split=%s …", args.split)
    ds = load_dataset(args.dataset, split=args.split)
    total = len(ds)

    # 按 rank 切片
    indices = list(range(args.rank, total, args.world_size))
    sessions = [ds[i] for i in indices]
    logger.info("Handling %d / %d sessions (rank=%d/%d)", len(sessions), total, args.rank, args.world_size)

    logger.info("Loading model on %s …", device)
    tokenizer, model = load_qwen(args.qwen, device)

    store: Dict[str, torch.Tensor] = {}
    pbar = tqdm(range(0, len(sessions), args.sessions_per_chunk),
                desc=f"gpu{args.rank}", position=0, leave=True)

    for chunk_start in pbar:
        chunk = sessions[chunk_start: chunk_start + args.sessions_per_chunk]
        added = process_sessions(chunk, store, tokenizer, model, device, args.batch)
        pbar.set_postfix(keys=len(store), new=added)
        torch.cuda.empty_cache()

    tmp_path = os.path.join(args.out_dir, f"_tmp_{args.split}_rank{args.rank}.pt")
    torch.save(store, tmp_path)
    logger.info("Saved %d entries → %s", len(store), tmp_path)


# ── 多卡调度（主进程用 subprocess 启动各 worker）────────────────────────────────

def run_split(split: str, args):
    logger = get_logger("main")
    out_path = os.path.join(args.out_dir, f"dialogue_embeddings_{split}.pt")
    os.makedirs(args.out_dir, exist_ok=True)

    num_gpus = min(args.num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    logger.info("split=%s  num_gpus=%d", split, num_gpus)

    if num_gpus <= 1:
        # 直接在当前进程跑
        single_args = argparse.Namespace(**vars(args), rank=0, world_size=1, split=split)
        run_worker(single_args)
        tmp = os.path.join(args.out_dir, f"_tmp_{split}_rank0.pt")
        os.rename(tmp, out_path)
        return

    # 用 subprocess 独立启动每个 GPU worker，完全进程隔离
    procs = []
    for rank in range(num_gpus):
        cmd = [
            sys.executable, __file__,
            "--dataset", args.dataset,
            "--split", split,
            "--qwen", args.qwen,
            "--out_dir", args.out_dir,
            "--batch", str(args.batch),
            "--sessions_per_chunk", str(args.sessions_per_chunk),
            "--rank", str(rank),
            "--world_size", str(num_gpus),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        logger.info("Launching rank=%d  CUDA_VISIBLE_DEVICES=%d", rank, rank)
        p = subprocess.Popen(cmd, env=env)
        procs.append(p)

    for rank, p in enumerate(procs):
        p.wait()
        if p.returncode != 0:
            raise RuntimeError(f"Worker rank={rank} failed with returncode={p.returncode}")

    # 合并
    logger.info("Merging %d shards …", num_gpus)
    merged: Dict[str, torch.Tensor] = {}
    for rank in range(num_gpus):
        tmp = os.path.join(args.out_dir, f"_tmp_{split}_rank{rank}.pt")
        shard = torch.load(tmp, map_location="cpu", weights_only=True)
        merged.update(shard)
        os.remove(tmp)

    torch.save(merged, out_path)
    logger.info("Merged %d entries → %s", len(merged), out_path)


def main(args):
    splits = ["train", "test"] if args.split == "all" else [args.split]
    for split in splits:
        run_split(split, args)
    get_logger("main").info("All done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="/bianxiaoding-default-ceph/guihaoyue/hf_datasets/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--split", type=str, default="train", help="train / test / all")
    parser.add_argument("--qwen", type=str,
                        default="/bianxiaoding-default-ceph/guihaoyue/models/Qwen3-Embedding-8B")
    parser.add_argument("--out_dir", type=str,
                        default="/bianxiaoding-default-ceph/guihaoyue/music_crs_local/embeddings")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--sessions_per_chunk", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=8)
    # worker 专用参数（子进程内部使用）
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    if args.rank >= 0:
        # 子进程模式：直接跑 worker
        run_worker(args)
    else:
        # 主进程模式：调度多卡
        main(args)
