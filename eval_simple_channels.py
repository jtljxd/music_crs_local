"""
eval_simple_channels.py
评估三路简单召回的 Recall@K 和 NDCG@20：
  ch7: user cf-bpr  cosine  item cf-bpr
  ch8: query qwen_emb  cosine  item attr_qwen_emb
  ch9: prev_music audio-laion_clap  cosine  item audio-laion_clap

数据集：
  - train: 随机100 session，区分 last_turn / non_last_turn
  - test:  随机100 session，区分 last_turn / non_last_turn
  - blinda: 所有 session，只看非最后 turn

结果写到 eval_simple_channels_result.txt
"""
import math, random, json, os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

K_LIST = [20, 50, 100]
RECALL_PER_CH = 100
SEED = 42
random.seed(SEED)

# ── 路径配置 ──────────────────────────────────────────────────────────────────
TRAIN_CE_PATH  = "qwen/hist_conversation_embeddings_train_0.6b.pt"
TEST_CE_PATH   = "qwen/hist_conversation_embeddings_test_0.6b.pt"
BLIND_CE_PATH  = "qwen/hist_conversation_embeddings_blindA_0.6b.pt"
USER_EMB_DB    = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings"
TRACK_EMB_DB   = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
CONV_DS        = "talkpl-ai/TalkPlayData-Challenge-Dataset"
BLIND_DS       = "talkpl-ai/TalkPlayData-Challenge-Blind-A"
OUT_TXT        = "eval_simple_channels_result.txt"
CF_DIM         = 128
QWEN_DIM       = 1024
CLAP_DIM       = 512   # audio-laion_clap dim (adjust if different)


# ── 1. 加载 track embeddings ──────────────────────────────────────────────────
print("Loading track embeddings ...")
ds_trk = load_dataset(TRACK_EMB_DB)
all_splits = list(ds_trk.keys())
trk_ds = concatenate_datasets([ds_trk[s] for s in all_splits])

cf_vecs   = []   # ch7 item side
attr_vecs = []   # ch8 item side
clap_vecs = []   # ch9 item side
track_ids = []

for row in tqdm(trk_ds, desc="Track embs"):
    tid = row["track_id"]
    cf   = row.get("cf-bpr")
    attr = row.get("attributes-qwen3_embedding_0.6b") or row.get("attr_qwen_emb")
    clap = row.get("audio-laion_clap")
    if cf is None or attr is None or clap is None:
        continue
    cf_t   = torch.tensor(cf,   dtype=torch.float32)
    attr_t = torch.tensor(attr, dtype=torch.float32)
    clap_t = torch.tensor(clap, dtype=torch.float32)
    # pad / truncate to expected dims
    def _fit(v, d):
        if v.shape[0] > d: return v[:d]
        if v.shape[0] < d: return F.pad(v, (0, d - v.shape[0]))
        return v
    cf_vecs.append(_fit(cf_t,   CF_DIM))
    attr_vecs.append(_fit(attr_t, QWEN_DIM))
    clap_vecs.append(_fit(clap_t, CLAP_DIM))
    track_ids.append(tid)

# L2 normalize
item_cf_mat   = F.normalize(torch.stack(cf_vecs),   dim=1)  # [N, CF_DIM]
item_attr_mat = F.normalize(torch.stack(attr_vecs), dim=1)  # [N, QWEN_DIM]
item_clap_mat = F.normalize(torch.stack(clap_vecs), dim=1)  # [N, CLAP_DIM]
tid2idx = {t: i for i, t in enumerate(track_ids)}
print(f"  Tracks: {len(track_ids)}  cf={item_cf_mat.shape}  attr={item_attr_mat.shape}  clap={item_clap_mat.shape}")

# ── 2. 加载 user embeddings (cf-bpr) ─────────────────────────────────────────
print("Loading user embeddings ...")
ds_usr = load_dataset(USER_EMB_DB)
usr_ds = concatenate_datasets([ds_usr[s] for s in ds_usr.keys()])
user_cf: Dict[str, torch.Tensor] = {}
for row in tqdm(usr_ds, desc="User embs"):
    uid = row["user_id"]
    cf  = row.get("cf-bpr")
    if cf is None: continue
    v = torch.tensor(cf, dtype=torch.float32)
    if v.shape[0] > CF_DIM: v = v[:CF_DIM]
    elif v.shape[0] < CF_DIM: v = F.pad(v, (0, CF_DIM - v.shape[0]))
    user_cf[uid] = v
print(f"  Users: {len(user_cf)}")


# ── 3. 召回函数 ───────────────────────────────────────────────────────────────
@torch.no_grad()
def retrieve_ch7(user_id: str) -> List[str]:
    """user cf-bpr cosine item cf-bpr"""
    if user_id not in user_cf:
        return []
    uv = F.normalize(user_cf[user_id].unsqueeze(0), dim=1)  # [1, CF_DIM]
    sc = (item_cf_mat * uv).sum(1)
    top = torch.topk(sc, min(RECALL_PER_CH, sc.shape[0])).indices.tolist()
    return [track_ids[i] for i in top]


@torch.no_grad()
def retrieve_ch8(conv_emb: torch.Tensor) -> List[str]:
    """query qwen_emb cosine item attr_qwen_emb"""
    if conv_emb is None:
        return []
    qv = F.normalize(conv_emb.float().unsqueeze(0), dim=1)  # [1, QWEN_DIM]
    # if conv_emb dim != QWEN_DIM, resize
    if qv.shape[1] != QWEN_DIM:
        qv = F.interpolate(qv.unsqueeze(0), size=QWEN_DIM, mode='linear', align_corners=False).squeeze(0)
        qv = F.normalize(qv, dim=1)
    sc = (item_attr_mat * qv).sum(1)
    top = torch.topk(sc, min(RECALL_PER_CH, sc.shape[0])).indices.tolist()
    return [track_ids[i] for i in top]


@torch.no_grad()
def retrieve_ch9(prev_tid: str) -> List[str]:
    """prev music audio-laion_clap cosine item audio-laion_clap"""
    if prev_tid not in tid2idx:
        return []
    idx = tid2idx[prev_tid]
    av  = item_clap_mat[idx].unsqueeze(0)  # [1, CLAP_DIM]
    sc  = (item_clap_mat * av).sum(1)
    top = torch.topk(sc, min(RECALL_PER_CH + 1, sc.shape[0])).indices.tolist()
    # exclude self
    return [track_ids[i] for i in top if track_ids[i] != prev_tid][:RECALL_PER_CH]


def merged_topk(c7, c8, c9, k):
    return list(dict.fromkeys(c7[:k] + c8[:k] + c9[:k]))[:k]


# ── 4. 评估逻辑 ───────────────────────────────────────────────────────────────
def compute_metrics(cands: List[str], gt: str) -> Dict:
    result = {}
    for k in K_LIST:
        result[f"hit@{k}"] = 1 if gt in cands[:k] else 0
    top20 = cands[:20]
    if gt in top20:
        rank = top20.index(gt) + 1
        result["ndcg@20"] = 1.0 / math.log2(rank + 1)
    else:
        result["ndcg@20"] = 0.0
    return result


def eval_dataset(
    ds_split,            # HF dataset
    conv_emb_store,      # dict: key → tensor
    label: str,
    max_sessions: int = 100,
    non_last_only: bool = False,   # True → blinda mode
):
    """
    对 ds_split 中随机抽 max_sessions 个 session 做评估。
    non_last_only=True: 只评估非最后 turn（blinda 用）
    non_last_only=False: 区分 last_turn / non_last_turn 分别统计
    返回 dict: split_name → {ch7/ch8/ch9/merged: {hit@K, ndcg@20 sum, n}}
    """
    # 随机抽 session
    items = list(ds_split)
    random.shuffle(items)
    items = items[:max_sessions]

    buckets = {}  # key: "last" or "non_last"
    for bk in (["non_last"] if non_last_only else ["last", "non_last"]):
        buckets[bk] = {ch: defaultdict(float) for ch in ["ch7","ch8","ch9","merged"]}
        for bk2 in buckets[bk].values():
            bk2["n"] = 0

    for item in tqdm(items, desc=f"Eval [{label}]"):
        session_id = item["session_id"]
        user_id    = item.get("user_id")
        convs      = item["conversations"]

        music_turns = {int(c["turn_number"]): c["content"]
                       for c in convs if c.get("role") == "music" and c.get("content")}
        if not music_turns:
            continue

        user_turns = [int(c["turn_number"]) for c in convs if c.get("role") == "user"]
        last_user  = max(user_turns) if user_turns else 0

        for turn_num, gt_tid in music_turns.items():
            is_last = (turn_num >= last_user)
            if non_last_only and is_last:
                continue

            bk = "last" if is_last else "non_last"

            # conv_emb for ch8
            emb_key = f"{session_id}_{turn_num}"
            conv_emb = conv_emb_store.get(emb_key)
            if conv_emb is None:
                for t in range(turn_num - 1, -1, -1):
                    k2 = f"{session_id}_{t}"
                    if k2 in conv_emb_store:
                        conv_emb = conv_emb_store[k2]; break

            # prev music tid for ch9
            prev_tids = sorted([t for t in music_turns if t < turn_num])
            prev_tid  = music_turns[prev_tids[-1]] if prev_tids else None

            c7 = retrieve_ch7(user_id)
            c8 = retrieve_ch8(conv_emb) if conv_emb is not None else []
            c9 = retrieve_ch9(prev_tid) if prev_tid else []

            for ch, cands in [("ch7", c7), ("ch8", c8), ("ch9", c9)]:
                m = compute_metrics(cands, gt_tid)
                buckets[bk][ch]["n"] += 1
                for kk, vv in m.items():
                    buckets[bk][ch][kk] += vv

            merged = merged_topk(c7, c8, c9, max(K_LIST))
            m = compute_metrics(merged, gt_tid)
            buckets[bk]["merged"]["n"] += 1
            for kk, vv in m.items():
                buckets[bk]["merged"][kk] += vv

    return buckets


def print_and_save(lines: List[str], out_path: str):
    text = "\n".join(lines)
    print(text)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def format_bucket(label: str, stats: Dict) -> List[str]:
    lines = [f"  [{label}]"]
    for ch in ["ch7","ch8","ch9","merged"]:
        n = max(stats[ch]["n"], 1)
        hits = "  ".join(f"Hit@{k}={stats[ch][f'hit@{k}']/n*100:.2f}%" for k in K_LIST)
        ndcg = stats[ch]["ndcg@20"] / n
        lines.append(f"    {ch:<8}  n={int(stats[ch]['n'])}  {hits}  NDCG@20={ndcg:.4f}")
    return lines


# ── 5. 主流程 ─────────────────────────────────────────────────────────────────
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("Simple Channel Recall Evaluation\n")
    f.write(f"K_LIST={K_LIST}  RECALL_PER_CH={RECALL_PER_CH}\n")
    f.write("ch7=user_cf_cosine_item_cf  ch8=conv_qwen_cosine_attr  ch9=prev_clap_cosine_clap\n\n")

# train
print("\n=== TRAIN ===")
train_ce = torch.load(TRAIN_CE_PATH, map_location="cpu", weights_only=True)
train_ds = load_dataset(CONV_DS, split="train")
train_buckets = eval_dataset(train_ds, train_ce, "train", max_sessions=100)
lines = ["=== TRAIN (100 sessions) ==="]
for bk in ["last", "non_last"]:
    lines += format_bucket(bk, train_buckets[bk])
print_and_save(lines, OUT_TXT)

# test
print("\n=== TEST ===")
test_ce = torch.load(TEST_CE_PATH, map_location="cpu", weights_only=True)
test_ds = load_dataset(CONV_DS, split="test")
test_buckets = eval_dataset(test_ds, test_ce, "test", max_sessions=100)
lines = ["\n=== TEST (100 sessions) ==="]
for bk in ["last", "non_last"]:
    lines += format_bucket(bk, test_buckets[bk])
print_and_save(lines, OUT_TXT)

# blinda
print("\n=== BLIND-A ===")
blind_ce = torch.load(BLIND_CE_PATH, map_location="cpu", weights_only=True)
blind_ds = load_dataset(BLIND_DS, split="test")
blind_buckets = eval_dataset(blind_ds, blind_ce, "blinda",
                              max_sessions=999, non_last_only=True)
lines = ["\n=== BLIND-A (all sessions, non_last turns only) ==="]
lines += format_bucket("non_last", blind_buckets["non_last"])
print_and_save(lines, OUT_TXT)

print(f"\nDone. Results saved to {OUT_TXT}")
