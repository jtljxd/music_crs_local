# Embedding Pre-computation Scripts

This directory contains scripts for generating pre-computed embeddings used downstream
by retrieval, reranking, and evaluation modules.

---

## Scripts Overview

| Script | Model | Input | Key Format | Output |
|--------|-------|-------|------------|--------|
| `precompute_dialogue_embeddings.py` | Qwen3-Embedding-0.6B | per-turn user query + history | `{session_id}_{turn}_{query\|history}` | `qwen/dialogue_embeddings_{split}_0.6b.pt` |
| `precompute_turn_embeddings.py` | Qwen3-Embedding-0.6B | per-turn user/system text | `{session_id}__{turn}_{role}` | `qwen/turn_embeddings_{split}.pt` |
| `precompute_goal_embeddings.py` | Qwen3-Embedding-0.6B | `conversation_goal.listener_goal` | `session_id` | `qwen/goal_embeddings_{split}_0.6b.pt` |
| `precompute_bge_embeddings.py` | BGE-Small-EN-v1.5 | query `genre`/`decade` fields | `{session_id}_{turn}_{field}` | `bge/query_{field}_embeddings_{split}.pt` |
| `precompute_bge_embeddings.py` | BGE-Small-EN-v1.5 | track `tag_list` | `track_id` | `bge/track_tag_embeddings.pt` |

---

## Script Details

### `precompute_dialogue_embeddings.py`
Pre-computes per-turn Qwen3-Embedding dialogue embeddings.

- **Key format**: `{session_id}_{turn}_query` / `{session_id}_{turn}_history`
- **Output**: `qwen/dialogue_embeddings_{split}_0.6b.pt`

```bash
# train
nohup python scripts/embedding/precompute_dialogue_embeddings.py \
    --dataset talkpl-ai/TalkPlayData-Challenge-Dataset \
    --split   train \
    --out     qwen/dialogue_embeddings_train_0.6b.pt \
    > logs/dialogue_emb_train.log 2>&1 &

# test
nohup python scripts/embedding/precompute_dialogue_embeddings.py \
    --dataset talkpl-ai/TalkPlayData-Challenge-Dataset \
    --split   test \
    --out     qwen/dialogue_embeddings_test_0.6b.pt \
    > logs/dialogue_emb_test.log 2>&1 &

# blind-A
nohup python scripts/embedding/precompute_dialogue_embeddings.py \
    --dataset talkpl-ai/TalkPlayData-Challenge-Blind-A \
    --split   test \
    --out     qwen/dialogue_embeddings_blindA_0.6b.pt \
    --batch   64 \
    > logs/dialogue_emb_blindA.log 2>&1 &
```

---

### `precompute_turn_embeddings.py`
Pre-computes per-turn Qwen embeddings with a different key schema (user / system / history_avg roles).

- **Key format**: `{session_id}__{turn}_{role}`
- **Output**: `qwen/turn_embeddings_{split}.pt`

```bash
# train
nohup python scripts/embedding/precompute_turn_embeddings.py \
    --dataset talkpl-ai/TalkPlayData-Challenge-Dataset \
    --split   train \
    --out     qwen/turn_embeddings_train.pt \
    > logs/turn_emb_train.log 2>&1 &

# test
nohup python scripts/embedding/precompute_turn_embeddings.py \
    --dataset talkpl-ai/TalkPlayData-Challenge-Dataset \
    --split   test \
    --out     qwen/turn_embeddings_test.pt \
    > logs/turn_emb_test.log 2>&1 &
```

---

### `precompute_goal_embeddings.py`
Pre-computes Qwen3-Embedding-0.6B embeddings for the `listener_goal` field inside
`conversation_goal`. One embedding per session.

- **Model**: Qwen3-Embedding-0.6B (1024-dim, fp16)
- **Key format**: `session_id`
- **Output**: `qwen/goal_embeddings_{split}_0.6b.pt`
- Missing / empty `listener_goal` → zero vector

```bash
# train
nohup python scripts/embedding/precompute_goal_embeddings.py \
    --dataset talkpl-ai/TalkPlayData-Challenge-Dataset \
    --split   train \
    --out     qwen/goal_embeddings_train_0.6b.pt \
    > logs/goal_emb_train.log 2>&1 &

# test
nohup python scripts/embedding/precompute_goal_embeddings.py \
    --dataset talkpl-ai/TalkPlayData-Challenge-Dataset \
    --split   test \
    --out     qwen/goal_embeddings_test_0.6b.pt \
    > logs/goal_emb_test.log 2>&1 &

# blind-A
nohup python scripts/embedding/precompute_goal_embeddings.py \
    --dataset talkpl-ai/TalkPlayData-Challenge-Blind-A \
    --split   test \
    --out     qwen/goal_embeddings_blindA_0.6b.pt \
    --batch   64 \
    > logs/goal_emb_blindA.log 2>&1 &
```

---

### `precompute_bge_embeddings.py`
Pre-computes BGE-Small-EN-v1.5 embeddings (384-dim, fp16) for two use cases:

#### Mode A — Query-level `genre` and `decade` fields
Source: query_split `.pt` stores (output of `scripts/retrieval/precompute_query_split.py`).

- **Key format**: `{session_id}_{turn}_{field}` where `field` ∈ `{genre, decade}`
- **Output**: `bge/query_genre_embeddings_{split_name}.pt`  
             `bge/query_decade_embeddings_{split_name}.pt`

```bash
# train
nohup python scripts/embedding/precompute_bge_embeddings.py \
    --mode        query \
    --query_split qwen/query_split_train.pt \
    --split_name  train \
    --out_dir     bge \
    > logs/bge_query_train.log 2>&1 &

# test
nohup python scripts/embedding/precompute_bge_embeddings.py \
    --mode        query \
    --query_split qwen/query_split_test.pt \
    --split_name  test \
    --out_dir     bge \
    > logs/bge_query_test.log 2>&1 &

# blind-A
nohup python scripts/embedding/precompute_bge_embeddings.py \
    --mode        query \
    --query_split qwen/query_split_blindA.pt \
    --split_name  blindA \
    --out_dir     bge \
    > logs/bge_query_blindA.log 2>&1 &
```

#### Mode B — Track `tag_list`
Source: track metadata dataset.

- **Key format**: `track_id`
- **Output**: `bge/track_tag_embeddings.pt`

```bash
# Run once (covers all tracks)
nohup python scripts/embedding/precompute_bge_embeddings.py \
    --mode          track \
    --track_dataset talkpl-ai/TalkPlayData-Challenge-Track-Metadata \
    --track_split   all_tracks \
    --out_dir       bge \
    > logs/bge_track_tags.log 2>&1 &
```

---

## Output Directory Structure

```
qwen/                                          # Qwen3-Embedding-0.6B outputs
├── dialogue_embeddings_train_0.6b.pt
├── dialogue_embeddings_test_0.6b.pt
├── dialogue_embeddings_blindA_0.6b.pt
├── turn_embeddings_train.pt
├── turn_embeddings_test.pt
├── goal_embeddings_train_0.6b.pt
├── goal_embeddings_test_0.6b.pt
└── goal_embeddings_blindA_0.6b.pt

bge/                                           # BGE-Small-EN-v1.5 outputs
├── query_genre_embeddings_train.pt
├── query_genre_embeddings_test.pt
├── query_genre_embeddings_blindA.pt
├── query_decade_embeddings_train.pt
├── query_decade_embeddings_test.pt
├── query_decade_embeddings_blindA.pt
└── track_tag_embeddings.pt

logs/                                          # nohup log files
├── dialogue_emb_train.log
├── dialogue_emb_test.log
├── dialogue_emb_blindA.log
├── turn_emb_train.log
├── turn_emb_test.log
├── goal_emb_train.log
├── goal_emb_test.log
├── goal_emb_blindA.log
├── bge_query_train.log
├── bge_query_test.log
├── bge_query_blindA.log
└── bge_track_tags.log
```

> **Tip**: 查看后台任务进度：`tail -f logs/<log_file>.log`  
> 查看所有后台进程：`jobs -l` 或 `ps aux | grep precompute`
