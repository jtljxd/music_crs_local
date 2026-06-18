# Evaluation Scripts

This directory contains scripts for evaluating retrieval and reranking quality.

## Scripts

### `eval_retrieval_hitrate.py`
Evaluates retrieval hit rate (recall@K) using BM25 or BERT.

```bash
python scripts/eval/eval_retrieval_hitrate.py \
    --retrieval_type bm25 \
    --topk           200 \
    --split          test \
    --max_sessions   0
```

**Output**: `exp/eval/retrieval_hitrate.txt`

---

### `eval_retrieval_accuracy.py`
Evaluates 3-channel retrieval accuracy at K = 20, 50, 100, 200.
Reports hit rate per channel (ch1, ch3, ch5) and for the union.

```bash
python scripts/eval/eval_retrieval_accuracy.py \
    --config            config/llama1b_multi_channel_devset.yaml \
    --conv_emb_store    qwen/dialogue_embeddings_test_0.6b.pt \
    --query_split_store qwen/query_split_test.pt \
    --max_sessions      200
```

---

### `eval_retrieval_recall.py`
Evaluates recall metrics (NDCG, HR) on the test split.

```bash
python scripts/eval/eval_retrieval_recall.py \
    --config  config/llama1b_bm25_devset.yaml \
    --split   test
```

---

### `eval_retrieval_recall_by_turn.py`
Breaks down retrieval recall by conversation turn number to analyze multi-turn performance.

```bash
python scripts/eval/eval_retrieval_recall_by_turn.py \
    --config  config/llama1b_bm25_devset.yaml \
    --split   test
```

---

### `eval_simple_channels.py`
Quick sanity-check evaluation for individual retrieval channels.

```bash
python scripts/eval/eval_simple_channels.py
```

---

### `eval_qwen_meta_tower.py`
Evaluates the trained QwenMeta dual-tower model (Recall@K, NDCG@K).

```bash
python scripts/eval/eval_qwen_meta_tower.py \
    --model_path qwen/qwen_meta_tower/model.pt \
    --conv_emb   qwen/dialogue_embeddings_test_0.6b.pt \
    --split      test
```

---

## Output
All evaluation reports are saved under `exp/eval/`.
