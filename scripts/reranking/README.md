# Reranking Scripts

This directory contains scripts for **training** and **inference** of reranking models.

## Training Scripts

### `train_bagging_reranker.py`
Trains a Bagging reranker with 5 sub-models:
- **FM** — Second-order feature crossing
- **DCN** — 3-layer Deep & Cross Network
- **xDeepFM** — CIN + DNN
- **LightGBM** — Gradient boosting tree
- **ThreeTowerGate** — Query-gate three-tower ranker

```bash
python scripts/reranking/train_bagging_reranker.py \
    --config      config/llama1b_multi_channel_devset.yaml \
    --conv_emb    qwen/dialogue_embeddings_train_0.6b.pt \
    --retrieval   qwen/retrieval_train_candidates.pt \
    --out_dir     qwen/bagging_ckpt
```

---

### `train_qwen_meta_tower.py`
Trains the QwenMeta dual-tower retrieval model.

- **Query tower**: `hist_conversation_emb (1024d)` → `[128d]`
- **Item tower**: `metadata-qwen3_embedding_0.6b (1024d)` → `[128d]`
- Loss: BPR (in-batch negatives)

```bash
python scripts/reranking/train_qwen_meta_tower.py \
    --conv_emb  qwen/dialogue_embeddings_train_0.6b.pt \
    --out_dir   qwen/qwen_meta_tower
```

---

### `train_three_tower.py`
Trains the CF-BPR three-tower retrieval model.

- **User tower**: CF-BPR emb + user profile
- **Query tower**: `hist_conversation_emb (1024d)`
- **Gate fusion**: Combines user and query towers
- **Item tower**: CF-BPR emb

```bash
python scripts/reranking/train_three_tower.py \
    --conv_emb  qwen/dialogue_embeddings_train_0.6b.pt \
    --out_dir   qwen/cf_bpr_retrieval
```

---

## Inference Scripts

### `infer_bagging_blindset.py`
Runs the trained Bagging reranker on the Blind-A set.

```bash
python scripts/reranking/infer_bagging_blindset.py \
    --config     config/llama1b_multi_channel_devset.yaml \
    --conv_emb   qwen/dialogue_embeddings_blindA_0.6b.pt \
    --retrieval  qwen/retrieval_blinda_candidates.pt \
    --checkpoint qwen/bagging_ckpt/epoch3 \
    --model      lgbm \
    --out        exp/inference/blindset_A/bagging_lgbm.json
```

---

### `infer_ch3_direct.py`
Directly uses QwenMeta (ch3) top-20 as predicted track IDs, skipping reranking.

```bash
python scripts/reranking/infer_ch3_direct.py \
    --conv_emb  qwen/dialogue_embeddings_blindA_0.6b.pt \
    --retrieval qwen/retrieval_blinda_candidates.pt \
    --topk      20 \
    --out       exp/inference/blindset_A/ch3_direct_top20.json
```

---

## Dependencies
These scripts require pre-computed files from `scripts/embedding/` and `scripts/retrieval/`.
