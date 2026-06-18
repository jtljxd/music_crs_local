# Retrieval Scripts

This directory contains scripts for retrieval pre-processing and candidate generation.
Retrieval is based on **BM25** or **BERT** — no multi-channel learned retrieval.

## Scripts

### `precompute_query_split.py`
Uses Qwen2.5-3B-Instruct to parse user intent for every turn across splits.

- Reads `mcrs/system_prompts/query_split.txt` as system prompt
- Stores: `{session_id}_{turn_number}` → JSON string of intent fields
- **Output files**: `qwen/query_split_{split}.pt`

```bash
python scripts/retrieval/precompute_query_split.py \
    --split train \
    --out   qwen/query_split_train.pt
```

---

### `precompute_retrieval_candidates.py`
Pre-caches retrieval candidate lists for each (session, turn) pair.

- **Output key format**: `{session_id}_{user_turn_number}` → `List[track_id]`
- **Output files**: `qwen/retrieval_{split}_candidates.pt`

```bash
python scripts/retrieval/precompute_retrieval_candidates.py \
    --split train \
    --out   qwen/retrieval_train_candidates.pt
```

---

## Notes
- These pre-computed files are consumed by `scripts/reranking/` and `scripts/inference/` scripts.
- BM25 index is built automatically on first run and cached under `qwen/retrieval_indices/bm25_index/`.
