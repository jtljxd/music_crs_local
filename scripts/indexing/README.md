# Indexing Scripts

This directory is reserved for scripts that **build retrieval indices** from scratch.

Currently, index building is handled automatically inside the retrieval modules:

| Index | Location | Built by |
|-------|----------|----------|
| BM25 index | `qwen/retrieval_indices/bm25_index/` | `mcrs/retrieval_modules/bm25.py` (auto-build on first run) |
| BERT index | `cache/` | `mcrs/retrieval_modules/bert.py` (auto-build on first run) |

## Adding a Custom Index Builder

If you need to pre-build indices offline (e.g. for large-scale production), add a script here with the naming convention:

```
build_{index_type}_index.py
```

Example:
```bash
python scripts/indexing/build_bm25_index.py \
    --dataset_name talkpl-ai/TalkPlayData-Challenge-Track-Metadata \
    --out_dir qwen/retrieval_indices/bm25_index
```
