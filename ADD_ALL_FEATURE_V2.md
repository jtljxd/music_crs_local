# Add All Feature V2 Implementation

This document describes the implementation of the `add_all_feature_v2` feature for the music CRS system.

## Overview

The V2 pipeline implements a comprehensive multi-channel retrieval and three-tower reranking architecture:

1. **Multi-Channel Retrieval** (~350 candidates)
   - CF-BPR user-music tower (100 tracks)
   - Similar users' liked music
   - Query-metadata semantic similarity (100 tracks)
   - Query-attributes semantic similarity (100 tracks)

2. **Three-Tower Reranking** (Top 20)
   - Intent Tower: Query + History + Conversation Goal
   - Item Tower: Multi-modal fusion with gating (6 modalities)
   - User Tower: User profile + CF-BPR

3. **LLM Generation** (Same as V1)
   - Llama-3.2-1B-Instruct for response generation

## Architecture Details

### Multi-Channel Retrieval

**File**: `mcrs/retrieval_modules/multi_channel_retrieval.py`

**Channels**:
1. **CF-BPR Tower**: Cosine similarity between user cf-bpr and track cf-bpr embeddings
2. **Collaborative Filtering**: Find 10 similar users via cf-bpr, collect their liked music
3. **Query-Metadata**: Semantic similarity using Qwen3-0.6B embeddings (first 256 dims)
4. **Query-Attributes**: Semantic similarity using Qwen3-0.6B embeddings (first 256 dims)

**Query Encoding**:
- Current query: 70% weight
- Historical queries: 30% weight (averaged)
- Encoded using Qwen3-Embedding-0.6B model

**User History Index**:
- Tracks user's liked music from conversation history
- Like score: 5 if `MOVES_TOWARD_GOAL`, 1 otherwise
- Sorted by like score and recency
- Top 5 tracks per user

### Three-Tower Reranking

**File**: `mcrs/reranking_modules/three_tower_reranker.py`

**Intent Tower**:
- Input: [query_emb(128), history_emb(128), listener_goal_emb(128), category_emb(128)]
- Architecture: MLP [512 → 256 → 128]
- Output: intent_repr (128 dims)

**Item Tower**:
- **Multi-modal inputs** (6 modalities, each 128 dims):
  - audio-laion_clap
  - image-siglip2
  - cf-bpr
  - attributes-qwen3_embedding_0.6b
  - lyrics-qwen3_embedding_0.6b
  - metadata-qwen3_embedding_0.6b

- **Gating mechanism**: For each modality, compute gate = sigmoid(MLP([intent_repr, modal_emb]))
- **Fusion**: Weighted sum of modalities using softmax-normalized gates
- **Categorical features**: track_id, ISRC, tag_list, artist_id, album_id, duration_bucket, release_year_bucket
- **Architecture**: MLP [fused_features → 256 → 128]
- Output: item_repr (128 dims)

**User Tower**:
- Input: user_cf_bpr (128) + user categorical features
- **Categorical features**: user_id, age, country_code, gender, preferred_language, preferred_musical_culture, year, month, is_workday, category, specificity
- **Architecture**: MLP [features → 128 → 128]
- Output: user_repr (128 dims)

**Final Scoring**:
- Concatenate: [intent_repr, item_repr, user_repr] (384 dims)
- MLP: [384 → 128 → 64 → 32 → 1]
- Output: predicted score for ranking

### Feature Engineering

**Duration Bucketing**:
```
0: <30s
1: 30s-1min
2: 1-2min
3: 2-3min
4: 3-3.5min
5: 3.5-4min
6: 4-6min
7: >6min
```

**Release Year Bucketing**:
```
0: <1950
1: 1950-1960
2: 1960-1970
3: 1970-1980
4: 1980-1985
5: 1985-1990
6: 1990-1995
7: 1995-2000
8: 2000-2005
9: 2005-2010
10: 2010-2015
11: 2015-2020
12: 2020-2025
13: 2025-2030
14: >2030
```

## Configuration

**File**: `config/llama1b_multi_channel_devset.yaml`

```yaml
pipeline_version: "v2"  # Use CRS_BASELINE_V2
lm_type: "/home/lijiatong06/music-crs-baselines/Llama-3.2-1B-Instruct"
qwen_model_path: "/home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B"
retrieval_topk: 350  # Multi-channel returns ~350 tracks
rerank_topk: 20  # Top 20 after reranking
reranker_lr: 0.001
```

## Usage

### Running Inference

```bash
python run_inference_devset_v2.py \
    --tid llama1b_multi_channel_devset \
    --batch_size 16
```

The script automatically detects `pipeline_version: "v2"` in the config and loads the V2 pipeline.

### Programmatic Usage

```python
from mcrs import load_crs_baseline_v2

# Initialize V2 pipeline
music_crs = load_crs_baseline_v2(
    lm_type="/home/lijiatong06/music-crs-baselines/Llama-3.2-1B-Instruct",
    qwen_model_path="/home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B",
    device="cuda",
    rerank_topk=20,
)

# Single query
result = music_crs.chat(
    user_query="I want upbeat pop music",
    user_id="user_123",
    session_memory=[],
)

# Batch queries
batch_data = [
    {"user_query": "relaxing jazz", "user_id": "user_1", "session_memory": []},
    {"user_query": "energetic rock", "user_id": "user_2", "session_memory": []},
]
results = music_crs.batch_chat(batch_data)
```

## Caching Strategy

To minimize redundant computation, the following data is cached:

1. **Track Embeddings**:
   - `cache/multi_channel_retrieval/track_cf_bpr.pt` (CF-BPR matrix)
   - `cache/multi_channel_retrieval/track_metadata_256.pt` (Metadata embeddings, first 256 dims)
   - `cache/multi_channel_retrieval/track_attributes_256.pt` (Attributes embeddings, first 256 dims)
   - `cache/multi_channel_retrieval/track_ids.json` (Track ID list)

2. **User History**:
   - `cache/multi_channel_retrieval/user_liked_music.json` (User liked music index)

3. **Model Checkpoints**:
   - `cache/three_tower_reranker/model.pt` (Three-tower model weights)

## Key Differences from V1

| Aspect | V1 (Original) | V2 (add_all_feature_v2) |
|--------|---------------|-------------------------|
| Retrieval | Single channel (BM25/BERT) | Multi-channel (4 channels) |
| Candidates | 20 tracks | ~350 tracks |
| Reranking | FM model | Three-tower model |
| Query Encoding | BM25/BERT | Qwen3-0.6B |
| User History | Not used | Used for CF and liked music |
| Modality Fusion | Simple concatenation | Gating mechanism |
| Intent Modeling | Not explicit | Explicit intent tower |

## Performance Considerations

1. **Memory**: V2 requires more GPU memory due to:
   - Larger candidate set (~350 vs 20)
   - Three-tower architecture
   - Qwen3-0.6B model for encoding

2. **Speed**: V2 is slower due to:
   - Multi-channel retrieval
   - More complex reranking model
   - Query encoding with Qwen3

3. **Recommendations**:
   - Use batch_size=8 or lower if GPU memory is limited
   - Pre-compute and cache embeddings before inference
   - Consider using mixed precision (bfloat16) for LLM

## Files Modified/Created

### New Files
- `mcrs/retrieval_modules/multi_channel_retrieval.py`
- `mcrs/reranking_modules/three_tower_reranker.py`
- `mcrs/crs_baseline_v2.py`
- `config/llama1b_multi_channel_devset.yaml`
- `run_inference_devset_v2.py`
- `ADD_ALL_FEATURE_V2.md` (this file)

### Modified Files
- `mcrs/__init__.py` (added `load_crs_baseline_v2`)
- `mcrs/retrieval_modules/__init__.py` (added multi-channel support)
- `mcrs/reranking_modules/__init__.py` (added three-tower support)

## Future Improvements

1. **Training**: Implement training loop for the three-tower model using BPR loss
2. **Negative Sampling**: Add hard negative sampling strategy
3. **Feature Engineering**: Add more sophisticated feature transformations
4. **Efficiency**: Optimize multi-channel retrieval with parallel processing
5. **Ablation**: Study contribution of each retrieval channel and tower component
