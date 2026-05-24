# Music-CRS-Baselines 项目详细文档

## 目录
- [项目概述](#项目概述)
- [项目结构](#项目结构)
- [根目录文件](#根目录文件)
- [核心模块详解](#核心模块详解)
  - [mcrs 主模块](#mcrs-主模块)
  - [lm_modules 语言模型模块](#lm_modules-语言模型模块)
  - [retrieval_modules 检索模块](#retrieval_modules-检索模块)
  - [db_item 音乐数据库模块](#db_item-音乐数据库模块)
  - [db_user 用户数据库模块](#db_user-用户数据库模块)
- [配置文件](#配置文件)
- [Baseline 方法](#baseline-方法)
- [扩展建议](#扩展建议)
- [函数索引](#函数索引)

---

## 项目概述

**Music-CRS-Baselines** 是 RecSys Challenge 2026 对话式音乐推荐系统挑战赛的官方评估框架。该项目为参赛者提供了一个标准化的基线系统，用于在 TalkPlay Data Challenge 数据集上评估音乐推荐系统。

### 核心功能
- **对话式推荐**：通过自然语言对话理解用户偏好并推荐音乐
- **双阶段架构**：
  1. **RecSys 阶段**：使用 BM25 或 BERT 检索候选曲目
  2. **LLM 阶段**：使用 Llama-3.2-1B 生成自然语言响应

### 技术栈
- **语言模型**：Llama-3.2-1B-Instruct (可扩展到其他模型)
- **检索方法**：BM25 (稀疏检索)、BERT (密集检索)
- **数据处理**：HuggingFace Datasets
- **深度学习框架**：PyTorch, Transformers

---

## 项目结构

```
music-crs-baselines/
├── mcrs/                          # 主要代码包
│   ├── __init__.py                # 包初始化和加载函数
│   ├── crs_baseline.py            # CRS 基线系统核心类
│   ├── db_item/                   # 音乐目录数据库
│   │   ├── __init__.py
│   │   └── music_catalog.py       # 音乐元数据访问
│   ├── db_user/                   # 用户资料数据库
│   │   ├── __init__.py
│   │   └── user_profile.py        # 用户档案访问
│   ├── lm_modules/                # 语言模型模块
│   │   ├── __init__.py
│   │   └── llama.py               # Llama 模型包装器
│   ├── retrieval_modules/         # 检索模块
│   │   ├── __init__.py
│   │   ├── bert.py                # BERT 嵌入检索
│   │   └── bm25.py                # BM25 稀疏检索
│   └── system_prompts/            # 系统提示模板
│       ├── personalization.txt    # 个性化提示
│       ├── response_generation.txt # 响应生成提示
│       └── roleplay.txt           # 角色扮演提示
├── config/                        # 配置文件目录
│   ├── llama1b_bert_devset.yaml
│   ├── llama1b_bm25_devset.yaml
│   ├── llama1b_bert_blindset_A.yaml
│   └── llama1b_bm25_blindset_A.yaml
├── lowerbound/                    # 基准方法
│   ├── popularity.py              # 流行度基线
│   └── random_sample.py           # 随机采样基线
├── tips/                          # 改进建议文档
│   ├── add_reranker.md            # 添加重排序模块
│   ├── improve_item_representation.md # 改进物品表示
│   └── use_genrec_semantic_ids.md # 生成式检索方法
├── run_inference_devset.py        # 开发集推理脚本
├── run_inference_blindset.py      # 盲测集推理脚本
├── pyproject.toml                 # 项目依赖配置
└── readme.md                      # 项目说明文档
```

---

## 根目录文件

### run_inference_devset.py
开发集推理脚本，用于在 TalkPlayData-2 测试数据集上运行批量推理。

#### 主要函数

**`chat_history_parser(conversations, music_crs, target_turn_number)`**
- **功能**：解析对话历史直到目标轮次
- **参数**：
  - `conversations`: 对话轮次字典列表 (包含 turn_number, role, content)
  - `music_crs`: CRS 基线实例（用于将曲目 ID 转换为元数据）
  - `target_turn_number`: 要预测的目标轮次
- **返回**：
  - `chat_history`: 之前的消息列表
  - `user_query`: 目标轮次的用户查询
- **逻辑**：
  1. 将对话转换为 DataFrame 进行过滤
  2. 提取目标轮次之前的所有对话
  3. 将 "music" 角色转换为 "assistant" 并解析曲目元数据
  4. 返回格式化的对话历史和当前用户查询

**`main(args)`**
- **功能**：在测试数据集上运行批量推理
- **参数**：
  - `args.tid`: 任务/配置标识符
  - `args.batch_size`: 批处理大小
  - `args.save_path`: 输出目录
- **处理流程**：
  1. 清除缓存目录防止内存问题
  2. 加载配置文件 (config/{tid}.yaml)
  3. 初始化 CRS 基线系统
  4. 加载测试数据集
  5. 为每个会话的每个轮次 (1-8) 准备批处理数据
  6. 执行批量推理并收集结果
  7. 保存结果到 exp/inference/devset/{tid}.json
- **输出格式**：
  ```json
  {
    "session_id": "会话ID",
    "user_id": "用户ID",
    "turn_number": 轮次编号,
    "predicted_track_ids": ["曲目ID列表"],
    "predicted_response": "生成的响应文本"
  }
  ```

### run_inference_blindset.py
盲测集推理脚本，用于提交评估。与 devset 脚本类似，但有以下区别：

**主要差异**：
1. **数据处理**：只处理对话的最后一轮（不是 1-8 所有轮次）
   ```python
   chat_history = item['conversations'][:-1]
   user_query = item['conversations'][-1]['content']
   ```
2. **输出路径**：保存到 `exp/inference/{eval_dataset}/{tid}.json`
3. **命令行参数**：增加了 `--eval_dataset` 参数指定评估数据集名称

---

## 核心模块详解

### mcrs 主模块

#### mcrs/\_\_init\_\_.py
模块入口文件，提供加载函数。

**`load_crs_baseline(...)`**
- **功能**：创建并返回 CRS_BASELINE 实例
- **参数**：
  - `lm_type`: 语言模型类型 (默认: "meta-llama/Llama-3.2-1B-Instruct")
  - `retrieval_type`: 检索类型 ("bm25" 或 "bert")
  - `item_db_name`: 曲目元数据数据集名称
  - `user_db_name`: 用户元数据数据集名称
  - `track_split_types`: 曲目数据集分割类型列表
  - `user_split_types`: 用户数据集分割类型列表
  - `corpus_types`: 检索使用的元数据字段列表
  - `cache_dir`: 缓存目录
  - `device`: 计算设备 ("cuda" 或 "cpu")
  - `attn_implementation`: 注意力机制实现方式
  - `dtype`: 模型数据类型
- **返回**：CRS_BASELINE 实例

#### mcrs/crs_baseline.py
对话推荐系统的核心实现类。

**类：`CRS_BASELINE`**

**属性**：
- `lm`: 语言模型模块
- `retrieval`: 检索模块
- `item_db`: 音乐目录数据库
- `user_db`: 用户资料数据库
- `role_prompt`: 角色提示模板字典
- `session_memory`: 当前会话的消息列表

**方法详解**：

**`__init__(...)`**
- **功能**：初始化 CRS 基线组件
- **流程**：
  1. 保存配置参数
  2. 加载语言模型模块
  3. 加载检索模块
  4. 初始化音乐目录数据库
  5. 初始化用户资料数据库
  6. 加载系统提示模板
  7. 初始化会话内存

**`_reset_session_memory()`**
- **功能**：清空当前会话内存中的所有消息
- **用途**：开始新会话时重置状态

**`_upload_session_memory(chat_history)`**
- **功能**：上传会话记忆到数据库
- **参数**：
  - `chat_history`: 聊天历史消息列表
- **用途**：恢复之前的对话上下文

**`_get_system_prompt(user_id)`**
- **功能**：构建系统提示，可选择性地包含用户个性化信息
- **参数**：
  - `user_id`: 可选的用户标识符
- **返回**：最终的系统提示字符串
- **逻辑**：
  1. 组合角色扮演和响应生成提示
  2. 如果提供 user_id，添加个性化提示和用户档案
  3. 返回完整的系统提示

**`chat(user_query, user_id)`**
- **功能**：运行单次 CRS 轮次：检索物品并生成响应
- **参数**：
  - `user_query`: 用户最新的消息或请求
  - `user_id`: 可选的用户标识符
- **返回**：包含以下键的字典
  - `user_id`: 用户标识符
  - `user_query`: 输入查询的回显
  - `retrieval_items`: 检索到的物品 ID 列表（top 20）
  - `recommend_item`: 推荐的顶部物品的元数据
  - `response`: 生成的助手响应字符串
- **流程**：
  1. 将用户查询添加到会话内存
  2. 生成系统提示
  3. 构建检索输入（对话历史的连接）
  4. 使用检索模块获取 top 20 候选曲目
  5. 获取第一个推荐曲目的元数据
  6. 使用语言模型生成响应
  7. 返回完整结果字典

**`batch_chat(batch_data)`**
- **功能**：批量运行多个 CRS 轮次：检索物品并生成响应
- **参数**：
  - `batch_data`: 字典列表，每个包含
    - `user_query`: 用户查询
    - `user_id`: 可选用户标识符
    - `session_memory`: 聊天历史消息列表
- **返回**：字典列表，每个包含与 `chat()` 相同的键
- **优化**：
  1. 批量准备所有系统提示
  2. 批量准备所有检索输入
  3. 使用批量检索（如果可用）
  4. 使用批量响应生成（如果可用）
  5. 如果批量方法不可用，回退到顺序处理

---

### lm_modules 语言模型模块

#### mcrs/lm_modules/\_\_init\_\_.py
语言模型模块加载器。

**`load_lm_module(lm_type, device, attn_implementation, dtype)`**
- **功能**：根据类型加载语言模型
- **参数**：
  - `lm_type`: 模型类型标识符
  - `device`: 计算设备
  - `attn_implementation`: 注意力实现
  - `dtype`: 数据类型
- **支持的模型**：
  - "meta-llama/Llama-3.2-1B-Instruct"
- **返回**：LLAMA_MODEL 实例
- **异常**：如果模型类型不支持，抛出 ValueError

#### mcrs/lm_modules/llama.py
Llama 模型包装器，用于响应生成。

**类：`LLAMA_MODEL`**

**方法详解**：

**`__init__(model_name, device, attn_implementation, dtype)`**
- **功能**：初始化 Llama 模型
- **流程**：
  1. 保存配置参数
  2. 加载模型和分词器
  3. 设置模型为评估模式
  4. 将模型移动到指定设备和数据类型

**`_load_model()`**
- **功能**：加载 Transformers 模型和分词器
- **返回**：(模型, 分词器) 元组
- **配置**：
  - 分词器使用左填充 (padding_side="left")
  - 模型使用指定的注意力实现和数据类型

**`_format_chat_history(sys_prompt, chat_history, recommend_item)`**
- **功能**：格式化对话历史为聊天模板
- **参数**：
  - `sys_prompt`: 系统提示
  - `chat_history`: 对话历史列表
  - `recommend_item`: 推荐物品字符串
- **返回**：格式化的聊天模板字符串
- **流程**：
  1. 创建包含系统提示的初始对话数据
  2. 添加对话历史
  3. 添加助手角色的推荐物品
  4. 应用分词器的聊天模板
  5. 添加生成提示

**`response_generation(sys_prompt, chat_history, recommend_item, max_new_tokens, response_format)`**
- **功能**：生成单个响应
- **参数**：
  - `sys_prompt`: 系统提示
  - `chat_history`: 对话历史
  - `recommend_item`: 推荐物品
  - `max_new_tokens`: 最大生成令牌数 (默认 512)
  - `response_format`: 响应格式 (可选)
- **返回**：生成的响应文本
- **流程**：
  1. 格式化对话历史
  2. 分词输入
  3. 将输入移动到设备
  4. 使用 no_grad 上下文生成
  5. 解码并返回生成的文本（仅新生成的令牌）

**`batch_response_generation(sys_prompts, chat_histories, recommend_items, max_new_tokens)`**
- **功能**：批量生成多个响应
- **参数**：
  - `sys_prompts`: 系统提示列表
  - `chat_histories`: 对话历史列表的列表
  - `recommend_items`: 推荐物品列表
  - `max_new_tokens`: 最大生成令牌数 (默认 64)
- **返回**：生成的响应文本列表
- **优化**：
  1. 批量格式化所有对话历史
  2. 使用填充进行批量分词
  3. 设置 pad_token（如果未设置）
  4. 批量生成所有响应
  5. 批量解码生成的文本

---

### retrieval_modules 检索模块

#### mcrs/retrieval_modules/\_\_init\_\_.py
检索模块加载器。

**`load_retrieval_module(retrieval_type, dataset_name, track_split_types, corpus_types, cache_dir)`**
- **功能**：根据类型加载检索模块
- **参数**：
  - `retrieval_type`: 检索类型 ("bm25" 或 "bert")
  - `dataset_name`: 数据集名称
  - `track_split_types`: 曲目分割类型列表
  - `corpus_types`: 语料字段列表
  - `cache_dir`: 缓存目录
- **支持的方法**：
  - "bm25": BM25 稀疏检索
  - "bert": BERT 密集检索
- **返回**：BM25_MODEL 或 BERT_MODEL 实例
- **异常**：如果检索类型不支持，抛出 ValueError

#### mcrs/retrieval_modules/bm25.py
基于 BM25 的曲目元数据检索工具。

**类：`BM25_MODEL`**

**方法详解**：

**`__init__(dataset_name, split_types, corpus_types, cache_dir)`**
- **功能**：初始化 BM25 检索器
- **流程**：
  1. 保存配置参数
  2. 生成语料名称（corpus_types 连接）
  3. 加载元数据语料
  4. 如果缓存存在，加载 BM25 索引
  5. 否则，构建索引并保存

**`_load_bm25(corpus_name)`**
- **功能**：加载缓存的 BM25 索引和曲目 ID 列表
- **参数**：
  - `corpus_name`: 缓存目录下的语料子目录名
- **返回**：(bm25_model, track_ids) 元组
- **文件**：
  - `{cache_dir}/bm25/{corpus_name}/model.pkl`
  - `{cache_dir}/bm25/{corpus_name}/track_ids.json`

**`_load_corpus()`**
- **功能**：从配置的数据集加载并组合元数据分割
- **返回**：从 track_id 到元数据字典的映射
- **流程**：
  1. 从 HuggingFace 加载数据集
  2. 连接所有指定的分割
  3. 创建 track_id 到元数据的字典映射

**`_stringify_metadata(metadata)`**
- **功能**：将元数据字典转换为多行字符串用于索引
- **参数**：
  - `metadata`: 包含 corpus_types 列出字段的曲目元数据
- **返回**：换行分隔的字符串，每个选定字段一行 "字段名: 值"
- **处理**：如果字段值是列表，使用逗号连接

**`build_index()`**
- **功能**：构建并持久化 BM25 索引
- **流程**：
  1. 提取所有 track_ids
  2. 为每个曲目生成元数据字符串
  3. 使用 bm25s.tokenize 分词语料
  4. 创建 BM25 检索器并建立索引
  5. 创建缓存目录
  6. 保存 BM25 模型和语料
  7. 保存 track_ids 到 JSON

**`text_to_item_retrieval(query, topk)`**
- **功能**：为自然语言查询检索 top-k 曲目 ID
- **参数**：
  - `query`: 用户文本查询
  - `topk`: 要检索的物品数量
- **返回**：按 BM25 分数降序排列的曲目 ID 列表
- **流程**：
  1. 将查询转换为小写并分词
  2. 使用 BM25 模型检索 top-k 结果
  3. 将文档 ID 映射回 track_ids
  4. 返回 track_id 列表

**`batch_text_to_item_retrieval(queries, topk)`**
- **功能**：批量检索多个查询的 top-k 曲目 ID
- **参数**：
  - `queries`: 用户文本查询列表
  - `topk`: 每个查询要检索的物品数量
- **返回**：列表的列表，每个内部列表包含按 BM25 分数排序的曲目 ID
- **优化**：批量分词和检索提高效率

#### mcrs/retrieval_modules/bert.py
基于 BERT 嵌入的曲目元数据检索工具。

**类：`BERT_MODEL`**

**方法详解**：

**`__init__(dataset_name, split_types, corpus_types, cache_dir, model_name, device, batch_size, max_length)`**
- **功能**：初始化 BERT 检索器
- **参数**：
  - `model_name`: HuggingFace 模型 ID（默认 "bert-base-uncased"）
  - `device`: Torch 设备（None 时自动选择 CUDA 或 CPU）
  - `batch_size`: 构建索引时的批处理大小（默认 32）
  - `max_length`: 分词的最大序列长度（默认 128）
- **流程**：
  1. 保存配置参数并设置索引目录
  2. 自动选择设备（优先 CUDA）
  3. 加载元数据语料
  4. 加载分词器和 BERT 模型
  5. 将模型移动到设备并设置为评估模式
  6. 如果缓存存在，加载嵌入索引
  7. 否则，构建索引并保存

**`_load_index()`**
- **功能**：加载缓存的嵌入矩阵和曲目 ID 列表
- **返回**：(embeddings [num_items, dim], track_ids) 元组
- **文件**：
  - `{index_dir}/embeddings.pt`
  - `{index_dir}/track_ids.json`

**`_load_corpus()`**
- **功能**：从配置的数据集加载并组合元数据分割
- **返回**：从 track_id 到元数据字典的映射

**`_stringify_metadata(metadata)`**
- **功能**：将元数据字典转换为多行字符串用于索引
- **格式**：每行 "字段名: 值"

**`_mean_pool(last_hidden_states, attention_mask)`**
- **功能**：使用注意力掩码对令牌嵌入进行平均池化
- **参数**：
  - `last_hidden_states`: [batch, seq_len, hidden] 张量
  - `attention_mask`: [batch, seq_len] 张量
- **返回**：[batch, hidden] 平均池化嵌入
- **算法**：
  1. 扩展注意力掩码到隐藏维度
  2. 对被掩码的令牌嵌入求和
  3. 除以有效令牌数（避免除零）
  4. 返回平均嵌入

**`build_index()`**
- **功能**：构建并持久化嵌入索引
- **流程**：
  1. 提取所有 track_ids
  2. 为每个曲目生成元数据字符串
  3. 创建索引目录
  4. 批量处理语料文本：
     - 分词（填充、截断）
     - 通过 BERT 模型前向传播
     - 平均池化令牌嵌入
     - L2 归一化（用于余弦相似度）
     - 移动到 CPU 并保存
  5. 连接所有批次的嵌入
  6. 保存嵌入矩阵和 track_ids

**`text_to_item_retrieval(query, topk)`**
- **功能**：通过余弦相似度为查询检索 top-k 曲目 ID
- **参数**：
  - `query`: 用户文本查询
  - `topk`: 要检索的物品数量
- **返回**：按余弦相似度降序排列的曲目 ID 列表
- **流程**：
  1. 分词查询
  2. 通过 BERT 模型获取嵌入
  3. 平均池化并 L2 归一化
  4. 计算与所有曲目嵌入的余弦相似度（矩阵乘法）
  5. 使用 topk 操作获取最高分数索引
  6. 映射回 track_ids

**`batch_text_to_item_retrieval(queries, topk)`**
- **功能**：批量检索多个查询的 top-k 曲目 ID
- **参数**：
  - `queries`: 用户文本查询列表
  - `topk`: 每个查询要检索的物品数量
- **返回**：列表的列表，每个内部列表包含按相似度排序的曲目 ID
- **优化**：
  1. 批量分词所有查询
  2. 批量前向传播
  3. 批量计算余弦相似度
  4. 为每个查询提取 top-k

---

### db_item 音乐数据库模块

#### mcrs/db_item/music_catalog.py
音乐目录数据库访问器。

**类：`MusicCatalogDB`**

**方法详解**：

**`__init__(dataset_name, split_types, corpus_types)`**
- **功能**：初始化音乐目录数据库
- **参数**：
  - `dataset_name`: HuggingFace 数据集名称
  - `split_types`: 要加载的数据集分割列表
  - `corpus_types`: 使用的元数据字段列表
- **流程**：
  1. 加载数据集
  2. 连接所有指定分割
  3. 创建 track_id 到元数据的字典映射
  4. 保存 corpus_types 供后续使用

**`id_to_metadata(track_id, use_semantic_id)`**
- **功能**：将曲目 ID 转换为格式化的元数据字符串
- **参数**：
  - `track_id`: 曲目标识符
  - `use_semantic_id`: 是否使用语义 ID（当前未使用）
- **返回**：格式化的曲目信息字符串
- **格式**：`"track_id: xxx, field1: value1, field2: value2, ..."`
- **处理**：
  1. 获取曲目元数据字典
  2. 从 track_id 开始构建字符串
  3. 遍历 corpus_types 字段
  4. 如果字段是列表，使用逗号连接并转换为小写
  5. 添加到结果字符串
  6. 返回完整的元数据字符串

---

### db_user 用户数据库模块

#### mcrs/db_user/user_profile.py
用户资料数据库访问器。

**类：`UserProfileDB`**

**方法详解**：

**`__init__(dataset_name, split_types)`**
- **功能**：初始化用户资料数据库
- **参数**：
  - `dataset_name`: HuggingFace 数据集名称
  - `split_types`: 要加载的数据集分割列表
- **流程**：
  1. 加载用户元数据数据集
  2. 连接所有指定分割
  3. 定义默认列 ['user_id', 'age_group', 'gender', 'country_name']
  4. 创建 user_id 到用户档案的字典映射

**`id_to_profile(user_id)`**
- **功能**：获取用户档案字典
- **参数**：
  - `user_id`: 用户标识符
- **返回**：用户档案字典（包含所有元数据字段）

**`id_to_profile_str(user_id)`**
- **功能**：将用户档案转换为格式化字符串
- **参数**：
  - `user_id`: 用户标识符
- **返回**：换行分隔的用户信息字符串
- **格式**：
  ```
  user_id: xxx
  age_group: xxx
  gender: xxx
  country_name: xxx
  ```
- **流程**：
  1. 获取用户档案
  2. 遍历默认列
  3. 构建 "key: value" 格式的字符串列表
  4. 使用换行符连接并返回

---

## 配置文件

### config/llama1b_bm25_devset.yaml
使用 Llama-3.2-1B 和 BM25 检索的开发集配置示例。

**配置项说明**：
- `lm_type`: 语言模型标识符
  - 值: "meta-llama/Llama-3.2-1B-Instruct"
  - 用途: 指定使用的 LLM 模型

- `retrieval_type`: 检索方法
  - 值: "bm25"
  - 可选: "bm25" 或 "bert"

- `test_dataset_name`: 测试数据集名称
  - 值: "talkpl-ai/TalkPlayData-Challenge-Dataset"

- `item_db_name`: 曲目元数据数据集
  - 值: "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"

- `user_db_name`: 用户元数据数据集
  - 值: "talkpl-ai/TalkPlayData-Challenge-User-Metadata"

- `track_split_types`: 曲目数据分割
  - 值: ["all_tracks"]
  - **重要**：评估时必须使用 "all_tracks"，不能过滤或限制曲目

- `user_split_types`: 用户数据分割
  - 值: ["all_users"]

- `corpus_types`: 检索使用的元数据字段
  - 值: ["track_name", "artist_name", "album_name", "release_date"]
  - 用途: 定义用于构建检索索引的曲目信息字段

- `cache_dir`: 缓存目录
  - 值: "./cache"
  - 用途: 存储索引、模型和中间结果

- `device`: 计算设备
  - 值: "cuda"
  - 可选: "cuda" 或 "cpu"

- `attn_implementation`: 注意力机制实现
  - 值: "flash_attention_2"
  - 可选: "flash_attention_2" 或 "eager"
  - 用途: flash_attention_2 更快但需要安装额外依赖

---

## Baseline 方法

### lowerbound/popularity.py
基于流行度的基线方法。

**函数详解**：

**`load_popularity_track()`**
- **功能**：从训练集中加载最流行的 20 首曲目
- **返回**：流行曲目 ID 列表
- **流程**：
  1. 加载训练集数据
  2. 遍历所有对话
  3. 收集所有推荐的曲目 ID（role == 'music'）
  4. 使用 Counter 统计频率
  5. 返回出现频率最高的 20 首曲目

**`main()`**
- **功能**：生成流行度基线预测
- **流程**：
  1. 加载流行曲目列表
  2. 加载测试集
  3. 为每个会话的每个轮次 (1-8) 生成预测
  4. 所有预测都使用相同的 20 首流行曲目
  5. 不生成响应文本（空字符串）
  6. 保存结果到 exp/inference/popularity.json
- **用途**：提供最简单的基线，评估仅推荐流行曲目的效果

### lowerbound/random_sample.py
基于随机采样的基线方法。

**函数详解**：

**`load_track_pools()`**
- **功能**：加载所有可用的曲目 ID
- **返回**：曲目 ID 列表
- **流程**：
  1. 加载曲目元数据数据集的 "all_tracks" 分割
  2. 提取所有 track_id
  3. 返回列表

**`main()`**
- **功能**：生成随机采样基线预测
- **流程**：
  1. 加载所有可用曲目池
  2. 加载测试集
  3. 为每个会话的每个轮次 (1-8)：
     - 从曲目池中随机采样 20 首曲目
     - 创建预测条目
  4. 不生成响应文本（空字符串）
  5. 保存结果到 exp/inference/random.json
- **用途**：提供纯随机基线，作为最低性能标准

---

## 扩展建议

项目在 `tips/` 目录下提供了三个改进方向的文档：

### 1. 添加重排序模块 (tips/add_reranker.md)

**方案 A：基于嵌入的重排序**
- 使用用户嵌入进行个性化
  - 从听歌历史计算用户档案
  - 根据用户-物品相似度评分候选项
- 跨模态重排序：结合多个信号
  - 文本相关性 + 音频相似度 + 用户偏好

**方案 B：基于 LLM 的重排序**
- 使用 LLM 判断 top-k 候选项的相关性
- 提示示例："根据相关性对这些曲目进行排序: {user_query}"
- 模型选择：Llama-3-8B、Qwen-7B 或专门的排序器

**实现方法**：
```python
# 在检索后添加到 CRS 管道
retrieval_items = self.retrieval.text_to_item_retrieval(query, topk=100)

# 重排序 top 候选项
if self.reranker:
    retrieval_items = self.reranker.rerank(
        query=query,
        candidates=retrieval_items[:50],
        user_profile=user_profile,
        topk=20
    )
```

### 2. 改进物品表示 (tips/improve_item_representation.md)

**选项 A：添加更多文本字段**
- 当前仅使用：曲目名、艺术家名、专辑名
- 可以添加：流派标签、情绪标签、发行年份、流行度分数
- 在配置文件中编辑 `corpus_types` 以包含 `tag_list`

**选项 B：使用音频特征**
- 不仅使用文本，还使用音乐的实际声音
- 尝试 CLAP（理解文本和音频的模型）
- 有助于找到声音相似的歌曲，而不仅仅是描述相似

**更好的文本模型**：
- Qwen2.5-Embedding - 适用于多种语言
- Contriever - 无需训练即可找到相关物品
- E5 或 BGE - 当前最佳文本嵌入模型
- ColBERT - 更精确的词匹配

### 3. 生成式检索 (tips/use_genrec_semantic_ids.md)

**概念**：用端到端生成替代嵌入相似度检索

**语义 ID 方法**：
- 为曲目分配层次语义 ID（例如 `jazz/smooth/piano/0042`）
- 训练 LLM 根据用户查询生成相关的曲目 ID
- 单一模型替代检索和生成两个阶段

**优势**：
- 统一架构
- 可以建模复杂的用户意图
- 利用 LLM 推理能力

**实现步骤**：
1. 为曲目创建语义 ID 系统
2. 微调 LLM 以生成曲目 ID
3. 可选：使用协同过滤进行 ID 分配

---

## 函数索引

### 根目录脚本

#### run_inference_devset.py
- `chat_history_parser(conversations, music_crs, target_turn_number)` - 解析对话历史
- `main(args)` - 开发集批量推理主函数

#### run_inference_blindset.py
- `chat_history_parser(conversations, music_crs, target_turn_number)` - 解析对话历史
- `main(args)` - 盲测集批量推理主函数

### mcrs 模块

#### mcrs/\_\_init\_\_.py
- `load_crs_baseline(...)` - 创建 CRS 基线实例

#### mcrs/crs_baseline.py
**类 CRS_BASELINE**：
- `__init__(...)` - 初始化 CRS 组件
- `_reset_session_memory()` - 清空会话内存
- `_upload_session_memory(chat_history)` - 上传会话历史
- `_get_system_prompt(user_id)` - 构建系统提示
- `chat(user_query, user_id)` - 单次对话处理
- `batch_chat(batch_data)` - 批量对话处理

#### mcrs/lm_modules/\_\_init\_\_.py
- `load_lm_module(lm_type, device, attn_implementation, dtype)` - 加载语言模型

#### mcrs/lm_modules/llama.py
**类 LLAMA_MODEL**：
- `__init__(model_name, device, attn_implementation, dtype)` - 初始化模型
- `_load_model()` - 加载 Transformers 模型
- `_format_chat_history(sys_prompt, chat_history, recommend_item)` - 格式化对话
- `response_generation(...)` - 单个响应生成
- `batch_response_generation(...)` - 批量响应生成

#### mcrs/retrieval_modules/\_\_init\_\_.py
- `load_retrieval_module(...)` - 加载检索模块

#### mcrs/retrieval_modules/bm25.py
**类 BM25_MODEL**：
- `__init__(dataset_name, split_types, corpus_types, cache_dir)` - 初始化 BM25
- `_load_bm25(corpus_name)` - 加载缓存索引
- `_load_corpus()` - 加载元数据语料
- `_stringify_metadata(metadata)` - 元数据转字符串
- `build_index()` - 构建 BM25 索引
- `text_to_item_retrieval(query, topk)` - 单查询检索
- `batch_text_to_item_retrieval(queries, topk)` - 批量查询检索

#### mcrs/retrieval_modules/bert.py
**类 BERT_MODEL**：
- `__init__(...)` - 初始化 BERT 检索器
- `_load_index()` - 加载嵌入索引
- `_load_corpus()` - 加载元数据语料
- `_stringify_metadata(metadata)` - 元数据转字符串
- `_mean_pool(last_hidden_states, attention_mask)` - 平均池化
- `build_index()` - 构建嵌入索引
- `text_to_item_retrieval(query, topk)` - 单查询检索
- `batch_text_to_item_retrieval(queries, topk)` - 批量查询检索

#### mcrs/db_item/music_catalog.py
**类 MusicCatalogDB**：
- `__init__(dataset_name, split_types, corpus_types)` - 初始化音乐目录
- `id_to_metadata(track_id, use_semantic_id)` - ID 转元数据字符串

#### mcrs/db_user/user_profile.py
**类 UserProfileDB**：
- `__init__(dataset_name, split_types)` - 初始化用户档案库
- `id_to_profile(user_id)` - 获取用户档案字典
- `id_to_profile_str(user_id)` - 获取用户档案字符串

### Baseline 方法

#### lowerbound/popularity.py
- `load_popularity_track()` - 加载流行曲目
- `main()` - 生成流行度基线预测

#### lowerbound/random_sample.py
- `load_track_pools()` - 加载所有曲目
- `main()` - 生成随机采样基线预测

---

## 使用流程

### 1. 安装依赖
```bash
uv venv .venv --python=3.10
source .venv/bin/activate
uv pip install -e .
uv pip install flash-attn --no-build-isolation  # 可选，用于快速推理
```

### 2. 运行开发集推理
```bash
# BM25 基线
python run_inference_devset.py --tid llama1b_bm25_devset --batch_size 16

# BERT 基线
python run_inference_devset.py --tid llama1b_bert_devset --batch_size 16
```

### 3. 运行盲测集推理（用于提交）
```bash
# BM25 基线
python run_inference_blindset.py --tid llama1b_bm25_blindset_A --batch_size 16

# BERT 基线
python run_inference_blindset.py --tid llama1b_bert_blindset_A --batch_size 16
```

### 4. 自定义配置
创建自己的配置文件 `config/my_model.yaml`，然后运行：
```bash
python run_inference_devset.py --tid my_model
```

### 5. 评估结果
使用官方评估工具：https://github.com/nlp4musa/music-crs-evaluator

---

## 数据流图

```
用户查询
    ↓
[会话历史处理]
    ↓
[构建系统提示] ← 用户档案（可选）
    ↓
[检索阶段]
    ├→ BM25 检索 → Top 20 曲目
    └→ BERT 检索 → Top 20 曲目
    ↓
[获取推荐曲目元数据]
    ↓
[LLM 响应生成]
    ↓
[返回结果]
    ├→ 检索物品列表
    └→ 自然语言响应
```

---

## 关键设计决策

### 1. 双阶段架构
- **优势**：检索和生成分离，便于独立优化
- **检索阶段**：快速从大规模候选集中筛选
- **生成阶段**：提供自然、个性化的解释

### 2. 批量处理支持
- 所有模块都支持批量推理
- 显著提高推理效率
- 如果批量方法不可用，自动回退到顺序处理

### 3. 缓存机制
- BM25 索引和 BERT 嵌入都缓存到磁盘
- 避免重复计算，加快启动速度
- 缓存目录可配置

### 4. 模块化设计
- 语言模型、检索方法、数据库访问器都是独立模块
- 易于扩展和替换组件
- 支持添加新的模型和检索方法

### 5. 个性化支持
- 可选地使用用户档案信息
- 系统提示动态构建
- 支持年龄、性别、国家等属性

---

## 性能优化建议

### 1. 内存优化
- 使用 `torch.bfloat16` 减少内存占用
- 批处理大小根据 GPU 内存调整
- 定期清理缓存目录

### 2. 速度优化
- 使用 `flash_attention_2` 加速 LLM 推理
- 批量处理提高吞吐量
- GPU 推理优先于 CPU

### 3. 检索优化
- BM25 更快但可能不够精确
- BERT 更慢但语义理解更好
- 考虑混合检索策略

---

## 常见问题

### Q1: 如何更换语言模型？
A: 修改配置文件中的 `lm_type`，并在 `mcrs/lm_modules/__init__.py` 中添加支持。

### Q2: 如何添加新的检索方法？
A: 在 `mcrs/retrieval_modules/` 下创建新模块，实现 `text_to_item_retrieval` 和 `batch_text_to_item_retrieval` 方法。

### Q3: 为什么必须使用 all_tracks？
A: 评估时系统必须能够推荐任何曲目，不能预先过滤候选集，否则评估无效。

### Q4: 如何处理 GPU 内存不足？
A: 减小 `batch_size` 参数，或使用更小的模型。

### Q5: 如何改进推荐质量？
A: 参考 `tips/` 目录的文档，考虑添加重排序、改进物品表示或使用生成式检索。

---

**文档版本**：1.0  
**最后更新**：2026年5月  
**联系方式**：参考挑战赛官网
