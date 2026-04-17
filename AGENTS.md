# AGENTS.md — VLE 跨模态蕴含实验项目

This document provides instructions for AI agents working in this repository.

---

## 项目概述

本项目实现跨模态蕴含（Vision-Language Entailment）任务，基于 SNLI-VE 数据集进行图像-文本一致性判断实验。

**核心任务**：
- Entailment（蕴含）：文本由图像支持
- Contradiction（矛盾）：文本与图像冲突
- Neutral（无关）：图像无法支持文本

**主要方法**：CLIP、BLIP/LLaVA、链式推理（Chain-of-Thought）

---

## 环境配置

### 依赖安装

```bash
pip install torch torchvision
pip install transformers accelerate
pip install pillow matplotlib seaborn pandas numpy tqdm
pip install scikit-learn
pip install opencv-python
```

### 验证环境

```bash
python -c "import torch, transformers, CLIP; print('OK')"
```

---

## 运行命令

### 数据处理

```bash
# 下载 SNLI-VE 数据集
python -m src.data.download

# 下载数据集图像
python -m src.data.download_images --num_workers 8

# 可视化 20 条样本
python -m src.data.visualize --num_samples 20 --output outputs/visualizations/
```

### 模型推理

```bash
# CLIP 方法评估
python -m src.methods.clip_eval --split val --batch_size 32

# BLIP/LLaVA 方法评估
python -m src.methods.llm_eval --model blip --batch_size 8

# 链式推理方法
python -m src.methods.cot_eval --model llava --batch_size 4
```

### 结果分析

```bash
# 生成对比实验结果
python -m src.analysis.compare --output outputs/analysis/

# 错误分析
python -m src.analysis.errors --num_examples 4 --output outputs/analysis/
```

### 单个样本测试

```bash
# 测试 CLIP 模型（单样本）
python -m src.methods.clip_eval --mode single --image_path path/to/image.jpg --text "hypothesis text"

# 测试 LLM 模型（单样本）
python -m src.methods.llm_eval --model blip --mode single --image_path path/to/image.jpg --text "hypothesis text"
```

---

## 代码测试

### 运行全部测试

```bash
pytest tests/ -v
```

### 运行单个测试文件

```bash
pytest tests/test_data_loader.py -v
pytest tests/test_clip_method.py -v
pytest tests/test_llm_method.py -v
```

### 运行单个测试函数

```bash
pytest tests/test_clip_method.py::test_clip_similarity -v
pytest tests/test_llm_method.py::test_parse_response -v
```

### 测试覆盖率

```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term
```

---

## 代码风格规范

### 格式与导入

- **缩进**：4 空格（不用 Tab）
- **行长**：最多 120 字符
- **导入顺序**：
  1. 标准库
  2. 第三方库
  3. 本地模块
- **类型注解**：函数参数和返回值必须标注类型
- **字符串**：优先使用单引号，docstring 使用双引号

### 命名约定

| 类型 | 规范 | 示例 |
|------|------|------|
| 类名 | CapWords | `CLIPModel`, `VLEPipeline` |
| 函数名 | snake_case | `load_model`, `compute_similarity` |
| 变量名 | snake_case | `image_features`, `prediction_label` |
| 常量 | UPPER_SNAKE | `MAX_BATCH_SIZE`, `DEFAULT_DEVICE` |
| 私有成员 | _前缀 | `_cache`, `_model` |

### 错误处理

```python
# 正确示例
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    raise VLEError(f"Failed to process: {e}") from e

# 禁止示例
try:
    result = operation()
except:
    pass
```

### 日志规范

- 使用 `logging` 模块，配置全局 logger
- 日志级别：`DEBUG`（开发）、`INFO`（运行）、`WARNING`/`ERROR`（异常）
- 敏感信息（API keys、paths）使用占位符或脱敏

---

## 目录结构

```
vl-entailment-project/
├── src/
│   ├── __init__.py
│   ├── data/           # 数据加载与处理
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── download.py
│   │   └── visualize.py
│   ├── methods/        # 模型方法实现
│   │   ├── __init__.py
│   │   ├── clip.py
│   │   ├── blip.py
│   │   └── llava.py
│   ├── analysis/      # 结果分析
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── errors.py
│   └── utils/         # 工具函数
│       ├── __init__.py
│       ├── config.py
│       └── logger.py
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_clip_method.py
│   └── test_llm_method.py
├── data/               # 数据集存储
├── outputs/            # 实验输出
│   ├── visualizations/
│   ├── predictions/
│   └── analysis/
└── configs/            # 配置文件
```

---

## 架构原则

1. **单一职责**：每个模块/类/函数只做一件事
2. **依赖注入**：通过参数传递依赖，便于测试
3. **不可变输出**：数据处理函数返回新对象，不修改输入
4. **延迟加载**：大型模型按需加载，避免启动耗时
5. **结果缓存**：重复计算结果缓存至 `outputs/predictions/`

---

## 注释规范

- 类和公共方法必须有 docstring（英文）
- 复杂逻辑添加行内注释说明意图
- 禁用无意义注释（如 `# increment i`）

```python
def compute_similarity(image_embeds: Tensor, text_embeds: Tensor) -> Tensor:
    """
    Compute cosine similarity between image and text embeddings.

    Args:
        image_embeds: Normalized image feature vectors [B, D]
        text_embeds: Normalized text feature vectors [B, D]

    Returns:
        Similarity scores in range [-1, 1], shape [B]
    """
    return (image_embeds * text_embeds).sum(dim=-1)
```

---

## 性能注意事项

- GPU 可用时优先使用 GPU：`device = "cuda" if torch.cuda.is_available() else "cpu"`
- 大批量推理时设置 `pin_memory=True`
- 图像预处理使用 `torchvision.transforms` 而非 PIL（更高效）
- 避免在循环中创建大型张量

---

## 调试技巧

```bash
# 单步调试
python -m pdb -m src.methods.clip_eval --mode single --image_path test.jpg --text "test"

# 打印中间变量
python -m src.methods.clip_eval --debug --image_path test.jpg

# Profile 性能
python -m cProfile -o output.prof -m src.methods.clip_eval --split val
```

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| OOM（内存不足） | 减小 `batch_size`，或使用 `torch.cuda.empty_cache()` |
| 模型下载慢 | 设置 `HF_HUB_ENABLE_HF_TRANSFER=1` 或手动下载 |
| 图像下载失败 | 检查网络连接，重试或跳过损坏样本 |
| 输出格式错误 | 检查 `parse_response` 函数的正则匹配 |

---

*Last updated: 2026-03-20*
