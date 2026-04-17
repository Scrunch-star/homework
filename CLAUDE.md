# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 项目概述

跨模态蕴含（Vision-Language Entailment, VLE）课程大作业。基于 **SNLI-VE** 数据集，判断图像与文本假设之间的一致性关系（Entailment / Contradiction / Neutral）。

**技术路线**：CLIP zero-shot → BLIP/LLaVA prompt → Chain-of-Thought → LLM Judge

---

## 环境依赖

```bash
pip install torch torchvision
pip install transformers accelerate datasets
pip install pillow matplotlib seaborn pandas numpy tqdm scikit-learn
```

验证：`python -c "import torch, transformers; print(torch.__version__)"`

当前已安装版本：torch 2.9.1+cu126, transformers 4.56.1

---

## 常用命令

**重要**：所有命令需在 `vl-entailment-project/` 目录下执行。

### 数据处理

```bash
# 下载 SNLI-VE 标注文件
python -m src.data.download

# 批量下载图像（多线程加速）
python -m src.data.download_images --num_workers 8

# 可视化样本（用于数据集理解）
python -m src.data.visualize --num_samples 20 --output outputs/visualizations/

# 小样本测试（快速验证）
python -m src.data.visualize --num_samples 5 --output outputs/visualizations/test/
```

### 模型推理

```bash
# CLIP 评估（完整验证集）
python -m src.methods.clip_eval --split validation --batch_size 32

# CLIP 小样本测试
python -m src.methods.clip_eval --split validation --max_samples 100

# BLIP 评估
python -m src.methods.llm_eval --model blip --split validation --batch_size 8

# LLaVA 评估
python -m src.methods.llm_eval --model llava --split validation --batch_size 4

# Chain-of-Thought 推理（结构化输出）
python -m src.methods.cot_eval --model llava --split validation --batch_size 4

# 调整 CLIP 阈值
python -m src.methods.clip_eval --threshold_high 0.1 --threshold_low 0.1
```

### 结果分析

```bash
# 对比所有方法的性能指标
python -m src.analysis.compare --output outputs/analysis/

# 错误分析（需先生成预测结果）
python -m src.analysis.errors --num_examples 4 --output outputs/analysis/
```

### 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_clip_method.py -v

# 运行单个测试函数
pytest tests/test_clip_method.py::test_clip_similarity -v

# 生成覆盖率报告
pytest tests/ --cov=src --cov-report=term
```

---

## 项目结构

```
vl-entailment-project/
├── src/
│   ├── data/         # 数据下载、加载、可视化
│   ├── methods/      # CLIP / BLIP / LLaVA 推理实现
│   ├── analysis/     # 指标计算、错误分析
│   └── utils/        # config、logger
├── tests/
├── data/             # SNLI-VE JSON 标注 + 图像缓存
├── outputs/
│   ├── visualizations/
│   ├── predictions/  # 推理结果缓存（避免重复运行）
│   └── analysis/
└── configs/
```

所有入口脚本均以 `python -m src.<module>` 方式调用，从 `vl-entailment-project/` 目录下执行。

---

## 架构要点

### 数据流
`download.py` 下载标注 → `download_images.py` 下载图像 → `dataset.py` 封装 Dataset → 各 `*_eval.py` 消费

### 模型评估器设计
- **CLIP**：计算 image-hypothesis 和 image-premise 相似度差异，通过阈值判断蕴含关系
- **BLIP/LLaVA**：使用统一 prompt 模板要求模型输出 A/B/C，通过 `parse_response` 正则提取
- **CoT**：要求模型返回 JSON 格式，包含 `image_entities`、`statement_entities`、`conflict_detected`、`label` 字段

### 关键设计模式
- **延迟加载**：模型在 `__init__` 时加载，不在 import 时初始化，避免不必要的内存占用
- **推理缓存**：所有预测结果保存至 `outputs/predictions/`，包含方法名、准确率、完整结果列表
- **容错机制**：LLM 输出解析失败时回退到 "neutral"；图像加载失败时返回黑色占位图
- **设备自适应**：所有模型自动检测 CUDA 可用性，优先使用 GPU

### 标签映射
```python
LABELS = ["entailment", "neutral", "contradiction"]  # 索引 0/1/2
ANSWER2LABEL = {"A": "entailment", "B": "neutral", "C": "contradiction"}
```

---

## 代码规范

- 类型注解：所有函数参数和返回值必须标注
- 命名：类用 CapWords，函数/变量用 snake_case，常量用 UPPER_SNAKE
- 行长：120 字符上限
- GPU 优先：`device = "cuda" if torch.cuda.is_available() else "cpu"`
- 注释和文档用中文，代码标识符用英文

---

## 开发指南

### 添加新的评估方法

1. 在 `src/methods/` 创建新文件（如 `new_method.py`）
2. 实现 Evaluator 类，包含 `predict_label(image, hypothesis)` 方法
3. 在 `main()` 中保存结果至 `PRED_DIR / f"{method_name}_{split}.json"`
4. 结果格式必须包含：`method`、`split`、`accuracy`、`results` 字段

### 调试技巧

- 使用 `--max_samples 10` 快速验证流程
- 检查 `outputs/predictions/` 中的 JSON 文件确认输出格式
- 图像加载失败会返回黑色占位图，不会中断推理
- CLIP 阈值调优：先用小样本测试不同 `threshold_high/low` 组合

### 常见问题

**Q: 模型加载失败或 OOM？**
A: 降低 `batch_size`，或在 BLIP/LLaVA 中使用 `torch.float32` 替代 `float16`

**Q: 预测结果全是 neutral？**
A: 检查 `parse_response` 正则是否匹配模型输出格式；查看原始 response

**Q: 图像路径错误？**
A: 确保在 `vl-entailment-project/` 目录执行命令，检查 `config.py` 中的路径配置

---

## 当前进度

数据集已下载（31783 张图像），代码框架已完成。待执行完整评估并生成实验结果。详见 [progress.md](progress.md)。
