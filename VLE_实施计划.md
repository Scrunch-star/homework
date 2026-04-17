# 跨模态蕴含（VLE）课程大作业实施计划

---

## 第一阶段：环境准备与数据集获取

### 步骤 1.1：创建项目目录结构

创建以下目录层次：

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

**验证方法**：执行 `tree vl-entailment-project` 或 `ls -R`，确认目录结构完整。

---

### 步骤 1.2：安装基础依赖

安装 Python 环境依赖包：

- `torch` (PyTorch 核心)
- `transformers` (Hugging Face 模型库)
- `CLIP` (OpenAI 的 CLIP 模型)
- `pillow` (图像处理)
- `matplotlib` / `seaborn` (可视化)
- `pandas` (数据处理)
- `numpy` (数值计算)
- `tqdm` (进度条)

**验证方法**：在 Python 环境中执行 `import torch; import transformers; import CLIP; print("OK")`，确认无报错。

---

### 步骤 1.3：下载 SNLI-VE 数据集

从官方源下载 SNLI-VE 数据集（验证集和测试集）。

**数据集包含字段**：
- `image_id`：图像标识符
- `premise_image_url`：图像 URL
- `premise`：premise 文本（来自 Flickr30k）
- `hypothesis`：假设文本
- `gold_label`：真实标签（entailment / contradiction / neutral）

**验证方法**：
1. 统计数据集样本数量（验证集约 5000 条，测试集约 5000 条）
2. 检查标签分布是否符合预期（约各占 1/3）
3. 随机抽样 5 条，逐一核对图像 URL 可访问性

---

### 步骤 1.4：下载数据集图像

编写脚本批量下载数据集中的图像（可使用多线程加速）。

**验证方法**：
1. 统计下载成功率（目标 ≥ 95%）
2. 检查下载图像的格式、尺寸是否一致
3. 随机抽查 10 张图像，确认内容与描述基本对应

---

## 第二阶段：任务一 — 数据集理解与可视化

### 步骤 2.1：随机抽取样本

编写脚本从验证集中随机抽取 **20 条**样本，确保：
- 覆盖三种标签类型（entailment ≈ 7 条，contradiction ≈ 7 条，neutral ≈ 6 条）
- 样本之间互不重复

**验证方法**：打印抽取样本的索引列表，核对无重复且标签分布符合预期。

---

### 步骤 2.2：图像与文本可视化

对抽取的 20 条样本，每条展示：
1. 图像（调整至统一尺寸如 400×400）
2. premise 文本（原描述）
3. hypothesis 文本（待判断假设）
4. gold_label（真实标签）
5. 样本编号（01-20）

**输出格式**：生成一张包含 20 个子图的网格图，或 20 张独立图像文件。

**验证方法**：
1. 人工审查每张图的图像、文本、标签是否正确对应
2. 确认文字清晰可读、图像完整无截断

---

### 步骤 2.3：标签特点分析

对抽取的 20 条样本，分析每类标签的特点：

| 标签类型 | 核心特征 | 示例关键词/模式 |
|---------|---------|----------------|
| Entailment | hypothesis 是 premise 的具体化/重述 | "a dog"→"a brown dog" |
| Contradiction | hypothesis 与 premise 明确冲突 | "室内" vs "室外" |
| Neutral | hypothesis 无法从 premise 推导 | 主观判断/推测性描述 |

**验证方法**：撰写分析报告（300-500 字），至少列举每类的 3 个具体例子佐证。

---

## 第三阶段：任务二 — Baseline 方法实现

### 步骤 3.1：实现 CLIP 方法

#### 步骤 3.1.1：加载 CLIP 模型

使用 OpenCLIP 或 Hugging Face 的 `openai/clip-vit-base-patch32` 模型。

**验证方法**：加载模型后，用一张图和一段文本测试相似度计算，输出概率分布。

---

#### 步骤 3.1.2：设计 CLIP 预测逻辑

**方案**：构建三元分类 prompt：
- "An image showing {hypothesis}"
- 测量图像与假设文本的相似度

**备选方案**：计算 premise、hypothesis 与图像的相似度差值。

**验证方法**：对 20 条样本运行预测，输出预测标签与真实标签的对照表。

---

#### 步骤 3.1.3：评估 CLIP 性能

在验证集（约 5000 条）上计算：
- Accuracy（整体准确率）
- Per-class Precision / Recall / F1
- 混淆矩阵

**验证方法**：
1. 运行完整验证集，保存预测结果
2. 打印指标数值，确认计算逻辑正确
3. 保存混淆矩阵可视化图

---

### 步骤 3.2：实现 LLM 方法（BLIP 或 LLaVA）

#### 步骤 3.2.1：加载多模态大模型

选择其一：

- **BLIP**：使用 `Salesforce/blip-vqa-vqa` 或 `blip-itm` 模块
- **LLaVA**：使用 `llava-hf/llava-1.5-7b-hf`

**验证方法**：加载模型后，用示例图像和文本测试推理输出，确认模型正常工作。

---

#### 步骤 3.2.2：设计 Prompt 模板

设计结构化 Prompt：

```
Given the image description and the following statement, determine if the statement is supported by the image.

Description: {premise}
Statement: {hypothesis}

Options:
A) Entailment - The statement is supported by the image
B) Contradiction - The statement contradicts the image
C) Neutral - The image neither supports nor contradicts the statement

Answer with ONLY the letter (A, B, or C) followed by a brief explanation.
```

**验证方法**：用同一张图和同一段文本测试 3 次，确认输出稳定性。

---

#### 步骤 3.2.3：实现批量推理

编写批量推理脚本，处理验证集全部样本。

**注意**：
- 设置合理的 batch_size（如 4-8）
- 添加进度条和日志
- 保存每条样本的模型输出（包含推理过程和最终答案）

**验证方法**：
1. 检查输出文件是否完整（应有 5000 条记录）
2. 抽查 10 条样本的模型输出，验证格式正确

---

#### 步骤 3.2.4：解析模型输出

编写后处理脚本：
1. 从模型输出中提取字母标签（A/B/C）
2. 处理解析失败的情况（回退策略：默认 Neutral 或重新推理）
3. 统计解析成功率

**验证方法**：
1. 对 5000 条输出运行解析脚本
2. 检查解析失败率（目标 < 5%）
3. 人工抽查解析结果的准确性

---

#### 步骤 3.2.5：评估 LLM 性能

在验证集上计算与 CLIP 相同的指标：
- Accuracy
- Per-class Precision / Recall / F1
- 混淆矩阵

**验证方法**：输出格式与步骤 3.1.3 完全一致，便于后续对比。

---

## 第四阶段：任务三 — 跨模态一致性检测模块设计

### 步骤 4.1：设计模块架构

设计"文本-图像一致性检测模块"，包含以下组件：

1. **图像理解组件**：提取图像关键实体和关系
2. **文本理解组件**：提取文本关键实体和关系
3. **对齐与比较组件**：判断一致性程度
4. **输出组件**：输出判断结果与置信度

**验证方法**：绘制模块架构图（可用 Mermaid 或手绘拍照），确保组件关系清晰。

---

### 步骤 4.2：实现链式推理（Chain-of-Thought）

设计多步推理 Prompt：

```
Step 1: Identify the key entities and actions in the image.
Step 2: Identify the key entities and actions in the statement.
Step 3: Compare Step 1 and Step 2. Are there conflicts?
Step 4: Can the statement be derived from the image?
Final Answer: Entailment / Contradiction / Neutral
```

**验证方法**：用 5 条样本测试，对比单步推理 vs 链式推理的效果差异。

---

### 步骤 4.3：实现结构化输出

要求模型输出 JSON 格式：

```json
{
  "image_entities": ["entity1", "entity2"],
  "statement_entities": ["entity1", "entity3"],
  "conflict_detected": true,
  "reason": "entity2 and entity3 are different",
  "label": "Contradiction"
}
```

**验证方法**：验证输出 JSON 的字段完整性（不允许缺失关键字段）。

---

### 步骤 4.4：集成 LLM Judge

使用 GPT-4 或同类模型作为裁判：
1. 输入：图像描述 + 假设文本 + 模型预测
2. 输出：裁判认为的正确标签
3. 用于提升最终准确率或分析模型错误

**验证方法**：设计 10 条"边界案例"，人工标注正确标签，对比 LLM Judge 与人工判断的一致率。

---

## 第五阶段：任务四 — 实验对比

### 步骤 5.1：整理实验结果

汇总各方法的性能指标：

| 方法 | Accuracy | F1 (Entailment) | F1 (Contradiction) | F1 (Neutral) | 可解释性 |
|------|----------|-----------------|-------------------|--------------|----------|
| CLIP |  |  |  |  | 否 |
| BLIP/LLaVA |  |  |  |  | 是 |
| CoT + BLIP/LLaVA |  |  |  |  | 是 |
| LLM Judge |  |  |  |  | 是 |

**验证方法**：检查表格数值来源，逐一对应到实验日志。

---

### 步骤 5.2：可视化对比

生成对比图：
1. Accuracy 柱状图
2. Per-class F1 对比图
3. 混淆矩阵热力图（每个方法一张）

**验证方法**：图表中数据与步骤 5.1 表格完全一致。

---

### 步骤 5.3：定性对比分析

对比各方法在以下维度的表现：
- **准确性**：哪种方法最高
- **可解释性**：哪种方法输出最有逻辑
- **稳定性**：同一输入多次运行结果是否一致
- **速度**：推理耗时对比

**验证方法**：撰写分析报告（500-800 字），论点需有数据支撑。

---

## 第六阶段：任务五 — 错误分析

### 步骤 6.1：收集错误样本

从验证集预测结果中筛选：
- 预测错误的所有样本
- 按错误类型分类

**验证方法**：统计错误总数和各类占比。

---

### 步骤 6.2：分析"看错图像"错误（≥4 例）

**定义**：模型对图像内容的理解与实际不符，导致判断错误。

**要求**：每例包含：
1. 图像 + premise + hypothesis
2. 真实标签 vs 预测标签
3. 错误原因分析
4. 模型的推理过程（截取）

**验证方法**：确保至少 4 个例子，每个都有图像证据。

---

### 步骤 6.3：分析"过度推理"错误（≥4 例）

**定义**：模型基于图像添加了文本未提及的信息，超出图像内容范围。

**要求**：每例包含：
1. 图像 + premise + hypothesis
2. 真实标签 vs 预测标签
3. 模型添加了什么"过度"信息
4. 分析为什么会过度推理

**验证方法**：确保至少 4 个例子，证明模型确实添加了不存在的信息。

---

### 步骤 6.4：分析"幻觉"错误（≥4 例）

**定义**：模型生成的内容与图像和文本都无关，完全捏造。

**要求**：每例包含：
1. 图像 + premise + hypothesis
2. 真实标签 vs 预测标签
3. 模型的幻觉内容
4. 分析幻觉类型（实体幻觉/关系幻觉/属性幻觉）

**验证方法**：确保至少 4 个例子，证明模型生成了不存在的信息。

---

### 步骤 6.5：总结错误模式

归纳三类错误的共同特征：

| 错误类型 | 典型场景 | 根本原因 | 可能的改进方向 |
|---------|---------|---------|---------------|
| 看错图像 | 模糊图像、复杂场景 | 视觉编码器不足 | 更高分辨率输入 |
| 过度推理 | 需要保守判断时 | 推理策略过于激进 | 添加"不确定"选项 |
| 幻觉 | 罕见实体 | 训练数据偏差 | 检索增强 |

**验证方法**：确保每个改进方向都有对应的错误案例支撑。

---

## 第七阶段：实验报告撰写

### 步骤 7.1：撰写报告正文

报告结构：

```
1. 任务介绍
   - 跨模态蕴含的定义
   - SNLI-VE 数据集简介
   - 实验目标

2. 方法描述
   - CLIP 方法原理与实现
   - 多模态大模型方法原理与实现
   - 一致性检测模块设计

3. 实验设置
   - 环境配置
   - 模型版本
   - 超参数设置

4. 结果分析
   - 定量指标对比
   - 定性分析
   - 各方法优缺点

5. 错误分析
   - 三类错误的具体案例
   - 错误原因总结
   - 改进建议

6. 改进方法
   - 短期可实现方案
   - 长期研究方向的思考
```

**验证方法**：
1. 检查各章节是否完整
2. 检查图表编号与正文引用是否对应
3. 检查参考文献格式（如有）

---

### 步骤 7.2：整理提交文件

提交文件清单：

| 文件名 | 内容 | 格式 |
|--------|------|------|
| `code/` | 全部代码文件 | .py |
| `results/predictions.json` | 模型预测结果 | JSON |
| `results/metrics.csv` | 性能指标汇总 | CSV |
| `results/errors_analysis.json` | 错误分析详情 | JSON |
| `visualizations/` | 可视化图表 | PNG |
| `report.pdf` | 最终实验报告 | PDF |

**验证方法**：
1. 检查每个文件是否存在
2. 检查代码文件是否可以独立运行（不含缺失 import）
3. 检查报告页数是否符合要求（通常 8-15 页）

---

## 进度检查清单

完成每个阶段后，在对应方框打勾：

- [ ] **阶段一**：环境准备与数据集获取
- [ ] **阶段二**：任务一（数据集可视化）完成
- [ ] **阶段三**：任务二（Baseline 方法）完成
- [ ] **阶段四**：任务四（实验对比）完成
- [ ] **阶段五**：任务五（错误分析）完成
- [ ] **阶段六**：任务三（一致性检测模块）完成
- [ ] **阶段七**：实验报告撰写完成
- [ ] **最终**：所有文件整理完毕，代码可复现

---

**计划结束**
