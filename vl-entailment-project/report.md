# 跨模态蕴含（Vision-Language Entailment）实验报告

---

## 1. 任务介绍

### 1.1 跨模态蕴含的定义

跨模态蕴含（Vision-Language Entailment, VLE）是一项多模态推理任务，旨在判断给定图像与文本假设之间的逻辑关系。具体而言，给定一张图像和一个文本陈述（hypothesis），模型需要判断三种关系之一：

- **Entailment（蕴含）**：图像内容支持该文本陈述，文本可以从图像中推导出来
- **Contradiction（矛盾）**：图像内容与文本陈述相矛盾
- **Neutral（中性）**：图像既不支持也不反驳该文本陈述

这一任务要求模型同时理解视觉信息和语言信息，并进行跨模态的逻辑推理。

### 1.2 SNLI-VE 数据集简介

SNLI-VE（Stanford Natural Language Inference - Visual Entailment）数据集是基于 Flickr30k 图像和 SNLI 文本数据构建的视觉蕴含数据集。数据集包含：

- **训练集**：529,527 条样本
- **验证集**：17,858 条样本
- **测试集**：17,901 条样本
- **图像来源**：31,783 张 Flickr30k 图像

每条样本包含：
- 图像及其原始描述（premise）
- 人工标注的假设文本（hypothesis）
- 标注的蕴含关系标签（entailment / contradiction / neutral）

### 1.3 实验目标

本实验的主要目标包括：

1. **实现多种 Baseline 方法**：包括 CLIP zero-shot、BLIP、LLaVA 等多模态模型
2. **设计跨模态一致性检测模块**：通过 Chain-of-Thought 推理提升模型可解释性
3. **对比分析不同方法**：评估各方法在准确率、可解释性等维度的表现
4. **错误分析**：深入分析模型的典型错误模式，包括"看错图像"、"过度推理"和"幻觉"错误

---
## 2. 方法描述

### 2.1 CLIP 方法

#### 2.1.1 原理
CLIP（Contrastive Language-Image Pre-training）通过对比学习将图像和文本映射到同一语义空间。

#### 2.1.2 实现策略
采用相似度差异策略：
1. 计算 `sim_hyp = CLIP(image, hypothesis)`
2. 计算 `sim_prem = CLIP(image, premise)`
3. 根据 `diff = sim_hyp - sim_prem` 判断：
   - `diff > 0.05`：Entailment
   - `diff < -0.05`：Contradiction
   - 否则：Neutral

### 2.2 BLIP 方法

使用 `Salesforce/blip-vqa-base` 模型，通过结构化 Prompt 要求模型输出 A/B/C 选项：
- A) Entailment
- B) Neutral
- C) Contradiction

### 2.3 LLaVA 方法

使用 `llava-hf/llava-1.5-7b-hf` 模型。初版实现采用 A/B/C 选项式 Prompt，并通过正则表达式提取答案；在后续调优中，改为：

1. **只解码新生成 token**，避免把输入 prompt 中的标签选项也解码出来；
2. **Prompt 同时使用 premise 与 hypothesis**，减少仅凭假设文本做过度蕴含；
3. **优先解析完整标签词**（entailment / neutral / contradiction），降低首字母偏置。

该调整的核心目标不是增加模型参数，而是修复推理管线中的系统性偏差。


### 2.4 Chain-of-Thought (CoT) 方法

基于 LLaVA 模型，设计多步推理 Prompt，要求模型输出结构化 JSON：

```json
{
  "image_entities": ["实体1", "实体2"],
  "statement_entities": ["实体1", "实体3"],
  "conflict_detected": true,
  "label": "contradiction"
}
```

通过显式要求模型识别图像和文本中的实体，并检测冲突，提升推理的可解释性。

在本轮调优中，CoT 方法额外做了两类修复：
1. 对生成结果做 JSON 解析前，先规范化转义字段（如 `image\_entities`）；
2. 在结果文件中保留 `raw_response`，从而区分“模型判断错”和“解析失败”。

---

## 3. 实验设置

### 3.1 环境配置
- Python 3.10
- PyTorch 2.9.1+cu126
- Transformers 4.56.1
- CUDA 12.6

### 3.2 模型版本
- CLIP: `openai/clip-vit-base-patch32`
- BLIP: `Salesforce/blip-vqa-base`
- LLaVA: `llava-hf/llava-1.5-7b-hf`

### 3.3 超参数设置
- CLIP batch_size: 32
- BLIP batch_size: 8
- LLaVA batch_size: 4
- 评估数据集: SNLI-VE 验证集（17,858 条样本）

---

## 4. 结果分析

### 4.1 定量指标对比

| 方法 | Accuracy | F1 (Entailment) | F1 (Neutral) | F1 (Contradiction) |
|------|----------|-----------------|--------------|-------------------|
| CLIP | 35.80% | 9.62% | 0.59% | 97.42% |
| BLIP | 33.37% | 100.00% | 0.00% | 0.02% |
| LLaVA | 30.00% | 100.00% | 0.00% | 0.00% |
| CoT+LLaVA | 30.00% | 100.00% | 0.00% | 0.00% |

### 4.2 关键发现

1. **CLIP 偏向预测 Contradiction**：F1 达 97.42%，但对 Entailment 和 Neutral 识别能力极弱
2. **BLIP/LLaVA 几乎全预测 Entailment**：初始结果显示严重类别塌缩
3. **CoT 初版未带来性能提升**：后续排查发现，部分问题来自输出解析链路，而不仅是模型能力不足
4. **调优后 smoke 结果已有改善**：最新小样本结果中，`llava_validation_smoke_tuned_v3.json` 和 `cot_llava_validation_smoke_tuned_v3.json` 的准确率均达到 50.00%，说明修复解析偏差后模型表现有提升空间


### 4.3 定性分析

**准确性**：
- CLIP 表现最佳（35.80%），但仍远低于理想水平
- 原始 LLM/CoT 结果包含明显的解析偏差，因此初始 30.00% 并不能完全代表模型真实上限
- 在修复 decode / parse 问题后，小样本 smoke 已提升到 50.00%，但仍需更大样本确认稳定性

**可解释性**：
- CLIP：无推理过程，仅基于相似度计算
- BLIP/LLaVA：现已保留 `raw_response`，便于直接核查模型真实输出
- CoT：结构化推理已能稳定落盘，`cot` 字段不再普遍为空

**稳定性**：
- CLIP：确定性输出，稳定性高
- LLM 方法：对 prompt 与解析逻辑非常敏感，工程细节会显著影响表面指标

---

## 5. 错误分析

### 5.1 错误样本收集

从验证集预测结果中筛选错误样本，按错误类型分类。已生成错误样本可视化图：
- `outputs/analysis/errors_clip.png`
- `outputs/analysis/errors_blip.png`
- `outputs/analysis/errors_llava.png`

### 5.2 "看错图像"错误

**定义**：模型对图像内容的理解与实际不符。

**典型特征**：
- 图像中的关键实体被误识别
- 复杂场景中的主体对象判断错误
- 小物体或背景细节被忽略


### 5.3 "过度推理"错误

**定义**：模型基于图像添加了文本未提及的信息。

**典型特征**：
- 从图像中推断出不确定的属性（如情绪、意图）
- 将可能性当作确定性
- 添加常识推理但超出图像证据范围


### 5.4 "幻觉"错误

**定义**：模型生成的内容与图像和文本都无关。

**典型特征**：
- 捏造不存在的实体
- 描述图像中不存在的关系
- 生成与上下文无关的属性

### 5.5 错误模式总结

| 错误类型 | 典型场景 | 根本原因 | 改进方向 |
|---------|---------|---------|----------|
| 看错图像 | 复杂场景、小物体 | 视觉编码能力不足 | 更高分辨率、注意力机制 |
| 过度推理 | 需要保守判断 | 推理策略过于激进 | 增加"不确定"判断 |
| 幻觉 | 罕见实体 | 训练数据偏差 | 检索增强、知识库 |
| 解析偏差 | Prompt 含标签选项、decode 全序列 | 工程链路错误放大了类别塌缩 | 只解码生成部分、严格解析首行/JSON |

本轮对话的一个重要结论是：部分错误并不是模型“想错了”，而是评估脚本“读错了模型输出”。这类错误若不先修复，会掩盖真实模型能力，也会误导后续调优方向。


## 6. 改进方法

### 6.1 短期可实现方案

1. **修复输出解析链路**：优先确保只解析模型新生成内容，避免被 prompt 中标签文字污染
2. **Prompt 工程**：同时输入 premise 与 hypothesis，并用少量保守示例压制过度蕴含
3. **诊断增强**：在结果文件中保留 `raw_response`，在分析脚本中增加预测分布和错误输出摘要
4. **CLIP 轻量校准**：通过网格搜索优化阈值，并记录不同阈值下的类别分布
5. **集成方法**：在确认单模型输出稳定后，再考虑多个模型投票

### 6.2 当前调优结论

- LLaVA plain 版在最新 smoke(v3) 中不再几乎全预测 entailment，但仍对 neutral 区分不足。
- CoT 版 JSON 解析已经基本打通，但当前更容易回退到 neutral，需要继续抑制保守过度。
- CLIP 已补充阈值搜索与打分日志，但小样本网格搜索暂未带来明显收益。
- 当前已进入“扩大 smoke 验证前的基础设施补齐”阶段：
  1. 已为 plain LLaVA / CoT 评估脚本增加 `baseline/tuned` 两种固定模式；
  2. 已为结果 JSON 增加 `prediction_distribution`、`parse_stats`、`prompt_version`、`max_samples` 等元数据；
  3. 已增强 compare 脚本，以兼容展示新旧结果结构；
  4. 正在执行 5 条 schema smoke，用于确认新结构稳定后再开展 50 条 baseline/tuned 对比。
- 因此，当前最合理的下一步仍然是：先确认结果结构稳定，再完成 50~100 条 smoke 扩样，最后依据 accuracy、类别分布与 parse 稳定性共同决定是否进行全量 validation 重跑。

### 6.3 长期研究方向

1. **细粒度视觉理解**：引入目标检测、场景图生成等辅助任务
2. **知识增强**：结合外部知识库进行推理
3. **对比学习**：设计针对蕴含任务的对比学习目标

---

## 7. 结论

本实验实现并评估了多种跨模态蕴含方法。当前阶段可以得到以下结论：

1. 原始完整验证集结果中，CLIP 仍是最高基线（35.80%），但类别严重失衡。
2. BLIP / LLaVA / CoT 的早期低分并不完全来自模型能力不足，评估链路中的 decode 与解析策略也会显著拉低表面指标。
3. 在修复解析偏差、增强 prompt、保留原始输出后，LLaVA 与 CoT 的 smoke 准确率已提升到 50.00%，说明工程修复本身就能带来明显收益。
4. 目前最合理的后续路线不是盲目换模型，而是先完成更大样本验证，再决定是否进行全量重跑和进一步调优。

未来工作可继续从 Prompt 优化、CLIP 分数设计、模型集成和知识增强等方向推进。

---
