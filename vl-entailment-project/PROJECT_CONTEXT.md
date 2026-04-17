# PROJECT_CONTEXT.md

## 文档元信息（Document Metadata）
- 文档名称：Project Context Document
- 适用用途：跨会话复用 / 项目管理 / 后续 Prompt 注入 / 项目交接
- 当前状态：进行中
- 最后更新时间：2026-03-29
- 更新来源：当前对话上下文、项目代码、`progress.md`、`report.md`、现有预测结果文件
- 说明：若字段在当前上下文中未明确出现或无法合理推断，统一填写为“暂无信息”

---

## 1. 项目概要（Project Overview）
- 项目名称：跨模态蕴含（Vision-Language Entailment, VLE）课程大作业 / `vl-entailment-project`
- 项目背景：课程大作业，基于 SNLI-VE 数据集完成图像-文本蕴含关系判断实验，并形成实验分析与报告
- 目标与目的：实现并对比 CLIP zero-shot、BLIP/LLaVA prompt、Chain-of-Thought（CoT）等方法，生成预测结果、误差分析、对比分析和实验报告
- 要解决的问题：给定图像、premise 与 hypothesis，判断其关系属于 entailment / neutral / contradiction 三分类中的哪一类
- 整体愿景：形成完整实验闭环，包括数据集准备、模型评估、错误分析、调优验证、报告撰写与提交材料整理

---

## 2. 范围定义（Scope Definition）
- 当前范围：
  - SNLI-VE 数据下载与图像缓存
  - 数据集可视化与理解
  - CLIP / BLIP / LLaVA / CoT 方法实现与验证集评估
  - 结果对比分析与错误分析
  - LLM / CoT 输出解析修复与 prompt 调优
  - smoke 级别的小样本调优验证
  - 实验报告与项目进度文档更新
- 非本次范围：
  - 训练新模型
  - 大规模参数微调
  - 线上服务部署
  - 产品化前端/后端系统开发
- 约束条件：
  - 所有命令需在 `vl-entailment-project/` 目录下执行
  - 所有入口脚本统一使用 `python -m src.<module>`
  - 项目当前不是 git repository
  - 代码规范要求：函数参数和返回值有类型注解，注释/文档使用中文，代码标识符使用英文，行长不超过 120
  - 推理默认 GPU 优先：`cuda` 可用时优先使用
  - 当前调优策略以低成本工程修复与 prompt 调整为主，不引入训练流程

---

## 3. 关键实体与关系（Key Entities & Relationships）

### 3.1 核心实体
- SNLI-VE 数据集
- 图像（image）
- premise
- hypothesis
- gold_label
- pred_label
- CLIP Evaluator
- BLIP Evaluator
- LLaVA Evaluator
- CoT Evaluator
- 预测结果文件（`outputs/predictions/`）
- 分析结果文件（`outputs/analysis/`）
- 项目进度文档（`progress.md`）
- 实验报告（`report.md`）
- 项目上下文文档（`PROJECT_CONTEXT.md`）

### 3.2 实体职责
| 实体名称 | 职责说明 |
|---|---|
| SNLI-VE 数据集 | 提供 train / validation / test 标注与图像样本 |
| image | 多模态输入中的视觉部分 |
| premise | 图像对应的描述性前提信息 |
| hypothesis | 需要判断与图像/前提关系的假设文本 |
| gold_label | 数据集真实标签 |
| pred_label | 各方法生成的预测标签 |
| CLIP Evaluator | 基于图文相似度差值进行 zero-shot 三分类 |
| BLIP Evaluator | 通过 prompt 直接生成标签文本并解析 |
| LLaVA Evaluator | 通过多模态大模型生成标签文本并解析 |
| CoT Evaluator | 生成结构化 JSON 推理结果并输出最终标签 |
| 预测结果文件 | 缓存方法级推理结果，避免重复运行 |
| 分析结果文件 | 汇总 accuracy、per-class 指标、混淆矩阵与错误可视化 |
| progress.md | 记录项目阶段进展、结果摘要与下一步建议 |
| report.md | 记录实验背景、方法、结果、错误分析与改进方向 |
| PROJECT_CONTEXT.md | 沉淀跨会话复用的项目长期上下文 |

### 3.3 实体关系描述
SNLI-VE 数据集通过 `src/data/dataset.py` 被封装为可迭代样本，每条样本包含 image、premise、hypothesis 与 gold_label。各评估脚本读取样本并生成 pred_label 和原始输出，结果写入 `outputs/predictions/`。分析脚本进一步读取预测文件，生成对比分析、预测分布、混淆矩阵与错误样本图。`progress.md`、`report.md` 与 `PROJECT_CONTEXT.md` 用于沉淀项目过程、实验结论与跨会话上下文。

---

## 4. 功能模块拆解（Functional Decomposition）

### 4.1 模块列表
- 数据准备模块
- 数据加载模块
- 数据可视化模块
- CLIP 评估模块
- BLIP / LLaVA 评估模块
- CoT 推理模块
- 结果对比分析模块
- 错误分析模块
- 文档沉淀模块

### 4.2 模块详情

#### 数据准备模块
- 输入：SNLI-VE 原始标注来源、Flickr30k 图像来源
- 输出：本地标注文件与图像缓存
- 核心逻辑：下载并整理 train / validation / test 标注文件与图像资源，供后续评估与分析使用

#### 数据加载模块
- 输入：本地标注文件、图像目录
- 输出：标准化样本对象，至少包含 `image_id`、`image`、`premise`、`hypothesis`、`label`
- 核心逻辑：封装数据集读取流程，为各方法统一提供输入格式

#### 数据可视化模块
- 输入：数据样本
- 输出：样本可视化图片与标签观察结果
- 核心逻辑：随机采样并渲染图像、文本和标签，用于理解数据分布与任务形式

#### CLIP 评估模块
- 输入：image、premise、hypothesis、阈值参数
- 输出：预测标签、相似度分数、预测结果 JSON
- 核心逻辑：计算 `sim(image, hypothesis)` 与 `sim(image, premise)`，基于差值和阈值判定 entailment / neutral / contradiction，并支持阈值搜索与预测分布记录

#### BLIP / LLaVA 评估模块
- 输入：image、premise、hypothesis、prompt 模板
- 输出：预测标签、`raw_response`、预测结果 JSON
- 核心逻辑：构造 prompt，调用模型生成标签文本，只解析新生成 token，并通过更严格的标签提取逻辑获得最终标签

#### CoT 推理模块
- 输入：image、premise、hypothesis、结构化 JSON prompt
- 输出：预测标签、结构化 `cot` 字段、`raw_response`
- 核心逻辑：要求模型输出包含 `image_entities`、`statement_entities`、`conflict_detected`、`reasoning`、`label` 的 JSON；对转义字段进行兼容处理，并保留原始输出用于诊断

#### 结果对比分析模块
- 输入：多个方法的预测结果文件
- 输出：summary、混淆矩阵、预测分布
- 核心逻辑：汇总 accuracy、per-class 指标与预测标签分布，帮助判断是否出现单类塌缩

#### 错误分析模块
- 输入：预测结果文件与数据样本
- 输出：错误样本可视化与错误类型观察结果
- 核心逻辑：筛选误判样本，展示真实标签、预测标签、hypothesis 与模型原始输出摘要，用于区分模型判断错误和解析问题

#### 文档沉淀模块
- 输入：项目进展、实验结果、调优结论、当前待办
- 输出：`progress.md`、`report.md`、`PROJECT_CONTEXT.md`
- 核心逻辑：将实验与工程过程结构化沉淀，便于跨会话复用和后续工作衔接

### 4.3 典型用户场景
- 在课程实验阶段，先运行小样本 smoke 验证方法是否可用，再运行完整验证集评估
- 若结果异常塌缩，优先检查解析链路与 prompt，而不是立即更换模型
- 在生成预测文件后，运行 compare / errors 脚本做对比分析与错误分析
- 在阶段性工作完成后，及时更新 `progress.md`、`report.md` 与 `PROJECT_CONTEXT.md`

---

## 5. 技术方向与关键决策（Technical Direction & Decisions）
- 客户端：暂无信息
- 服务端：暂无信息
- 模型或算法层：
  - CLIP zero-shot 相似度差值分类
  - BLIP / LLaVA 基于 prompt 的生成式分类
  - CoT + LLaVA 结构化 JSON 推理
  - CLIP 阈值搜索与轻量校准
- 数据流与架构：
  - `src.data.download` / `src.data.download_images` 下载数据
  - `src.data.dataset` 统一样本读取
  - `src.methods.*_eval` 生成预测结果并保存到 `outputs/predictions/`
  - `src.analysis.compare` / `src.analysis.errors` 消费预测结果并生成分析产物
  - `progress.md` / `report.md` / `PROJECT_CONTEXT.md` 记录阶段结论与项目上下文
- 已做技术决策：
  - 使用 SNLI-VE 作为课程实验数据集
  - 采用 CLIP、BLIP、LLaVA、CoT 作为主要方法路线
  - 对 LLM / CoT 仅解码新生成 token，避免把输入 prompt 一并解码
  - prompt 从仅使用 hypothesis 调整为同时使用 premise + hypothesis
  - 优先解析完整标签词（entailment / neutral / contradiction），降低 A/B/C 首项偏置
  - CoT 输出兼容 `image\_entities` 这类转义字段，提升 JSON 解析稳定性
  - 结果文件中保留 `raw_response`，用于后续诊断和错误归因
  - 在 compare / errors 中增加预测分布与原始输出摘要
  - 先做 smoke 调优闭环，再决定是否进行全量 validation 重跑
- 可替代方案：
  - 继续优化 CLIP score 设计，而不只依赖当前单一阈值策略
  - 使用 prompt ensemble / 模板集成
  - 使用多模型投票或集成方法
  - 结合外部知识增强或检索增强

---

## 6. 交互、风格与输出约定（Interaction & Style Conventions）
- AI 输出风格：结构清晰、层级明确、工程化表达
- 表达规范：统一使用 Markdown；必要时使用列表、表格与伪代码式结构
- 格式要求：严谨、有序、模块化、可迁移、可长期维护
- 用户特殊偏好：
  - 希望基于当前对话沉淀完整上下文文档
  - 明确要求字段齐全，未知信息必须写“暂无信息”
  - 更偏好先完成错误分析、调优与文档更新，再决定下一步执行
  - 当前希望将项目状态整理成可跨会话复用的稳定文档

---

## 7. 当前进展总结（Current Status）

### 7.1 已确认事实
- 项目为跨模态蕴含（VLE）课程大作业，主目录为 `/root/智能信息实验大作业/vl-entailment-project`
- 数据集已下载完成，图像已缓存，共 31783 张 Flickr30k 图像
- 数据可视化、基础评估、对比分析、错误分析、实验报告初稿均已完成
- `progress.md` 已更新至包含第八阶段“模型调优与诊断修复”
- `report.md` 已补充本轮调优背景、解析偏差结论与最新 smoke 结果
- CLIP、BLIP/LLaVA、CoT 的基础脚本已存在于 `src/methods/`
- 原始完整验证集记录中的主要基线结果为：
  - CLIP：35.80%
  - BLIP：33.37%
  - LLaVA：30.00%
  - CoT + LLaVA：30.00%
- 初始 LLaVA / CoT 低分并非完全来自模型能力不足，存在明显的输出解析链路偏差
- 已完成的关键修复包括：
  - 只解码新生成 token
  - 严格化标签解析逻辑
  - prompt 同时使用 premise + hypothesis
  - CoT 兼容转义 JSON 字段
  - 结果文件保留 `raw_response`
  - 分析脚本支持预测分布与输出摘要
- 最新 smoke 结果（v3）为：
  - `outputs/predictions/llava_validation_smoke_tuned_v3.json`：50.00%
  - `outputs/predictions/cot_llava_validation_smoke_tuned_v3.json`：50.00%
- 从最新 smoke 样本观察：
  - LLaVA 已不再几乎全预测 entailment，但 neutral 区分仍弱
  - CoT 的结构化输出已可稳定落盘，但当前更偏向 neutral 回退
- CLIP 已增加阈值搜索、分数记录与预测分布统计，但当前小样本搜索尚未带来明显提升
- 项目当前不是 git repository
- `tests/` 下未发现已有测试用例

### 7.2 未解决问题
- tuned LLaVA / CoT 的 50.00% 结果仅来自小样本 smoke，尚未验证是否在更大样本或完整验证集上稳定成立
- CoT 当前仍存在“过度保守、偏向 neutral”的倾向
- LLaVA plain 版虽然缓解了单类塌缩，但对 neutral / contradiction 的边界仍不稳定
- CLIP 的阈值搜索暂未带来明显收益，neutral / contradiction 失衡仍是问题
- 尚未完成 tuned 版本的更大样本复核与全量 validation 重跑
- 提交文件整理仍在进行中

---

## 8. 后续计划与风险（Next Steps & Risks）

### 8.1 待讨论主题
- tuned LLaVA / CoT 是否先扩大到 50~100 条 smoke，再决定是否全量重跑
- 是否继续针对 CoT 的 neutral 偏置做进一步 prompt 调整
- 是否继续针对 CLIP 引入更合理的 score 设计或模板集成
- 是否在单模型稳定后引入多模型投票或集成策略
- 实验报告是否需要整理为最终提交版

### 8.2 潜在风险与不确定性
- 当前 50.00% 的提升样本量较小，存在偶然性风险
- 模型输出对 prompt 与解析逻辑高度敏感，工程改动可能显著改变表面指标
- CoT 虽已修复解析问题，但 reasoning 与 label 仍可能存在不一致现象
- CLIP 当前类别分布失衡问题尚未真正解决
- 若直接全量重跑而不先扩大 smoke，可能浪费计算资源且难以快速定位问题

### 8.3 推荐的后续初始化 Prompt
- 推荐 Prompt：

```markdown
请基于当前 `PROJECT_CONTEXT.md` 继续推进 VLE 项目，不要重复做已完成工作。优先执行以下顺序：
1. 检查 tuned LLaVA / CoT 的最新结果文件与预测分布；
2. 将 smoke 样本扩大到 50~100 条，验证 50% 提升是否稳定；
3. 若结果稳定，再决定是否重跑完整 validation；
4. 同步回填 `progress.md` 与 `report.md`；
5. 若 LLaVA/CoT 暂无明显收益，再继续优化 CLIP 的 score 设计或模板集成。

输出时请保持工程化、结构化表达，并明确区分：模型能力问题、解析链路问题、数据规模不确定性。
```

---

## 9. 可直接复用的项目快照（Quick Reuse Snapshot）
- 当前最重要结论：LLaVA / CoT 的早期低分被“decode 全序列 + prompt 标签误匹配”显著放大，修复解析链路后 smoke 指标已提升到 50.00%
- 当前最优先下一步：扩大 tuned smoke 样本，而不是立刻全量重跑
- 当前文档状态：`progress.md`、`report.md`、`PROJECT_CONTEXT.md` 均已更新到本轮对话上下文
- 当前工程判断：先验证调优稳定性，再决定继续调 prompt / CoT / CLIP
