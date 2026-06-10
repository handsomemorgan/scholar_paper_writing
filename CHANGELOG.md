# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v3.0.0] — 2026-06-10

### 🎯 重大升级：融合 Skills 系列设计哲学，双模式 + 论证质量革命

本次更新系统性地吸收了「流水线式论文写作 Skills 系列」的核心优势，在不改变 8-Agent 串行架构的前提下，为系统注入学术界通行的论证规范和评审标准。

---

### ✨ 新增功能

#### 双模式支持
- **教学版 (teaching)**：面向本科生课程论文，强调批判性思维、竞争性假设、六段对话式文献综述
- **申报版 (application)**：面向课题申报/基金申请，强调政策价值、匿名合规、因果识别设计
- CLI 新增 `--mode` / `-m` 参数，Web GUI 同步支持模式切换
- 默认 `teaching` 模式，确保向后兼容

#### 五句叙事结构 (Agent 5)
- `innovation_proposal.rationale` 字段强制执行五句叙事：
  1. 现实痛点 + 关键数据
  2. 制度悖论 / 理论张力（因果承接词咬合）
  3. 独特视角 + 核心概念锚定（"——即..."当场定义）
  4. 预期反常识发现（"不是...而是..."强制约束）
  5. 研究意义
- 新增 `concept_anchor_map` 和 `counter_intuitive_claim` 字段
- 新增 `quality_self_check` 质量自检

#### 竞争性假设体系 (Agent 5)
- **教学版**：3个真正竞争的假设（不同因果判断，非互补关系）
- **申报版**：1个主假设 + 2个备用假设（机制检验 + 稳健性检验，递进关系）
- 每个假设标注理论深度、数据可行性、与其他假设的关系

#### 六段对话式文献综述 (Agent 6)
- 文献综述/相关研究部分按六段结构撰写：
  1. 英文文献路径与核心主张
  2. 英文文献内部挑战与盲区
  3. 中文文献独特发现（≥40%占比要求）
  4. 双方文献的对话与沉默
  5. 用理论视角重读文献（发现三个盲区）
  6. 批判性综合（从盲区推导研究问题与假设）
- 模式和核心概念一致性提醒自动注入 prompt

#### 文献质量门槛与增强 (Agent 4)
- **质量指标计算**：中文文献占比、CSSCI/SSCI 估算占比、近5年文献占比
- **定向补充检索**：中文不足→中文定向；高质量不足→高引用检索；近期不足→年份过滤
- **引文链扩展**：基于 top-5 高引种子文献，向前（被引）+ 向后（参考文献）扩展
- **Mock 依赖降低**：TARGET_MIN 从 10 降至 5，质量增强在 Mock 之前执行
- 质量指标存储在 `last_quality_metrics` 实例属性，供 orchestrator 读取

#### 评审视角3分钟测试 (Agent 7)
- 模拟评审专家阅读过程：
  - 第1分钟：研究问题能否在30秒内理解？
  - 第2分钟：文献缺口真实性和创新点可记忆性？
  - 第3分钟：整体框架清晰度和方法可行性？
- 测试结果写入 `reviewer_3min_test` 字段

#### 论证质量检查 (Agent 7)
- **论证深度**：引用支撑论点比 (cited_claim_ratio) ≥ 0.5
- **概念一致性**：核心概念在全文中的使用一致性
- **反常识展开**：检查"不是...而是..."是否在讨论/结论部分展开

#### 匿名合规检查 (Agent 7)
- 申报版自动检测：身份暴露词（"笔者""我的研究""我校"等）
- 空话检测："填补了国内空白""首次研究"等
- 文献对手检测：创新点是否点名了具体文献

#### 自动修复增强 (Agent 7)
- 高优先级问题不仅记录，自动执行文本替换：
  - "笔者" → "本研究"
  - "我的研究" → "本研究"
  - "我校" → "所在单位"
- 评分 < 80 时自动附加改进建议到论文末尾

#### 模式特定写作指令 (Agent 6)
- 教学版：批判性思维、文献对话、逻辑链完整性
- 申报版：活页风格、政策价值、匿名合规、避免空话
- 中英文 Prompt 同步增强
- 创新方向中的反常识发现和核心概念自动注入写作 prompt

### 🛠 配置变更 (config/settings.yaml)

新增配置段：
- `mode`: 运行模式 (teaching / application)
- `quality_thresholds`: 文献质量门槛 (cssci_ratio, chinese_literature_ratio, recent_5year_ratio, min_high_quality, enable_citation_chain)
- `review_check`: 评审视角检查参数
- `argument_quality`: 论证质量约束开关

### 📁 修改文件

| 文件 | 变更 | 说明 |
|:---|:---|---|
| `config/settings.yaml` | +68行 | 新增4个配置段 |
| `orchestrator.py` | +108行 | mode参数贯穿全流程，质量指标集成 |
| `agents/literature_searcher.py` | +465行 | 质量评估、定向补充、引文扩展、Mock降级 |
| `agents/literature_analyzer.py` | +403行 | 五句叙事、竞争性假设、概念锚定、质量自检 |
| `agents/paper_writer.py` | +242行 | 双模式写作、六段对话式综述、模式指令 |
| `agents/format_checker.py` | +421行 | 评审测试、论证深度、概念一致性、匿名合规、自动修复 |
| `agents/figure_agent.py` | 新增 | Agent 8: 图表生成 (v2 已存在，首次纳入版本管理) |
| `utils/literature_sources.py` | 新增 | 多源文献库路由 (v2 已存在，首次纳入版本管理) |
| `docs/figures/` | 新增 | 系统运行数据配图 |
| `docs/底层原理解释_欧阳月粼.md` | 新增 | 系统技术详解文档 |

### 🔄 向后兼容

- 所有新增参数均有默认值 (`mode="teaching"`)
- 原有调用方式无需修改
- `pipeline_log` 新增字段为增量添加

---

## [v2.0.0] — 2026-06-08 (历史记录)

### 新增
- DeepSeek API 集成（LLM 多后端工厂模式）
- 多源文献库路由（arXiv + DOAJ + OpenAlex + SocArXiv + RePEc + Socolar）
- Agent 8 (FigureAgent)：文献图表提取 + 自主统计运算 + 9种图表类型
- Web GUI + SSE 流式进度推送
- 中英双版输出
- 字数自适应（短篇/中篇/长篇/超长篇）
- GitHub Pages 落地页

---

## [v1.0.0] — 2026-06-01 (历史记录)

### 初始版本
- 7-Agent 串行流水线架构
- 论文类型分类（5类）
- RAG 格式模板检索（精确匹配 + 向量语义检索）
- 三级回退链文献检索（arXiv → Semantic Scholar → 自主模拟）
- 文献分析与创新方向发现
- 一次性整体论文生成
- 双层格式校验（规则引擎 + LLM 语义校验）
