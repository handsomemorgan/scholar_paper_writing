"""
Agent 6: 论文撰写Agent

根据RAG检索的格式模板、文献分析结果和创新方向，
撰写一篇完整的、高质量的论文或报告。

全部采用一次性整体生成，避免分章节导致的重复内容。
"""

import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)


PAPER_WRITER_SYSTEM_PROMPT = """你是一位经验丰富的学术论文写作者。你擅长撰写各类学术论文——理科论文、文科论文、工科论文、实验报告和调研报告。

## 你的任务

根据以下输入，撰写一篇完整的论文/报告：
1. **格式模板**：严格遵循提供的格式模板的每一个部分
2. **课程要求**：确保内容完全响应课程论文要求
3. **文献基础**：基于提供的文献分析，使用真实文献支撑论点
4. **创新方向**：围绕建议的创新方向展开论述

## 写作原则

### 内容质量
- **学术规范性**：使用规范的学术语言，避免口语化表达
- **逻辑严密性**：每个论点都要有充分的论据支撑
- **文献引用**：在合适的地方引用文献，标注[1][2]等序号
- **原创性**：虽然是基于已有文献，但要体现独立的思考和整合
- **批判性思维**：不盲目接受已有观点，要有自己的分析和判断

### 文献综述结构：六段对话式（重要）
文献综述/相关研究部分应按以下六段结构组织：
1. **英文文献路径与核心主张**：该领域的国际研究主流观点
2. **英文文献内部挑战与盲区**：同一传统内的批评与未解决的问题
3. **中文文献独特发现**：中文学界的独特贡献（不可缺席，占比不低于40%）
4. **双方文献的对话与沉默**：中英文文献之间的交汇点与彼此忽略之处
5. **用理论视角重读文献**：用本文的理论透镜重新审视已有研究，发现三个盲区
6. **批判性综合**：从盲区推导出本文的研究问题与假设

### 格式要求
- **严格按照格式模板的结构组织文章**
- 每个部分的比例要协调（引言不宜过长，正文要充实）
- 图表标注规范：论文中已自动生成了配图，请用 `![图X](figures/filename.png)` 引用它们，并在图下方添加图注（如"**图X：标题描述**"）
- 参考文献格式完全统一

## 特别注意

1. 如果课程要求中指定了具体的主题或方向，必须严格遵循
2. 如果格式模板中有"实验数据"等要求，但输入中没有数据，创造合理的示例数据
3. 最终输出必须是完整的论文，包含所有部分
4. 使用Markdown格式输出论文
5. 论文中的引用要和参考文献列表一一对应
6. 确保内容充实、论述深入，每个章节都要有足够的深度
7. 使用具体的数据、案例和文献对比来支撑论点
8. **如果额外指令中包含了"论文配图清单"，请务必在论文正文中适当位置引用这些配图**。每张配图都应该在相关章节出现，并用文字分析图表内容
9. **创新方向中的"反常识发现"（"不是...而是..."）必须在论文的讨论/结论部分明确展开论述**
10. **创新方向中锚定的核心概念必须在论文正文中保持一致的定义和使用**

## 输出格式

直接输出完整的论文，用Markdown格式。
不要添加"这是生成的论文"之类的说明文字。
直接从论文标题开始。"""


ENGLISH_PAPER_WRITER_SYSTEM_PROMPT = """You are an experienced academic paper writer. You are skilled at writing various types of academic papers — science papers, engineering papers, lab reports, and research reports.

## Your Task

Based on the following inputs, write a complete academic paper in **English**:
1. **Format Template**: Strictly follow every section of the provided format template
2. **Requirements**: Ensure the content fully addresses the course paper requirements
3. **Literature Foundation**: Use the provided literature analysis and real references to support arguments
4. **Innovation Direction**: Develop the discussion around the suggested innovation direction
5. **Chinese Reference Paper**: A Chinese version of this paper has already been generated; use it as a content reference to ensure consistency in structure, arguments, and data. However, write in natural academic English — do NOT translate word-for-word. Adapt expressions, idioms, and sentence structures to sound like a native English academic paper.

## Writing Principles

### Content Quality
- **Academic Standards**: Use formal academic English, avoid colloquial expressions
- **Logical Rigor**: Every argument must be well-supported by evidence
- **Literature Citations**: Cite references at appropriate places using [1][2] notation
- **Originality**: While based on existing literature, demonstrate independent thinking and synthesis
- **Critical Thinking**: Do not blindly accept existing views; provide your own analysis and judgment

### Literature Review Structure: Six-Paragraph Dialogic Model
The literature review / related work section should follow this six-paragraph structure:
1. **English Literature: Core Claims** — Mainstream findings and dominant narratives in international research
2. **English Literature: Internal Challenges** — Critiques and unresolved issues within the same traditions
3. **Chinese Literature: Unique Contributions** — Distinctive findings from Chinese-language scholarship (must not be absent; ≥40% of cited works)
4. **Cross-Tradition Dialogue & Silences** — Points of convergence and mutual blind spots between English and Chinese scholarship
5. **Re-reading Literature Through Your Theoretical Lens** — Use the paper's chosen theoretical perspective to identify three specific gaps
6. **Critical Synthesis** — Derive the paper's research question and hypotheses from the identified gaps

### Format Requirements
- **Strictly follow the format template structure**
- Balance the proportion of each section (introduction should not be too long, body should be substantial)
- Use proper notation for figures: the paper has auto-generated figures; reference them with `![Figure X](figures/filename.png)` and add captions below (e.g., "**Figure X: Title description**")
- Ensure consistent reference formatting throughout

## Important Notes

1. Follow the specified topic or direction strictly if given in the requirements
2. If the format template requires "experimental data" but none is provided, create reasonable example data
3. The final output must be a complete paper with all sections
4. Output the paper in Markdown format
5. All citations in the text must correspond one-to-one with the reference list
6. Ensure substantial content and in-depth discussion in every section
7. Use specific data, cases, and literature comparisons to support arguments
8. Write in natural, fluent academic English — this should read like an original English paper, not a translation
9. **If the extra instructions contain a "Figure Manifest", you MUST reference those figures in the appropriate sections of the paper.** Each figure should appear in a relevant section with textual analysis of its content
10. **The innovation proposal's "counter-intuitive discovery" ("not X, but rather Y") must be explicitly developed in the Discussion/Conclusion section**
11. **Core concepts anchored in the innovation proposal must be used consistently throughout the paper**

## Output Format

Output the complete paper directly in Markdown format.
Do NOT add any explanatory text like "Here is the generated paper".
Start directly from the paper title."""


class PaperWriter:
    """论文撰写 Agent — 一次性整体生成

    v2 增强（吸收 Skills 系列设计）：
      - 双模式支持：教学版（批判性思维展示） / 申报版（政策价值+匿名合规）
      - 六段对话式文献综述结构
      - 反常识发现展开 + 核心概念一致性
      - 质量提醒清单
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        self.last_mode = "teaching"

    def write(
        self,
        requirements: str,
        format_template: Dict[str, Any],
        keyword_result: Dict[str, Any],
        literature_list: list,
        analysis_result: Dict[str, Any],
        extra_instructions: str = "",
        target_word_count: int = 0,
        mode: str = "teaching",
    ) -> str:
        """
        一次性撰写完整论文。

        Args:
            requirements: 原始课程论文要求
            format_template: Agent 2的格式模板检索结果
            keyword_result: Agent 3的关键词提取结果
            literature_list: Agent 4的文献检索结果
            analysis_result: Agent 5的文献分析结果
            extra_instructions: 额外写作指令（可选）
            target_word_count: 目标字数（仅作参考，不强制）
            mode: "teaching"（教学版）或 "application"（申报版）

        Returns:
            完整的论文文本（Markdown格式）
        """
        self.last_mode = mode
        logger.info(f"Agent 6: 开始撰写论文... (模式: {mode})")

        user_message = self._build_writing_prompt(
            requirements,
            format_template,
            keyword_result,
            literature_list,
            analysis_result,
            extra_instructions,
            target_word_count,
            mode,
        )

        try:
            paper = self.llm.chat(
                system_prompt=PAPER_WRITER_SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.7,
                max_tokens=16000,
            )

            paper = self._post_process(paper)
            logger.info(f"Agent 6 完成: 生成论文约 {len(paper)} 字符")
            return paper

        except Exception as e:
            logger.error(f"Agent 6 论文撰写失败: {e}")
            return self._fallback_write(requirements, format_template)

    def write_english(
        self,
        chinese_paper: str,
        requirements: str,
        format_template: Dict[str, Any],
        keyword_result: Dict[str, Any],
        literature_list: list,
        analysis_result: Dict[str, Any],
        extra_instructions: str = "",
        target_word_count: int = 0,
        mode: str = "teaching",
    ) -> str:
        """
        生成英文版论文。

        Args:
            chinese_paper: 已生成的中文论文全文
            requirements: 原始课程论文要求
            format_template: Agent 2的格式模板检索结果
            keyword_result: Agent 3的关键词提取结果
            literature_list: Agent 4的文献检索结果
            analysis_result: Agent 5的文献分析结果
            extra_instructions: 额外写作指令（可选）
            target_word_count: 目标字数（仅作参考，不强制）
            mode: "teaching"（教学版）或 "application"（申报版）

        Returns:
            完整的英文论文文本（Markdown格式）
        """
        self.last_mode = mode
        logger.info(f"Agent 6 (English): 开始撰写英文版论文... (模式: {mode})")

        user_message = self._build_english_writing_prompt(
            chinese_paper,
            requirements,
            format_template,
            keyword_result,
            literature_list,
            analysis_result,
            extra_instructions,
            target_word_count,
            mode,
        )

        try:
            paper_en = self.llm.chat(
                system_prompt=ENGLISH_PAPER_WRITER_SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.7,
                max_tokens=16000,
            )

            paper_en = self._post_process(paper_en)
            logger.info(f"Agent 6 (English) 完成: 生成英文论文约 {len(paper_en)} 字符")
            return paper_en

        except Exception as e:
            logger.error(f"Agent 6 (English) 英文论文撰写失败: {e}")
            return self._fallback_write_english(requirements, format_template)

    def _build_mode_specific_instruction(self, mode: str) -> str:
        """构建模式特定的写作指令"""
        if mode == "application":
            return """## 申报版写作要求（application mode）

1. **活页风格**：使用简洁有力的学术语言，每段有明确的论点-论据-文献支撑
2. **政策价值**：在"研究意义""结论"部分明确阐述研究对政策制定的启示
3. **匿名合规**：不使用"笔者""我的研究""本人"等第一人称，不出现学校/院系名称
4. **方法可行性**：在方法部分具体说明数据来源、样本策略和分析方法
5. **创新点突出**：每个创新点必须点名与具体文献的对话（如"与XX（2020）不同，本研究..."）
6. **避免空话**：不使用"填补了空白""首次研究""具有重要意义"等空洞表述
7. **字数控制**：控制在活页标准字数范围内，重要部分不压缩，次要部分简洁"""
        else:
            return """## 教学版写作要求（teaching mode）

1. **批判性思维**：不盲目接受已有观点，对每篇核心文献进行评价性引用
2. **文献对话**：展示不同文献之间的共识与分歧，而非简单罗列
3. **论证深度**：每个章节都要有"论点→论据→文献支撑→小结"的完整逻辑链
4. **概念一致性**：核心概念在全文中保持统一，避免同义替换导致的歧义
5. **反常识展开**：在讨论部分明确展开创新方向中的"不是...而是..."发现
6. **参考文献一一对应**：正文中引用的每一篇文献都必须在参考文献列表中"""

    def _add_quality_reminders(self, mode: str) -> str:
        """添加质量提醒清单到prompt"""
        reminders = [
            "\n## 质量自检清单（输出前自查）\n",
        ]
        if mode == "teaching":
            reminders.extend([
                "- [ ] 文献综述是否展示了中英文文献的对话与张力？",
                "- [ ] 每个章节是否有充分的论据和文献支撑？",
                "- [ ] 创新方向中的反常识发现是否在讨论部分展开？",
                "- [ ] 核心概念在全文中是否定义一致？",
                "- [ ] 参考文献列表是否与正文引用一一对应？",
                "- [ ] 全文是否无bullet list（除本清单外）？",
            ])
        else:
            reminders.extend([
                "- [ ] 是否避免了'笔者''我的研究'等身份暴露词？",
                "- [ ] 创新点是否点名了具体文献对手？",
                "- [ ] 是否避免了'填补空白''首次研究'等空话？",
                "- [ ] 方法部分是否具体说明了数据来源和样本策略？",
                "- [ ] 研究意义是否包含理论价值+政策价值双重表述？",
                "- [ ] 参考文献中近五年文献占比是否≥60%？",
            ])
        return "\n".join(reminders)

    def _build_english_writing_prompt(
        self,
        chinese_paper: str,
        requirements: str,
        format_template: Dict[str, Any],
        keyword_result: Dict[str, Any],
        literature_list: list,
        analysis_result: Dict[str, Any],
        extra_instructions: str,
        target_word_count: int = 0,
        mode: str = "teaching",
    ) -> str:
        """构建英文论文写作prompt（含模式特定指令）"""
        template = format_template.get("template_content", "")
        innovation = analysis_result.get("innovation_proposal", {})
        hypothesis_system = analysis_result.get("hypothesis_system", {})

        max_lit_display = min(len(literature_list), 30)
        lit_summary_lines = []
        for i, lit in enumerate(literature_list[:max_lit_display], 1):
            authors = ", ".join(lit.get("authors", [])[:3])
            if len(lit.get("authors", [])) > 3:
                authors += " et al."
            abstract_preview = lit.get("abstract", "N/A")[:200]
            lit_summary_lines.append(
                f"[{i}] {lit.get('title', 'N/A')} "
                f"({lit.get('year', 'N/A')}) - {authors} "
                f"- {lit.get('journal', 'N/A')}\n"
                f"    Abstract: {abstract_preview}"
            )
        lit_summary = "\n".join(lit_summary_lines)

        core_findings = analysis_result.get("core_findings", [])
        gaps = analysis_result.get("research_gaps", [])

        word_count_hint = ""
        if target_word_count > 0:
            word_count_hint = f"\n## Target Word Count (Reference)\nApproximately {target_word_count:,} words. Ensure substantial content."

        # 假设体系
        hypothesis_text = self._format_hypothesis_system(hypothesis_system, mode)

        # 模式指令
        if mode == "application":
            mode_note = (
                "\n## Application Mode Notes\n"
                "1. Use grant-proposal academic style — concise, authoritative, policy-relevant\n"
                "2. Avoid first-person language ('I found', 'my research')\n"
                "3. Each innovation point should name a specific literature opponent\n"
                "4. Include both theoretical value and policy implications in the significance section\n"
                "5. Avoid empty claims like 'fills a gap' — use 'unlike X (2020), this study...'\n"
            )
        else:
            mode_note = (
                "\n## Teaching Mode Notes\n"
                "1. Demonstrate critical thinking — engage with literature dialogically, not just list them\n"
                "2. Use the six-paragraph dialogic structure for the literature review section\n"
                "3. Develop the innovation's counter-intuitive finding in the discussion section\n"
                "4. Maintain consistent use of core concepts throughout\n"
            )

        # 核心概念提醒
        counter_intuitive = innovation.get("counter_intuitive_claim", "")
        concept_map = innovation.get("concept_anchor_map", {})
        concept_reminder = ""
        if concept_map:
            concept_reminder = "\n## Core Concept Anchors (use consistently)\n"
            for concept_name, definition in concept_map.items():
                concept_reminder += f"- **{concept_name}**: {definition}\n"
        if counter_intuitive:
            concept_reminder += f"\n**Counter-intuitive finding (must develop in Discussion/Conclusion)**: {counter_intuitive}\n"

        return f"""Please write a complete academic paper in English based on all the following information.

## Course Paper Requirements
{requirements}

## Format Template (MUST follow this format strictly)
{template}

## Paper Category
{format_template.get('category_name', 'Academic Paper')}
{word_count_hint}

## Search Keywords
{self._format_keywords(keyword_result)}

## Reference List (Total {len(literature_list)} papers — cite appropriately in the paper)
{lit_summary}

## Core Findings from Literature Analysis
{self._format_core_findings(core_findings)}

## Research Gaps
{self._format_gaps(gaps)}

## Suggested Innovation Direction
- Title: {innovation.get('title', '')}
- Research Question: {innovation.get('research_question', '')}
- Novelty: {innovation.get('novelty', '')}
- Suggested Methodology: {innovation.get('methodology', '')}
- Expected Contribution: {innovation.get('expected_contribution', '')}
- Rationale: {innovation.get('rationale', '')}
{concept_reminder}

## Hypothesis System
{hypothesis_text}
{mode_note}

{extra_instructions}

## Chinese Reference Paper (for content consistency)
The following is the Chinese version of this paper. Use it as a content reference to maintain consistency in structure, key arguments, and data. However, write in natural academic English — adapt expressions and sentence structures to sound native. Do NOT translate word-for-word.

---
{chinese_paper[:8000]}
---

## Writing Requirements
1. Strictly follow the format template structure
2. Cite references at appropriate places using [1][2] notation
3. Ensure formal academic English and logical rigor
4. **Literature review should follow the six-paragraph dialogic structure**
5. Demonstrate independent thinking and analysis
6. Ensure reference list corresponds one-to-one with in-text citations
7. Output the paper directly without any explanatory text
8. Every section must have sufficient depth; avoid repetitive content
9. Use specific data, cases, and literature comparisons to support arguments
10. Write in natural, fluent academic English — this should read as an original English paper
11. Maintain content consistency with the Chinese version while using natural English expression"""

    def _fallback_write_english(
        self, requirements: str, format_template: Dict[str, Any]
    ) -> str:
        """English paper generation fallback"""
        template = format_template.get("template_content", "")
        return f"""# Paper

> Due to LLM call failure, this is a framework output. Please check API configuration and retry.

## Requirements
{requirements}

## Format Reference
{template[:500]}...

---

**Note**: This is a framework output. A complete English paper requires successful LLM invocation. Please check API key configuration and network connection.
"""

    def _build_writing_prompt(
        self,
        requirements: str,
        format_template: Dict[str, Any],
        keyword_result: Dict[str, Any],
        literature_list: list,
        analysis_result: Dict[str, Any],
        extra_instructions: str,
        target_word_count: int = 0,
        mode: str = "teaching",
    ) -> str:
        """构建完整的写作prompt（含模式特定指令和六段对话式结构）"""
        template = format_template.get("template_content", "")
        innovation = analysis_result.get("innovation_proposal", {})
        hypothesis_system = analysis_result.get("hypothesis_system", {})

        # 展示全部文献摘要
        max_lit_display = min(len(literature_list), 30)
        lit_summary_lines = []
        for i, lit in enumerate(literature_list[:max_lit_display], 1):
            authors = ", ".join(lit.get("authors", [])[:3])
            if len(lit.get("authors", [])) > 3:
                authors += " et al."
            abstract_preview = lit.get("abstract", "N/A")[:200]
            lit_summary_lines.append(
                f"[{i}] {lit.get('title', 'N/A')} "
                f"({lit.get('year', 'N/A')}) - {authors} "
                f"- {lit.get('journal', 'N/A')}\n"
                f"    Abstract: {abstract_preview}"
            )
        lit_summary = "\n".join(lit_summary_lines)

        core_findings = analysis_result.get("core_findings", [])
        gaps = analysis_result.get("research_gaps", [])

        word_count_hint = ""
        if target_word_count > 0:
            word_count_hint = f"\n## 目标字数（参考）\n约 {target_word_count:,} 字，请确保内容充实。"

        # 构建假设体系展示
        hypothesis_text = self._format_hypothesis_system(hypothesis_system, mode)

        # 模式特定指令
        mode_instruction = self._build_mode_specific_instruction(mode)
        quality_reminders = self._add_quality_reminders(mode)

        # 反常识发现和核心概念（用于一致性提醒）
        counter_intuitive = innovation.get("counter_intuitive_claim", "")
        concept_map = innovation.get("concept_anchor_map", {})
        concept_reminder = ""
        if concept_map:
            concept_reminder = "\n## 核心概念锚定（必须在论文中一致使用）\n"
            for concept_name, definition in concept_map.items():
                concept_reminder += f"- **{concept_name}**：{definition}\n"
        if counter_intuitive:
            concept_reminder += f"\n**反常识发现（必须在讨论/结论中展开）**：{counter_intuitive}\n"

        return f"""请根据以下全部信息，撰写一篇完整的论文。

## 课程论文要求
{requirements}

## 格式模板（必须严格遵循此格式）
{template}

## 论文类型
{format_template.get('category_name', '学术论文')}
{word_count_hint}

## 检索关键词
{self._format_keywords(keyword_result)}

## 参考文献列表（共 {len(literature_list)} 篇，请在论文中恰当引用）
{lit_summary}

## 文献分析核心发现
{self._format_core_findings(core_findings)}

## 研究空白
{self._format_gaps(gaps)}

## 建议的创新方向
- 题目：{innovation.get('title', '')}
- 研究问题：{innovation.get('research_question', '')}
- 创新点：{innovation.get('novelty', '')}
- 建议方法：{innovation.get('methodology', '')}
- 预期贡献：{innovation.get('expected_contribution', '')}
- 论证：{innovation.get('rationale', '')}
{concept_reminder}

## 假设体系
{hypothesis_text}

{mode_instruction}

{extra_instructions}

## 写作要求
1. 严格按照格式模板的结构组织论文
2. 在合适的位置引用文献（用[1][2]标注）
3. **文献综述部分按六段对话式结构撰写**（英文主张→内部挑战→中文独特发现→对话与沉默→理论重读→批判性综合）
4. 确保学术语言规范、逻辑严密
5. 体现独立的思考和分析
6. 参考文献列表与正文引用一一对应
7. 直接输出论文，不要添加任何说明文字
8. 每个章节都要有足够的深度，避免内容重复
9. 使用具体的数据、案例和文献对比来支撑论点
{quality_reminders}"""

    def _format_keywords(self, keyword_result: Dict[str, Any]) -> str:
        primary = [
            kw.get("zh", "")
            for kw in keyword_result.get("primary_keywords", [])
        ]
        return "、".join(primary)

    def _format_core_findings(self, findings: list) -> str:
        lines = []
        for f in findings[:10]:
            theme = f.get("theme", "")
            consensus = f.get("consensus", "")
            lines.append(f"- **{theme}**: {consensus}")
        return "\n".join(lines) if lines else "无"

    def _format_gaps(self, gaps: list) -> str:
        lines = []
        for g in gaps[:8]:
            lines.append(
                f"- {g.get('gap', '')} "
                f"(重要性: {g.get('significance', 'N/A')})"
            )
        return "\n".join(lines) if lines else "无"

    def _format_hypothesis_system(
        self, hypothesis_system: Dict[str, Any], mode: str
    ) -> str:
        """格式化假设体系为可读文本"""
        if not hypothesis_system:
            return "（无假设体系）"

        lines = []
        hs_mode = hypothesis_system.get("mode", mode)

        if hs_mode == "teaching" or "teaching_hypotheses" in hypothesis_system:
            hyps = hypothesis_system.get("teaching_hypotheses", [])
            if hyps:
                lines.append(f"**教学版竞争性假设（{len(hyps)}个）**：")
                for h in hyps:
                    lines.append(
                        f"- {h.get('id', '?')}: {h.get('statement', '')} "
                        f"[理论传统: {h.get('theoretical_tradition', 'N/A')}] "
                        f"[深度: {h.get('depth', '?')}] "
                        f"[可行性: {h.get('data_feasibility', '?')}]"
                    )
                    vs = h.get('vs_other_hypotheses', '')
                    if vs:
                        lines.append(f"  竞争关系: {vs}")
                lines.append("（论文中请选择其中一个假设深入论证）")

        if hs_mode == "application" or "application_hypotheses" in hypothesis_system:
            hyps = hypothesis_system.get("application_hypotheses", [])
            if hyps:
                lines.append(f"**申报版假设体系（递进关系）**：")
                for h in hyps:
                    lines.append(
                        f"- {h.get('id', '?')} [{h.get('type', '')}]: "
                        f"{h.get('statement', '')} "
                        f"[深度: {h.get('depth', '?')}] "
                        f"[可行性: {h.get('data_feasibility', '?')}]"
                    )

        return "\n".join(lines) if lines else "（无假设体系）"

    def _post_process(self, paper: str) -> str:
        """后处理生成的论文"""
        paper = re.sub(r'^.*?(?:以下是|这是|下面).*?论文.*?\n', '', paper)

        if not paper.strip().startswith("#"):
            lines = paper.strip().split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("#"):
                    paper = "\n".join(lines[i:])
                    break

        return paper.strip()

    def _fallback_write(
        self, requirements: str, format_template: Dict[str, Any]
    ) -> str:
        """简化的论文生成fallback"""
        template = format_template.get("template_content", "")
        return f"""# 论文

> 由于LLM调用失败，以下为框架性输出。请检查API配置后重试。

## 课程要求
{requirements}

## 格式参考
{template[:500]}...

---

**注意**: 这是一份框架性输出，完整的论文需要LLM成功调用后生成。请检查API密钥配置和网络连接。
"""
