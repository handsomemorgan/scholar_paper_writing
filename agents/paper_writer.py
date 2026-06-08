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

### 格式要求
- **严格按照格式模板的结构组织文章**
- 每个部分的比例要协调（引言不宜过长，正文要充实）
- 图表标注规范（如有需要，用文字描述图表内容和位置）
- 参考文献格式完全统一

## 特别注意

1. 如果课程要求中指定了具体的主题或方向，必须严格遵循
2. 如果格式模板中有"实验数据"等要求，但输入中没有数据，创造合理的示例数据
3. 最终输出必须是完整的论文，包含所有部分
4. 使用Markdown格式输出论文
5. 论文中的引用要和参考文献列表一一对应
6. 确保内容充实、论述深入，每个章节都要有足够的深度
7. 使用具体的数据、案例和文献对比来支撑论点

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

### Format Requirements
- **Strictly follow the format template structure**
- Balance the proportion of each section (introduction should not be too long, body should be substantial)
- Use proper notation for figures and tables (describe content and placement in text if needed)
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

## Output Format

Output the complete paper directly in Markdown format.
Do NOT add any explanatory text like "Here is the generated paper".
Start directly from the paper title."""


class PaperWriter:
    """论文撰写 Agent — 一次性整体生成"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def write(
        self,
        requirements: str,
        format_template: Dict[str, Any],
        keyword_result: Dict[str, Any],
        literature_list: list,
        analysis_result: Dict[str, Any],
        extra_instructions: str = "",
        target_word_count: int = 0,
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

        Returns:
            完整的论文文本（Markdown格式）
        """
        logger.info("Agent 6: 开始撰写论文...")

        user_message = self._build_writing_prompt(
            requirements,
            format_template,
            keyword_result,
            literature_list,
            analysis_result,
            extra_instructions,
            target_word_count,
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
    ) -> str:
        """
        生成英文版论文。基于与中文版相同的源材料，以中文论文为内容参考，
        撰写自然流畅的英文学术论文。

        Args:
            chinese_paper: 已生成的中文论文全文（作为内容一致性参考）
            requirements: 原始课程论文要求
            format_template: Agent 2的格式模板检索结果
            keyword_result: Agent 3的关键词提取结果
            literature_list: Agent 4的文献检索结果
            analysis_result: Agent 5的文献分析结果
            extra_instructions: 额外写作指令（可选）
            target_word_count: 目标字数（仅作参考，不强制）

        Returns:
            完整的英文论文文本（Markdown格式）
        """
        logger.info("Agent 6 (English): 开始撰写英文版论文...")

        user_message = self._build_english_writing_prompt(
            chinese_paper,
            requirements,
            format_template,
            keyword_result,
            literature_list,
            analysis_result,
            extra_instructions,
            target_word_count,
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
            # Fallback: 返回一个简单的英文框架
            return self._fallback_write_english(requirements, format_template)

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
    ) -> str:
        """构建英文论文写作prompt"""
        template = format_template.get("template_content", "")
        innovation = analysis_result.get("innovation_proposal", {})

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
4. Demonstrate independent thinking and analysis
5. Ensure reference list corresponds one-to-one with in-text citations
6. Output the paper directly without any explanatory text
7. Every section must have sufficient depth; avoid repetitive content
8. Use specific data, cases, and literature comparisons to support arguments
9. Write in natural, fluent academic English — this should read as an original English paper
10. Maintain content consistency with the Chinese version (same structure, arguments, data) while using natural English expression"""

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
    ) -> str:
        """构建完整的写作prompt"""
        template = format_template.get("template_content", "")
        innovation = analysis_result.get("innovation_proposal", {})

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

{extra_instructions}

## 写作要求
1. 严格按照格式模板的结构组织论文
2. 在合适的位置引用文献（用[1][2]标注）
3. 确保学术语言规范、逻辑严密
4. 体现独立的思考和分析
5. 参考文献列表与正文引用一一对应
6. 直接输出论文，不要添加任何说明文字
7. 每个章节都要有足够的深度，避免内容重复
8. 使用具体的数据、案例和文献对比来支撑论点"""

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
