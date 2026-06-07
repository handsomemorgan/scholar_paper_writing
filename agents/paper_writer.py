"""
Agent 6: 论文撰写Agent

根据RAG检索的格式模板、文献分析结果和创新方向，
撰写一篇完整的、高质量的论文或报告。

这是系统中最核心的Agent——它将所有前置分析结果转化为最终论文。
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

### 字数要求
- 摘要：200-300字
- 正文：根据课程要求，一般3000-8000字
- 参考文献：至少10篇

## 特别注意

1. 如果课程要求中指定了具体的主题或方向，必须严格遵循
2. 如果格式模板中有"实验数据"等要求，但输入中没有数据，创造合理的示例数据
3. 最终输出必须是完整的论文，包含所有部分
4. 使用Markdown格式输出论文
5. 论文中的引用要和参考文献列表一一对应

## 输出格式

直接输出完整的论文，用Markdown格式。
不要添加"这是生成的论文"之类的说明文字。
直接从论文标题开始。"""


class PaperWriter:
    """论文撰写 Agent — 将所有分析结果转化为完整论文"""

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
    ) -> str:
        """
        撰写完整论文。

        Args:
            requirements: 原始课程论文要求
            format_template: Agent 2的格式模板检索结果
            keyword_result: Agent 3的关键词提取结果
            literature_list: Agent 4的文献检索结果
            analysis_result: Agent 5的文献分析结果
            extra_instructions: 额外写作指令（可选）

        Returns:
            完整的论文文本（Markdown格式）
        """
        logger.info("Agent 6: 开始撰写论文...")

        # 构建写作prompt
        user_message = self._build_writing_prompt(
            requirements,
            format_template,
            keyword_result,
            literature_list,
            analysis_result,
            extra_instructions,
        )

        try:
            paper = self.llm.chat(
                system_prompt=PAPER_WRITER_SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.7,
                max_tokens=8000,  # 论文需要较多token
            )

            # 后处理：清理可能的格式问题
            paper = self._post_process(paper)

            logger.info(
                f"Agent 6 完成: 生成论文约 {len(paper)} 字符"
            )
            return paper

        except Exception as e:
            logger.error(f"Agent 6 论文撰写失败: {e}")
            return self._fallback_write(requirements, format_template)

    def _build_writing_prompt(
        self,
        requirements: str,
        format_template: Dict[str, Any],
        keyword_result: Dict[str, Any],
        literature_list: list,
        analysis_result: Dict[str, Any],
        extra_instructions: str,
    ) -> str:
        """构建完整的写作prompt"""

        # 格式模板
        template = format_template.get("template_content", "")

        # 创新方向
        innovation = analysis_result.get("innovation_proposal", {})

        # 文献摘要
        lit_summary_lines = []
        for i, lit in enumerate(literature_list[:15], 1):
            authors = ", ".join(lit.get("authors", [])[:3])
            if len(lit.get("authors", [])) > 3:
                authors += " et al."
            lit_summary_lines.append(
                f"[{i}] {lit.get('title', 'N/A')} "
                f"({lit.get('year', 'N/A')}) - {authors} "
                f"- {lit.get('journal', 'N/A')}"
            )
        lit_summary = "\n".join(lit_summary_lines)

        # 文献分析
        core_findings = analysis_result.get("core_findings", [])
        gaps = analysis_result.get("research_gaps", [])

        prompt = f"""请根据以下全部信息，撰写一篇完整的论文。

## 课程论文要求
{requirements}

## 格式模板（必须严格遵循此格式）
{template}

## 论文类型
{format_template.get('category_name', '学术论文')}

## 检索关键词
{self._format_keywords(keyword_result)}

## 参考文献列表（请在论文中恰当引用）
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
"""
        return prompt

    def _format_keywords(self, keyword_result: Dict[str, Any]) -> str:
        primary = [
            kw.get("zh", "") for kw in keyword_result.get("primary_keywords", [])
        ]
        return "、".join(primary)

    def _format_core_findings(self, findings: list) -> str:
        lines = []
        for f in findings[:5]:
            theme = f.get("theme", "")
            consensus = f.get("consensus", "")
            lines.append(f"- **{theme}**: {consensus}")
        return "\n".join(lines) if lines else "无"

    def _format_gaps(self, gaps: list) -> str:
        lines = []
        for g in gaps[:3]:
            lines.append(f"- {g.get('gap', '')} (重要性: {g.get('significance', 'N/A')})")
        return "\n".join(lines) if lines else "无"

    def _post_process(self, paper: str) -> str:
        """后处理生成的论文"""
        # 移除可能的开头说明文本
        paper = re.sub(r'^.*?(?:以下是|这是|下面).*?论文.*?\n', '', paper)

        # 确保以 # 标题开头
        if not paper.strip().startswith("#"):
            lines = paper.strip().split("\n")
            # 找到第一个标题
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
