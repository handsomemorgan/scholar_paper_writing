"""
Agent 7: 格式校验Agent

检查生成的论文是否严格遵循了RAG检索到的格式模板。
这是质量的最后一道关卡——确保输出格式统一、规范。

如果发现格式偏差，自动修正或标注需要修正的部分。
"""

import logging
import re
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


FORMAT_CHECKER_SYSTEM_PROMPT = """你是一个学术论文格式审查专家。你的任务是检查论文是否符合格式模板的要求，并指出任何偏差。

## 检查项目

1. **结构完整性**：论文是否包含格式模板要求的所有部分？
2. **层级正确性**：标题层级（一级、二级、三级）是否符合模板？
3. **内容匹配度**：每个部分的内容是否与其标题匹配？
4. **引用规范性**：参考文献格式是否统一？正文引用是否与文献列表对应？
5. **排版要素**：摘要长度、关键词数量、图表标注等是否符合要求？

## 输出格式

请严格输出以下JSON：
{
    "overall_score": 85,
    "is_compliant": true,
    "check_items": [
        {
            "item": "结构完整性",
            "score": 90,
            "status": "pass",
            "comments": "所有必需部分均已包含"
        },
        {
            "item": "引用规范性",
            "score": 70,
            "status": "warning",
            "comments": "参考文献[3]无对应正文引用",
            "suggestion": "请检查参考文献[3]是否在正文中被引用"
        }
    ],
    "format_deviations": [
        {
            "section": "引言",
            "expected": "应包含研究背景、国内外现状、存在问题、本文工作四个子部分",
            "actual": "缺少'本文工作'子部分",
            "severity": "medium",
            "fix_suggestion": "在引言末尾添加本文工作与结构安排的说明"
        }
    ],
    "improvement_suggestions": [
        "建议在正文中增加对文献[2]的引用",
        "摘要建议控制在300字以内，当前350字"
    ]
}

只输出JSON。"""


class FormatChecker:
    """格式校验 Agent — 确保输出严格符合格式模板"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def check(
        self, paper: str, template_content: str, category_name: str
    ) -> Dict[str, Any]:
        """
        检查论文格式是否符合模板要求。

        Args:
            paper: Agent 6生成的论文全文
            template_content: RAG检索到的格式模板
            category_name: 论文类型名称

        Returns:
            格式检查报告
        """
        logger.info("Agent 7: 正在进行格式校验...")

        # 先做规则化的基础检查
        basic_issues = self._basic_format_check(paper, template_content)

        user_message = f"""请检查以下论文是否符合格式模板的要求。

## 论文类型
{category_name}

## 格式模板（标准）
{template_content}

## 待检查论文
{paper[:5000]}...

## 基础规则检查发现的问题
{self._format_basic_issues(basic_issues)}

请进行全面的格式审查并给出JSON报告。"""

        try:
            result = self.llm.chat_with_json_output(
                system_prompt=FORMAT_CHECKER_SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.2,
            )

            # 合并基础检查问题
            if basic_issues:
                for issue in basic_issues:
                    result.setdefault("format_deviations", []).append(issue)

            logger.info(
                f"Agent 7 完成: 总体评分 {result.get('overall_score', 'N/A')}, "
                f"合规: {result.get('is_compliant', 'N/A')}"
            )
            return result

        except Exception as e:
            logger.error(f"Agent 7 格式校验失败: {e}")
            return self._fallback_check(paper, template_content, basic_issues)

    def _basic_format_check(
        self, paper: str, template_content: str
    ) -> List[Dict[str, Any]]:
        """基于规则的基础格式检查"""
        issues = []

        # 从模板中提取预期的章节标题
        expected_sections = re.findall(r'###?\s+\d+\.\s*(.+?)(?:\n|$)', template_content)
        if not expected_sections:
            expected_sections = re.findall(r'^#{1,3}\s+(.+?)$', template_content, re.MULTILINE)

        # 检查各章节
        for section in expected_sections:
            section = section.strip().rstrip("：:")

            # 构建可能的匹配模式
            patterns = [
                rf'#+\s+\d+\.\s*{re.escape(section)}',
                rf'#+\s+{re.escape(section)}',
            ]
            found = False
            for pattern in patterns:
                if re.search(pattern, paper, re.IGNORECASE):
                    found = True
                    break

            if not found:
                issues.append({
                    "section": section,
                    "expected": f"应包含'{section}'章节",
                    "actual": "未找到对应章节或标题不匹配",
                    "severity": "medium",
                    "fix_suggestion": f"添加 '{section}' 章节",
                })

        # 检查摘要长度
        abstract_match = re.search(r'摘要.*?\n(.*?)(?=\n#|\n##|\Z)', paper, re.DOTALL)
        if abstract_match:
            abstract_text = abstract_match.group(1).strip()
            if len(abstract_text) > 500:
                issues.append({
                    "section": "摘要",
                    "expected": "摘要200-400字",
                    "actual": f"摘要约{len(abstract_text)}字",
                    "severity": "low",
                    "fix_suggestion": "精简摘要至400字以内",
                })

        # 检查参考文献数量
        ref_section = re.search(r'(?:参考文献|References).*?\n(.*?)(?=\n#|\Z)', paper, re.DOTALL)
        if ref_section:
            ref_count = len(re.findall(r'\[\d+\]', ref_section.group(1)))
            if ref_count < 8:
                issues.append({
                    "section": "参考文献",
                    "expected": "至少10篇参考文献",
                    "actual": f"检测到约{ref_count}篇参考文献",
                    "severity": "high",
                    "fix_suggestion": "补充参考文献至至少10篇",
                })

        return issues

    def _format_basic_issues(self, issues: List[Dict[str, Any]]) -> str:
        if not issues:
            return "无基础规则问题"
        lines = []
        for i, issue in enumerate(issues, 1):
            lines.append(
                f"{i}. [{issue['severity']}] {issue.get('section', '')}: "
                f"{issue.get('expected', '')} → {issue.get('actual', '')}"
            )
        return "\n".join(lines)

    def _fallback_check(
        self, paper: str, template_content: str, basic_issues: list
    ) -> Dict[str, Any]:
        """格式检查fallback"""
        score = max(50, 100 - len(basic_issues) * 10)
        return {
            "overall_score": score,
            "is_compliant": len(basic_issues) == 0,
            "check_items": [
                {
                    "item": "结构完整性",
                    "score": score,
                    "status": "pass" if len(basic_issues) == 0 else "warning",
                    "comments": f"基础规则检查发现{len(basic_issues)}个问题",
                }
            ],
            "format_deviations": basic_issues,
            "improvement_suggestions": [],
        }

    def auto_fix(
        self, paper: str, check_report: Dict[str, Any]
    ) -> str:
        """
        根据检查报告自动修正格式问题。

        Args:
            paper: 原始论文
            check_report: Agent 7的格式检查报告

        Returns:
            修正后的论文
        """
        fixed_paper = paper
        deviations = check_report.get("format_deviations", [])

        for deviation in deviations:
            severity = deviation.get("severity", "low")
            if severity == "high":
                logger.warning(
                    f"需要手动修正: {deviation.get('section', '')} - "
                    f"{deviation.get('fix_suggestion', '')}"
                )

        return fixed_paper
