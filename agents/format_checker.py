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


FORMAT_CHECKER_SYSTEM_PROMPT = """你是一个学术论文格式审查专家。你的任务是从论文格式和论证质量层面进行全面检查。

## 检查项目

### 格式层面
1. **结构完整性**：论文是否包含格式模板要求的所有部分？
2. **层级正确性**：标题层级（一级、二级、三级）是否符合模板？
3. **内容匹配度**：每个部分的内容是否与其标题匹配？
4. **引用规范性**：参考文献格式是否统一？正文引用是否与文献列表对应？
5. **排版要素**：摘要长度、关键词数量、图表标注等是否符合要求？

### 论证质量层面（新增）
6. **论证深度**：每个章节是否有充分的论据和文献支撑？引用支撑论点比例是否足够？
7. **概念一致性**：核心概念在全文中是否使用一致？有无同义替换导致的歧义？
8. **反常识发现展开**：创新方向中的"不是...而是..."是否在讨论/结论部分得到充分展开？
9. **文献对话性**：文献综述是批评性对话还是简单罗列？

### 评审视角测试（新增）
10. **3分钟可读性**：评审专家在3分钟内能否抓住核心信息？
  - 第1分钟：能否在30秒内理解研究问题？
  - 第2分钟：文献缺口的真实性是否可感知？创新点是否可记忆？
  - 第3分钟：整体框架是否清晰？方法是否可行？

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
    "argument_quality": {
        "argument_depth_score": 75,
        "cited_claim_ratio": 0.6,
        "concept_consistency_score": 80,
        "counter_intuitive_developed": true,
        "literature_dialogic_score": 70
    },
    "reviewer_3min_test": {
        "minute_1_pass": true,
        "minute_2_pass": true,
        "minute_3_pass": true,
        "overall_pass": true,
        "comments": "评审视角测试通过"
    },
    "improvement_suggestions": [
        {
            "priority": "high",
            "section": "文献综述",
            "issue": "文献罗列多于对话",
            "suggestion": "增加'X主张...但Y质疑...'的批评性衔接"
        }
    ]
}

只输出JSON。"""


class FormatChecker:
    """格式校验 Agent — 确保输出严格符合格式模板

    v2 增强（吸收 Skills 系列设计）：
      - 评审视角3分钟测试
      - 论证深度检查（引用支撑论点比例）
      - 核心概念一致性检查
      - 反常识发现展开检查
      - 增强的 auto_fix（具体改写建议）
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        self.last_mode = "teaching"
        self.last_improvement_report = {}

    def check(
        self,
        paper: str,
        template_content: str,
        category_name: str,
        mode: str = "teaching",
    ) -> Dict[str, Any]:
        """
        检查论文格式是否符合模板要求。

        Args:
            paper: Agent 6生成的论文全文
            template_content: RAG检索到的格式模板
            category_name: 论文类型名称
            mode: "teaching"（教学版）或 "application"（申报版）

        Returns:
            格式检查报告（含论证质量和评审视角测试）
        """
        self.last_mode = mode
        logger.info(f"Agent 7: 正在进行格式校验... (模式: {mode})")

        # 先做规则化的基础检查
        basic_issues = self._basic_format_check(paper, template_content)

        # 论证深度检查（新增）
        argument_depth = self._check_argument_depth(paper)

        # 概念一致性检查（新增）
        concept_consistency = self._check_concept_consistency(paper)

        # 反常识发现检查（新增）
        counter_intuitive_check = self._check_counter_intuitive(paper)

        # 评审视角3分钟测试（新增）
        reviewer_test = self._reviewer_3min_test(paper, mode)

        # 模式特定的额外检查
        mode_specific_issues = self._mode_specific_check(paper, mode)

        user_message = f"""请检查以下论文是否符合格式模板的要求。

## 论文类型
{category_name}

## 运行模式
{mode}

## 格式模板（标准）
{template_content}

## 待检查论文
{paper[:5000]}...

## 基础规则检查发现的问题
{self._format_basic_issues(basic_issues + mode_specific_issues)}

## 论证质量预检结果
- 引用支撑论点比: {argument_depth.get('cited_claim_ratio', 'N/A')}
- 概念一致性: {'通过' if concept_consistency.get('is_consistent', True) else '需关注'}
- 反常识展开: {'是' if counter_intuitive_check.get('is_developed', False) else '否'}

## 评审3分钟测试预检
- 第1分钟（研究问题）: {'通过' if reviewer_test.get('minute_1_pass', True) else '需改进'}
- 第2分钟（文献缺口+创新）: {'通过' if reviewer_test.get('minute_2_pass', True) else '需改进'}
- 第3分钟（框架+可行性）: {'通过' if reviewer_test.get('minute_3_pass', True) else '需改进'}

请进行全面的格式审查并给出JSON报告（含argument_quality和reviewer_3min_test字段）。"""

        try:
            result = self.llm.chat_with_json_output(
                system_prompt=FORMAT_CHECKER_SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.2,
            )

            # 合并基础检查问题
            all_issues = basic_issues + mode_specific_issues
            if all_issues:
                for issue in all_issues:
                    result.setdefault("format_deviations", []).append(issue)

            # 合并论证质量预检结果
            if "argument_quality" not in result:
                result["argument_quality"] = {}
            result["argument_quality"].update({
                "cited_claim_ratio": argument_depth.get("cited_claim_ratio", 0),
                "concept_consistency_score": concept_consistency.get("score", 80),
                "counter_intuitive_developed": counter_intuitive_check.get("is_developed", False),
            })

            # 合并评审测试结果
            if "reviewer_3min_test" not in result:
                result["reviewer_3min_test"] = reviewer_test

            # 生成改进报告
            self.last_improvement_report = self._generate_improvement_report(result)

            logger.info(
                f"Agent 7 完成: 总体评分 {result.get('overall_score', 'N/A')}, "
                f"合规: {result.get('is_compliant', 'N/A')}, "
                f"评审测试: {'通过' if reviewer_test.get('overall_pass', True) else '需改进'}"
            )
            return result

        except Exception as e:
            logger.error(f"Agent 7 格式校验失败: {e}")
            return self._fallback_check(
                paper, template_content,
                basic_issues + mode_specific_issues,
                argument_depth, concept_consistency,
                counter_intuitive_check, reviewer_test,
            )

    # ================================================================
    # 新增检查方法
    # ================================================================

    def _check_argument_depth(self, paper: str) -> Dict[str, Any]:
        """检查论证深度：引用支撑论点 vs 无引用断言的比例"""
        # 统计正文引用标记 [N]
        citations = len(re.findall(r'\[\d+\]', paper))
        # 统计段落数（近似论点数）
        paragraphs = [p for p in paper.split("\n\n") if len(p.strip()) > 50]
        para_count = len(paragraphs)
        # 引用支撑论点比
        cited_claim_ratio = round(citations / max(para_count, 1), 3)

        depth_ok = cited_claim_ratio >= 0.5
        if not depth_ok:
            logger.warning(
                f"论证深度不足: 引用/段落比 = {cited_claim_ratio:.2f} "
                f"(建议 ≥ 0.5)"
            )

        return {
            "citation_count": citations,
            "paragraph_count": para_count,
            "cited_claim_ratio": cited_claim_ratio,
            "depth_adequate": depth_ok,
        }

    def _check_concept_consistency(self, paper: str) -> Dict[str, Any]:
        """检查核心概念在全文中的一致性（简化版：检查关键术语的重复模式）"""
        # 提取可能的理论概念（引号中的术语、破折号后的解释）
        concepts = re.findall(r'[「「]([^」」]+)[」」]', paper)
        concepts += re.findall(r'\*\*([^*]+)\*\*', paper)

        # 去重
        unique_concepts = list(set(concepts))[:10]

        # 检查每个概念的出现模式
        consistency_issues = []
        for concept in unique_concepts[:5]:
            occurrences = len(re.findall(re.escape(concept), paper))
            if occurrences < 2:
                consistency_issues.append(
                    f"概念'{concept}'仅出现{occurrences}次，可能未被充分使用"
                )

        return {
            "detected_concepts": unique_concepts[:5],
            "consistency_issues": consistency_issues,
            "is_consistent": len(consistency_issues) == 0,
            "score": max(60, 100 - len(consistency_issues) * 10),
        }

    def _check_counter_intuitive(self, paper: str) -> Dict[str, Any]:
        """检查反常识发现是否在论文中得到展开"""
        has_not_but = "不是" in paper and "而是" in paper
        # 检查是否在讨论/结论部分（通常在后半部分）
        paper_half = len(paper) // 2
        latter_half = paper[paper_half:]
        developed_in_discussion = "不是" in latter_half and "而是" in latter_half

        return {
            "has_counter_intuitive_marker": has_not_but,
            "developed_in_discussion": developed_in_discussion,
            "is_developed": has_not_but and developed_in_discussion,
        }

    def _reviewer_3min_test(
        self, paper: str, mode: str
    ) -> Dict[str, Any]:
        """评审视角3分钟测试"""
        # 提取关键部分
        lines = paper.split("\n")
        first_300_chars = paper[:300]
        abstract = ""
        intro_start = paper.find("引言") if "引言" in paper else paper.find("Introduction")
        if intro_start >= 0:
            abstract = paper[:intro_start]

        # 分钟1：研究问题能否快速理解？
        minute_1_pass = (
            len(first_300_chars) > 100
            and ("?" in first_300_chars or "问题" in first_300_chars
                 or "研究" in first_300_chars)
        )

        # 分钟2：文献缺口和创新点
        has_gap_marker = any(
            kw in paper[:len(paper)//2]
            for kw in ["然而", "但是", "不足", "缺乏", "空白", "忽视",
                       "however", "gap", "lack", "overlook"]
        )
        has_innovation = any(
            kw in paper
            for kw in ["创新", "新颖", "不同", "不是", "novel", "不同于",
                       "contribution", "本文提出", "本研究"]
        )
        minute_2_pass = has_gap_marker and has_innovation

        # 分钟3：框架清晰 + 方法可行
        has_framework = any(
            kw in paper
            for kw in ["框架", "结构", "章节", "framework", "structure",
                       "本文安排", "组织如下"]
        )
        has_method = any(
            kw in paper
            for kw in ["方法", "数据", "样本", "实验", "调查", "访谈",
                       "method", "data", "sample", "experiment", "survey"]
        )
        minute_3_pass = has_framework and has_method

        overall = minute_1_pass and minute_2_pass and minute_3_pass

        comments = []
        if not minute_1_pass:
            comments.append("研究问题不够清晰——评审在30秒内无法抓住核心问题")
        if not minute_2_pass:
            comments.append("文献缺口或创新点不够突出——评审难以记住本文的独特贡献")
        if not minute_3_pass:
            comments.append("研究框架或方法不够具体——评审对可行性存疑")

        return {
            "minute_1_pass": minute_1_pass,
            "minute_2_pass": minute_2_pass,
            "minute_3_pass": minute_3_pass,
            "overall_pass": overall,
            "comments": "; ".join(comments) if comments else "评审视角测试通过",
        }

    def _mode_specific_check(
        self, paper: str, mode: str
    ) -> List[Dict[str, Any]]:
        """模式特定的格式检查"""
        issues = []

        if mode == "application":
            # 检查身份暴露
            identity_markers = ["笔者", "我的研究", "我的导师", "我在", "我们学校",
                              "我校", "本院", "本系"]
            for marker in identity_markers:
                if marker in paper:
                    issues.append({
                        "section": "全文",
                        "expected": "活页中不得出现身份暴露信息",
                        "actual": f"发现身份暴露词'{marker}'",
                        "severity": "high",
                        "fix_suggestion": f"将'{marker}'替换为'本研究'或删除",
                    })

            # 检查空话
            empty_claims = ["填补了国内空白", "首次研究", "具有重要理论意义和现实意义"]
            for claim in empty_claims:
                if claim in paper:
                    issues.append({
                        "section": "全文",
                        "expected": "避免空洞表述",
                        "actual": f"发现空话'{claim}'",
                        "severity": "medium",
                        "fix_suggestion": f"将'{claim}'替换为具体的文献对话表述",
                    })

            # 检查是否有具体文献对手
            if "不同于" not in paper and "不像" not in paper and "unlike" not in paper.lower():
                issues.append({
                    "section": "创新之处",
                    "expected": "创新点应点名具体文献对手",
                    "actual": "未发现'不同于XX(2020)'类的文献对手表述",
                    "severity": "medium",
                    "fix_suggestion": "在每个创新点后添加具体文献对比",
                })

        return issues

    def _generate_improvement_report(
        self, check_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """基于检查结果生成优先级排序的改进建议"""
        suggestions = []

        # 格式偏差 → 高优先级修正
        for dev in check_result.get("format_deviations", []):
            if dev.get("severity") == "high":
                suggestions.append({
                    "priority": "high",
                    "category": "format",
                    "section": dev.get("section", ""),
                    "action": dev.get("fix_suggestion", ""),
                })

        # 论证质量 → 中优先级
        aq = check_result.get("argument_quality", {})
        if aq.get("cited_claim_ratio", 1) < 0.5:
            suggestions.append({
                "priority": "medium",
                "category": "argument_depth",
                "section": "全文",
                "action": "增加文献引用支撑论点。当前引用支撑比为"
                         f"{aq.get('cited_claim_ratio', 0):.0%}，建议≥50%",
            })

        if not aq.get("counter_intuitive_developed", True):
            suggestions.append({
                "priority": "medium",
                "category": "counter_intuitive",
                "section": "讨论/结论",
                "action": "在讨论部分明确展开创新方向中的反常识发现",
            })

        # 评审测试失败 → 高优先级
        rt = check_result.get("reviewer_3min_test", {})
        if not rt.get("minute_1_pass", True):
            suggestions.append({
                "priority": "high",
                "category": "clarity",
                "section": "摘要/引言",
                "action": "在摘要或引言的第一段明确写出核心研究问题",
            })
        if not rt.get("minute_2_pass", True):
            suggestions.append({
                "priority": "high",
                "category": "innovation_visibility",
                "section": "文献综述/创新之处",
                "action": "用'不同于XX(2020)的主张，本文发现...'模式突出创新点",
            })

        return {
            "total_suggestions": len(suggestions),
            "high_priority": len([s for s in suggestions if s["priority"] == "high"]),
            "suggestions": suggestions,
        }

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

        # 检查参考文献数量（最低要求10篇）
        ref_section = re.search(
            r'(?:参考文献|References).*?\n(.*?)(?=\n#|\Z)', paper, re.DOTALL
        )
        if ref_section:
            ref_count = len(re.findall(r'\[\d+\]', ref_section.group(1)))
            if ref_count < 10:
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
        self, paper: str, template_content: str, basic_issues: list,
        argument_depth: Dict[str, Any] = None,
        concept_consistency: Dict[str, Any] = None,
        counter_intuitive_check: Dict[str, Any] = None,
        reviewer_test: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """格式检查fallback（含新增字段）"""
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
            "argument_quality": argument_depth or {"cited_claim_ratio": 0},
            "reviewer_3min_test": reviewer_test or {"overall_pass": True, "comments": ""},
            "improvement_suggestions": [],
        }

    def auto_fix(
        self, paper: str, check_report: Dict[str, Any]
    ) -> str:
        """
        根据检查报告自动修正格式问题（增强版）。

        不仅记录问题，对特定类型问题提供具体的文本替换建议。
        """
        fixed_paper = paper
        deviations = check_report.get("format_deviations", [])
        suggestions = check_report.get("improvement_suggestions", [])

        # 收集改进报告中的建议
        if self.last_improvement_report:
            suggestions.extend(
                self.last_improvement_report.get("suggestions", [])
            )

        fix_applied = 0
        for deviation in deviations:
            severity = deviation.get("severity", "low")
            section = deviation.get("section", "")
            fix = deviation.get("fix_suggestion", "")

            if severity == "high":
                logger.warning(
                    f"⚠️ 高优先级修正建议 [{section}]: {fix}"
                )

                # 身份暴露词的自动替换（申报版）
                if "笔者" in fixed_paper:
                    fixed_paper = fixed_paper.replace("笔者", "本研究")
                    fix_applied += 1
                if "我的研究" in fixed_paper:
                    fixed_paper = fixed_paper.replace("我的研究", "本研究")
                    fix_applied += 1
                if "我的导师" in fixed_paper:
                    fixed_paper = fixed_paper.replace("我的导师", "")
                    fix_applied += 1
                if "我校" in fixed_paper:
                    fixed_paper = fixed_paper.replace("我校", "所在单位")
                    fix_applied += 1

            elif severity == "medium":
                logger.info(f"  中优先级建议 [{section}]: {fix}")

        # 应用改进报告中的操作建议
        improvement_notes = []
        for sug in suggestions:
            improvement_notes.append(
                f"[{sug.get('priority', 'medium')}][{sug.get('category', 'general')}] "
                f"{sug.get('section', '')}: {sug.get('action', '')}"
            )

        logger.info(
            f"auto_fix 完成: 自动修正 {fix_applied} 处，"
            f"生成 {len(improvement_notes)} 条改进建议"
        )

        # 将改进建议附加到论文末尾（作为注释）
        if improvement_notes and check_report.get("overall_score", 100) < 80:
            notes_section = "\n\n---\n\n## 格式改进建议（由Agent 7自动生成）\n\n"
            notes_section += "\n".join(f"- {note}" for note in improvement_notes)
            fixed_paper += notes_section

        return fixed_paper
