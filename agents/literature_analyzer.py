"""
Agent 5: 文献分析Agent

分析检索到的文献，提取核心观点，发现研究空白，
并提出创新的研究方向。

这是整个系统中"创新性"的核心——不是简单拼接文献，
而是通过分析文献之间的关系，找到一个有价值的新角度。
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


LITERATURE_ANALYSIS_SYSTEM_PROMPT = """你是一位资深的学术文献分析专家。你的任务是对检索到的文献进行深度分析，提炼核心观点，发现研究空白，并提出一个创新的研究方向。

## 分析框架

### 1. 文献脉络梳理
- 识别该领域的研究主流和分支
- 标注里程碑式的经典文献
- 绘制研究演进路径：早期基础 → 中期发展 → 前沿热点

### 2. 核心观点提取
- 从每篇文献中提取1-2个核心观点
- 标注不同文献之间观点的共识与分歧
- 识别被广泛引用的"共识性结论"

### 3. 研究空白发现
- 现有研究未覆盖的领域
- 方法论上的不足
- 应用场景的局限
- 跨学科切入的可能

### 4. 创新方向提议
- 结合研究空白和现有基础
- 提出一个具体可行的创新方向
- 论证其价值和可行性
- 给出研究方法建议

## 输出格式

请严格输出以下JSON格式：
{
    "research_landscape": {
        "mainstream": "主流研究方向描述",
        "branches": ["分支1", "分支2"],
        "milestone_works": [
            {"title": "经典文献标题", "contribution": "核心贡献"}
        ]
    },
    "core_findings": [
        {
            "theme": "主题",
            "consensus": "学术界共识",
            "debates": "争议与分歧",
            "supporting_literature": ["文献标题1", "文献标题2"]
        }
    ],
    "research_gaps": [
        {
            "gap": "研究空白描述",
            "significance": "重要性评估 (high/medium/low)",
            "feasibility": "研究可行性分析"
        }
    ],
    "innovation_proposal": {
        "title": "建议的创新研究题目",
        "research_question": "核心研究问题",
        "novelty": "创新点阐述",
        "methodology": "建议的研究方法",
        "expected_contribution": "预期学术贡献",
        "rationale": "为什么这是一个有价值的方向（300字）"
    },
    "literature_matrix": [
        {
            "title": "文献标题",
            "core_argument": "核心论点（50字）",
            "methodology_used": "使用的研究方法",
            "relevance_to_proposal": "与本研究的关联"
        }
    ]
}

只输出JSON，不要包含任何其他文字。"""


class LiteratureAnalyzer:
    """文献分析 Agent — 提炼观点、发现空白、提出创新方向"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def analyze(
        self,
        literature_list: List[Dict[str, Any]],
        requirements: str,
        keyword_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        分析文献并提出创新方向。

        Args:
            literature_list: Agent 4检索到的文献列表
            requirements: 原始课程论文要求
            keyword_result: Agent 3的关键词提取结果

        Returns:
            包含文献分析矩阵和创新方向的字典
        """
        logger.info(f"Agent 5: 正在分析 {len(literature_list)} 篇文献...")

        # 构建分析输入
        literature_summary = self._format_literature_for_analysis(literature_list)
        keywords_formatted = self._format_keywords(keyword_result)

        user_message = f"""请分析以下文献并发现研究空白与创新方向。

## 课程论文要求
{requirements}

## 关键词
{keywords_formatted}

## 检索到的文献（共{len(literature_list)}篇）
{literature_summary}

请按照分析框架进行系统分析，特别注意：
1. 找出这些文献之间的内在联系
2. 发现它们共同忽略的研究角度
3. 提出一个本科生能力范围内可以完成的创新研究方向
4. 确保创新方向与课程论文要求紧密相关
"""

        try:
            result = self.llm.chat_with_json_output(
                system_prompt=LITERATURE_ANALYSIS_SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.6,
            )

            logger.info(
                f"Agent 5 完成: 发现 {len(result.get('research_gaps', []))} 个研究空白, "
                f"提出了创新方向 '{result.get('innovation_proposal', {}).get('title', 'N/A')}'"
            )
            return result

        except Exception as e:
            logger.error(f"Agent 5 文献分析失败: {e}")
            return self._fallback_analyze(literature_list, requirements)

    def _format_literature_for_analysis(
        self, literature_list: List[Dict[str, Any]]
    ) -> str:
        """将文献列表格式化为分析输入"""
        lines = []
        for i, lit in enumerate(literature_list, 1):
            authors = ", ".join(lit.get("authors", [])[:3])
            if len(lit.get("authors", [])) > 3:
                authors += " et al."

            lines.append(
                f"[{i}] {lit.get('title', 'N/A')}\n"
                f"    作者: {authors}\n"
                f"    年份: {lit.get('year', 'N/A')} | "
                f"期刊: {lit.get('journal', 'N/A')} | "
                f"引用: {lit.get('citation_count', 0)}\n"
                f"    摘要: {lit.get('abstract', 'N/A')[:200]}\n"
            )

        return "\n".join(lines)

    def _format_keywords(self, keyword_result: Dict[str, Any]) -> str:
        """格式化关键词"""
        primary = [
            kw.get("zh", "") for kw in keyword_result.get("primary_keywords", [])
        ]
        return "、".join(primary)

    def _fallback_analyze(
        self,
        literature_list: List[Dict[str, Any]],
        requirements: str,
    ) -> Dict[str, Any]:
        """简化的文献分析fallback"""
        titles = [lit.get("title", "") for lit in literature_list[:5]]

        return {
            "research_landscape": {
                "mainstream": "根据已有文献推断的主流研究方向",
                "branches": [],
                "milestone_works": [
                    {"title": t, "contribution": "该领域的重要研究"}
                    for t in titles[:3]
                ],
            },
            "core_findings": [
                {
                    "theme": "从文献中识别的核心主题",
                    "consensus": "文献中体现的共识",
                    "debates": "文献中存在的争议",
                    "supporting_literature": titles[:3],
                }
            ],
            "research_gaps": [
                {
                    "gap": "现有研究尚未充分探索的领域",
                    "significance": "medium",
                    "feasibility": "在本科生能力范围内可完成",
                }
            ],
            "innovation_proposal": {
                "title": "基于文献综述的探索性研究",
                "research_question": "结合课程要求与文献空白的核心问题",
                "novelty": "从新的角度审视现有问题",
                "methodology": "文献分析法 + 案例研究法",
                "expected_contribution": "为理解该问题提供新的视角",
                "rationale": "已有研究多集中在传统视角，缺乏新维度的探索。本研究将尝试填补这一空白。",
            },
            "literature_matrix": [
                {
                    "title": lit.get("title", ""),
                    "core_argument": lit.get("abstract", "")[:50],
                    "methodology_used": "文献研究",
                    "relevance_to_proposal": "提供理论基础",
                }
                for lit in literature_list[:8]
            ],
        }
