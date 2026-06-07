"""
Agent 1: 论文类型分类器

根据输入的课程论文要求，自动分类论文类型。
分类结果将直接影响后续RAG格式模板的选择。
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


# 分类Prompt模板
CLASSIFIER_SYSTEM_PROMPT = """你是一个学术论文分类专家。你的任务是根据课程论文/报告的要求，将其准确分类到以下5种类别之一。

## 可选的类别

1. **science_paper (理科论文)**
   - 特征：涉及数学、物理、化学、生物等自然科学
   - 关键词：定理、证明、实验、数据、公式、自然规律、科学方法
   - 典型课程：高等数学、大学物理、有机化学、分子生物学

2. **liberal_arts_paper (文科论文)**
   - 特征：涉及文学、历史、哲学、社会学、教育学、法学等
   - 关键词：理论、思想、分析、论述、观点、社会、文化、价值
   - 典型课程：文学概论、中国历史、西方哲学、社会学原理

3. **engineering_paper (工科论文)**
   - 特征：涉及计算机、机械、电气、土木等应用技术
   - 关键词：系统、设计、实现、算法、架构、技术、开发、工程
   - 典型课程：数据结构、操作系统、机械设计、电路原理

4. **lab_report (实验报告)**
   - 特征：需要记录实验过程、数据、分析
   - 关键词：实验目的、实验器材、实验步骤、数据记录、误差分析
   - 典型课程：大学物理实验、化学实验、电子实验

5. **research_report (调研报告)**
   - 特征：对某个现象/行业/问题进行调查研究
   - 关键词：调查、问卷、访谈、现状、问题、对策、建议
   - 典型课程：社会调查方法、市场营销、思政课社会实践

## 输出格式

请严格输出以下JSON格式：
{
    "category_id": "science_paper",
    "category_name": "理科论文",
    "confidence": 0.85,
    "reasoning": "分类理由（简短说明为什么归入此类别）",
    "discipline": "具体学科（如：物理学、计算机科学等）",
    "suggested_keywords": ["关键词1", "关键词2", "关键词3"]
}

只输出JSON，不要包含任何其他文字。"""


class PaperClassifier:
    """论文类型分类器 Agent"""

    def __init__(self, llm_client):
        """
        Args:
            llm_client: LLMClient 实例
        """
        self.llm = llm_client

    def classify(self, requirements: str) -> Dict[str, Any]:
        """
        分类论文类型。

        Args:
            requirements: 课程论文/报告的要求文本

        Returns:
            分类结果字典，包含类别ID、名称、置信度等
        """
        logger.info("Agent 1: 正在分类论文类型...")

        user_message = f"请分类以下课程论文要求：\n\n{requirements}"

        try:
            result = self.llm.chat_with_json_output(
                system_prompt=CLASSIFIER_SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.1,  # 低温度确保分类稳定
            )

            # 验证必需的字段
            required_fields = ["category_id", "category_name", "confidence", "reasoning"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Classification result missing field: {field}")

            logger.info(
                f"Agent 1 完成: 分类为 '{result['category_name']}' "
                f"(置信度: {result['confidence']})"
            )
            return result

        except Exception as e:
            logger.error(f"Agent 1 分类失败: {e}")

            # 返回基于规则的fallback分类
            return self._fallback_classify(requirements)

    def _fallback_classify(self, requirements: str) -> Dict[str, Any]:
        """基于关键词的规则分类（LLM失败时的fallback）"""
        text = requirements.lower()

        # 关键词规则匹配
        rules = [
            ("实验目的", "实验器材", "实验报告", "lab_report", "实验报告"),
            ("调查", "问卷", "调研报告", "research_report", "调研报告"),
            ("系统设计", "算法", "实现", "engineering_paper", "工科论文"),
            ("公式推导", "定理", "自然界", "science_paper", "理科论文"),
            ("理论", "思想", "论述", "liberal_arts_paper", "文科论文"),
        ]

        for kw1, kw2, kw3, cat_id, cat_name in rules:
            if kw1 in text or kw2 in text or kw3 in text:
                return {
                    "category_id": cat_id,
                    "category_name": cat_name,
                    "confidence": 0.5,
                    "reasoning": f"基于关键词 '{kw1}/{kw2}' 的规则匹配(fallback)",
                    "discipline": "未确定",
                    "suggested_keywords": [],
                }

        # 默认
        return {
            "category_id": "liberal_arts_paper",
            "category_name": "文科论文",
            "confidence": 0.3,
            "reasoning": "无法确定，默认为文科论文 (fallback)",
            "discipline": "未确定",
            "suggested_keywords": [],
        }
