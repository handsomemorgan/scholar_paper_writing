"""
Agent 3: 关键词提取器

从课程论文要求中提取核心学术关键词，用于后续文献检索。
提取中英文关键词，覆盖主要研究方向和子领域。
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


KEYWORD_SYSTEM_PROMPT = """你是一个学术关键词提取专家。你的任务是从课程论文要求中提取用于文献检索的关键词。

## 提取原则

1. **覆盖核心技术/概念术语**：提取领域专有名词
2. **中英文兼顾**：提供中文关键词和对应的英文关键词
3. **层次化**：从宽泛到具体
   - 一级关键词（领域级别）：2-3个，用于广泛检索
   - 二级关键词（主题级别）：3-5个，用于聚焦检索
   - 三级关键词（细节级别）：3-5个，用于深度检索
4. **考虑同义词**：列出可替代的关键词表述

## 检索策略提示

为每个一级关键词建议一个英文检索组合（arXiv 公开接口），使用英文关键词可获得最佳检索效果：
- "deep learning AND image recognition AND CNN"

## 输出格式

请严格输出以下JSON格式：
{
    "primary_keywords": [
        {"zh": "深度学习", "en": "deep learning"},
        {"zh": "图像识别", "en": "image recognition"}
    ],
    "secondary_keywords": [
        {"zh": "卷积神经网络", "en": "CNN"},
        {"zh": "迁移学习", "en": "transfer learning"}
    ],
    "tertiary_keywords": [
        {"zh": "注意力机制", "en": "attention mechanism"}
    ],
    "search_queries": [
        {
            "query": "deep learning AND image recognition AND CNN",
            "source": "arxiv",
            "priority": "high"
        }
    ],
    "discipline_detected": "检测到的学科领域"
}

只输出JSON，不要包含任何其他文字。"""


class KeywordExtractor:
    """关键词提取 Agent"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def extract(self, requirements: str) -> Dict[str, Any]:
        """
        从论文要求中提取关键词。

        Args:
            requirements: 课程论文要求文本

        Returns:
            包含多层次关键词和检索查询的字典
        """
        logger.info("Agent 3: 正在提取关键词...")

        user_message = f"请从以下课程论文要求中提取关键词：\n\n{requirements}"

        try:
            result = self.llm.chat_with_json_output(
                system_prompt=KEYWORD_SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.3,
            )

            # 验证必要字段
            if "primary_keywords" not in result:
                result["primary_keywords"] = []
            if "search_queries" not in result:
                result["search_queries"] = []

            keyword_count = sum(
                len(result.get(k, []))
                for k in ["primary_keywords", "secondary_keywords", "tertiary_keywords"]
            )
            logger.info(
                f"Agent 3 完成: 提取了 {keyword_count} 个关键词, "
                f"{len(result.get('search_queries', []))} 个检索查询"
            )
            return result

        except Exception as e:
            logger.error(f"Agent 3 关键词提取失败: {e}")
            return self._fallback_extract(requirements)

    def _fallback_extract(self, requirements: str) -> Dict[str, Any]:
        """基于文本分析的基础关键词提取（LLM失败时使用）"""
        # 简单的关键词候选词
        import re

        # 提取引号中的术语
        quoted = re.findall(r'[《「『"]([^》」』"]+)[》」』"]', requirements)
        # 提取括号中的英文缩写
        abbreviations = re.findall(r'\(([A-Z]{2,})\)', requirements)

        primary = []
        for term in quoted[:3]:
            primary.append({"zh": term, "en": ""})

        return {
            "primary_keywords": primary,
            "secondary_keywords": [],
            "tertiary_keywords": [],
            "search_queries": [
                {"query": kw["zh"], "source": "arxiv", "priority": "high"}
                for kw in primary
            ],
            "discipline_detected": "未确定",
        }

    def get_all_keywords_flat(self, keyword_result: Dict[str, Any]) -> List[str]:
        """将所有层次的关键词展平为一个列表"""
        all_kw = []
        for level in ["primary_keywords", "secondary_keywords", "tertiary_keywords"]:
            for kw in keyword_result.get(level, []):
                if isinstance(kw, dict):
                    if kw.get("zh"):
                        all_kw.append(kw["zh"])
                    if kw.get("en"):
                        all_kw.append(kw["en"])
                elif isinstance(kw, str):
                    all_kw.append(kw)
        return all_kw
