"""
Agent 3: 关键词提取器

从课程论文要求中提取核心学术关键词，用于后续文献检索。
提取中英文关键词，覆盖主要研究方向和子领域。
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


KEYWORD_SYSTEM_PROMPT = """你是一个学术关键词提取专家。你的任务是从课程论文要求中提取用于文献检索的关键词。

## 提取原则

1. **覆盖核心技术/概念术语**：提取领域专有名词
2. **中英文兼顾**：提供中文关键词和对应的英文关键词
3. **层次化**：从宽泛到具体（总共不超过20个关键词）
   - 一级关键词（领域级别）：2-3个，用于广泛检索
   - 二级关键词（主题级别）：3-5个，用于聚焦检索
   - 三级关键词（细节级别）：3-5个，用于深度检索
4. **考虑同义词和变体**：列出可替代的关键词表述、缩写、全称
5. ⚠️ 关键词总数控制在20个以内，质量为上

## 检索查询生成规则 ⚠️ 重要

1. **arXiv 查询必须全部使用英文**：arXiv 不支持中文检索，中文查询会返回0结果
2. **生成5-10个高质量英文检索查询**，覆盖：
   - 核心主题组合: "deep learning AND image recognition AND CNN"
   - 方法论角度: "transfer learning AND few-shot learning AND survey"
   - 应用场景: "medical image analysis AND deep learning AND review"
   - 交叉学科: "computer vision AND cognitive science AND neural networks"
   - 对比/综述: "survey OR review AND deep learning AND computer vision"
3. 每个查询标记 priority (high/medium/normal) 和 source ("arxiv"/"web")
4. **绝对不能**在 search_queries 中使用中文关键词

## 输出格式

请严格输出以下JSON格式：
{
    "primary_keywords": [
        {"zh": "深度学习", "en": "deep learning", "synonyms": ["deep neural networks", "DNN"]},
        {"zh": "图像识别", "en": "image recognition", "synonyms": ["visual recognition", "image classification"]}
    ],
    "secondary_keywords": [...],
    "tertiary_keywords": [...],
    "search_queries": [
        {
            "query": "deep learning AND image recognition AND CNN",
            "source": "arxiv",
            "priority": "high"
        }
    ],
    "discipline_detected": "检测到的学科领域",
    "sub_disciplines": ["子领域1", "子领域2"],
    "related_domains": ["相关领域1", "相关领域2"]
}

只输出JSON，不要包含任何其他文字。"""

KEYWORD_EXPANSION_PROMPT = """你是一个学术关键词扩展专家。基于以下已有关键词和研究主题，请进一步扩展出更多检索角度。

## 扩展方向

1. **同义词与变体**：每个核心关键词的英文同义词、缩写、全称、不同拼写
2. **上下位词**：更广泛的上位概念和更具体的下位概念
3. **方法论扩展**：该领域常用的研究方法、技术、框架名称
4. **应用场景**：该领域的不同应用场景和案例
5. **交叉学科**：可能的跨学科关联领域
6. **时间维度**：经典研究（早期奠基）+ 最新前沿（近2年）
7. **地理/学派**：不同学术传统下的表述差异

## 输出格式

请严格输出以下JSON格式：
{
    "expanded_queries": [
        {
            "query": "新的检索查询字符串",
            "source": "arxiv",
            "priority": "high",
            "angle": "该查询的检索角度说明"
        }
    ],
    "new_keywords": [
        {"zh": "中文", "en": "English", "synonyms": ["variant1", "variant2"]}
    ],
    "broadened_terms": ["更广泛的上位术语"],
    "narrowed_terms": ["更具体的下位术语"]
}

只输出JSON。"""


class KeywordExtractor:
    """关键词提取 Agent — 按字数级别扩展检索覆盖面"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def extract(
        self, requirements: str, expansion_rounds: int = 1
    ) -> Dict[str, Any]:
        """
        从论文要求中提取关键词，并按需进行多轮扩展。

        Args:
            requirements: 课程论文要求文本
            expansion_rounds: 关键词扩展轮数（依据目标字数级别）

        Returns:
            包含多层次关键词和检索查询的字典
        """
        logger.info(f"Agent 3: 正在提取关键词... (扩展轮数: {expansion_rounds})")

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
            if "sub_disciplines" not in result:
                result["sub_disciplines"] = []
            if "related_domains" not in result:
                result["related_domains"] = []

            # 后处理：限制关键词总数 ≤ 20
            result = self._cap_keywords(result, max_total=20)

            # 后处理：过滤中文查询（arXiv 不支持中文，中文查询返回0结果）
            result["search_queries"] = self._filter_chinese_queries(
                result["search_queries"]
            )

            # 多轮关键词扩展
            for round_idx in range(expansion_rounds - 1):
                logger.info(f"  关键词扩展轮次 {round_idx + 1}/{expansion_rounds - 1}...")
                expanded = self._expand_keywords(requirements, result)
                if expanded:
                    # 合并扩展结果
                    result["search_queries"].extend(
                        expanded.get("expanded_queries", [])
                    )
                    for level in ["primary_keywords", "secondary_keywords", "tertiary_keywords"]:
                        if expanded.get("new_keywords"):
                            result.setdefault(level, []).extend(
                                expanded.get("new_keywords", [])
                            )
                    # 去重 search_queries
                    seen_queries = set()
                    unique_queries = []
                    for q in result["search_queries"]:
                        q_text = q.get("query", "").lower()
                        if q_text not in seen_queries:
                            seen_queries.add(q_text)
                            unique_queries.append(q)
                    result["search_queries"] = unique_queries

            # 扩展后再次限制关键词总数
            result = self._cap_keywords(result, max_total=20)
            # 扩展后再次过滤中文查询
            result["search_queries"] = self._filter_chinese_queries(
                result["search_queries"]
            )

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

    def _expand_keywords(
        self, requirements: str, current_result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """基于已有结果进行关键词扩展"""
        try:
            existing_queries = [
                q.get("query", "") for q in current_result.get("search_queries", [])
            ]
            existing_kw = []
            for level in ["primary_keywords", "secondary_keywords", "tertiary_keywords"]:
                for kw in current_result.get(level, []):
                    if isinstance(kw, dict):
                        existing_kw.append(kw.get("en", "") or kw.get("zh", ""))

            expansion_input = (
                f"原始论文要求：{requirements}\n\n"
                f"已有关键词：{', '.join(existing_kw[:20])}\n"
                f"已有检索查询：{', '.join(existing_queries[:10])}\n"
                f"已发现学科：{current_result.get('discipline_detected', '')}\n"
                f"子领域：{', '.join(current_result.get('sub_disciplines', []))}\n\n"
                f"请从不同角度（同义词、上下位词、方法论、应用场景、交叉学科、最新前沿）"
                f"扩展出更多检索查询，确保覆盖面更广。"
            )

            return self.llm.chat_with_json_output(
                system_prompt=KEYWORD_EXPANSION_PROMPT,
                user_message=expansion_input,
                temperature=0.5,
            )
        except Exception as e:
            logger.warning(f"  关键词扩展失败: {e}")
            return None

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

    @staticmethod
    def _has_chinese(text: str) -> bool:
        """检测文本是否包含中文字符"""
        import re as _re
        return bool(_re.search(r'[一-鿿]', text))

    def _cap_keywords(
        self, result: Dict[str, Any], max_total: int = 20
    ) -> Dict[str, Any]:
        """限制关键词总数不超过 max_total"""
        levels = ["primary_keywords", "secondary_keywords", "tertiary_keywords"]
        total = sum(len(result.get(level, [])) for level in levels)

        if total > max_total:
            logger.info(f"  关键词总数 ({total}) 超过上限 ({max_total})，进行截断...")
            # 从低优先级开始截断
            remaining = max_total
            for level in levels:
                items = result.get(level, [])
                if len(items) > remaining:
                    result[level] = items[:remaining]
                    remaining = 0
                else:
                    remaining -= len(items)

        return result

    @staticmethod
    def _filter_chinese_queries(
        queries: list,
    ) -> list:
        """过滤掉包含中文的 arXiv 查询（中文在 arXiv 返回0结果）。

        对于包含中文的查询，尝试提取英文部分；如果完全没有英文，
        则标记为 web 查询（web 搜索可以处理中文）。
        """
        import re as _re

        # 停用词：布尔运算符和常见的无意义单词
        STOP_WORDS = {"AND", "OR", "NOT", "THE", "A", "AN", "OF", "IN", "ON", "TO",
                      "FOR", "WITH", "BY", "AT", "IS", "ARE", "WAS", "WERE", "BEEN"}

        filtered = []
        for q in queries:
            if not isinstance(q, dict):
                continue

            query_text = q.get("query", "")
            if not query_text:
                continue

            has_chinese = bool(_re.search(r'[一-鿿]', query_text))
            has_english = bool(_re.search(r'[a-zA-Z]{2,}', query_text))

            if has_chinese:
                if has_english:
                    # 包含中文但也有英文：提取英文部分
                    # 匹配连续的英文单词序列（>=3个字符）
                    english_parts = _re.findall(
                        r'[a-zA-Z]{3,}(?:\s+[a-zA-Z]{3,})*', query_text
                    )
                    if english_parts:
                        # 过滤掉纯停用词的片段
                        meaningful_parts = []
                        for part in english_parts:
                            words = part.split()
                            content_words = [
                                w for w in words
                                if w.upper() not in STOP_WORDS
                            ]
                            if len(content_words) >= 1:
                                meaningful_parts.append(part)

                        if meaningful_parts:
                            # 保留最长的有意义的英文部分
                            best_part = max(meaningful_parts, key=len)
                            new_q = dict(q)
                            new_q["query"] = best_part.strip()
                            new_q["note"] = "英文部分提取自混合查询"
                            filtered.append(new_q)
                            continue

                # 中文查询无法提取有效英文 → 标记为 web 查询
                new_q = dict(q)
                new_q["source"] = "web"
                new_q["note"] = "中文查询（arXiv不支持）→ Web搜索"
                filtered.append(new_q)
            else:
                filtered.append(q)

        return filtered
