"""
Agent 4: 文献检索Agent

使用关键词通过 arXiv 公开 API 进行学术文献检索。
自动筛选高质量文献，去重合并。
"""

import logging
import random
from typing import List, Dict, Any

import yaml

from utils.web_search import LiteratureSearcher, LiteratureItem

logger = logging.getLogger(__name__)


class LiteratureSearchAgent:
    """文献检索 Agent — 通过 arXiv API 搜索学术文献"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.searcher = LiteratureSearcher(self.config)

    def search(self, keyword_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        根据关键词检索文献。

        Args:
            keyword_result: Agent 3的关键词提取结果

        Returns:
            文献列表，每条包含完整的文献信息
        """
        logger.info("Agent 4: 正在通过 arXiv API 检索文献...")

        # 收集所有检索查询
        queries = keyword_result.get("search_queries", [])
        if not queries:
            # 如果没有预设查询，使用关键词构建
            primary_kw = [
                kw.get("en", "") or kw.get("zh", "")
                for kw in keyword_result.get("primary_keywords", [])
            ]
            queries = [{"query": " AND ".join(primary_kw), "source": "arxiv", "priority": "high"}]

        all_literature: List[LiteratureItem] = []
        seen_titles = set()

        # 按优先级排序
        high_priority = [q for q in queries if q.get("priority") == "high"]
        normal_priority = [q for q in queries if q.get("priority") != "high"]

        for query_info in high_priority + normal_priority:
            query = query_info.get("query", "")
            if not query:
                continue

            logger.info(f"检索查询: '{query}'")

            try:
                results = self.searcher.search([query])
                for item in results:
                    norm_title = item.title.lower().strip()
                    if norm_title not in seen_titles and norm_title:
                        seen_titles.add(norm_title)
                        all_literature.append(item)
            except Exception as e:
                logger.error(f"查询 '{query}' 检索失败: {e}")
                continue

        # arXiv 不提供引用数，按年份排序
        all_literature.sort(
            key=lambda x: (x.year if x.year else 0), reverse=True
        )

        # 转换为字典格式（便于后续处理）
        literature_list = [
            {
                "title": item.title,
                "authors": item.authors,
                "year": item.year,
                "journal": item.journal,
                "abstract": item.abstract,
                "citation_count": item.citation_count,
                "url": item.url,
                "source": item.source,
                "arxiv_id": item.arxiv_id,
                "primary_category": item.primary_category,
            }
            for item in all_literature
        ]

        if len(literature_list) < 5:
            logger.warning(
                f"检索结果不足 ({len(literature_list)}篇)，"
                "请检查网络连接或尝试更宽泛的关键词"
            )

        logger.info(
            f"Agent 4 完成: 共检索到 {len(literature_list)} 篇文献"
        )
        return literature_list

    def filter_high_quality(
        self, literature_list: List[Dict[str, Any]], min_citations: int = 0
    ) -> List[Dict[str, Any]]:
        """筛选高质量文献

        对于 arXiv 数据源，主要根据期刊信息（journal-ref）筛选，
        同时可结合 Semantic Scholar 等外部引用数据。
        """
        # arXiv 预印本通常没有引用数，按是否有期刊引用信息筛选
        high_quality = []
        for lit in literature_list:
            journal = lit.get("journal", "")
            # 有正式期刊引用信息的视为高质量
            if journal and journal != "arXiv preprint":
                high_quality.append(lit)
                continue
            # 或者 primary_category 是核心领域
            primary_cat = lit.get("primary_category", "")
            if primary_cat in [
                "cs.AI", "cs.CL", "cs.CV", "cs.LG", "stat.ML",
                "physics.soc-ph", "q-bio", "q-fin",
            ]:
                high_quality.append(lit)
                continue

        return high_quality if high_quality else literature_list

    def generate_mock_literature(
        self, keyword_result: Dict[str, Any], count: int = 5
    ) -> List[Dict[str, Any]]:
        """
        生成模拟文献数据（用于演示和测试）。

        在实际部署中，此方法仅在 arXiv API 不可用时作为 fallback。
        """
        primary_kw = keyword_result.get("primary_keywords", [])
        topic_en = (
            primary_kw[0].get("en", "research topic")
            if primary_kw
            else "research topic"
        )
        topic_zh = (
            primary_kw[0].get("zh", "研究主题")
            if primary_kw
            else "研究主题"
        )

        mock_templates = [
            {
                "title": f"A Comprehensive Survey of {topic_en.title()}: Methods, Challenges, and Future Directions",
                "authors": ["Wang, X.", "Li, Y.", "Zhang, H."],
                "year": 2023,
                "journal": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
                "abstract": f"This paper presents a comprehensive survey of {topic_en}, covering the latest advances in methodology, key challenges, and promising future research directions.",
                "citation_count": random.randint(50, 200),
                "source": "arxiv",
                "url": f"https://arxiv.org/abs/2301.{random.randint(10000, 99999)}",
                "arxiv_id": f"2301.{random.randint(10000, 99999)}",
            },
            {
                "title": f"{topic_zh}的理论基础与实践应用研究",
                "authors": ["张明", "李华", "王强"],
                "year": 2024,
                "journal": "中国科学",
                "abstract": f"本文系统梳理了{topic_zh}的理论基础，结合实践案例分析了其在多个领域的应用效果，并提出了一种改进的分析框架。",
                "citation_count": random.randint(30, 100),
                "source": "arxiv",
                "url": f"https://arxiv.org/abs/2401.{random.randint(10000, 99999)}",
                "arxiv_id": f"2401.{random.randint(10000, 99999)}",
            },
            {
                "title": f"Deep Learning Approaches for {topic_en.title()}: A Review",
                "authors": ["Chen, L.", "Liu, J."],
                "year": 2022,
                "journal": "Nature Machine Intelligence",
                "abstract": f"We review deep learning approaches applied to {topic_en}, comparing performance across different architectures and datasets.",
                "citation_count": random.randint(80, 300),
                "source": "arxiv",
                "url": f"https://arxiv.org/abs/2206.{random.randint(10000, 99999)}",
                "arxiv_id": f"2206.{random.randint(10000, 99999)}",
            },
            {
                "title": f"基于大数据的{topic_zh}分析方法研究",
                "authors": ["刘伟", "陈静"],
                "year": 2023,
                "journal": "计算机学报",
                "abstract": f"提出了一种基于大数据技术的{topic_zh}分析方法，通过海量数据挖掘揭示{topic_zh}的内在规律。",
                "citation_count": random.randint(20, 60),
                "source": "arxiv",
                "url": f"https://arxiv.org/abs/2306.{random.randint(10000, 99999)}",
                "arxiv_id": f"2306.{random.randint(10000, 99999)}",
            },
            {
                "title": f"Recent Advances and Trends in {topic_en.title()}: 2020-2024",
                "authors": ["Johnson, M.", "Park, S.", "Wu, T."],
                "year": 2024,
                "journal": "ACM Computing Surveys",
                "abstract": f"This survey covers the most recent advances in {topic_en}, identifying emerging trends and open problems in the field.",
                "citation_count": random.randint(15, 40),
                "source": "arxiv",
                "url": f"https://arxiv.org/abs/2405.{random.randint(10000, 99999)}",
                "arxiv_id": f"2405.{random.randint(10000, 99999)}",
            },
        ]

        return mock_templates[:count]
