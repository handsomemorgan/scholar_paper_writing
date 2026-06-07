"""
Agent 4: 文献检索Agent

检索策略（三级回退链）：
  1. arXiv API —— 主力数据源，免费开放的学术预印本数据库
  2. 浏览器网页搜索 —— arXiv 不足时，通过搜索引擎查阅学术资料
  3. 自主模拟文献 —— 仅在无法收集到相关资料时作为最后手段

自动筛选高质量文献，去重合并。
"""

import logging
import random
from typing import List, Dict, Any

import yaml

from utils.web_search import LiteratureSearcher, LiteratureItem

logger = logging.getLogger(__name__)

# 最少需要的文献数量，低于此值触发下一级 fallback
MIN_LITERATURE_THRESHOLD = 5


class LiteratureSearchAgent:
    """文献检索 Agent — 多级回退，确保总能获取文献"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.searcher = LiteratureSearcher(self.config)

    def search(self, keyword_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        根据关键词检索文献。使用三级回退链：

        Phase 4a: arXiv API 检索
           ↓ 结果不足 {MIN_LITERATURE_THRESHOLD} 篇？
        Phase 4b: 浏览器网页搜索
           ↓ 结果仍不足？
        Phase 4c: 自主模拟文献（最后手段）

        Args:
            keyword_result: Agent 3的关键词提取结果

        Returns:
            文献列表，每条包含完整的文献信息
        """
        # ---------- 收集检索查询 ----------
        queries = keyword_result.get("search_queries", [])
        if not queries:
            primary_kw = [
                kw.get("en", "") or kw.get("zh", "")
                for kw in keyword_result.get("primary_keywords", [])
            ]
            queries = [
                {
                    "query": " AND ".join(primary_kw),
                    "source": "arxiv",
                    "priority": "high",
                }
            ]

        # 按优先级排序
        high_priority = [q for q in queries if q.get("priority") == "high"]
        normal_priority = [q for q in queries if q.get("priority") != "high"]
        ordered_queries = high_priority + normal_priority

        # 收集所有关键词（用于 web search 和 mock）
        all_keywords = self._collect_all_keywords(keyword_result)
        if not all_keywords:
            all_keywords = [q.get("query", "research") for q in ordered_queries]

        all_literature: List[LiteratureItem] = []
        seen_titles = set()

        # ============================================================
        # Phase 4a: arXiv API 检索（主力数据源）
        # ============================================================
        logger.info("=" * 50)
        logger.info("Phase 4a: arXiv API 检索（主力数据源）")
        logger.info("=" * 50)

        for query_info in ordered_queries:
            query = query_info.get("query", "")
            if not query:
                continue

            logger.info(f"  arXiv 查询: '{query}'")
            try:
                results = self.searcher.search([query])
                for item in results:
                    norm_title = item.title.lower().strip()
                    if norm_title not in seen_titles and norm_title:
                        seen_titles.add(norm_title)
                        all_literature.append(item)
            except Exception as e:
                logger.error(f"  arXiv 查询失败: {e}")
                continue

        arxiv_count = len(all_literature)
        logger.info(
            f"Phase 4a 完成: arXiv 检索到 {arxiv_count} 篇文献"
        )

        # ============================================================
        # Phase 4b: 浏览器网页搜索（arXiv 不足时的 fallback）
        # ============================================================
        if arxiv_count < MIN_LITERATURE_THRESHOLD:
            logger.warning(
                f"arXiv 检索结果不足 ({arxiv_count} < {MIN_LITERATURE_THRESHOLD})，"
                "启动浏览器网页搜索..."
            )
            logger.info("=" * 50)
            logger.info("Phase 4b: 浏览器网页搜索（查阅学术资料）")
            logger.info("=" * 50)

            # 使用不同粒度的关键词组合进行多轮搜索
            web_search_queries = self._build_web_search_queries(
                all_keywords, ordered_queries
            )

            for web_query in web_search_queries:
                # 如果已有足够文献，停止搜索
                if len(all_literature) >= MIN_LITERATURE_THRESHOLD * 2:
                    break

                logger.info(f"  Web 查询: '{web_query}'")
                try:
                    web_results = self.searcher.search_web_for_academic(
                        [web_query]
                    )
                    for item in web_results:
                        norm_title = item.title.lower().strip()
                        if norm_title not in seen_titles and norm_title:
                            seen_titles.add(norm_title)
                            item.source = item.source or "web_search"
                            all_literature.append(item)
                except Exception as e:
                    logger.error(f"  Web 搜索失败: {e}")
                    continue

            web_count = len(all_literature) - arxiv_count
            logger.info(
                f"Phase 4b 完成: Web 搜索新增 {web_count} 篇，"
                f"总计 {len(all_literature)} 篇"
            )
        else:
            logger.info(
                f"arXiv 结果充足 ({arxiv_count} >= {MIN_LITERATURE_THRESHOLD})，"
                "跳过浏览器搜索"
            )

        # ============================================================
        # Phase 4c: 自主模拟文献（最后手段）
        # ============================================================
        if len(all_literature) < MIN_LITERATURE_THRESHOLD:
            logger.warning(
                f"真实检索结果仍不足 ({len(all_literature)} < "
                f"{MIN_LITERATURE_THRESHOLD})，启动自主模拟文献生成..."
            )
            logger.info("=" * 50)
            logger.info("Phase 4c: 自主模拟文献（最后手段）")
            logger.info("=" * 50)

            mock_count = MIN_LITERATURE_THRESHOLD * 2 - len(all_literature)
            mock_lit = self.generate_mock_literature(
                keyword_result, count=mock_count
            )

            # 去重后加入
            for mock_item in mock_lit:
                norm_title = mock_item.get("title", "").lower().strip()
                if norm_title not in seen_titles and norm_title:
                    seen_titles.add(norm_title)
                    # 直接添加字典（mock 返回的已经是 dict）
                    all_literature.append(
                        LiteratureItem(
                            title=mock_item["title"],
                            authors=mock_item.get("authors", []),
                            year=mock_item.get("year"),
                            journal=mock_item.get("journal", ""),
                            abstract=mock_item.get("abstract", ""),
                            citation_count=mock_item.get("citation_count", 0),
                            url=mock_item.get("url", ""),
                            source="mock",
                            arxiv_id=mock_item.get("arxiv_id", ""),
                        )
                    )

            logger.info(
                f"Phase 4c 完成: 模拟生成 {mock_count} 篇，"
                f"总计 {len(all_literature)} 篇"
            )

        # ============================================================
        # 排序和输出
        # ============================================================
        # 按年份降序排列（较新的在前）
        all_literature.sort(
            key=lambda x: (x.year if x.year else 0), reverse=True
        )

        # 转换为字典格式（便于后续 Agent 处理）
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

        # 统计各来源数量
        source_counts = {}
        for lit in literature_list:
            src = lit.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
        source_summary = ", ".join(
            f"{src}: {cnt}篇" for src, cnt in sorted(source_counts.items())
        )

        logger.info(
            f"\nAgent 4 完成: 共检索到 {len(literature_list)} 篇文献 "
            f"({source_summary})"
        )
        return literature_list

    def _collect_all_keywords(
        self, keyword_result: Dict[str, Any]
    ) -> List[str]:
        """从关键词结果中收集所有关键词"""
        all_kw = []
        for level in [
            "primary_keywords",
            "secondary_keywords",
            "tertiary_keywords",
        ]:
            for kw in keyword_result.get(level, []):
                if isinstance(kw, dict):
                    if kw.get("en"):
                        all_kw.append(kw["en"])
                elif isinstance(kw, str):
                    all_kw.append(kw)
        return all_kw

    def _build_web_search_queries(
        self,
        all_keywords: List[str],
        existing_queries: List[Dict[str, Any]],
    ) -> List[str]:
        """
        构建多组网页搜索查询，提高命中率。

        策略：
        - 用优先级最高的查询原样搜索
        - 用去 AND 简化后的关键词组合搜索
        - 用单独的核心关键词搜索
        """
        queries = []

        # 保留已有的高优先级查询
        for q in existing_queries:
            if q.get("priority") == "high":
                # 去掉 AND 和引号，适合网页搜索
                clean = q.get("query", "").replace("AND", "").replace('"', "")
                clean = " ".join(clean.split())  # 规范化空格
                if clean and clean not in queries:
                    queries.append(clean)

        # 用前 3-5 个核心关键词组合
        if len(all_keywords) >= 2:
            queries.append(" ".join(all_keywords[:3]))
        if len(all_keywords) >= 1:
            queries.append(all_keywords[0])

        # 去重，最多 5 个查询
        seen = set()
        unique_queries = []
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)

        return unique_queries[:5]

    def filter_high_quality(
        self, literature_list: List[Dict[str, Any]], min_citations: int = 0
    ) -> List[Dict[str, Any]]:
        """筛选高质量文献

        对于 arXiv 数据源，主要根据期刊信息（journal-ref）筛选。
        web_search 和 mock 来源的文献按来源可信度排序。
        """
        high_quality = []
        for lit in literature_list:
            journal = lit.get("journal", "")
            source = lit.get("source", "")

            # arXiv 且有正式期刊引用 → 高质量
            if source == "arxiv" and journal and journal != "arXiv preprint":
                high_quality.append(lit)
                continue

            # Web 搜索结果中来自 .edu / 学术域名的
            if source.startswith("web_") and source != "web_search":
                high_quality.append(lit)
                continue

            # arXiv 核心领域
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
        自主模拟文献数据 —— 仅在 arXiv 和 Web 搜索均无法获取
        足够资料时作为最后手段使用。

        按 topic 生成合理的文献条目，确保后续流程可以运行。
        来源标记为 "mock"，便于区分真实文献和模拟文献。
        """
        logger.warning(
            "⚠️  正在使用自主模拟文献（最后手段）。"
            "这些文献为 AI 自动生成，不保证真实存在，"
            "请在实际提交前替换为真实引文。"
        )

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
                "abstract": f"This paper presents a comprehensive survey of {topic_en}, covering the latest advances in methodology, key challenges, and promising future research directions. The authors systematically review over 200 papers and identify major research trends.",
                "citation_count": random.randint(50, 200),
                "source": "mock",
                "url": "",
                "arxiv_id": "",
            },
            {
                "title": f"{topic_zh}的理论基础与实践应用研究",
                "authors": ["张明", "李华", "王强"],
                "year": 2024,
                "journal": "中国科学",
                "abstract": f"本文系统梳理了{topic_zh}的理论基础，结合实践案例分析了其在多个领域的应用效果，并提出了一种改进的分析框架。研究表明，该框架在效率和准确性方面均有显著提升。",
                "citation_count": random.randint(30, 100),
                "source": "mock",
                "url": "",
                "arxiv_id": "",
            },
            {
                "title": f"Deep Learning Approaches for {topic_en.title()}: A Review",
                "authors": ["Chen, L.", "Liu, J."],
                "year": 2022,
                "journal": "Nature Machine Intelligence",
                "abstract": f"We review deep learning approaches applied to {topic_en}, comparing performance across different architectures and datasets. Key findings suggest that transformer-based models outperform traditional approaches in most benchmarks.",
                "citation_count": random.randint(80, 300),
                "source": "mock",
                "url": "",
                "arxiv_id": "",
            },
            {
                "title": f"基于大数据的{topic_zh}分析方法研究",
                "authors": ["刘伟", "陈静"],
                "year": 2023,
                "journal": "计算机学报",
                "abstract": f"提出了一种基于大数据技术的{topic_zh}分析方法，通过海量数据挖掘揭示{topic_zh}的内在规律。实验结果表明该方法相比传统方法准确率提升约15%。",
                "citation_count": random.randint(20, 60),
                "source": "mock",
                "url": "",
                "arxiv_id": "",
            },
            {
                "title": f"Recent Advances and Trends in {topic_en.title()}: 2020-2024",
                "authors": ["Johnson, M.", "Park, S.", "Wu, T."],
                "year": 2024,
                "journal": "ACM Computing Surveys",
                "abstract": f"This survey covers the most recent advances in {topic_en}, identifying emerging trends and open problems in the field. The authors highlight the growing importance of interpretability and fairness in {topic_en} research.",
                "citation_count": random.randint(15, 40),
                "source": "mock",
                "url": "",
                "arxiv_id": "",
            },
            {
                "title": f"A Novel Framework for {topic_en.title()} Using Hybrid Models",
                "authors": ["Kim, S.", "Park, J.", "Lee, H."],
                "year": 2023,
                "journal": "NeurIPS",
                "abstract": f"We propose a novel hybrid framework combining the strengths of multiple approaches for {topic_en}. Extensive experiments on standard benchmarks demonstrate state-of-the-art performance.",
                "citation_count": random.randint(40, 120),
                "source": "mock",
                "url": "",
                "arxiv_id": "",
            },
            {
                "title": f"{topic_zh}研究的热点与趋势——基于文献计量学的分析",
                "authors": ["赵丽", "孙强", "周敏"],
                "year": 2024,
                "journal": "科学学研究",
                "abstract": f"运用文献计量学方法，对近五年{topic_zh}领域的研究热点和趋势进行分析。结果发现，{topic_zh}的研究正从单一方法向多学科交叉融合方向发展。",
                "citation_count": random.randint(10, 50),
                "source": "mock",
                "url": "",
                "arxiv_id": "",
            },
            {
                "title": f"Ethical Considerations in {topic_en.title()}: Challenges and Solutions",
                "authors": ["Brown, A.", "Davis, R."],
                "year": 2024,
                "journal": "Science",
                "abstract": f"As {topic_en} technologies become more prevalent, ethical considerations have gained increasing attention. This paper discusses key ethical challenges and proposes practical solutions for responsible development.",
                "citation_count": random.randint(20, 80),
                "source": "mock",
                "url": "",
                "arxiv_id": "",
            },
        ]

        return mock_templates[:count]
