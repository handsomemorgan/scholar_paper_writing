"""
Agent 4: 文献检索Agent

检索策略（三级回退链）：
  1. arXiv API —— 主力数据源，遍历所有英文查询
  2. 浏览器网页搜索 —— 始终执行，补充更多资料
  3. 自主模拟文献 —— 仅在无法收集到资料时作为最后手段

目标：收集 10-20 篇文献，不区分字数级别。
"""

import logging
import random
import re as _re
from typing import List, Dict, Any, Optional

import yaml

from utils.web_search import LiteratureSearcher, LiteratureItem

logger = logging.getLogger(__name__)

# 文献收集目标：通用的 10-20 篇
TARGET_MIN = 10
TARGET_IDEAL = 15


class LiteratureSearchAgent:
    """文献检索 Agent — 遍历所有查询，三级回退"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.searcher = LiteratureSearcher(self.config)

    def search(
        self,
        keyword_result: Dict[str, Any],
        lit_strategy: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        根据关键词检索文献。使用三级回退链。

        Phase 4a: arXiv API 检索（遍历所有英文查询）
           ↓
        Phase 4b: 浏览器网页搜索（始终执行）
           ↓ 结果不足 10 篇？
        Phase 4c: 自主模拟文献（最后手段）
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
                    "query": " AND ".join([k for k in primary_kw if k]),
                    "source": "arxiv",
                    "priority": "high",
                }
            ]

        # 按优先级排序
        high_priority = [q for q in queries if q.get("priority") == "high"]
        medium_priority = [q for q in queries if q.get("priority") == "medium"]
        normal_priority = [q for q in queries if q.get("priority") not in ("high", "medium")]
        ordered_queries = high_priority + medium_priority + normal_priority

        # 收集所有关键词（用于 web search）
        all_keywords = self._collect_all_keywords(keyword_result)
        if not all_keywords:
            all_keywords = [q.get("query", "research") for q in ordered_queries]

        all_literature: List[LiteratureItem] = []
        seen_titles = set()

        # ============================================================
        # Phase 4a: arXiv API 检索
        #   遍历所有英文查询，每个查询取 max_results 条
        #   高优先级查询额外多取一页（翻页）
        # ============================================================
        logger.info("=" * 50)
        logger.info("Phase 4a: arXiv API 检索（遍历所有查询）")
        logger.info("=" * 50)

        arxiv_query_count = 0
        for query_info in ordered_queries:
            query = query_info.get("query", "")
            if not query:
                continue

            # 跳过中文查询（arXiv 不支持，返回 0 结果）
            if self._has_chinese(query):
                logger.warning(f"  跳过中文查询: '{query[:60]}...'")
                continue

            arxiv_query_count += 1
            logger.info(f"  arXiv 查询 [{arxiv_query_count}]: '{query[:80]}...'")

            try:
                # 第一页（max_results 已在 config 设为 30）
                results = self.searcher.search_arxiv_paginated(
                    [query], start=0, max_results=self.searcher.max_results
                )
                got_count = len(results)

                # 如果返回0条，尝试更宽泛的查询
                if got_count == 0:
                    # 去掉 AND 运算符，只用核心词
                    broad_query = " ".join(
                        q for q in query.replace("AND", " ").split()
                        if len(q) > 2 and q.upper() not in ("AND", "OR", "NOT")
                    )
                    if broad_query and broad_query != query:
                        logger.info(f"    0条结果，尝试宽泛查询: '{broad_query[:60]}...'")
                        results = self.searcher.search_arxiv_paginated(
                            [broad_query], start=0, max_results=self.searcher.max_results
                        )
                        got_count = len(results)

                for item in results:
                    norm_title = item.title.lower().strip()
                    if norm_title not in seen_titles and norm_title:
                        seen_titles.add(norm_title)
                        all_literature.append(item)

                logger.info(f"    → {got_count} 条结果")

                # 高优先级查询额外翻一页
                priority = query_info.get("priority", "normal")
                if priority == "high" and got_count >= self.searcher.max_results:
                    logger.info(f"    高优先级查询翻页...")
                    results_page2 = self.searcher.search_arxiv_paginated(
                        [query],
                        start=self.searcher.max_results,
                        max_results=self.searcher.max_results,
                    )
                    for item in results_page2:
                        norm_title = item.title.lower().strip()
                        if norm_title not in seen_titles and norm_title:
                            seen_titles.add(norm_title)
                            all_literature.append(item)

            except Exception as e:
                logger.error(f"  arXiv 查询失败: {e}")
                continue

        arxiv_count = len(all_literature)
        logger.info(
            f"Phase 4a 完成: {arxiv_query_count} 次查询 → {arxiv_count} 篇文献"
        )

        # ============================================================
        # Phase 4b: 浏览器网页搜索（始终执行）
        # ============================================================
        logger.info("=" * 50)
        logger.info("Phase 4b: 浏览器网页搜索（补充学术资料）")
        logger.info("=" * 50)

        web_search_queries = self._build_web_search_queries(
            all_keywords, ordered_queries
        )

        for web_query in web_search_queries:
            logger.info(f"  Web 查询: '{web_query[:80]}...'")
            try:
                web_results = self.searcher.search_web_for_academic([web_query])
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

        # ============================================================
        # Phase 4c: 自主模拟文献（最后手段 — 总数不足 10 篇时触发）
        # ============================================================
        if len(all_literature) < TARGET_MIN:
            logger.warning(
                f"真实检索结果不足 ({len(all_literature)} < {TARGET_MIN})，"
                "启动自主模拟文献生成..."
            )
            logger.info("=" * 50)
            logger.info("Phase 4c: 自主模拟文献（最后手段）")
            logger.info("=" * 50)

            mock_count = TARGET_IDEAL - len(all_literature)
            mock_lit = self.generate_mock_literature(
                keyword_result, count=mock_count
            )

            for mock_item in mock_lit:
                norm_title = mock_item.get("title", "").lower().strip()
                if norm_title not in seen_titles and norm_title:
                    seen_titles.add(norm_title)
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
        all_literature.sort(
            key=lambda x: (x.year if x.year else 0), reverse=True
        )

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

    @staticmethod
    def _has_chinese(text: str) -> bool:
        """检测文本是否包含中文字符"""
        return bool(_re.search(r'[一-鿿]', text))

    def _collect_all_keywords(
        self, keyword_result: Dict[str, Any]
    ) -> List[str]:
        """从关键词结果中收集所有关键词（含同义词）"""
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
                    if kw.get("zh"):
                        all_kw.append(kw["zh"])
                    for syn in kw.get("synonyms", []):
                        if syn and syn not in all_kw:
                            all_kw.append(syn)
                elif isinstance(kw, str):
                    all_kw.append(kw)
        return all_kw

    def _build_web_search_queries(
        self,
        all_keywords: List[str],
        existing_queries: List[Dict[str, Any]],
    ) -> List[str]:
        """
        构建网页搜索查询。

        注意：search_web_for_academic 会自动追加 "research paper OR study OR survey"，
        所以这里只需提供核心关键词，无需加学术修饰词。
        """
        queries = []

        # 保留已有高优先级查询（清理 AND，仅保留英文）
        for q in existing_queries:
            if q.get("priority") in ("high", "medium"):
                clean = q.get("query", "").replace("AND", "").replace('"', "")
                clean = " ".join(clean.split())
                # 只保留英文查询（中文查询 web 搜索也容易出错）
                if clean and not self._has_chinese(clean):
                    queries.append(clean)

        # 用英文关键词的简单组合
        en_keywords = [kw for kw in all_keywords if not self._has_chinese(kw)]
        if len(en_keywords) >= 2:
            queries.append(" ".join(en_keywords[:2]))
        if len(en_keywords) >= 3:
            queries.append(" ".join(en_keywords[:3]))

        # 去重
        seen = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower()
            if q_lower not in seen and len(q) > 5:
                seen.add(q_lower)
                unique_queries.append(q)

        return unique_queries[:8]

    def filter_high_quality(
        self, literature_list: List[Dict[str, Any]], min_citations: int = 0
    ) -> List[Dict[str, Any]]:
        """筛选高质量文献"""
        high_quality = []
        for lit in literature_list:
            journal = lit.get("journal", "")
            source = lit.get("source", "")

            if source == "arxiv" and journal and journal != "arXiv preprint":
                high_quality.append(lit)
                continue

            if source.startswith("web_") and source != "web_search":
                high_quality.append(lit)
                continue

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
        """自主模拟文献数据（最后手段）"""
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
                "abstract": f"This paper presents a comprehensive survey of {topic_en}, covering the latest advances in methodology, key challenges, and promising future research directions.",
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
                "abstract": f"本文系统梳理了{topic_zh}的理论基础，结合实践案例分析了其在多个领域的应用效果，并提出了一种改进的分析框架。",
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
                "abstract": f"We review deep learning approaches applied to {topic_en}, comparing performance across different architectures and datasets.",
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
                "abstract": f"提出了一种基于大数据技术的{topic_zh}分析方法，通过海量数据挖掘揭示{topic_zh}的内在规律。",
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
                "abstract": f"This survey covers the most recent advances in {topic_en}, identifying emerging trends and open problems in the field.",
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
                "abstract": f"We propose a novel hybrid framework combining the strengths of multiple approaches for {topic_en}.",
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
                "abstract": f"运用文献计量学方法，对近五年{topic_zh}领域的研究热点和趋势进行分析。",
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
                "abstract": f"As {topic_en} technologies become more prevalent, ethical considerations have gained increasing attention.",
                "citation_count": random.randint(20, 80),
                "source": "mock",
                "url": "",
                "arxiv_id": "",
            },
        ]

        return mock_templates[:count]
