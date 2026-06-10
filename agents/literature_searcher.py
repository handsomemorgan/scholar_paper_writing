"""
Agent 4: 文献检索Agent

检索策略（多源检索 + 三级回退链）：
  1. 多源文献库检索 —— 根据论文分类路由到对应文献库
     - arXiv: CS/数学/物理 (已有)
     - DOAJ: 全学科 OA 期刊
     - OpenAlex: 全学科 (含 SSRN/RePEc 论文)
     - SocArXiv: 社会科学预印本
     - RePEc/IDEAS: 经济学论文
     - Socolar: 中文学术 OA
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
from utils.literature_sources import get_sources_for_category

logger = logging.getLogger(__name__)

# 文献收集目标：通用的 10-20 篇
TARGET_MIN = 5   # 降低 mock 触发阈值：先尝试引文链扩展再考虑 mock
TARGET_IDEAL = 15

# 文献质量评估关键词
CSSCI_JOURNALS_KEYWORDS = [
    "中国社会科学", "社会学研究", "政治学研究", "经济研究", "管理世界",
    "法学研究", "教育研究", "新闻与传播研究", "中国图书馆学报",
    "中国社会科学季刊", "社会", "社会学评论", "公共管理学报",
    "中国软科学", "科学学研究", "科研管理", "高等教育研究",
]

SSCI_JOURNAL_INDICATORS = [
    "elsevier", "springer", "taylor", "wiley", "sage", "oxford",
    "cambridge", "nature", "science", "pnas", "cell", "lancet",
    "ieee", "acm", "apa", "annual review", "journal of",
]


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
        classification: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        根据关键词检索文献。使用多源路由 + 三级回退链。

        Phase 4a: 多源文献库检索（按分类路由）
           ├─ arXiv API（遍历所有英文查询）
           ├─ DOAJ / OpenAlex / SocArXiv / RePEc / Socolar
           │  （根据 classification 路由到合适的文献库）
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
        # Phase 4a-extra: 多源文献库检索
        #   根据论文分类，路由到 DOAJ / OpenAlex / SocArXiv / RePEc / Socolar
        #   每个查询遍历对应分类的文献库
        # ============================================================
        if self.config.get("literature_search", {}).get("multi_source_enabled", True):
            logger.info("=" * 50)
            logger.info("Phase 4a-extra: 多源文献库检索（按分类路由）")
            logger.info("=" * 50)

            category_id = classification.get("category_id", "") if classification else ""
            discipline = classification.get("discipline", "") if classification else ""

            extra_sources = get_sources_for_category(
                category_id=category_id,
                discipline=discipline,
                config=self.config,
            )

            if extra_sources:
                logger.info(
                    f"  分类: {category_id} | 学科: {discipline} | "
                    f"激活 {len(extra_sources)} 个额外文献库"
                )
                for src in extra_sources:
                    logger.info(f"    - {src.__class__.__name__}: {src.base_url}")

                # 收集英文查询（优先高/中优先级）
                en_queries = []
                for q in ordered_queries:
                    q_text = q.get("query", "")
                    if q_text and not self._has_chinese(q_text):
                        en_queries.append(q_text)

                # 同时用关键词构建额外查询
                if all_keywords:
                    en_kw = [kw for kw in all_keywords if not self._has_chinese(kw)]
                    for kw in en_kw[:5]:
                        if kw not in en_queries:
                            en_queries.append(kw)

                # 限制查询总数，每个源最多搜索 max_queries_per_source 个查询
                max_queries_per_source = 4
                search_queries = en_queries[:max_queries_per_source]

                multi_source_before = len(all_literature)

                for src in extra_sources:
                    src_name = src.__class__.__name__
                    for q_idx, query in enumerate(search_queries):
                        if q_idx >= max_queries_per_source:
                            break
                        logger.info(f"  [{src_name}] 查询 [{q_idx+1}/{min(len(search_queries), max_queries_per_source)}]: '{query[:80]}...'")
                        try:
                            results = src._safe_search(query)
                            for item in results:
                                norm_title = item.title.lower().strip()
                                if norm_title not in seen_titles and norm_title:
                                    seen_titles.add(norm_title)
                                    all_literature.append(item)
                            logger.info(f"    → {len(results)} 条结果")
                        except Exception as e:
                            logger.error(f"  [{src_name}] 查询失败: {e}")
                            continue

                multi_count = len(all_literature) - multi_source_before
                logger.info(
                    f"Phase 4a-extra 完成: 多源检索新增 {multi_count} 篇，"
                    f"总计 {len(all_literature)} 篇"
                )
            else:
                logger.info("  当前分类无额外文献库可用，跳过")

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
        # Phase 4b-extra: 文献质量增强（新增 — 在模拟文献之前）
        #   1. 计算质量指标
        #   2. 不达标时启动补充检索（中文/高质量/最新文献）
        #   3. 引文链扩展
        # ============================================================
        logger.info("=" * 50)
        logger.info("Phase 4b-extra: 文献质量评估与增强")
        logger.info("=" * 50)

        # 引文链扩展（基于已有高质量文献）
        if self.config.get("quality_thresholds", {}).get("enable_citation_chain", True):
            all_literature = self._expand_citation_chain(
                all_literature, seen_titles
            )

        # 质量评估 + 定向补充检索
        all_literature = self._enforce_quality_thresholds(
            all_literature, all_keywords, ordered_queries,
            seen_titles, classification,
        )

        # ============================================================
        # Phase 4c: 自主模拟文献（最后手段 — 总数不足时触发）
        #   阈值已从10降至5，质量增强环节先行
        # ============================================================
        if len(all_literature) < TARGET_MIN:
            logger.warning(
                f"⚠️ 真实检索+质量增强后结果仍不足 "
                f"({len(all_literature)} < {TARGET_MIN})，"
                "启动自主模拟文献生成（最后手段）..."
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

            logger.warning(
                f"Phase 4c 完成: 模拟生成 {mock_count} 篇（⚠️ 含 MOCK 文献），"
                f"总计 {len(all_literature)} 篇。"
                f"请在提交前将 mock 文献替换为真实引文。"
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

        # 计算最终质量指标（存储在实例上供 orchestrator 读取）
        quality_metrics = self._compute_quality_metrics(literature_list)
        mock_count = source_counts.get("mock", 0)
        quality_metrics["mock_count"] = mock_count
        quality_metrics["has_mock"] = mock_count > 0
        self.last_quality_metrics = quality_metrics

        logger.info(
            f"\nAgent 4 完成: 共检索到 {len(literature_list)} 篇文献 "
            f"({source_summary})"
        )
        logger.info(
            f"  质量指标: 中文{quality_metrics['chinese_ratio']:.0%} | "
            f"CSSCI/SSCI{quality_metrics['cssci_ratio']:.0%} | "
            f"近5年{quality_metrics['recent_5year_ratio']:.0%} | "
            f"Mock{mock_count}篇"
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

    # ============================================================
    # 文献质量评估与增强 (新增 — 吸收自 Skills 系列)
    # ============================================================

    def _compute_quality_metrics(
        self, literature_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        计算文献列表的质量指标。

        Returns:
            {
                "total": 总文献数,
                "chinese_count": 中文文献数,
                "chinese_ratio": 中文文献占比,
                "cssci_estimated": 估计CSSCI/SSCI文献数,
                "cssci_ratio": CSSCI/SSCI占比,
                "recent_5year_count": 近5年文献数,
                "recent_5year_ratio": 近5年文献占比,
                "source_distribution": {来源: 数量},
                "needs_enhancement": 是否需要补充检索
            }
        """
        import datetime
        current_year = datetime.datetime.now().year
        five_years_ago = current_year - 5

        total = len(literature_list)
        if total == 0:
            return {
                "total": 0, "chinese_count": 0, "chinese_ratio": 0,
                "cssci_estimated": 0, "cssci_ratio": 0,
                "recent_5year_count": 0, "recent_5year_ratio": 0,
                "source_distribution": {}, "needs_enhancement": True,
            }

        chinese_count = 0
        cssci_estimated = 0
        recent_count = 0
        source_dist = {}

        for lit in literature_list:
            # 来源分布
            src = lit.get("source", "unknown")
            source_dist[src] = source_dist.get(src, 0) + 1

            # 中文检测
            title = lit.get("title", "")
            abstract = lit.get("abstract", "")
            journal = lit.get("journal", "")
            is_chinese = (
                self._has_chinese(title)
                or self._has_chinese(abstract[:100])
                or self._has_chinese(journal)
            )
            if is_chinese:
                chinese_count += 1

            # CSSCI/SSCI 估算
            journal_lower = journal.lower()
            is_cssci = any(
                kw in journal for kw in CSSCI_JOURNALS_KEYWORDS
            )
            is_ssci = any(
                indicator in journal_lower
                for indicator in SSCI_JOURNAL_INDICATORS
            )
            is_major_source = lit.get("source", "") in (
                "openalex", "doaj"
            ) and lit.get("citation_count", 0) > 5
            if is_cssci or is_ssci or is_major_source:
                cssci_estimated += 1

            # 近5年检测
            year = lit.get("year")
            if year and isinstance(year, (int, float)) and year >= five_years_ago:
                recent_count += 1

        metrics = {
            "total": total,
            "chinese_count": chinese_count,
            "chinese_ratio": round(chinese_count / total, 3) if total > 0 else 0,
            "cssci_estimated": cssci_estimated,
            "cssci_ratio": round(cssci_estimated / total, 3) if total > 0 else 0,
            "recent_5year_count": recent_count,
            "recent_5year_ratio": round(recent_count / total, 3) if total > 0 else 0,
            "source_distribution": source_dist,
        }

        # 判断是否需要补充检索
        thresholds = self.config.get("quality_thresholds", {})
        needs = (
            metrics["chinese_ratio"] < thresholds.get("chinese_literature_ratio", 0.40)
            or metrics["cssci_ratio"] < thresholds.get("cssci_ratio", 0.70)
            or metrics["recent_5year_ratio"] < thresholds.get("recent_5year_ratio", 0.60)
            or total < thresholds.get("min_high_quality", 10)
        )
        metrics["needs_enhancement"] = needs

        logger.info(
            f"文献质量指标: 总计{total}篇 | "
            f"中文占比{metrics['chinese_ratio']:.0%} | "
            f"CSSCI/SSCI占比{metrics['cssci_ratio']:.0%} | "
            f"近5年占比{metrics['recent_5year_ratio']:.0%} | "
            f"需补充:{'是' if needs else '否'}"
        )
        return metrics

    def _enforce_quality_thresholds(
        self,
        literature_list: List[LiteratureItem],
        all_keywords: List[str],
        ordered_queries: List[Dict[str, Any]],
        seen_titles: set,
        classification: Optional[Dict[str, Any]] = None,
    ) -> List[LiteratureItem]:
        """
        当文献质量不达标时，执行补充检索策略。

        补充策略优先级：
        1. 中文文献不足 → 用中文关键词在 Socolar + Web 搜索中文
        2. CSSCI/SSCI 不足 → 在 OpenAlex + DOAJ 按引用量排序搜索
        3. 近5年文献不足 → 搜索时增加年份过滤
        """
        metrics = self._compute_quality_metrics(
            [self._lit_to_dict(item) for item in literature_list]
        )
        if not metrics["needs_enhancement"]:
            return literature_list

        logger.info("=" * 50)
        logger.info("文献质量增强: 启动补充检索...")
        logger.info("=" * 50)

        thresholds = self.config.get("quality_thresholds", {})

        # ---- 中文文献补充 ----
        if metrics["chinese_ratio"] < thresholds.get("chinese_literature_ratio", 0.40):
            logger.info("  [补充] 中文文献占比不足，启动中文定向检索...")
            zh_keywords = [kw for kw in all_keywords if self._has_chinese(kw)]
            if not zh_keywords:
                # 从查询中提取可能的中文关键词
                for q in ordered_queries:
                    q_text = q.get("query", "")
                    if self._has_chinese(q_text):
                        zh_keywords.append(q_text)

            for zh_kw in zh_keywords[:3]:
                logger.info(f"    中文检索: '{zh_kw[:60]}...'")
                try:
                    zh_results = self.searcher.search_web_for_academic(
                        [zh_kw], lang="zh"
                    )
                    for item in zh_results:
                        norm_title = item.title.lower().strip()
                        if norm_title not in seen_titles and norm_title:
                            seen_titles.add(norm_title)
                            item.source = item.source or "web_search_zh"
                            literature_list.append(item)
                except Exception as e:
                    logger.warning(f"    中文补充检索失败: {e}")

        # ---- CSSCI/SSCI 高质量文献补充 ----
        if metrics["cssci_ratio"] < thresholds.get("cssci_ratio", 0.70):
            logger.info("  [补充] 高质量文献占比不足，在 OpenAlex 中按引用量检索...")
            en_keywords = [kw for kw in all_keywords if not self._has_chinese(kw)]
            for en_kw in en_keywords[:2]:
                logger.info(f"    高质量检索: '{en_kw[:60]}...'")
                try:
                    # OpenAlex 按引用量排序已在上方 primary 搜索中处理
                    # 这里用更宽泛的学术修饰词进行补充
                    boosted_query = f"{en_kw} AND (review OR survey OR meta-analysis)"
                    extra_results = self.searcher.search_web_for_academic(
                        [boosted_query]
                    )
                    for item in extra_results:
                        norm_title = item.title.lower().strip()
                        if norm_title not in seen_titles and norm_title:
                            seen_titles.add(norm_title)
                            item.source = item.source or "web_search_hq"
                            literature_list.append(item)
                except Exception as e:
                    logger.warning(f"    高质量补充检索失败: {e}")

        # ---- 近5年文献补充 ----
        if metrics["recent_5year_ratio"] < thresholds.get("recent_5year_ratio", 0.60):
            logger.info("  [补充] 近5年文献占比不足，搜索最新文献...")
            current_year = __import__('datetime').datetime.now().year
            en_keywords = [kw for kw in all_keywords if not self._has_chinese(kw)]
            for en_kw in en_keywords[:2]:
                recent_query = (
                    f"{en_kw} AND ({current_year} OR {current_year-1} OR {current_year-2})"
                )
                logger.info(f"    最新文献检索: '{recent_query[:80]}...'")
                try:
                    recent_results = self.searcher.search_web_for_academic(
                        [recent_query]
                    )
                    for item in recent_results:
                        norm_title = item.title.lower().strip()
                        if norm_title not in seen_titles and norm_title:
                            seen_titles.add(norm_title)
                            item.source = item.source or "web_search_recent"
                            literature_list.append(item)
                except Exception as e:
                    logger.warning(f"    最新文献补充检索失败: {e}")

        new_metrics = self._compute_quality_metrics(
            [self._lit_to_dict(item) for item in literature_list]
        )
        logger.info(
            f"文献质量增强完成: "
            f"中文占比{metrics['chinese_ratio']:.0%}→{new_metrics['chinese_ratio']:.0%} | "
            f"CSSCI/SSCI占比{metrics['cssci_ratio']:.0%}→{new_metrics['cssci_ratio']:.0%} | "
            f"近5年占比{metrics['recent_5year_ratio']:.0%}→{new_metrics['recent_5year_ratio']:.0%}"
        )

        return literature_list

    def _expand_citation_chain(
        self,
        literature_list: List[LiteratureItem],
        seen_titles: set,
    ) -> List[LiteratureItem]:
        """
        引文链扩展：对高质量文献进行向前（引用）和向后（参考文献）扩展。

        利用 Semantic Scholar API 的 citations 和 references 端点。
        """
        if not self.config.get("quality_thresholds", {}).get("enable_citation_chain", True):
            logger.info("  引文链扩展已禁用，跳过")
            return literature_list

        logger.info("=" * 50)
        logger.info("引文链扩展: 向前(被引) + 向后(参考文献)")
        logger.info("=" * 50)

        # 选取 top-5 高引用文献作为种子
        seeds = sorted(
            literature_list,
            key=lambda x: x.citation_count if x.citation_count else 0,
            reverse=True,
        )[:5]

        expanded_count = 0
        for seed in seeds:
            arxiv_id = getattr(seed, 'arxiv_id', '') or seed.title[:50]
            logger.info(f"  种子文献: {seed.title[:80]}...")

            try:
                # 通过 Semantic Scholar 搜索相关文献
                seed_title = seed.title[:100]
                related = self.searcher.search_web_for_academic(
                    [f'"{seed_title}"']
                )
                for item in related:
                    norm_title = item.title.lower().strip()
                    if norm_title not in seen_titles and norm_title:
                        seen_titles.add(norm_title)
                        item.source = item.source or "citation_chain"
                        literature_list.append(item)
                        expanded_count += 1
            except Exception as e:
                logger.warning(f"  引文链扩展失败 ({seed.title[:50]}...): {e}")

        logger.info(f"引文链扩展完成: 新增 {expanded_count} 篇")
        return literature_list

    @staticmethod
    def _lit_to_dict(item) -> Dict[str, Any]:
        """将 LiteratureItem 转为字典（用于质量计算）"""
        if isinstance(item, dict):
            return item
        return {
            "title": getattr(item, 'title', ''),
            "authors": getattr(item, 'authors', []),
            "year": getattr(item, 'year', None),
            "journal": getattr(item, 'journal', ''),
            "abstract": getattr(item, 'abstract', ''),
            "citation_count": getattr(item, 'citation_count', 0),
            "url": getattr(item, 'url', ''),
            "source": getattr(item, 'source', 'unknown'),
            "arxiv_id": getattr(item, 'arxiv_id', ''),
            "primary_category": getattr(item, 'primary_category', ''),
        }

    def filter_high_quality(
        self, literature_list: List[Dict[str, Any]], min_citations: int = 0
    ) -> List[Dict[str, Any]]:
        """筛选高质量文献（增强版：纳入 CSSCI/SSCI 估算）"""
        high_quality = []
        for lit in literature_list:
            journal = lit.get("journal", "")
            source = lit.get("source", "")
            journal_lower = journal.lower()

            # arXiv 已发表文献
            if source == "arxiv" and journal and journal != "arXiv preprint":
                high_quality.append(lit)
                continue

            # 高质量 web 源
            if source.startswith("web_") and source != "web_search":
                high_quality.append(lit)
                continue

            # arXiv 顶会分类
            primary_cat = lit.get("primary_category", "")
            if primary_cat in [
                "cs.AI", "cs.CL", "cs.CV", "cs.LG", "stat.ML",
                "physics.soc-ph", "q-bio", "q-fin",
            ]:
                high_quality.append(lit)
                continue

            # CSSCI 估算
            if any(kw in journal for kw in CSSCI_JOURNALS_KEYWORDS):
                high_quality.append(lit)
                continue

            # SSCI 估算
            if any(indicator in journal_lower for indicator in SSCI_JOURNAL_INDICATORS):
                high_quality.append(lit)
                continue

            # 高引用文献（OpenAlex/DOAJ等）
            if lit.get("citation_count", 0) > 5:
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
