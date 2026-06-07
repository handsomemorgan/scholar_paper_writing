"""
Web搜索模块 - arXiv 公开接口检索学术文献

使用 arXiv API (https://export.arxiv.org/api/query) 进行文献检索。
arXiv 是免费的开放获取预印本数据库，覆盖物理学、计算机科学、
数学、生物学、经济学等多个学科领域。

API 文档: https://info.arxiv.org/help/api/index.html
"""

import logging
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# arXiv API 基础 URL
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# arXiv 要求的速率限制：至少 3 秒一次请求
ARXIV_RATE_LIMIT = 3.0


@dataclass
class LiteratureItem:
    """文献条目"""
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    journal: Optional[str] = None
    abstract: str = ""
    citation_count: int = 0
    url: str = ""
    source: str = ""  # arxiv / cnki
    arxiv_id: str = ""  # arXiv 论文 ID，如 "2301.00001"
    primary_category: str = ""  # 主分类，如 "cs.AI"


class LiteratureSearcher:
    """学术文献检索器 — 多数据源（arXiv → Web → Fallback）"""

    def __init__(self, config: Dict):
        search_config = config.get("literature_search", {})
        self.sources = search_config.get("sources", ["arxiv"])
        self.max_results = search_config.get("max_results_per_source", 10)
        self.timeout = search_config.get("timeout_seconds", 30)
        # 是否启用浏览器网页搜索作为 fallback
        self.enable_web_fallback = search_config.get("enable_web_fallback", True)

    def search(self, keywords: List[str]) -> List[LiteratureItem]:
        """
        根据关键词列表检索文献。

        Args:
            keywords: 检索关键词列表

        Returns:
            去重合并后的文献列表
        """
        all_results: List[LiteratureItem] = []
        seen_titles = set()

        for source in self.sources:
            try:
                if source == "arxiv":
                    results = self._search_arxiv(keywords)
                elif source == "cnki":
                    results = self._search_cnki(keywords)
                elif source == "web":
                    results = self._search_web_academic(keywords)
                else:
                    logger.warning(f"Unknown source: {source}, skipping")
                    continue

                # 去重
                for item in results:
                    normalized_title = item.title.lower().strip()
                    if normalized_title not in seen_titles:
                        seen_titles.add(normalized_title)
                        all_results.append(item)

                logger.info(f"Retrieved {len(results)} results from {source}")

            except Exception as e:
                logger.error(f"Search failed for source {source}: {e}")
                continue

        logger.info(f"Total unique results: {len(all_results)}")
        return all_results

    def search_web_for_academic(
        self, keywords: List[str]
    ) -> List[LiteratureItem]:
        """
        通过浏览器搜索引擎查找学术文献。用作 arXiv 检索不足时的 fallback。

        搜索策略：
        - 在 DuckDuckGo 上搜索 "keyword research paper OR study OR survey"
        - 从搜索结果中提取标题、摘要片段和 URL
        - 尝试识别来自学术域名的结果（arxiv.org, semanticscholar.org 等）
        """
        return self._search_web_academic(keywords)

    def _search_web_academic(self, keywords: List[str]) -> List[LiteratureItem]:
        """
        通过 DuckDuckGo 搜索引擎查找学术文献。

        将多个关键词组合成搜索查询，尝试找到学术论文相关网页。
        """
        results: List[LiteratureItem] = []

        # 构建学术搜索查询
        query = " ".join(keywords)
        # 添加学术相关限定词
        academic_query = f"{query} research paper OR study OR survey"

        logger.info(f"Web 学术搜索: '{academic_query}'")

        try:
            import requests
            from bs4 import BeautifulSoup

            search_url = "https://html.duckduckgo.com/html/"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            }

            response = requests.post(
                search_url,
                data={"q": academic_query},
                headers=headers,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                logger.warning(
                    f"DuckDuckGo returned status {response.status_code}"
                )
                return results

            soup = BeautifulSoup(response.text, "lxml")
            result_items = soup.select(".result")

            for item in result_items[:self.max_results]:
                try:
                    title_el = item.select_one(".result__title a")
                    snippet_el = item.select_one(".result__snippet")
                    link_el = item.select_one(".result__url")

                    if not title_el:
                        continue

                    title = title_el.get_text(strip=True)
                    snippet = (
                        snippet_el.get_text(strip=True)
                        if snippet_el
                        else ""
                    )
                    url = (
                        link_el.get("href", "")
                        if link_el
                        else ""
                    )

                    # 过滤明显不是学术内容的结果
                    skip_patterns = [
                        "wikipedia.org", "youtube.com", "amazon.",
                        "reddit.com", "twitter.com", "facebook.",
                        "instagram.", "pinterest.", "ebay.",
                    ]
                    if any(p in url.lower() for p in skip_patterns):
                        continue

                    if not title:
                        continue

                    # 尝试从 URL 或标题推断年份
                    year = None
                    import re
                    year_match = re.search(r'(19|20)(\d{2})', title + url)
                    if year_match:
                        year = int(year_match.group(0))

                    # 识别来源
                    source = "web_search"
                    if "arxiv.org" in url:
                        source = "arxiv"
                    elif "semanticscholar.org" in url:
                        source = "web_semantic_scholar"
                    elif "researchgate.net" in url:
                        source = "web_researchgate"
                    elif any(
                        d in url
                        for d in [".edu", "springer.com", "ieee.org", "acm.org"]
                    ):
                        source = "web_academic"

                    results.append(
                        LiteratureItem(
                            title=title,
                            authors=[],  # 网页搜索不提供作者信息
                            year=year,
                            journal="",  # 网页搜索不提供期刊信息
                            abstract=snippet[:500] if snippet else "",
                            citation_count=0,
                            url=url,
                            source=source,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error parsing web search result: {e}")
                    continue

            logger.info(
                f"Web academic search returned {len(results)} results"
            )

        except ImportError as e:
            logger.error(
                f"Web search dependencies missing: {e}. "
                "Install with: pip install requests beautifulsoup4 lxml"
            )
        except Exception as e:
            logger.error(f"Web academic search error: {e}")

        return results

    def _build_arxiv_query(self, keywords: List[str]) -> str:
        """
        将用户友好的关键词列表转换为 arXiv API 查询语法。

        处理两种情况：
        1. 简单短语（无布尔运算符）→ all:"phrase"
        2. 含 AND/OR 的复杂查询 → (all:word1+AND+all:word2)+AND+(all:word3)

        Args:
            keywords: 关键词列表，如 ["deep learning AND image classification"]
                      或 ["deep learning", "image recognition"]

        Returns:
            arXiv API 格式的查询字符串
        """
        import re

        # 先用 " AND " 连接所有关键词（用户可能传多个）
        raw_query = " AND ".join(keywords)

        # 按 AND/OR 分割（保留运算符）
        # 使用正则：匹配两侧有空格的 AND/OR（不区分大小写）
        tokens = re.split(r'\s+(AND|OR)\s+', raw_query, flags=re.IGNORECASE)

        # tokens 现在交替为 [term, operator, term, operator, term, ...]
        arxiv_parts = []
        for token in tokens:
            upper = token.upper()
            if upper in ("AND", "OR"):
                arxiv_parts.append(upper)
            else:
                # 清理术语：去除引号和首尾空格
                term = token.strip().strip('"\'')
                if not term:
                    continue

                # 将术语拆分为单词
                words = term.split()
                if len(words) == 1:
                    # 单个词 → all:word
                    arxiv_parts.append(f"all:{words[0]}")
                else:
                    # 多个词 → (all:word1 AND all:word2 AND ...)
                    # 使用空格连接，urlencode 会自动转义
                    sub_query = " AND ".join(
                        f"all:{w}" for w in words
                    )
                    arxiv_parts.append(f"({sub_query})")

        return " ".join(arxiv_parts)

    def _search_arxiv(self, keywords: List[str]) -> List[LiteratureItem]:
        """
        使用 arXiv 公开 API 检索文献。

        arXiv API 完全免费开放，无需 API Key，无需注册。
        覆盖学科：物理学、计算机科学、数学、生物学、经济学、统计学等。

        速率限制：建议至少间隔 3 秒，请遵守 arXiv 使用政策。
        """
        results = []

        # 构建查询字符串
        # 支持两种输入：
        #   1. 含布尔运算符的复杂查询: "deep learning AND image classification"
        #   2. 简单关键词短语: "deep learning"
        query_string = self._build_arxiv_query(keywords)

        if not query_string:
            query_string = "all:research"

        # 构建请求参数
        params = {
            "search_query": query_string,
            "start": 0,
            "max_results": self.max_results,
            "sortBy": "relevance",
        }
        url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"

        logger.info(f"arXiv API 请求: {url}")

        try:
            # 速率限制
            time.sleep(ARXIV_RATE_LIMIT)

            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "PaperWritingAssistant/1.0",
                },
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                xml_data = response.read().decode("utf-8")

            # 解析 Atom XML 响应
            root = ET.fromstring(xml_data)

            # XML 命名空间
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }

            entries = root.findall("atom:entry", ns)
            logger.info(f"arXiv 返回 {len(entries)} 条结果")

            for entry in entries:
                try:
                    item = self._parse_arxiv_entry(entry, ns)
                    if item:
                        results.append(item)
                except Exception as e:
                    logger.warning(f"Error parsing arXiv entry: {e}")
                    continue

        except urllib.error.URLError as e:
            logger.error(f"arXiv API 网络请求失败: {e}")
        except ET.ParseError as e:
            logger.error(f"arXiv API 响应 XML 解析失败: {e}")
        except Exception as e:
            logger.error(f"arXiv search error: {e}")

        return results

    def _parse_arxiv_entry(
        self, entry: ET.Element, ns: Dict[str, str]
    ) -> Optional[LiteratureItem]:
        """解析单个 arXiv Atom entry 为 LiteratureItem"""
        # 标题
        title_el = entry.find("atom:title", ns)
        title = title_el.text.strip().replace("\n", " ") if title_el is not None and title_el.text else ""
        if not title:
            return None

        # 作者
        authors = []
        for author_el in entry.findall("atom:author", ns):
            name_el = author_el.find("atom:name", ns)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        # 摘要
        summary_el = entry.find("atom:summary", ns)
        abstract = summary_el.text.strip().replace("\n", " ") if summary_el is not None and summary_el.text else ""

        # 出版日期（提取年份）
        published_el = entry.find("atom:published", ns)
        year = None
        if published_el is not None and published_el.text:
            try:
                year = int(published_el.text[:4])
            except (ValueError, IndexError):
                pass

        # arXiv ID 和 URL
        id_el = entry.find("atom:id", ns)
        arxiv_id = ""
        url = ""
        if id_el is not None and id_el.text:
            full_url = id_el.text.strip()
            url = full_url
            # 从 URL 中提取 arXiv ID
            # 格式: http://arxiv.org/abs/2301.00001v1
            arxiv_id = full_url.split("/abs/")[-1] if "/abs/" in full_url else ""

        # 主分类
        primary_cat_el = entry.find("arxiv:primary_category", ns)
        primary_category = ""
        if primary_cat_el is not None:
            primary_category = primary_cat_el.get("term", "")

        # 期刊/会议信息（arXiv 的 comment 字段或 journal-ref）
        journal = ""
        journal_ref_el = entry.find("arxiv:journal_ref", ns)
        if journal_ref_el is not None and journal_ref_el.text:
            journal = journal_ref_el.text.strip()

        # 评论信息（可能包含发表信息）
        comment_el = entry.find("arxiv:comment", ns)
        comment = comment_el.text.strip() if comment_el is not None and comment_el.text else ""

        # citation_count 在 arXiv 中不可用，设为 0
        # （可以通过 Semantic Scholar API 补充，但不是必须的）

        return LiteratureItem(
            title=title,
            authors=authors,
            year=year,
            journal=journal or comment or "arXiv preprint",
            abstract=abstract,
            citation_count=0,  # arXiv 不提供引用数
            url=url,
            source="arxiv",
            arxiv_id=arxiv_id,
            primary_category=primary_category,
        )

    def _search_cnki(self, keywords: List[str]) -> List[LiteratureItem]:
        """
        检索中国知网(CNKI) — 备用数据源。

        注意：CNKI有严格的反爬机制。实际使用时建议：
        1. 使用CNKI官方API（需机构授权）
        2. 使用学校图书馆的CNKI代理
        3. 默认不使用此数据源（主要使用 arXiv）
        """
        results = []

        try:
            import requests
            from bs4 import BeautifulSoup

            query = " ".join(keywords)
            search_url = "https://kns.cnki.net/kns8/defaultresult/index"

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            }

            params = {
                "kwd": query,
                "dbcode": "CJFD",
            }

            response = requests.get(
                search_url,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "lxml")
                result_items = soup.select(".result-item, tr.result-item")

                for item in result_items[:self.max_results]:
                    try:
                        title_el = item.select_one(".title, a.title")
                        title = title_el.get_text(strip=True) if title_el else ""

                        authors_el = item.select_one(".author, .authors")
                        authors_text = authors_el.get_text(strip=True) if authors_el else ""
                        authors = [a.strip() for a in authors_text.split(";")]

                        journal_el = item.select_one(".source, .journal")
                        journal = journal_el.get_text(strip=True) if journal_el else ""

                        year_el = item.select_one(".year, .date")
                        year = None
                        if year_el:
                            year_text = year_el.get_text(strip=True)
                            try:
                                year = int(year_text[:4])
                            except ValueError:
                                pass

                        abstract_el = item.select_one(".abstract, .summary")
                        abstract = abstract_el.get_text(strip=True) if abstract_el else ""

                        if title:
                            results.append(
                                LiteratureItem(
                                    title=title,
                                    authors=authors,
                                    year=year,
                                    journal=journal,
                                    abstract=abstract,
                                    source="cnki",
                                )
                            )
                    except Exception as e:
                        logger.warning(f"Error parsing CNKI result: {e}")
                        continue

            else:
                logger.warning(
                    f"CNKI returned status {response.status_code}"
                )

        except ImportError as e:
            logger.error(f"Missing dependency for CNKI search: {e}")
        except Exception as e:
            logger.error(f"CNKI search error: {e}")

        return results


class WebSearcher:
    """通用网页搜索 — 作为文献检索的补充"""

    @staticmethod
    def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        执行通用网页搜索。

        注意：生产环境建议使用 SerpAPI、Bing Search API 等。
        此实现为占位，可替换为实际搜索API。
        """
        results = []

        try:
            import requests
            from bs4 import BeautifulSoup

            search_url = "https://html.duckduckgo.com/html/"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36"
                )
            }

            response = requests.post(
                search_url,
                data={"q": query},
                headers=headers,
                timeout=10,
            )

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "lxml")
                result_items = soup.select(".result")

                for item in result_items[:max_results]:
                    title_el = item.select_one(".result__title a")
                    snippet_el = item.select_one(".result__snippet")
                    link_el = item.select_one(".result__url")

                    if title_el:
                        results.append({
                            "title": title_el.get_text(strip=True),
                            "snippet": (
                                snippet_el.get_text(strip=True)
                                if snippet_el
                                else ""
                            ),
                            "url": (
                                link_el.get("href", "")
                                if link_el
                                else ""
                            ),
                        })

        except Exception as e:
            logger.error(f"Web search failed: {e}")

        return results
