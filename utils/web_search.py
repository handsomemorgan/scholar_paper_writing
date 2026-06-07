"""
Web搜索模块 - 谷歌学术、中国知网等学术数据库检索

使用 scholarly 库访问 Google Scholar，
使用 requests + BeautifulSoup 抓取 CNKI 等中文数据库。
"""

import logging
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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
    source: str = ""  # google_scholar / cnki


class LiteratureSearcher:
    """学术文献检索器 — 支持多数据源"""

    def __init__(self, config: Dict):
        search_config = config.get("literature_search", {})
        self.sources = search_config.get("sources", ["google_scholar"])
        self.max_results = search_config.get("max_results_per_source", 10)
        self.timeout = search_config.get("timeout_seconds", 30)

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
                if source == "google_scholar":
                    results = self._search_google_scholar(keywords)
                elif source == "cnki":
                    results = self._search_cnki(keywords)
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

    def _search_google_scholar(self, keywords: List[str]) -> List[LiteratureItem]:
        """
        使用 scholarly 库检索 Google Scholar。

        注意：scholarly 可能因 Google 反爬机制而不稳定，
        生产环境建议使用 SerpAPI 或官方 Google Scholar API。
        """
        results = []
        query = " ".join(keywords)

        try:
            from scholarly import scholarly

            search_query = scholarly.search_pubs(query)
            count = 0

            for pub in search_query:
                if count >= self.max_results:
                    break

                try:
                    item = LiteratureItem(
                        title=pub.get("bib", {}).get("title", "Unknown"),
                        authors=pub.get("bib", {}).get("author", []),
                        year=pub.get("bib", {}).get("pub_year", None),
                        journal=pub.get("bib", {}).get("journal", ""),
                        abstract=pub.get("bib", {}).get("abstract", ""),
                        citation_count=pub.get("num_citations", 0),
                        url=pub.get("pub_url", pub.get("eprint_url", "")),
                        source="google_scholar",
                    )
                    results.append(item)
                    count += 1

                    # 速率限制
                    time.sleep(1)

                except Exception as e:
                    logger.warning(f"Error parsing a Google Scholar result: {e}")
                    continue

        except ImportError:
            logger.error(
                "scholarly not installed. Install with: pip install scholarly"
            )
        except Exception as e:
            logger.error(f"Google Scholar search error: {e}")

        return results

    def _search_cnki(self, keywords: List[str]) -> List[LiteratureItem]:
        """
        检索中国知网(CNKI)。

        注意：CNKI有严格的反爬机制。以下为模拟实现，
        实际使用时建议：
        1. 使用CNKI官方API（需机构授权）
        2. 使用学校图书馆的CNKI代理
        3. 考虑使用CNKI镜像站点
        """
        results = []

        try:
            import requests
            from bs4 import BeautifulSoup

            query = " ".join(keywords)
            # CNKI 搜索URL（此为模拟，实际URL可能变化）
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
                "dbcode": "CJFD",  # 期刊论文
            }

            response = requests.get(
                search_url,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "lxml")
                # CNKI页面解析（具体选择器需根据实际页面结构调整）
                result_items = soup.select(".result-item, tr.result-item")

                for item in result_items[: self.max_results]:
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
        except requests.exceptions.RequestException as e:
            logger.error(f"CNKI request failed: {e}")
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

            # DuckDuckGo 作为备选（比 Google 更容易抓取）
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
