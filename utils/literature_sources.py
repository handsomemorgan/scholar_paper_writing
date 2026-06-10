"""
多源学术文献检索模块

支持的文献库:
  - arXiv (已有，通过 web_search.py)
  - Semantic Scholar (已有，通过 web_search.py)
  - DOAJ — 开放获取期刊目录 API v4
  - OpenAlex — 统一开放学术 API (代理 SSRN/RePEc 内容)
  - SocArXiv — 社会科学预印本 (OSF API v2)
  - RePEc/IDEAS — 经济学论文 (HTML 解析)
  - Socolar — 中文学术 OA 统一检索 (HTML 解析)

使用方式:
    from utils.literature_sources import get_sources_for_category

    sources = get_sources_for_category("liberal_arts_paper", "经济学")
    for source in sources:
        results = source.search("income inequality AND tax policy", max_results=20)
"""

import logging
import re
import time
import urllib.parse
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from utils.web_search import LiteratureItem

logger = logging.getLogger(__name__)


# ============================================================
# 分类 → 文献库映射
# ============================================================

# primary: 该分类优先使用的文献库
# secondary: 该分类可选的补充文献库（如特定学科触发）
CATEGORY_SOURCE_MAP = {
    "science_paper": {
        "primary": ["arxiv", "doaj"],
        "secondary": ["openalex"],
    },
    "liberal_arts_paper": {
        "primary": ["doaj", "openalex", "socarxiv"],
        "secondary": ["repec", "socolar"],
    },
    "engineering_paper": {
        "primary": ["arxiv", "doaj"],
        "secondary": ["openalex"],
    },
    "research_report": {
        "primary": ["doaj", "openalex", "socarxiv"],
        "secondary": ["repec", "socolar"],
    },
    "lab_report": {
        "primary": ["arxiv", "doaj"],
        "secondary": ["openalex"],
    },
}

# 学科关键词 → 额外激活的文献库
DISCIPLINE_EXTRA_SOURCES = {
    # 经济学 → 激活 RePEc
    "经济": ["repec"],
    "金融": ["repec"],
    "贸易": ["repec"],
    "宏观经济": ["repec"],
    "微观经济": ["repec"],
    "economy": ["repec"],
    "finance": ["repec"],
    "economics": ["repec"],
    # 社会科学 → 激活 SocArXiv
    "社会": ["socarxiv"],
    "政治": ["socarxiv"],
    "法律": ["socarxiv"],
    "教育": ["socarxiv"],
    "心理": ["socarxiv"],
    "sociology": ["socarxiv"],
    "political": ["socarxiv"],
    "law": ["socarxiv"],
    "education": ["socarxiv"],
    "psychology": ["socarxiv"],
    # 中国相关 → 激活 Socolar
    "中国": ["socolar"],
    "中文": ["socolar"],
    "china": ["socolar"],
}


def get_sources_for_category(
    category_id: str,
    discipline: str = "",
    config: Optional[Dict] = None,
) -> List["BaseLiteratureSource"]:
    """
    根据论文分类和学科，返回应使用的文献库列表。

    Args:
        category_id: 分类ID (science_paper / liberal_arts_paper / ...)
        discipline: 具体学科名称（可选，用于激活额外文献库）
        config: 配置字典（可选，用于读取各源的参数）

    Returns:
        BaseLiteratureSource 实例列表
    """
    mapping = CATEGORY_SOURCE_MAP.get(category_id, {})
    primary = mapping.get("primary", ["arxiv", "doaj"])
    secondary = list(mapping.get("secondary", []))

    # 根据学科关键词激活额外文献库
    if discipline:
        discipline_lower = discipline.lower()
        for keyword, extra_sources in DISCIPLINE_EXTRA_SOURCES.items():
            if keyword.lower() in discipline_lower:
                for src in extra_sources:
                    if src not in primary and src not in secondary:
                        secondary.append(src)

    all_source_names = primary + secondary

    # 实例化文献库
    sources = []
    source_configs = {}
    if config:
        source_configs = config.get("literature_search", {}).get("source_configs", {})

    for name in all_source_names:
        src_config = source_configs.get(name, {})
        source_instance = _create_source(name, src_config)
        if source_instance:
            sources.append(source_instance)

    return sources


def _create_source(name: str, config: Dict) -> Optional["BaseLiteratureSource"]:
    """工厂方法：根据名称创建文献库实例"""
    if name == "doaj":
        return DOAJSource(config)
    elif name == "openalex":
        return OpenAlexSource(config)
    elif name == "socarxiv":
        return SocArXivSource(config)
    elif name == "repec":
        return RePEcSource(config)
    elif name == "socolar":
        return SocolarSource(config)
    elif name == "arxiv":
        # arXiv 由 LiteratureSearcher 通过 web_search.py 直接处理
        return None
    else:
        logger.warning(f"未知文献库: {name}")
        return None


# ============================================================
# 抽象基类
# ============================================================

class BaseLiteratureSource(ABC):
    """文献库基类"""

    def __init__(self, config: Dict):
        self.config = config
        self.max_results = config.get("max_results", 20)
        self.timeout = config.get("timeout", 30)
        self.base_url = config.get("base_url", "")

    @abstractmethod
    def search(self, query: str, max_results: int = None) -> List[LiteratureItem]:
        """
        检索文献。

        Args:
            query: 检索查询字符串（英文）
            max_results: 最大返回数量

        Returns:
            LiteratureItem 列表
        """
        ...

    def _safe_search(self, query: str, max_results: int = None) -> List[LiteratureItem]:
        """带异常保护的搜索包装器"""
        try:
            return self.search(query, max_results)
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] 搜索失败: {e}")
            return []


# ============================================================
# DOAJ — Directory of Open Access Journals
# API: https://doaj.org/api/v4/docs
# ============================================================

class DOAJSource(BaseLiteratureSource):
    """DOAJ 文献库 — 免费开放获取期刊"""

    def search(self, query: str, max_results: int = None) -> List[LiteratureItem]:
        max_res = max_results or self.max_results

        # DOAJ 支持 Elasticsearch 查询语法
        # 清理查询中的特殊字符
        clean_query = query.replace("AND", "").replace("OR", "").replace('"', "")
        clean_query = " ".join(clean_query.split())  # 压缩空白

        # DOAJ search endpoint: /api/search/articles/{query}
        url = f"{self.base_url}/{urllib.parse.quote(clean_query)}"
        params = {
            "pageSize": min(max_res, 100),  # DOAJ 最大 100/页
        }
        # 注意：sort 参数在 search endpoint 格式为 bibjson.year:desc
        # 但某些版本可能不支持，先不加 sort

        logger.info(f"[DOAJ] 检索: '{clean_query[:80]}...' (max={max_res})")

        try:
            import requests
            headers = {
                "User-Agent": "PaperWritingAssistant/1.0 (mailto:research@example.com)",
                "Accept": "application/json",
            }
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            if response.status_code != 200:
                logger.warning(f"[DOAJ] HTTP {response.status_code}")
                return []

            data = response.json()
            results = data.get("results", [])
            logger.info(f"[DOAJ] 返回 {len(results)} 条结果")

            items = []
            for paper in results:
                try:
                    bibjson = paper.get("bibjson", {})
                    title = bibjson.get("title", "")
                    if not title:
                        continue

                    # 作者
                    authors_raw = bibjson.get("author", [])
                    authors = []
                    for a in authors_raw:
                        name = a.get("name", "") if isinstance(a, dict) else str(a)
                        if name:
                            authors.append(name)

                    # 年份
                    year = None
                    year_str = bibjson.get("year", "")
                    if year_str:
                        try:
                            year = int(str(year_str)[:4])
                        except (ValueError, TypeError):
                            pass

                    # 期刊
                    journal_info = bibjson.get("journal", {})
                    journal = journal_info.get("title", "") if isinstance(journal_info, dict) else ""

                    # 摘要
                    abstract = ""
                    abstract_raw = bibjson.get("abstract", "")
                    if abstract_raw:
                        abstract = abstract_raw[:500]

                    # URL
                    article_url = ""
                    for link in bibjson.get("link", []):
                        if isinstance(link, dict) and link.get("type") == "fulltext":
                            article_url = link.get("url", "")
                            break
                    if not article_url:
                        doi = bibjson.get("identifier", [{}])[0].get("id", "") if bibjson.get("identifier") else ""
                        if doi:
                            article_url = f"https://doi.org/{doi}"

                    items.append(LiteratureItem(
                        title=title,
                        authors=authors[:10],
                        year=year,
                        journal=journal,
                        abstract=abstract,
                        citation_count=0,
                        url=article_url,
                        source="doaj",
                    ))
                except Exception as e:
                    logger.warning(f"[DOAJ] 解析条目失败: {e}")
                    continue

            return items[:max_res]

        except ImportError:
            logger.error("[DOAJ] requests 未安装")
            return []
        except Exception as e:
            logger.error(f"[DOAJ] 请求失败: {e}")
            return []


# ============================================================
# OpenAlex — 统一开放学术 API
# API: https://docs.openalex.org
# 覆盖 SSRN、RePEc、PubMed 等多个来源的论文
# ============================================================

class OpenAlexSource(BaseLiteratureSource):
    """OpenAlex 文献库 — 全学科覆盖，含 SSRN/RePEc 论文"""

    def search(self, query: str, max_results: int = None) -> List[LiteratureItem]:
        max_res = max_results or self.max_results

        clean_query = query.replace("AND", " ").replace("OR", " ").replace('"', "")
        clean_query = " ".join(clean_query.split())

        params = {
            "search": clean_query,
            "per_page": min(max_res, 200),
            "sort": "cited_by_count:desc",
            "select": "id,doi,title,authorships,publication_year,"
                      "abstract_inverted_index,primary_location,cited_by_count",
        }

        logger.info(f"[OpenAlex] 检索: '{clean_query[:80]}...' (max={max_res})")

        try:
            import requests
            headers = {
                "User-Agent": "PaperWritingAssistant/1.0 (mailto:research@example.com)",
                "Accept": "application/json",
            }
            response = requests.get(self.base_url, params=params, headers=headers, timeout=self.timeout)
            if response.status_code != 200:
                logger.warning(f"[OpenAlex] HTTP {response.status_code}")
                return []

            data = response.json()
            results = data.get("results", [])
            logger.info(f"[OpenAlex] 返回 {len(results)} 条结果")

            items = []
            for paper in results:
                try:
                    title = paper.get("title", "")
                    if not title:
                        continue

                    # 作者
                    authors = []
                    for auth in paper.get("authorships", []):
                        author_info = auth.get("author", {})
                        name = author_info.get("display_name", "") if isinstance(author_info, dict) else ""
                        if name:
                            authors.append(name)

                    # 年份
                    year = paper.get("publication_year")

                    # 摘要（inverted index → 文本）
                    abstract = self._reconstruct_abstract(
                        paper.get("abstract_inverted_index", {})
                    )

                    # 期刊/来源
                    journal = ""
                    primary_loc = paper.get("primary_location", {}) or {}
                    source_info = primary_loc.get("source", {}) or {}
                    if isinstance(source_info, dict):
                        journal = source_info.get("display_name", "")

                    # URL
                    url = paper.get("doi", "")
                    if url:
                        url = f"https://doi.org/{url}"
                    else:
                        url = paper.get("id", "")

                    # 引用数
                    citations = paper.get("cited_by_count", 0) or 0

                    items.append(LiteratureItem(
                        title=title,
                        authors=authors[:10],
                        year=year,
                        journal=journal,
                        abstract=abstract[:500],
                        citation_count=citations,
                        url=url,
                        source="openalex",
                    ))
                except Exception as e:
                    logger.warning(f"[OpenAlex] 解析条目失败: {e}")
                    continue

            return items[:max_res]

        except ImportError:
            logger.error("[OpenAlex] requests 未安装")
            return []
        except Exception as e:
            logger.error(f"[OpenAlex] 请求失败: {e}")
            return []

    @staticmethod
    def _reconstruct_abstract(inverted_index: Dict) -> str:
        """
        将 OpenAlex 的 abstract_inverted_index 格式重构为文本。

        OpenAlex 返回: {"word": [position1, position2, ...], ...}
        需要还原为: "word1 word2 word3 ..."
        """
        if not inverted_index:
            return ""

        try:
            # 收集所有 (position, word) 对
            word_positions = []
            for word, positions in inverted_index.items():
                if isinstance(positions, list):
                    for pos in positions:
                        word_positions.append((pos, word))

            # 按位置排序
            word_positions.sort(key=lambda x: x[0])

            # 拼接
            return " ".join(w for _, w in word_positions)
        except Exception:
            return ""


# ============================================================
# SocArXiv — 社会科学预印本 (OSF API v2)
# API: https://api.osf.io/v2/preprints/
# ============================================================

class SocArXivSource(BaseLiteratureSource):
    """SocArXiv 文献库 — 社会科学预印本"""

    def search(self, query: str, max_results: int = None) -> List[LiteratureItem]:
        max_res = max_results or self.max_results

        clean_query = query.replace("AND", " ").replace("OR", " ").replace('"', "")
        clean_query = " ".join(clean_query.split())

        # OSF API filter[title] 是精确匹配，不适合搜索
        # 使用 filter[description] 进行全文搜索（匹配摘要/描述文本）
        params = {
            "filter[provider]": "socarxiv",
            "filter[description]": clean_query,
            "page[size]": min(max_res, 50),
        }

        logger.info(f"[SocArXiv] 检索: '{clean_query[:80]}...' (max={max_res})")

        try:
            import requests
            headers = {
                "User-Agent": "PaperWritingAssistant/1.0 (mailto:research@example.com)",
                "Accept": "application/vnd.api+json",
            }
            response = requests.get(self.base_url, params=params, headers=headers, timeout=self.timeout)
            if response.status_code != 200:
                logger.warning(f"[SocArXiv] HTTP {response.status_code}")
                return []

            data = response.json()
            papers = data.get("data", [])
            logger.info(f"[SocArXiv] 返回 {len(papers)} 条结果")

            items = []
            for paper in papers:
                try:
                    attrs = paper.get("attributes", {})
                    title = attrs.get("title", "")
                    if not title:
                        continue

                    # 作者
                    authors = []
                    for contributor in attrs.get("contributors", []):
                        name = ""
                        if isinstance(contributor, dict):
                            embeds = contributor.get("embeds", {}).get("users", {})
                            if isinstance(embeds, dict):
                                data_obj = embeds.get("data", {})
                                if isinstance(data_obj, dict):
                                    name = data_obj.get("attributes", {}).get("full_name", "")
                        if name:
                            authors.append(name)

                    # 日期
                    year = None
                    date_str = attrs.get("date_created", "") or attrs.get("date_published", "")
                    if date_str:
                        try:
                            year = int(date_str[:4])
                        except (ValueError, TypeError):
                            pass

                    # 摘要
                    abstract = attrs.get("description", "") or ""
                    abstract = abstract[:500]

                    # URL
                    url = ""
                    links = paper.get("links", {})
                    if isinstance(links, dict):
                        url = links.get("html", "") or links.get("self", "")

                    items.append(LiteratureItem(
                        title=title,
                        authors=authors[:10],
                        year=year,
                        journal="SocArXiv preprint",
                        abstract=abstract,
                        citation_count=0,
                        url=url,
                        source="socarxiv",
                    ))
                except Exception as e:
                    logger.warning(f"[SocArXiv] 解析条目失败: {e}")
                    continue

            return items[:max_res]

        except ImportError:
            logger.error("[SocArXiv] requests 未安装")
            return []
        except Exception as e:
            logger.error(f"[SocArXiv] 请求失败: {e}")
            return []


# ============================================================
# RePEc/IDEAS — 经济学论文
# 搜索 URL: https://ideas.repec.org/cgi-bin/htsearch
# ============================================================

class RePEcSource(BaseLiteratureSource):
    """RePEc/IDEAS 文献库 — 经济学论文 (HTML 解析)"""

    def search(self, query: str, max_results: int = None) -> List[LiteratureItem]:
        max_res = max_results or self.max_results

        clean_query = query.replace("AND", " ").replace("OR", " ").replace('"', "")
        clean_query = " ".join(clean_query.split())

        params = {
            "q": clean_query,
            "cmd": "Search!",
            "m": "all",
            "fmt": "long",
            "wm": "wrd",
        }

        logger.info(f"[RePEc] 检索: '{clean_query[:80]}...' (max={max_res})")

        try:
            import requests
            from bs4 import BeautifulSoup

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
            }
            response = requests.get(self.base_url, params=params, headers=headers, timeout=self.timeout)
            if response.status_code != 200:
                logger.warning(f"[RePEc] HTTP {response.status_code}")
                return []

            soup = BeautifulSoup(response.text, "lxml" if __import__("lxml") else "html.parser")

            # IDEAS 搜索结果结构：
            # <li class="listitem"> 包含标题、作者、年份等信息
            items = []
            for li in soup.select("li.listitem, div.listitem, .search-result"):
                try:
                    # 标题
                    title_el = li.select_one("a b, .title a, a[href*='paper']")
                    if not title_el:
                        title_el = li.find("a")
                    title = title_el.get_text(strip=True) if title_el else ""

                    if not title or len(title) < 5:
                        continue

                    # 获取完整文本，解析作者/年份
                    full_text = li.get_text(" ", strip=True)

                    # 作者通常在 "by" 之后
                    authors = []
                    if "by " in full_text.lower():
                        author_part = full_text.lower().split("by ", 1)[1]
                        author_part = author_part.split(". ")[0] if ". " in author_part else author_part
                        author_part = author_part.split(" (")[0] if " (" in author_part else author_part
                        authors = [a.strip() for a in author_part.split(",") if a.strip()]

                    # 年份
                    year = None
                    year_match = re.search(r'\b(19|20)\d{2}\b', full_text)
                    if year_match:
                        year = int(year_match.group(0))

                    # 链接
                    url = ""
                    link_el = li.find("a", href=True)
                    if link_el:
                        href = link_el.get("href", "")
                        if href.startswith("/"):
                            url = f"https://ideas.repec.org{href}"
                        elif href.startswith("http"):
                            url = href

                    items.append(LiteratureItem(
                        title=title,
                        authors=authors[:10],
                        year=year,
                        journal="RePEc/IDEAS",
                        abstract=full_text[:500],
                        citation_count=0,
                        url=url,
                        source="repec",
                    ))
                except Exception as e:
                    logger.warning(f"[RePEc] 解析条目失败: {e}")
                    continue

            logger.info(f"[RePEc] 返回 {len(items)} 条结果")
            return items[:max_res]

        except ImportError as e:
            logger.error(f"[RePEc] 依赖缺失: {e}")
            return []
        except Exception as e:
            logger.error(f"[RePEc] 请求失败: {e}")
            return []


# ============================================================
# Socolar — 中文学术 OA 统一检索平台
# URL: http://www.socolar.com
# ============================================================

class SocolarSource(BaseLiteratureSource):
    """Socolar 文献库 — 中文学术 OA 统一检索 (HTML 解析)"""

    def search(self, query: str, max_results: int = None) -> List[LiteratureItem]:
        max_res = max_results or self.max_results

        clean_query = " ".join(query.replace("AND", " ").replace("OR", " ").split())

        # Socolar 搜索 URL
        search_url = f"{self.base_url}/Search/SearchResult"
        params = {
            "searchField": "All",
            "keyword": clean_query,
            "pageNum": 1,
            "pageSize": min(max_res, 50),
        }

        logger.info(f"[Socolar] 检索: '{clean_query[:80]}...' (max={max_res})")

        try:
            import requests
            from bs4 import BeautifulSoup

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            }
            response = requests.get(search_url, params=params, headers=headers, timeout=self.timeout)

            if response.status_code != 200:
                logger.warning(f"[Socolar] HTTP {response.status_code}")
                return []

            soup = BeautifulSoup(response.text, "lxml" if __import__("lxml") else "html.parser")

            items = []
            # Socolar 结果项选择器（根据实际页面结构调整）
            selectors = [
                ".search-result-item",
                ".result-item",
                "li.result",
                ".article-item",
                "tr",  # 表格布局
            ]

            result_elements = []
            for sel in selectors:
                result_elements = soup.select(sel)
                if result_elements:
                    break

            # 如果以上选择器都不匹配，尝试查找包含标题链接的元素
            if not result_elements:
                result_elements = soup.select("a[href*='detail'], a[href*='article'], a[href*='paper']")

            for el in result_elements[:max_res]:
                try:
                    # 标题
                    title_el = (
                        el.select_one("a.title, h3 a, .title, h4 a")
                        or el.find("a")
                    )
                    title = title_el.get_text(strip=True) if title_el else ""
                    if not title or len(title) < 3:
                        # 如果元素本身就是链接
                        if el.name == "a":
                            title = el.get_text(strip=True)
                    if not title or len(title) < 3:
                        continue

                    # 获取周围文本作为元数据
                    parent_text = el.get_text(" ", strip=True) if el.name != "a" else ""

                    # 作者
                    authors = []
                    author_match = re.search(
                        r'(?:作者|Author)[：:]\s*([^。\n]+)',
                        parent_text, re.IGNORECASE
                    )
                    if author_match:
                        authors = [a.strip() for a in author_match.group(1).split(";") if a.strip()]

                    # 年份
                    year = None
                    year_match = re.search(r'\b(19|20)\d{2}\b', parent_text)
                    if year_match:
                        year = int(year_match.group(0))

                    # 期刊
                    journal = ""
                    journal_match = re.search(
                        r'(?:刊名|期刊|Journal|Source)[：:]\s*([^。\n]+)',
                        parent_text, re.IGNORECASE
                    )
                    if journal_match:
                        journal = journal_match.group(1).strip()

                    # URL
                    url = ""
                    if el.name == "a" and el.get("href"):
                        href = el.get("href", "")
                        if href.startswith("http"):
                            url = href
                        else:
                            url = f"{self.base_url}{href}"
                    else:
                        link = el.find("a", href=True)
                        if link:
                            href = link.get("href", "")
                            if href.startswith("http"):
                                url = href
                            else:
                                url = f"{self.base_url}{href}"

                    items.append(LiteratureItem(
                        title=title,
                        authors=authors[:10],
                        year=year,
                        journal=journal or "Socolar",
                        abstract=parent_text[:500],
                        citation_count=0,
                        url=url,
                        source="socolar",
                    ))
                except Exception as e:
                    logger.warning(f"[Socolar] 解析条目失败: {e}")
                    continue

            logger.info(f"[Socolar] 返回 {len(items)} 条结果")
            return items[:max_res]

        except ImportError as e:
            logger.error(f"[Socolar] 依赖缺失: {e}")
            return []
        except Exception as e:
            logger.error(f"[Socolar] 请求失败: {e}")
            return []
