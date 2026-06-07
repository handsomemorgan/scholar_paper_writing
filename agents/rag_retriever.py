"""
Agent 2: RAG格式模板检索器

基于分类结果，从格式模板库中检索匹配的论文格式标准。
使用向量相似度检索确保最佳匹配。

设计理念：
- 格式模板预先人工定义，确保格式标准统一
- RAG确保每次检索到的格式一致性
- 不需要AI去猜测格式，格式是人为框定的
"""

import os
import logging
import json
import threading
from typing import Dict, Any, Optional

import yaml

logger = logging.getLogger(__name__)


def _load_with_timeout(fn, timeout: int = 30):
    """
    在线程中执行函数，带超时保护。
    用于防止 HuggingFace 模型下载无限等待。
    """
    result = [None]
    error = [None]

    def _target():
        try:
            result[0] = fn()
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(
            f"操作超时（{timeout}秒）。"
            "如在国内请设置 HF_ENDPOINT=https://hf-mirror.com"
        )
    if error[0]:
        raise error[0]
    return result[0]


class RAGFormatRetriever:
    """RAG格式模板检索器 — 从模板库检索论文格式"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        rag_config = config["rag"]
        categories = config["paper_categories"]

        self.template_dir = rag_config["template_dir"]
        self.top_k = rag_config.get("top_k", 1)
        self.vector_db_path = rag_config.get("vector_db_path", "data/vector_db")

        # 构建类别ID到模板路径的映射
        self.category_template_map: Dict[str, str] = {}
        for cat in categories:
            self.category_template_map[cat["id"]] = cat["template"]

        # 预加载所有模板内容
        self.template_cache: Dict[str, str] = {}
        self._preload_templates()

        # 尝试初始化向量存储（用于语义检索增强）
        self.vector_store = None
        self.collection = None
        self._vector_store_attempted = False
        # 注意：不在此处初始化向量存储！
        # 精确匹配已覆盖100%场景，向量索引延迟到首次 _semantic_search 调用时。
        # 避免启动时阻塞下载 sentence-transformers 模型（尤其在网络受限环境）。

        logger.info(
            f"RAG Retriever 初始化完成，已加载 {len(self.template_cache)} 个模板"
        )

    def _preload_templates(self):
        """预加载所有格式模板到内存"""
        for cat_id, template_path in self.category_template_map.items():
            full_path = template_path
            if not os.path.isabs(full_path):
                full_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), template_path
                )

            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    self.template_cache[cat_id] = f.read()
                    logger.debug(f"加载模板: {cat_id} ({len(self.template_cache[cat_id])} chars)")
            else:
                logger.warning(f"模板文件不存在: {full_path}")

    def _init_vector_store(self):
        """
        初始化向量存储（用于语义检索增强）。

        注意：向量存储为可选增强功能。
        RAG主要依赖精确匹配（格式是人为框定的），
        向量检索仅在精确匹配失败时作为备用路径。
        """
        try:
            from chromadb import Client
            from chromadb.config import Settings

            os.makedirs(self.vector_db_path, exist_ok=True)

            self.vector_store = Client(Settings(
                persist_directory=self.vector_db_path,
                anonymized_telemetry=False,
            ))

            # 尝试获取或创建集合
            try:
                self.collection = self.vector_store.get_collection("paper_templates")
                logger.info("向量存储就绪（已有集合）")
            except Exception:
                self.collection = self.vector_store.create_collection("paper_templates")
                logger.info("创建新的向量集合")
                self._index_templates()

        except ImportError:
            logger.warning("chromadb未安装，向量检索不可用（不影响主流程）")
        except Exception as e:
            logger.warning(f"向量存储初始化失败: {e}（不影响主流程）")

    def _index_templates(self):
        """将模板索引到向量数据库（带HG镜像和超时保护）"""
        try:
            # 国内用户使用 HuggingFace 镜像，避免连接超时
            if "HF_ENDPOINT" not in os.environ:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

            from sentence_transformers import SentenceTransformer

            # 带超时的模型加载（避免无限等待）
            import signal

            def _load_model():
                return SentenceTransformer("all-MiniLM-L6-v2")

            # 30秒超时——对于这个小模型足够了
            model = _load_with_timeout(_load_model, timeout=60)

            for cat_id, content in self.template_cache.items():
                # 取模板前500字符作为索引（样本足够区分）
                sample = content[:500]
                embedding = model.encode(sample).tolist()

                self.collection.add(
                    ids=[cat_id],
                    embeddings=[embedding],
                    metadatas=[{"category_id": cat_id}],
                )

            logger.info(f"已将 {len(self.template_cache)} 个模板索引到向量数据库")

        except Exception as e:
            logger.warning(
                f"模板索引失败: {e}。RAG将使用精确匹配模式（不影响主流程）"
            )
            # 索引失败不阻塞——精确匹配足够覆盖所有场景

    def retrieve(self, classification_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据分类结果检索格式模板。

        Args:
            classification_result: Agent 1的分类结果

        Returns:
            包含模板内容和元信息的字典
        """
        logger.info("Agent 2: 正在检索格式模板...")

        category_id = classification_result.get("category_id", "")
        category_name = classification_result.get("category_name", "")

        # 精确匹配（主要路径 — 因为格式是人为框定的）
        template_content = self.template_cache.get(category_id)

        if not template_content:
            # 尝试语义检索（备用路径）
            logger.warning(
                f"未找到类别 '{category_id}' 的模板，尝试语义检索..."
            )
            template_content, category_id = self._semantic_search(
                classification_result
            )

        if not template_content:
            # 最后的fallback
            logger.error("无法找到合适的模板，使用默认模板")
            template_content = self._get_default_template()
            category_id = "liberal_arts_paper"

        result = {
            "category_id": category_id,
            "category_name": category_name,
            "template_content": template_content,
            "template_length": len(template_content),
            "retrieval_method": "exact_match" if category_id in self.template_cache else "fallback",
        }

        logger.info(
            f"Agent 2 完成: 检索到模板 '{category_id}' "
            f"({result['template_length']} 字符, 方法: {result['retrieval_method']})"
        )
        return result

    def _ensure_vector_store(self):
        """懒初始化向量存储——仅在首次需要语义检索时调用"""
        if self._vector_store_attempted:
            return
        self._vector_store_attempted = True
        self._init_vector_store()

    def _semantic_search(self, classification_result: Dict[str, Any]) -> tuple:
        """使用向量相似度检索最匹配的模板"""
        self._ensure_vector_store()  # 懒加载

        if self.vector_store is None:
            return None, None

        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")

            # 用分类理由作为查询文本
            query = (
                f"{classification_result.get('category_name', '')} "
                f"{classification_result.get('reasoning', '')} "
                f"{classification_result.get('discipline', '')}"
            )
            query_embedding = model.encode(query).tolist()

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.top_k,
            )

            if results["ids"] and results["ids"][0]:
                best_id = results["ids"][0][0]
                confidence = results["distances"][0][0] if results["distances"] else 0
                logger.info(f"语义检索匹配: {best_id} (距离: {confidence:.3f})")
                return self.template_cache.get(best_id), best_id

        except Exception as e:
            logger.error(f"语义检索失败: {e}")

        return None, None

    def _get_default_template(self) -> str:
        """获取默认模板（fallback）"""
        default = self.template_cache.get("liberal_arts_paper")
        if default:
            return default
        # 返回任意可用模板
        if self.template_cache:
            return list(self.template_cache.values())[0]
        return "# 默认论文格式\n\n## 摘要\n...\n\n## 正文\n...\n\n## 参考文献\n..."
