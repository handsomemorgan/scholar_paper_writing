"""
主编排器 (Orchestrator) — 串联论文写作全流程

7-Agent Pipeline:
  Agent 1 (分类器) → Agent 2 (RAG检索) → Agent 3 (关键词)
  → Agent 4 (文献检索) → Agent 5 (文献分析)
  → Agent 6 (论文撰写) → Agent 7 (格式校验)

每个Agent的输出是下一个Agent的输入，形成一个完整的流水线。
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
import requests  # 新增：DeepSeek API 需要 requests
from dotenv import load_dotenv  # 新增：加载环境变量（存储API Key）

# 移除 anthropic 相关导入，新增 DeepSeek 配置
load_dotenv()  # 加载 .env 文件中的 DEEPSEEK_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Orchestrator")

# ===================== 新增：DeepSeek LLM 客户端 =====================
class DeepSeekLLMClient:
    """适配 DeepSeek API 的 LLM 客户端

    兼容原 LLMClient 的全部接口 (chat / chat_with_json_output)，
    各 Agent 无需修改即可直接使用。
    """
    def __init__(self, config_path: str = None):
        # 从环境变量获取 API Key
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量（在 .env 文件中）")

        # DeepSeek API 基础配置
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.model = "deepseek-chat"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    # -------- 底层 API 调用 --------
    def _call_api(self, messages: list, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """
        调用 DeepSeek Chat Completion API
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"DeepSeek API 调用失败: {str(e)}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"DeepSeek API 响应格式异常: {str(e)}")
            raise

    # -------- 兼容原 LLMClient 的接口（各 Agent 直接调用） --------
    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """
        发送对话请求，返回文本响应。
        与原始 LLMClient.chat() 接口完全兼容。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        temp = temperature if temperature is not None else 0.7
        max_tok = max_tokens if max_tokens is not None else 4096
        return self._call_api(messages, temperature=temp, max_tokens=max_tok)

    def chat_with_json_output(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = None,
    ) -> dict:
        """
        发送对话请求，返回解析后的 JSON 字典。
        与原始 LLMClient.chat_with_json_output() 接口完全兼容。
        """
        json_instruction = (
            "\n\nIMPORTANT: You MUST respond with valid JSON only. "
            "No markdown, no code fences, no extra text. Just pure JSON."
        )
        full_system_prompt = system_prompt + json_instruction

        response_text = self.chat(
            system_prompt=full_system_prompt,
            user_message=user_message,
            temperature=temperature,
        )

        # 清理可能的 markdown 代码块标记
        response_text = response_text.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # 去掉首行 ```json 或 ```
            if len(lines) > 1:
                response_text = "\n".join(lines[1:])
            if response_text.endswith("```"):
                response_text = response_text[:-3]

        import json as _json
        return _json.loads(response_text.strip())

# ===================== 原有 Agent 导入保持不变 =====================
from agents.classifier import PaperClassifier
from agents.rag_retriever import RAGFormatRetriever
from agents.keyword_extractor import KeywordExtractor
from agents.literature_searcher import LiteratureSearchAgent
from agents.literature_analyzer import LiteratureAnalyzer
from agents.paper_writer import PaperWriter
from agents.format_checker import FormatChecker

class PaperWritingOrchestrator:
    """论文自动写作主编排器

    协调7个Agent完成从课程要求到最终论文的全流程。
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        logger.info("=" * 60)
        logger.info("论文自动写作助手 (Paper Writing Agent System)")
        logger.info("=" * 60)

        self.config_path = config_path

        # 初始化LLM客户端 —— 替换为 DeepSeek
        logger.info("初始化 DeepSeek LLM 客户端...")
        self.llm = DeepSeekLLMClient(config_path)  # 核心改动：替换 LLM 客户端

        # 初始化所有Agent
        logger.info("初始化 7 个 Agent...")
        self.classifier = PaperClassifier(self.llm)
        self.rag_retriever = RAGFormatRetriever(config_path)
        self.keyword_extractor = KeywordExtractor(self.llm)
        self.literature_searcher = LiteratureSearchAgent(config_path)
        self.literature_analyzer = LiteratureAnalyzer(self.llm)
        self.paper_writer = PaperWriter(self.llm)
        self.format_checker = FormatChecker(self.llm)

        logger.info("所有组件初始化完成，准备接受论文写作任务。")

    # ===================== 原有 run 方法逻辑完全保留 =====================
    def run(
        self,
        requirements: str,
        output_dir: Optional[str] = None,
        extra_instructions: str = "",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        执行完整的论文自动写作流程。

        Args:
            requirements: 课程论文要求（可以是纯文本或包含格式说明）
            output_dir: 输出目录（可选，默认使用配置中的output目录）
            extra_instructions: 额外的写作指令
            verbose: 是否输出详细信息

        Returns:
            包含整个流程结果的字典：
            {
                "paper": "最终论文文本",
                "metadata": {...},
                "format_check_report": {...},
                "pipeline_log": {...}
            }
        """
        start_time = time.time()
        pipeline_log = {}

        # =============================================
        # Phase 1: 论文类型分类
        # =============================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 1: 论文类型分类 (Agent 1)")
        logger.info("=" * 60)

        classification = self.classifier.classify(requirements)
        pipeline_log["classification"] = classification
        logger.info(f"  类型: {classification['category_name']}")
        logger.info(f"  置信度: {classification['confidence']}")
        logger.info(f"  理由: {classification['reasoning']}")

        # =============================================
        # Phase 2: RAG格式模板检索
        # =============================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 2: RAG格式模板检索 (Agent 2)")
        logger.info("=" * 60)

        format_template = self.rag_retriever.retrieve(classification)
        pipeline_log["format_template"] = {
            "category_id": format_template["category_id"],
            "retrieval_method": format_template["retrieval_method"],
            "template_length": format_template["template_length"],
        }
        logger.info(f"  检索方法: {format_template['retrieval_method']}")
        logger.info(f"  模板长度: {format_template['template_length']} 字符")

        # =============================================
        # Phase 3: 关键词提取
        # =============================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 3: 关键词提取 (Agent 3)")
        logger.info("=" * 60)

        keyword_result = self.keyword_extractor.extract(requirements)
        pipeline_log["keywords"] = {
            "primary_count": len(keyword_result.get("primary_keywords", [])),
            "total_count": sum(
                len(keyword_result.get(k, []))
                for k in ["primary_keywords", "secondary_keywords", "tertiary_keywords"]
            ),
            "search_queries": len(keyword_result.get("search_queries", [])),
        }
        all_kw = self.keyword_extractor.get_all_keywords_flat(keyword_result)
        logger.info(f"  关键词: {', '.join(all_kw[:10])}")

        # =============================================
        # Phase 4: 文献检索
        # =============================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 4: 文献检索 (Agent 4)")
        logger.info("=" * 60)

        literature_list = self.literature_searcher.search(keyword_result)

        pipeline_log["literature"] = {
            "total_found": len(literature_list),
            "high_quality": len(
                self.literature_searcher.filter_high_quality(literature_list)
            ),
        }
        logger.info(f"  检索到 {len(literature_list)} 篇文献")

        # =============================================
        # Phase 5: 文献分析 + 创新方向发现
        # =============================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 5: 文献分析与创新方向 (Agent 5)")
        logger.info("=" * 60)

        analysis_result = self.literature_analyzer.analyze(
            literature_list, requirements, keyword_result
        )
        pipeline_log["analysis"] = {
            "research_gaps": len(analysis_result.get("research_gaps", [])),
            "innovation_title": analysis_result.get("innovation_proposal", {}).get(
                "title", "N/A"
            ),
        }
        innovation = analysis_result.get("innovation_proposal", {})
        logger.info(f"  创新方向: {innovation.get('title', 'N/A')}")
        logger.info(f"  研究空白数: {len(analysis_result.get('research_gaps', []))}")

        # =============================================
        # Phase 6: 论文撰写
        # =============================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 6: 论文撰写 (Agent 6)")
        logger.info("=" * 60)

        paper = self.paper_writer.write(
            requirements=requirements,
            format_template=format_template,
            keyword_result=keyword_result,
            literature_list=literature_list,
            analysis_result=analysis_result,
            extra_instructions=extra_instructions,
        )
        pipeline_log["writing"] = {
            "paper_length": len(paper),
            "paper_lines": len(paper.split("\n")),
        }
        logger.info(f"  论文长度: {len(paper)} 字符, {len(paper.split(chr(10)))} 行")

        # =============================================
        # Phase 7: 格式校验
        # =============================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 7: 格式校验 (Agent 7)")
        logger.info("=" * 60)

        check_report = self.format_checker.check(
            paper=paper,
            template_content=format_template["template_content"],
            category_name=format_template["category_name"],
        )
        pipeline_log["format_check"] = {
            "overall_score": check_report.get("overall_score", 0),
            "is_compliant": check_report.get("is_compliant", False),
            "deviations": len(check_report.get("format_deviations", [])),
        }
        logger.info(f"  格式评分: {check_report.get('overall_score', 'N/A')}/100")
        logger.info(f"  合规: {'是' if check_report.get('is_compliant') else '否'}")
        logger.info(
            f"  格式偏差: {len(check_report.get('format_deviations', []))} 处"
        )

        # =============================================
        # 保存输出
        # =============================================
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("output", timestamp)

        os.makedirs(output_dir, exist_ok=True)

        # 保存论文
        paper_path = os.path.join(output_dir, "paper.md")
        with open(paper_path, "w", encoding="utf-8") as f:
            f.write(paper)

        # 保存元数据
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "category": classification["category_name"],
            "category_id": classification["category_id"],
            "paper_title": self._extract_title(paper),
            "paper_length": len(paper),
            "format_score": check_report.get("overall_score", 0),
            "pipeline_log": pipeline_log,
        }
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # 保存格式检查报告
        check_path = os.path.join(output_dir, "format_check_report.json")
        with open(check_path, "w", encoding="utf-8") as f:
            json.dump(check_report, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 60)
        logger.info(f"全流程完成! 耗时: {elapsed:.1f} 秒")
        logger.info(f"论文已保存至: {paper_path}")
        logger.info(f"元数据已保存至: {metadata_path}")
        logger.info("=" * 60)

        return {
            "paper": paper,
            "metadata": metadata,
            "format_check_report": check_report,
            "pipeline_log": pipeline_log,
            "output_dir": output_dir,
        }

    def _extract_title(self, paper: str) -> str:
        """从论文中提取标题"""
        lines = paper.strip().split("\n")
        for line in lines[:5]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return "未找到标题"

    def run_interactive(self):
        """交互式运行模式 — 逐步引导用户输入"""
        print("\n" + "=" * 60)
        print("  论文自动写作助手 - 交互模式")
        print("=" * 60)
        print()
        print("请粘贴您的课程论文要求（可以包含格式说明、字数要求等）。")
        print("输入完成后，请输入 'END' (单独一行) 结束输入。")
        print()

        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            except EOFError:
                break

        requirements = "\n".join(lines).strip()

        if not requirements:
            print("错误: 未输入任何要求。")
            return

        print(f"\n收到论文要求，共 {len(requirements)} 字符。")
        print("开始自动写作流程...\n")

        extra = input("是否有额外的写作要求？(直接回车跳过): ").strip()

        return self.run(requirements, extra_instructions=extra)


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="论文自动写作助手 - 自动生成高质量学术论文",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 从文件读取论文要求
  python orchestrator.py --input requirements.txt

  # 交互式输入
  python orchestrator.py --interactive

  # 指定输出目录
  python orchestrator.py --input requirements.txt --output my_paper/

  # 添加额外指令
  python orchestrator.py --input requirements.txt --extra "请使用APA引用格式"
        """,
    )

    parser.add_argument(
        "--input", "-i", type=str, help="论文要求文件路径"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="输出目录"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="交互式模式"
    )
    parser.add_argument(
        "--extra", "-e", type=str, default="", help="额外写作指令"
    )
    parser.add_argument(
        "--config", "-c", type=str, default="config/settings.yaml",
        help="配置文件路径"
    )

    args = parser.parse_args()

    orchestrator = PaperWritingOrchestrator(config_path=args.config)

    if args.interactive:
        orchestrator.run_interactive()
    elif args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            requirements = f.read()
        orchestrator.run(
            requirements=requirements,
            output_dir=args.output,
            extra_instructions=args.extra,
        )
    else:
        orchestrator.run_interactive()


if __name__ == "__main__":
    main()