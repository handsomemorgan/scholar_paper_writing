"""
主编排器 (Orchestrator) — 串联论文写作全流程

7-Agent Pipeline:
  Agent 1 (分类器) → Agent 2 (RAG检索) → Agent 3 (关键词)
  → Agent 4 (文献检索) → Agent 5 (文献分析)
  → Agent 6 (论文撰写) → Agent 7 (格式校验)

每个Agent的输出是下一个Agent的输入，形成一个完整的流水线。

字数自适应：
  - 自动从需求中解析目标字数
  - 按字数级别动态调整文献收集量（10/30/50篇）
  - 论文撰写后严格校验字数（±20%容差）
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
    def __init__(self, config_path: str = None, model: str = "deepseek-chat"):
        # 从环境变量获取 API Key
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量（在 .env 文件中）")

        # DeepSeek API 基础配置
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.model = model
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


# ===================== LLM 客户端工厂 =====================
def create_llm_client(provider: str = "deepseek", config_path: Optional[str] = None, model: Optional[str] = None):
    """创建 LLM 客户端（支持多后端切换）

    Args:
        provider: "deepseek" | "anthropic" | "openai"
        config_path: 配置文件路径
        model: 模型名称（可选，覆盖默认值）

    Returns:
        LLM 客户端实例（兼容 chat / chat_with_json_output 接口）
    """
    if provider == "deepseek":
        return DeepSeekLLMClient(config_path=config_path, model=model or "deepseek-chat")
    elif provider in ("anthropic", "openai"):
        # 尝试使用 utils/llm_client 中的通用客户端
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from utils.llm_client import LLMClient
            client = LLMClient(config_path=config_path or "config/settings.yaml")
            if model:
                client.model = model
            return client
        except ImportError:
            raise ValueError(f"使用 {provider} 需要安装对应 SDK: pip install anthropic openai")
    else:
        raise ValueError(f"不支持的 LLM 提供商: {provider}。可选: deepseek, anthropic, openai")

# 导入 sys（提前引用）
import sys

# ===================== 原有 Agent 导入保持不变 =====================
from agents.classifier import PaperClassifier
from agents.rag_retriever import RAGFormatRetriever
from agents.keyword_extractor import KeywordExtractor
from agents.literature_searcher import LiteratureSearchAgent
from agents.literature_analyzer import LiteratureAnalyzer
from agents.paper_writer import PaperWriter
from agents.format_checker import FormatChecker
from agents.figure_agent import FigureAgent  # Agent 8: 图表生成

class PaperWritingOrchestrator:
    """论文自动写作主编排器

    协调8个Agent完成从课程要求到最终论文的全流程。
    Agent 8 (图表Agent) 在文献分析后自动提取论文图表并生成数据配图。
    """

    def __init__(self, config_path: str = "config/settings.yaml",
                 provider: str = "deepseek", model: Optional[str] = None,
                 mode: str = "teaching"):
        logger.info("=" * 60)
        logger.info("论文自动写作助手 (Paper Writing Agent System)")
        logger.info(f"运行模式: {mode}")
        logger.info("=" * 60)

        self.config_path = config_path
        self.mode = mode

        # 初始化LLM客户端 —— 支持多后端切换
        logger.info(f"初始化 LLM 客户端 ({provider})...")
        self.llm = create_llm_client(provider=provider, config_path=config_path, model=model)

        # 初始化所有Agent
        logger.info("初始化 8 个 Agent...")
        self.classifier = PaperClassifier(self.llm)
        self.rag_retriever = RAGFormatRetriever(config_path)
        self.keyword_extractor = KeywordExtractor(self.llm)
        self.literature_searcher = LiteratureSearchAgent(config_path)
        self.literature_analyzer = LiteratureAnalyzer(self.llm)
        self.figure_agent = FigureAgent(self.llm)  # Agent 8: 图表生成
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
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        执行完整的论文自动写作流程。

        Args:
            requirements: 课程论文要求（可以是纯文本或包含格式说明）
            output_dir: 输出目录（可选，默认使用配置中的output目录）
            extra_instructions: 额外的写作指令
            verbose: 是否输出详细信息
            progress_callback: 进度回调函数，签名: callback(phase, status, message, details_dict)

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

        # 提前初始化 output_dir（Agent 8 图表生成需要提前使用）
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("output", timestamp)
        os.makedirs(output_dir, exist_ok=True)

        def _notify(phase, status, message, **details):
            if progress_callback:
                progress_callback(phase, status, message, details)

        # =============================================
        # Phase 1: 论文类型分类
        # =============================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 1: 论文类型分类 (Agent 1)")
        logger.info("=" * 60)

        _notify("phase1", "running", "正在识别论文类型...")
        classification = self.classifier.classify(requirements)
        pipeline_log["classification"] = classification
        _notify("phase1", "done", f"类型: {classification['category_name']}",
                category=classification['category_name'], confidence=classification['confidence'])
        logger.info(f"  类型: {classification['category_name']}")
        logger.info(f"  置信度: {classification['confidence']}")
        logger.info(f"  理由: {classification['reasoning']}")

        # =============================================
        # Phase 2: RAG格式模板检索
        # =============================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 2: RAG格式模板检索 (Agent 2)")
        logger.info("=" * 60)

        _notify("phase2", "running", "正在检索格式模板...")
        format_template = self.rag_retriever.retrieve(classification)
        pipeline_log["format_template"] = {
            "category_id": format_template["category_id"],
            "retrieval_method": format_template["retrieval_method"],
            "template_length": format_template["template_length"],
        }
        _notify("phase2", "done", f"模板: {format_template['category_name']}",
                category=format_template['category_name'])
        logger.info(f"  检索方法: {format_template['retrieval_method']}")
        logger.info(f"  模板长度: {format_template['template_length']} 字符")

        # =============================================
        # Phase 3: 关键词提取
        # =============================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 3: 关键词提取 (Agent 3)")
        logger.info("=" * 60)

        _notify("phase3", "running", "正在提取关键词...")
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
        _notify("phase3", "done", f"提取 {len(all_kw)} 个关键词",
                keywords=all_kw[:5])
        logger.info(f"  关键词: {', '.join(all_kw[:10])}")
        logger.info(f"  检索查询数: {len(keyword_result.get('search_queries', []))}")

        # =============================================
        # Phase 4: 文献检索
        # =============================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 4: 文献检索 (Agent 4)")
        logger.info("=" * 60)

        _notify("phase4", "running", "正在检索学术文献（多源路由）...")
        # 传入分类信息，使检索路由到对应学科文献库
        literature_list = self.literature_searcher.search(
            keyword_result,
            classification={
                "category_id": classification["category_id"],
                "category_name": classification["category_name"],
                "discipline": classification.get("discipline", ""),
            },
        )

        # 获取文献质量指标
        lit_quality = getattr(self.literature_searcher, 'last_quality_metrics', {})
        pipeline_log["literature"] = {
            "total_found": len(literature_list),
            "high_quality": len(
                self.literature_searcher.filter_high_quality(literature_list)
            ),
            "quality_metrics": lit_quality,
        }
        _notify("phase4", "done", f"检索到 {len(literature_list)} 篇文献",
                count=len(literature_list))
        logger.info(f"  检索到 {len(literature_list)} 篇文献")

        # =============================================
        # Phase 5: 文献分析 + 创新方向发现
        # =============================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 5: 文献分析与创新方向 (Agent 5)")
        logger.info("=" * 60)

        _notify("phase5", "running", "正在分析文献、发现创新方向...")
        analysis_result = self.literature_analyzer.analyze(
            literature_list, requirements, keyword_result,
            mode=self.mode,  # 传递运行模式
        )
        pipeline_log["analysis"] = {
            "research_gaps": len(analysis_result.get("research_gaps", [])),
            "innovation_title": analysis_result.get("innovation_proposal", {}).get(
                "title", "N/A"
            ),
            "quality_self_check": analysis_result.get("quality_self_check", {}),
            "hypothesis_mode": analysis_result.get("hypothesis_system", {}).get("mode", self.mode),
        }
        innovation = analysis_result.get("innovation_proposal", {})
        _notify("phase5", "done", f"创新方向: {innovation.get('title', 'N/A')}",
                gaps=len(analysis_result.get('research_gaps', [])))
        logger.info(f"  创新方向: {innovation.get('title', 'N/A')}")
        logger.info(f"  研究空白数: {len(analysis_result.get('research_gaps', []))}")
        qc = analysis_result.get("quality_self_check", {})
        logger.info(f"  质量自检: 五句完整={qc.get('five_sentence_complete')}, "
                    f"反常识={qc.get('counter_intuitive_present')}, "
                    f"概念锚定={qc.get('concept_anchored')}")

        # =============================================
        # Phase 5b (Agent 8): 图表生成
        #   - Phase A: 从检索论文中提取图表（综述用）
        #   - Phase B: 基于主题自动生成数据图表
        # =============================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 5b: 论文配图生成 (Agent 8: FigureAgent)")
        logger.info("=" * 60)

        _notify("phase5b", "running", "正在提取文献图表并生成数据配图...")
        figure_result = self.figure_agent.generate_figures(
            literature_list=literature_list,
            analysis_result=analysis_result,
            keyword_result=keyword_result,
            classification=classification,
            output_dir=output_dir,
        )
        figure_manifest = figure_result.get("manifest_text", "")
        figure_count = len(figure_result.get("figures", []))
        fig_stats = figure_result.get("stats", {})
        pipeline_log["figures"] = {
            "total": figure_count,
            "extracted": fig_stats.get("extracted", 0),
            "generated": fig_stats.get("generated", 0),
            "failed": fig_stats.get("failed", 0),
        }
        _notify("phase5b", "done",
                f"配图完成: {figure_count} 张 (提取{fig_stats.get('extracted',0)}+生成{fig_stats.get('generated',0)})",
                figures=figure_count,
                extracted=fig_stats.get('extracted', 0),
                generated=fig_stats.get('generated', 0))
        logger.info(f"  配图总数: {figure_count}")
        logger.info(f"  提取: {fig_stats.get('extracted', 0)} | "
                    f"生成: {fig_stats.get('generated', 0)} | "
                    f"失败: {fig_stats.get('failed', 0)}")

        # 将配图清单追加到 extra_instructions，传递给论文撰写Agent
        if figure_manifest:
            full_extra = extra_instructions + "\n\n" + figure_manifest if extra_instructions else figure_manifest
        else:
            full_extra = extra_instructions

        # =============================================
        # Phase 6: 论文撰写
        # =============================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 6: 论文撰写 (Agent 6)")
        logger.info("=" * 60)

        _notify("phase6", "running", "正在撰写中文论文（耗时较长）...")
        paper = self.paper_writer.write(
            requirements=requirements,
            format_template=format_template,
            keyword_result=keyword_result,
            literature_list=literature_list,
            analysis_result=analysis_result,
            extra_instructions=full_extra if figure_manifest else extra_instructions,
            mode=self.mode,  # 传递运行模式
        )
        pipeline_log["writing"] = {
            "paper_length": len(paper),
            "paper_lines": len(paper.split("\n")),
            "mode": self.mode,
        }
        _notify("phase6", "done", f"中文论文完成 ({len(paper)} 字符)",
                length=len(paper))
        logger.info(f"  论文长度: {len(paper)} 字符")

        # =============================================
        # Phase 6b: 英文版论文撰写
        # =============================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 6b: 英文版论文撰写 (Agent 6 - English)")
        logger.info("=" * 60)

        _notify("phase6b", "running", "正在撰写英文论文...")
        paper_en = self.paper_writer.write_english(
            chinese_paper=paper,
            requirements=requirements,
            format_template=format_template,
            keyword_result=keyword_result,
            literature_list=literature_list,
            analysis_result=analysis_result,
            extra_instructions=full_extra if figure_manifest else extra_instructions,
            mode=self.mode,  # 传递运行模式
        )
        pipeline_log["writing_en"] = {
            "paper_length": len(paper_en),
            "paper_lines": len(paper_en.split("\n")),
        }
        _notify("phase6b", "done", f"英文论文完成 ({len(paper_en)} 字符)")
        logger.info(f"  英文论文长度: {len(paper_en)} 字符")

        # =============================================
        # Phase 7: 格式校验
        # =============================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 7: 格式校验 (Agent 7)")
        logger.info("=" * 60)

        _notify("phase7", "running", "正在校验论文格式...")
        check_report = self.format_checker.check(
            paper=paper,
            template_content=format_template["template_content"],
            category_name=format_template["category_name"],
            mode=self.mode,  # 传递运行模式
        )
        pipeline_log["format_check"] = {
            "overall_score": check_report.get("overall_score", 0),
            "is_compliant": check_report.get("is_compliant", False),
            "deviations": len(check_report.get("format_deviations", [])),
            "argument_quality": check_report.get("argument_quality", {}),
            "reviewer_3min_test": check_report.get("reviewer_3min_test", {}),
        }
        _notify("phase7", "done", f"格式评分: {check_report.get('overall_score', 'N/A')}/100",
                score=check_report.get('overall_score', 0))
        logger.info(f"  格式评分: {check_report.get('overall_score', 'N/A')}/100")
        logger.info(f"  合规: {'是' if check_report.get('is_compliant') else '否'}")
        logger.info(
            f"  格式偏差: {len(check_report.get('format_deviations', []))} 处"
        )
        rt = check_report.get("reviewer_3min_test", {})
        if rt:
            logger.info(
                f"  评审3分钟测试: {'通过' if rt.get('overall_pass') else '需改进'}"
            )

        # =============================================
        # 保存输出
        # =============================================
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("output", timestamp)

        os.makedirs(output_dir, exist_ok=True)

        # 保存中文论文
        paper_path = os.path.join(output_dir, "paper.md")
        with open(paper_path, "w", encoding="utf-8") as f:
            f.write(paper)

        # 保存英文论文
        paper_en_path = os.path.join(output_dir, "paper_en.md")
        with open(paper_en_path, "w", encoding="utf-8") as f:
            f.write(paper_en)

        # 保存元数据
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "category": classification["category_name"],
            "category_id": classification["category_id"],
            "paper_title": self._extract_title(paper),
            "paper_title_en": self._extract_title(paper_en),
            "paper_length": len(paper),
            "paper_length_en": len(paper_en),
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
        logger.info(f"中文论文已保存至: {paper_path}")
        logger.info(f"英文论文已保存至: {paper_en_path}")
        logger.info(f"元数据已保存至: {metadata_path}")
        logger.info("=" * 60)

        return {
            "paper": paper,
            "paper_en": paper_en,
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
    parser.add_argument(
        "--mode", "-m", type=str, default="teaching",
        choices=["teaching", "application"],
        help="运行模式: teaching（教学版）或 application（申报版）"
    )

    args = parser.parse_args()

    orchestrator = PaperWritingOrchestrator(
        config_path=args.config, mode=args.mode
    )

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