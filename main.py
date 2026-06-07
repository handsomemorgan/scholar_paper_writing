#!/usr/bin/env python3
"""
论文自动写作助手 (Paper Writing Agent System)
==============================================

一个基于多Agent协作的学术论文自动写作系统。

架构：7-Agent Pipeline
  Agent 1 (分类器) → Agent 2 (RAG检索) → Agent 3 (关键词提取)
  → Agent 4 (文献检索) → Agent 5 (文献分析)
  → Agent 6 (论文撰写) → Agent 7 (格式校验)

核心特性：
  - RAG回溯式格式管理：格式模板人为框定，确保输出格式统一
  - 自动文献检索：通过 arXiv 公开 API 检索学术文献（免费开放）
  - 创新方向发现：基于文献分析自动提出创新研究角度
  - 严格格式校验：7步验证确保输出符合模板要求

使用方法：
  python main.py --interactive          # 交互式输入论文要求
  python main.py --input req.txt        # 从文件读取要求
  python main.py --input req.txt --output ./my_paper/  # 指定输出目录
"""

import sys
import os

# 确保项目根目录在path中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import PaperWritingOrchestrator, main


if __name__ == "__main__":
    main()
