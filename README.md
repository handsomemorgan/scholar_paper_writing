# 论文自动写作助手 (Paper Writing Agent System)

基于多Agent协作的学术论文自动写作系统。

## 架构概览

```
┌──────────────────────────────────────────────────────────┐
│                    7-Agent Pipeline                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  输入: 课程论文要求                                        │
│     │                                                    │
│     ▼                                                    │
│  ┌─────────────────┐                                     │
│  │ Agent 1: 分类器  │  识别论文类型                        │
│  │ (理科/文科/工科/  │  (5大类: 理科论文、文科论文、         │
│  │  实验/调研报告)   │   工科论文、实验报告、调研报告)       │
│  └────────┬────────┘                                     │
│           ▼                                              │
│  ┌─────────────────┐                                     │
│  │ Agent 2: RAG检索 │  从模板库检索对应格式标准              │
│  │ (格式模板匹配)    │  (格式为人框定，不依赖AI猜测)         │
│  └────────┬────────┘                                     │
│           ▼                                              │
│  ┌─────────────────┐                                     │
│  │ Agent 3: 关键词  │  提取中英文学术关键词                 │
│  │ (多层次关键词)    │  (一级/二级/三级 + 检索查询式)        │
│  └────────┬────────┘                                     │
│           ▼                                              │
│  ┌─────────────────┐                                     │
│  │ Agent 4: 文献检索│  通过 arXiv API 搜索学术文献          │
│  │ (学术数据库检索)  │  (去重、排序、质量筛选)              │
│  └────────┬────────┘                                     │
│           ▼                                              │
│  ┌─────────────────┐                                     │
│  │ Agent 5: 文献分析│  提炼核心观点、发现研究空白            │
│  │ (创新方向发现)    │  (提出创新的研究方向)                │
│  └────────┬────────┘                                     │
│           ▼                                              │
│  ┌─────────────────┐                                     │
│  │ Agent 6: 论文撰写│  按格式模板撰写完整论文               │
│  │ (完整论文生成)    │  (引用文献、逻辑严密、学术规范)       │
│  └────────┬────────┘                                     │
│           ▼                                              │
│  ┌─────────────────┐                                     │
│  │ Agent 7: 格式校验│  检查格式合规性                      │
│  │ (质量保障)       │  (自动修正格式偏差)                  │
│  └────────┬────────┘                                     │
│           ▼                                              │
│  输出: 格式统一的完整论文                                   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## 目录结构

```
scholar_paper_writing/
├── config/
│   └── settings.yaml           # 全局配置（LLM、RAG、类别定义）
├── templates/                   # 格式模板库（RAG知识库）
│   ├── science_paper.md         # 理科论文格式
│   ├── liberal_arts_paper.md    # 文科论文格式
│   ├── engineering_paper.md     # 工科论文格式
│   ├── lab_report.md            # 实验报告格式
│   └── research_report.md       # 调研报告格式
├── agents/                      # 7个Agent模块
│   ├── classifier.py            # Agent 1: 论文类型分类
│   ├── rag_retriever.py         # Agent 2: RAG格式检索
│   ├── keyword_extractor.py     # Agent 3: 关键词提取
│   ├── literature_searcher.py   # Agent 4: 文献检索
│   ├── literature_analyzer.py   # Agent 5: 文献分析
│   ├── paper_writer.py          # Agent 6: 论文撰写
│   └── format_checker.py        # Agent 7: 格式校验
├── utils/                       # 工具模块
│   ├── llm_client.py            # LLM客户端（Anthropic/OpenAI）
│   └── web_search.py            # Web搜索（arXiv API / CNKI备选）
├── orchestrator.py              # 主编排器（串联7个Agent）
├── main.py                      # CLI入口
├── output/                      # 生成论文输出目录
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/handsomemorgan/scholar_paper_writing.git
cd scholar_paper_writing

# 创建虚拟环境（推荐）
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 设置API密钥

本项目默认使用 **DeepSeek API** 作为 LLM 后端。

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 DeepSeek API Key
# 获取地址: https://platform.deepseek.com/api_keys
```

`.env` 文件内容：
```bash
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
```

也可以使用其他 LLM 后端，详见 [配置说明](#配置说明)。

### 3. 运行

#### 方式 A：Web 图形界面（推荐）

启动后在浏览器中打开 **http://127.0.0.1:5000** 即可使用可视化界面：

```bash
python web_server.py
```

自定义端口：
```bash
python web_server.py --port 8080
```

> 打开浏览器访问 **http://127.0.0.1:5000**，在对话框中输入论文需求，点击"开始写作"即可。
> 界面支持切换 DeepSeek / Claude / GPT 模型，以及星空、雨林、雪山、樱花、海洋五款背景主题。

#### 方式 B：命令行模式

```bash
# 交互式模式
python main.py --interactive

# 从文件读取论文要求
python main.py --input requirements_system_paper.txt

# 指定输出目录和额外指令
python main.py --input requirements.txt --output ./my_paper/ --extra "请使用APA引用格式，字数5000左右"
```

### 4. 查看输出

生成的文件保存在 `output/<时间戳>/` 目录下：
- `paper.md` — 完整论文（Markdown 格式）
- `paper_en.md` — 英文版本（如启用）
- `metadata.json` — 流程元数据
- `format_check_report.json` — 格式校验报告

## 核心设计理念

### RAG回溯式格式管理

论文格式是**人为预先框定**的五大类模板，不依赖AI临时猜测。

- **模板统一性**：同类型论文使用完全相同的格式标准
- **向量检索增强**：通过ChromaDB实现语义级别的模板匹配
- **格式回溯**：每次写作时都从模板库中RAG检索，确保输出一致性

### 创新性保障

系统不只做文献堆砌，而是通过Agent 5的文献分析：
1. 识别研究脉络和主流方向
2. 发现现有研究的空白和不足
3. 提出有学术价值的创新研究方向
4. 给出可行的研究方法论建议

### 质量保障机制

- 7步流水线，每步有结果验证
- 格式校验（Agent 7）作为最后一道质量关卡
- 每个Agent都有LLM失败时的fallback机制
- 完整的日志记录便于调试

## 配置说明

编辑 `config/settings.yaml` 自定义系统行为：

```yaml
# LLM 配置（默认使用 DeepSeek，由 orchestrator 直接调用）
llm:
  provider: "anthropic"        # 备选: anthropic / openai
  model: "claude-sonnet-4-6"
  api_key_env: "ANTHROPIC_API_KEY"
  temperature: 0.7

# 文献检索配置
literature_search:
  sources:
    - "arxiv"                  # arXiv 公开 API（免费，无需注册）
  max_results_per_source: 30
  search_depth: "comprehensive"
  enable_web_fallback: true    # arXiv 不足时自动网页搜索
  min_literature_threshold: 10

# 输出配置
output:
  formats:
    - "markdown"
    - "docx"
  timestamp_dir: true
```

### 切换 LLM 后端

- **DeepSeek**（默认）：在 `.env` 中设置 `DEEPSEEK_API_KEY`，无需额外配置
- **Anthropic Claude**：在 `.env` 中设置 `ANTHROPIC_API_KEY`，需 `pip install anthropic`
- **OpenAI**：在 `.env` 中设置 `OPENAI_API_KEY`，需 `pip install openai`

## 扩展指南

### 添加新的论文类型

1. 在 `templates/` 下创建新的格式模板（如 `case_study.md`）
2. 在 `config/settings.yaml` 的 `paper_categories` 中添加新类别

### 添加新的文献数据源

在 `utils/web_search.py` 的 `LiteratureSearcher` 类中：
1. 添加新的 `_search_<source>()` 方法
2. 在 `config/settings.yaml` 的 `sources` 列表中注册

### 添加自定义Agent

1. 在 `agents/` 下创建新的Agent模块
2. 在 `orchestrator.py` 的 `run()` 方法中集成

## 依赖项

- **LLM**: DeepSeek API（通过 `requests`）+ openai SDK（备选）
- **RAG**: chromadb + sentence-transformers
- **Web搜索**: arXiv API（标准库）+ beautifulsoup4 + requests
- **文档处理**: PyYAML + python-docx + markdown
- **CLI**: click + rich
- **其他**: tiktoken + tenacity + lxml + python-dotenv

## 注意事项

1. **API密钥**: 需要有效的LLM API密钥才能运行完整流程
2. **学术搜索**: 默认使用 arXiv 公开 API（免费开放、无需注册），如需要可扩展 Semantic Scholar 等数据源
3. **学术规范**: 本工具为研究和学习目的开发，使用时应遵守相关学术规范
4. **离线模式**: 系统设计为需要LLM API支持，部分Agent在LLM失败时有规则fallback

## License

MIT
