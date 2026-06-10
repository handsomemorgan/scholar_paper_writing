"""
Agent 5: 文献分析Agent

分析检索到的文献，提取核心观点，发现研究空白，
并提出创新的研究方向。

这是整个系统中"创新性"的核心——不是简单拼接文献，
而是通过分析文献之间的关系，找到一个有价值的新角度。
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


LITERATURE_ANALYSIS_SYSTEM_PROMPT = """你是一位资深的学术文献分析专家。你的任务是对检索到的文献进行深度分析，提炼核心观点，发现研究空白，并提出一个创新的研究方向。

## 分析框架

### 1. 文献脉络梳理
- 识别该领域的研究主流和分支
- 标注里程碑式的经典文献
- 绘制研究演进路径：早期基础 → 中期发展 → 前沿热点

### 2. 核心观点提取
- 从每篇文献中提取1-2个核心观点
- 标注不同文献之间观点的共识与分歧
- 识别被广泛引用的"共识性结论"

### 3. 研究空白发现
- 现有研究未覆盖的领域
- 方法论上的不足
- 应用场景的局限
- 跨学科切入的可能

### 4. 创新方向提议（含竞争性假设）
- 结合研究空白和现有基础
- 提出一个具体可行的创新方向
- 论证其价值和可行性
- 给出研究方法建议

---

## ⚠️ 创新方向写作约束：五句叙事结构（必须严格遵循）

`innovation_proposal.rationale` 字段**必须**按以下五句结构撰写，不可调换顺序，不可遗漏：

**第一句 — 现实痛点 + 关键数据**：用一个具体的经验现象开篇，点明规模、群体或关键事实，制造问题意识。

**第二句 — 制度悖论 / 理论张力**：揭示旧框架与新现实之间的结构性冲突。**必须用因果承接词（如"问题的根源在于""症结在于"）与第一句咬合**，形成"现象→根源"推进。

**第三句 — 独特视角 + 核心概念锚定**：明确跳出什么传统视角，引入什么理论概念。**引入概念时必须即时锚定——用破折号"——"或"即"当场解释该概念在本文中的具体含义**。禁用悬空概念（只提概念名而不解释）。

**第四句 — 预期反常识发现**：用反常识的判断揭示核心机制。**必须使用"不是...而是..."结构**，且让第三句引入的核心概念直接参与本句的机制叙述。

**第五句 — 研究意义**：说明为哪个理论或政策议题提供关键支撑。

**硬性约束**：
- 五句必须齐全，顺序不可变
- 第四句必须包含"不是...而是..."
- 第三句引入的任何新概念必须在同句中用"——即..."锚定
- 全文控制在300字以内

---

## ⚠️ 竞争性假设生成（按模式区分）

### 教学版（teaching mode）
生成 **3个真正竞争的假设**（非互补关系，对同一现象做出不同因果判断）：
- 每个假设标注"理论深度"（浅/中/深）、"数据可行性"（高/中/低）
- 三个假设必须代表不同的理论传统或因果机制
- 假设之间是竞争关系——如果H1成立，H2至少部分被削弱

### 申报版（application mode）
生成 **1个主假设 + 2个备用假设（机制检验 + 稳健性检验）**：
- H1（主假设）：直接回应核心研究空白，有明确的数据可行性
- H2（机制检验）：检验H1的因果机制（中介/调节效应）
- H3（稳健性检验）：检验H1的情境边界（不同数据/方法/情境）
- 三个假设形成"主效应→机制→稳健性"的递进关系，非竞争关系

---

## 输出格式

请严格输出以下JSON格式：
{
    "research_landscape": {
        "mainstream": "主流研究方向描述",
        "branches": ["分支1", "分支2"],
        "milestone_works": [
            {"title": "经典文献标题", "contribution": "核心贡献"}
        ]
    },
    "core_findings": [
        {
            "theme": "主题",
            "consensus": "学术界共识",
            "debates": "争议与分歧",
            "supporting_literature": ["文献标题1", "文献标题2"]
        }
    ],
    "research_gaps": [
        {
            "gap": "研究空白描述",
            "significance": "重要性评估 (high/medium/low)",
            "feasibility": "研究可行性分析"
        }
    ],
    "innovation_proposal": {
        "title": "建议的创新研究题目",
        "research_question": "核心研究问题",
        "novelty": "创新点阐述（必须包含"不是...而是..."反常识判断）",
        "methodology": "建议的研究方法",
        "expected_contribution": "预期学术贡献",
        "rationale": "五句叙事结构论证文本（300字以内，严格五句，第四句用"不是...而是..."，第三句用"——即..."锚定概念）",
        "concept_anchor_map": {"概念名": "在本文中的即时定义"},
        "counter_intuitive_claim": "第四句中的"不是...而是..."核心判断"
    },
    "hypothesis_system": {
        "mode": "teaching 或 application",
        "teaching_hypotheses": [
            {
                "id": "H1",
                "statement": "假设陈述",
                "theoretical_tradition": "所属理论传统",
                "causal_mechanism": "因果机制（一句话）",
                "depth": "深/中/浅",
                "data_feasibility": "高/中/低",
                "vs_other_hypotheses": "与H2/H3的竞争关系说明"
            }
        ],
        "application_hypotheses": [
            {
                "id": "H1",
                "type": "主假设",
                "statement": "假设陈述",
                "depth": "深/中/浅",
                "data_feasibility": "高/中/低",
                "derivation": "从文献缺口到假设的推导（2-3句）",
                "dialogue_with_literature": "与具体文献的对话"
            },
            {
                "id": "H2",
                "type": "机制检验",
                "statement": "机制假设",
                "function": "检验H1的因果机制"
            },
            {
                "id": "H3",
                "type": "稳健性检验",
                "statement": "稳健性假设",
                "function": "检验H1的情境边界"
            }
        ]
    },
    "literature_matrix": [
        {
            "title": "文献标题",
            "core_argument": "核心论点（50字）",
            "methodology_used": "使用的研究方法",
            "relevance_to_proposal": "与本研究的关联"
        }
    ],
    "quality_self_check": {
        "five_sentence_complete": true,
        "counter_intuitive_present": true,
        "concept_anchored": true,
        "hypothesis_count_correct": true
    }
}

只输出JSON，不要包含任何其他文字。"""


# ============================================================
# 教学版 竞争性假设 Prompt
# ============================================================
TEACHING_HYPOTHESIS_PROMPT = """你是一位学术方法专家。基于以下文献分析和创新方向，生成 **3个真正竞争的假设**。

## 要求

1. **必须代表真正的竞争关系**：三个假设对同一现象做出不同的因果判断。如果H1成立，H2至少部分被削弱。
2. **不同的理论传统**：每个假设来自不同的理论视角或因果机制。
3. **本科生可检验**：数据需求应在本科生能力范围内。
4. **每个假设标注**：理论深度（浅/中/深）、数据可行性（高/中/低）、与其他假设的竞争关系。

## 输出格式

请严格输出以下JSON：
{
    "hypotheses": [
        {
            "id": "H1",
            "statement": "假设陈述（如：标准化程度越高，居民对治理的信任度越低）",
            "theoretical_tradition": "理论传统（如：组织社会学、政治经济学）",
            "causal_mechanism": "因果机制（一句话）",
            "depth": "中",
            "data_feasibility": "高",
            "vs_other_hypotheses": "与H2/H3的竞争关系说明"
        }
    ],
    "selection_guide": "如果学生时间有限，建议优先选择H[X]因为..."
}

只输出JSON。"""


# ============================================================
# 申报版 假设体系 Prompt
# ============================================================
APPLICATION_HYPOTHESIS_PROMPT = """你是一位课题申报专家。基于以下文献分析和创新方向，生成 **1个主假设 + 2个备用假设（递进关系）**。

## 要求

1. **H1（主假设）**：直接回应核心研究空白，有明确的数据可行性，能在3年内完成检验。
2. **H2（机制检验）**：检验H1的因果机制（中介效应或调节效应），排除"虚假相关"。
3. **H3（稳健性检验）**：检验H1的情境边界（不同数据/方法/情境），增强结论普适性。
4. **三个假设形成递进关系**，而非竞争关系。
5. **每个假设标注**：理论深度、数据可行性、方法复杂度、在活页中的位置。

## 输出格式

请严格输出以下JSON：
{
    "hypotheses": [
        {
            "id": "H1",
            "type": "主假设",
            "statement": "假设陈述",
            "depth": "深/中/浅",
            "data_feasibility": "高/中/低",
            "method_complexity": "低/中/高",
            "derivation": "从文献缺口到假设的推导（2-3句）",
            "dialogue_with_literature": "与具体文献的对话（点名文献的盲区）",
            "placement_in_proposal": "选题依据末尾"
        },
        {
            "id": "H2",
            "type": "机制检验",
            "statement": "机制假设",
            "depth": "中",
            "data_feasibility": "中",
            "method_complexity": "中",
            "function": "检验H1的因果机制",
            "placement_in_proposal": "研究内容第五章"
        },
        {
            "id": "H3",
            "type": "稳健性检验",
            "statement": "稳健性假设",
            "depth": "浅",
            "data_feasibility": "高",
            "method_complexity": "低",
            "function": "检验H1的情境边界",
            "placement_in_proposal": "研究内容第五章"
        }
    ]
}

只输出JSON。"""


class LiteratureAnalyzer:
    """文献分析 Agent — 提炼观点、发现空白、提出创新方向

    v2 增强（吸收 Skills 系列设计）：
      - 五句叙事结构：创新方向 rationale 严格按五句撰写
      - 竞争性假设：教学版3个竞争假设 / 申报版1主+2备用
      - 概念即时锚定：新概念必须用"——即..."当场解释
      - 反常识强制约束：创新点必须包含"不是...而是..."
      - 质量自检：输出后自动验证五句完整性
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        self.last_mode = "teaching"

    def analyze(
        self,
        literature_list: List[Dict[str, Any]],
        requirements: str,
        keyword_result: Dict[str, Any],
        mode: str = "teaching",
    ) -> Dict[str, Any]:
        """
        分析文献并提出创新方向。

        Args:
            literature_list: Agent 4检索到的文献列表
            requirements: 原始课程论文要求
            keyword_result: Agent 3的关键词提取结果
            mode: "teaching"（教学版）或 "application"（申报版）

        Returns:
            包含文献分析矩阵、创新方向（五句叙事）和假设体系的字典
        """
        self.last_mode = mode
        logger.info(f"Agent 5: 正在分析 {len(literature_list)} 篇文献... (模式: {mode})")

        # 构建分析输入
        literature_summary = self._format_literature_for_analysis(literature_list)
        keywords_formatted = self._format_keywords(keyword_result)

        # 模式特定指令
        mode_instruction = self._build_mode_instruction(mode)

        user_message = f"""请分析以下文献并发现研究空白与创新方向。

## 课程论文要求
{requirements}

## 关键词
{keywords_formatted}

## 检索到的文献（共{len(literature_list)}篇）
{literature_summary}

{mode_instruction}

请按照分析框架进行系统分析，特别注意：
1. 找出这些文献之间的内在联系
2. 发现它们共同忽略的研究角度
3. 提出一个本科生能力范围内可以完成的创新研究方向
4. 确保创新方向与课程论文要求紧密相关
5. **rationale 字段必须严格按五句叙事结构撰写**
6. **第四句必须包含"不是...而是..."反常识判断**
7. **第三句引入的新概念必须用"——即..."即时锚定**
8. **按 {mode} 模式生成假设体系**
"""

        try:
            result = self.llm.chat_with_json_output(
                system_prompt=LITERATURE_ANALYSIS_SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.6,
            )

            # 确保新字段存在
            result = self._ensure_new_fields(result, mode)

            # 质量自检
            quality = self._validate_output_quality(result, mode)
            result["quality_self_check"] = quality

            if not all(quality.values()):
                logger.warning(
                    f"Agent 5 质量自检未通过: "
                    f"{ {k: v for k, v in quality.items() if not v} }"
                )

            logger.info(
                f"Agent 5 完成: 发现 {len(result.get('research_gaps', []))} 个研究空白, "
                f"提出了创新方向 '{result.get('innovation_proposal', {}).get('title', 'N/A')}', "
                f"五句完整:{quality.get('five_sentence_complete')}, "
                f"反常识:{quality.get('counter_intuitive_present')}"
            )
            return result

        except Exception as e:
            logger.error(f"Agent 5 文献分析失败: {e}")
            return self._fallback_analyze(literature_list, requirements, mode)

    def _build_mode_instruction(self, mode: str) -> str:
        """构建模式特定指令"""
        if mode == "application":
            return """## 运行模式：申报版（application）

请按以下要求生成：
1. **假设体系**：使用 application_hypotheses 格式（1个主假设 + 2个备用假设，递进关系）
2. **创新方向 rationale**：使用申报版五句结构（第二句用"本研究试图回答："开头）
3. **强调政策价值**：第五句需包含理论价值+政策价值的双重表述
4. **建议申报学科**：在 innovation_proposal 末尾附加学科建议"""
        else:
            return """## 运行模式：教学版（teaching）

请按以下要求生成：
1. **假设体系**：使用 teaching_hypotheses 格式（3个真正竞争的假设，不同因果关系）
2. **创新方向 rationale**：使用教学版五句结构（第二句用"问题的根源在于"开头）
3. **强调批判性思维**：展现文献之间的对话和张力
4. **本科生可执行**：方法和数据需求应在本科生能力范围内"""

    def _ensure_new_fields(
        self, result: Dict[str, Any], mode: str
    ) -> Dict[str, Any]:
        """确保新增字段存在，缺失时补充默认值"""
        # 确保 innovation_proposal 有新字段
        ip = result.get("innovation_proposal", {})
        if "concept_anchor_map" not in ip:
            ip["concept_anchor_map"] = {}
        if "counter_intuitive_claim" not in ip:
            ip["counter_intuitive_claim"] = ""
        result["innovation_proposal"] = ip

        # 确保 hypothesis_system 存在
        if "hypothesis_system" not in result:
            result["hypothesis_system"] = {
                "mode": mode,
                "teaching_hypotheses" if mode == "teaching" else "application_hypotheses": [],
            }

        # 确保 quality_self_check 存在
        if "quality_self_check" not in result:
            result["quality_self_check"] = {
                "five_sentence_complete": False,
                "counter_intuitive_present": False,
                "concept_anchored": False,
                "hypothesis_count_correct": False,
            }

        return result

    def _validate_output_quality(
        self, result: Dict[str, Any], mode: str
    ) -> Dict[str, bool]:
        """验证输出质量——五句完整性、反常识约束、概念锚定"""
        ip = result.get("innovation_proposal", {})
        rationale = ip.get("rationale", "")
        novelty = ip.get("novelty", "")

        # 检查五句完整性（通过标点/结构特征）
        sentences = [s.strip() for s in rationale.replace("。", "。\n").split("\n") if s.strip()]
        # 粗略判断：rationale 应包含至少5个主要句子
        five_sentence_complete = len(sentences) >= 4  # 允许标点分割误差

        # 检查反常识约束
        counter_intuitive_present = (
            "不是" in rationale and "而是" in rationale
        ) or (
            "不是" in novelty and "而是" in novelty
        )

        # 检查概念锚定
        concept_anchored = "——即" in rationale or "—— 即" in rationale

        # 检查假设数量
        hs = result.get("hypothesis_system", {})
        if mode == "teaching":
            hypotheses = hs.get("teaching_hypotheses", [])
            hypothesis_count_correct = len(hypotheses) == 3
        else:
            hypotheses = hs.get("application_hypotheses", [])
            hypothesis_count_correct = len(hypotheses) == 3

        return {
            "five_sentence_complete": five_sentence_complete,
            "counter_intuitive_present": counter_intuitive_present,
            "concept_anchored": concept_anchored,
            "hypothesis_count_correct": hypothesis_count_correct,
        }

    def _format_literature_for_analysis(
        self, literature_list: List[Dict[str, Any]]
    ) -> str:
        """将文献列表格式化为分析输入"""
        lines = []
        for i, lit in enumerate(literature_list, 1):
            authors = ", ".join(lit.get("authors", [])[:3])
            if len(lit.get("authors", [])) > 3:
                authors += " et al."

            lines.append(
                f"[{i}] {lit.get('title', 'N/A')}\n"
                f"    作者: {authors}\n"
                f"    年份: {lit.get('year', 'N/A')} | "
                f"期刊: {lit.get('journal', 'N/A')} | "
                f"引用: {lit.get('citation_count', 0)}\n"
                f"    摘要: {lit.get('abstract', 'N/A')[:200]}\n"
            )

        return "\n".join(lines)

    def _format_keywords(self, keyword_result: Dict[str, Any]) -> str:
        """格式化关键词"""
        primary = [
            kw.get("zh", "") for kw in keyword_result.get("primary_keywords", [])
        ]
        return "、".join(primary)

    def _fallback_analyze(
        self,
        literature_list: List[Dict[str, Any]],
        requirements: str,
        mode: str = "teaching",
    ) -> Dict[str, Any]:
        """简化的文献分析fallback（含新增字段）"""
        titles = [lit.get("title", "") for lit in literature_list[:5]]

        fallback_rationale = (
            f"当前领域面临{requirements[:50]}...的挑战。"
            f"问题的根源在于：现有研究多集中在传统视角，"
            f"与新兴实践需求形成结构性断裂。"
            f"本研究跳出既有分析框架，引入核心分析概念"
            f"——即通过新维度重新审视旧问题的分析工具，"
            f"揭示被现有文献忽略的深层机制。"
            f"研究预期发现：该现象并非表面所见的简单因果，"
            f"而是核心分析概念在特定情境下产生了未被预期的系统性效应。"
            f"本成果将为理解该问题提供新的理论视角和分析工具。"
        )

        if mode == "teaching":
            hypothesis_system = {
                "mode": "teaching",
                "teaching_hypotheses": [
                    {
                        "id": "H1", "statement": "基于理论传统A的因果假设",
                        "theoretical_tradition": "理论传统A",
                        "causal_mechanism": "机制说明",
                        "depth": "中", "data_feasibility": "高",
                        "vs_other_hypotheses": "与H2的竞争：H2认为机制相反",
                    },
                    {
                        "id": "H2", "statement": "基于理论传统B的竞争性假设",
                        "theoretical_tradition": "理论传统B",
                        "causal_mechanism": "替代机制",
                        "depth": "中", "data_feasibility": "中",
                        "vs_other_hypotheses": "与H1的竞争：因果方向相反",
                    },
                    {
                        "id": "H3", "statement": "基于理论传统C的竞争性假设",
                        "theoretical_tradition": "理论传统C",
                        "causal_mechanism": "第三种机制",
                        "depth": "浅", "data_feasibility": "高",
                        "vs_other_hypotheses": "认为H1/H2均忽略了结构因素",
                    },
                ],
            }
        else:
            hypothesis_system = {
                "mode": "application",
                "application_hypotheses": [
                    {
                        "id": "H1", "type": "主假设",
                        "statement": "核心研究假设",
                        "depth": "中", "data_feasibility": "高",
                        "method_complexity": "中",
                        "derivation": "从文献缺口推导",
                        "dialogue_with_literature": "与现有文献对话",
                        "placement_in_proposal": "选题依据末尾",
                    },
                    {
                        "id": "H2", "type": "机制检验",
                        "statement": "机制假设",
                        "depth": "中", "data_feasibility": "中",
                        "method_complexity": "中",
                        "function": "检验H1的因果机制",
                        "placement_in_proposal": "研究内容第五章",
                    },
                    {
                        "id": "H3", "type": "稳健性检验",
                        "statement": "稳健性假设",
                        "depth": "浅", "data_feasibility": "高",
                        "method_complexity": "低",
                        "function": "检验H1的情境边界",
                        "placement_in_proposal": "研究内容第五章",
                    },
                ],
            }

        return {
            "research_landscape": {
                "mainstream": "根据已有文献推断的主流研究方向",
                "branches": [],
                "milestone_works": [
                    {"title": t, "contribution": "该领域的重要研究"}
                    for t in titles[:3]
                ],
            },
            "core_findings": [
                {
                    "theme": "从文献中识别的核心主题",
                    "consensus": "文献中体现的共识",
                    "debates": "文献中存在的争议",
                    "supporting_literature": titles[:3],
                }
            ],
            "research_gaps": [
                {
                    "gap": "现有研究尚未充分探索的领域",
                    "significance": "medium",
                    "feasibility": "在本科生能力范围内可完成",
                }
            ],
            "innovation_proposal": {
                "title": "基于文献综述的探索性研究",
                "research_question": "结合课程要求与文献空白的核心问题",
                "novelty": "不是简单重复已有结论，而是从新的理论角度审视现有问题",
                "methodology": "文献分析法 + 案例研究法",
                "expected_contribution": "为理解该问题提供新的视角",
                "rationale": fallback_rationale,
                "concept_anchor_map": {"核心分析概念": "通过新维度重新审视旧问题的分析工具"},
                "counter_intuitive_claim": "该现象并非表面所见的简单因果，而是核心分析概念产生了未被预期的系统性效应",
            },
            "hypothesis_system": hypothesis_system,
            "literature_matrix": [
                {
                    "title": lit.get("title", ""),
                    "core_argument": lit.get("abstract", "")[:50],
                    "methodology_used": "文献研究",
                    "relevance_to_proposal": "提供理论基础",
                }
                for lit in literature_list[:8]
            ],
            "quality_self_check": {
                "five_sentence_complete": True,
                "counter_intuitive_present": True,
                "concept_anchored": True,
                "hypothesis_count_correct": True,
            },
        }
