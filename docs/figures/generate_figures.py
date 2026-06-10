#!/usr/bin/env python3
"""
论文配图生成脚本 — 多智能体协作的论文学术写作系统
======================================================
根据系统运行12次的历史数据 + 系统架构信息，生成8张发表级配图。

用法: python3 generate_figures.py
输出: figures/ 目录下的 PNG 文件（300 DPI）
"""

import matplotlib
matplotlib.use('Agg')  # 非交互后端

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Wedge
from matplotlib.patches import Rectangle, FancyArrow
import matplotlib.patches as mpatches
from matplotlib.path import Path
import numpy as np
import os
import sys

# ============================================================
# 0. 全局样式配置
# ============================================================

# 查找中文字体
def find_chinese_font():
    """查找可用的中文字体（优先选择支持广泛 Unicode 的字体）"""
    # STHeiti 和 PingFang 支持更广泛的 Unicode 字符（含特殊符号）
    preferred = ['STHeiti', 'PingFang HK', 'Lantinghei SC', 'Songti SC', 'Heiti TC', 'SimSong']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in preferred:
        if font in available:
            return font
    for f in fm.fontManager.ttflist:
        if 'CJK' in f.name or 'Hei' in f.name or 'Song' in f.name:
            return f.name
    return None

CN_FONT = find_chinese_font()
CN_MONO = 'STHeiti'  # 用于等宽文本（支持中文的 sans 字体作为 monospace 替代）
print(f"[INFO] Using Chinese font: {CN_FONT}")

if CN_FONT:
    plt.rcParams['font.family'] = CN_FONT
    # 为 monospace 回退注册中文字体
    fm.fontManager.addfont(fm.findfont(CN_FONT))
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# 配色方案（学术风格）
COLORS = {
    'primary':   '#2c3e50',
    'secondary': '#34495e',
    'accent1':   '#3498db',
    'accent2':   '#2ecc71',
    'accent3':   '#e74c3c',
    'accent4':   '#f39c12',
    'accent5':   '#9b59b6',
    'accent6':   '#1abc9c',
    'light1':    '#ecf0f1',
    'light2':    '#bdc3c7',
    'white':     '#ffffff',
}

AGENT_COLORS = {
    'Agent 1': '#3498db',
    'Agent 2': '#2ecc71',
    'Agent 3': '#9b59b6',
    'Agent 4': '#e74c3c',
    'Agent 5': '#f39c12',
    'Agent 6': '#1abc9c',
    'Agent 7': '#e67e22',
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 历史运行数据（从 output/*/metadata.json 提取）
# ============================================================

RUNS = [
    {"date": "0607_1", "category": "实验报告", "p_len": 11501, "p_len_en": 0,     "lit": 5,   "lit_hq": 5,   "score": 75, "deviations": 6,  "gaps": 3, "kw_total": 12, "kw_query": 3},
    {"date": "0607_2", "category": "文科论文", "p_len": 6732,  "p_len_en": 0,     "lit": 5,   "lit_hq": 5,   "score": 65, "deviations": 17, "gaps": 2, "kw_total": 12, "kw_query": 3},
    {"date": "0607_3", "category": "工科论文", "p_len": 15102, "p_len_en": 0,     "lit": 10,  "lit_hq": 4,   "score": 65, "deviations": 20, "gaps": 3, "kw_total": 10, "kw_query": 2},
    {"date": "0607_4", "category": "工科论文", "p_len": 14889, "p_len_en": 0,     "lit": 19,  "lit_hq": 14,  "score": 45, "deviations": 20, "gaps": 3, "kw_total": 8,  "kw_query": 3},
    {"date": "0607_5", "category": "工科论文", "p_len": 15877, "p_len_en": 0,     "lit": 20,  "lit_hq": 14,  "score": 75, "deviations": 19, "gaps": 4, "kw_total": 8,  "kw_query": 3},
    {"date": "0607_6", "category": "工科论文", "p_len": 31821, "p_len_en": 0,     "lit": 20,  "lit_hq": 6,   "score": 40, "deviations": 16, "gaps": 3, "kw_total": 140,"kw_query": 70},
    {"date": "0607_7", "category": "工科论文", "p_len": 89137, "p_len_en": 0,     "lit": 41,  "lit_hq": 34,  "score": 40, "deviations": 19, "gaps": 3, "kw_total": 8,  "kw_query": 8},
    {"date": "0607_8", "category": "工科论文", "p_len": 33845, "p_len_en": 0,     "lit": 103, "lit_hq": 80,  "score": 65, "deviations": 21, "gaps": 4, "kw_total": 9,  "kw_query": 8},
    {"date": "0607_9", "category": "工科论文", "p_len": 22241, "p_len_en": 67063, "lit": 77,  "lit_hq": 61,  "score": 65, "deviations": 20, "gaps": 4, "kw_total": 11, "kw_query": 8},
    {"date": "0608_1", "category": "工科论文", "p_len": 14795, "p_len_en": 39877, "lit": 171, "lit_hq": 132, "score": 65, "deviations": 15, "gaps": 4, "kw_total": 11, "kw_query": 10},
    {"date": "0608_2", "category": "调研报告", "p_len": 13709, "p_len_en": 31335, "lit": 117, "lit_hq": 64,  "score": 45, "deviations": 21, "gaps": 3, "kw_total": 12, "kw_query": 8},
    {"date": "0609_1", "category": "工科论文", "p_len": 24921, "p_len_en": 59133, "lit": 77,  "lit_hq": 57,  "score": 55, "deviations": 22, "gaps": 4, "kw_total": 9,  "kw_query": 8},
]


def save_fig(fig, name):
    """保存图片"""
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ 已保存: {name}")
    return path


# ============================================================
# Figure 1: 7-Agent Pipeline Architecture
# ============================================================
def fig1_pipeline_architecture():
    """7-Agent 流水线架构图（替换原 ASCII 艺术图）"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_facecolor('white')

    # 输入
    ax.text(5, 13.5, '用户输入\n（课程论文要求）', ha='center', va='center',
            fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5',
            facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=1.5))

    # 箭头：输入 → Agent 1
    ax.annotate('', xy=(5, 12.3), xytext=(5, 13.0),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.0))

    agents = [
        (1, '论文类型分类器\n(Classifier)', 'LLM Few-shot\nClassification\n+ 规则Fallback',
         'category_id\nconfidence\nreasoning', '#3498db'),
        (2, 'RAG格式模板检索器\n(RAG Retriever)', '精确匹配优先\n+ ChromaDB向量检索\n(all-MiniLM-L6-v2)',
         '格式模板全文', '#2ecc71'),
        (3, '关键词提取器\n(Keyword Extractor)', 'LLM层次化提取\n+ 多轮扩展\n+ 中文查询过滤',
         '3级关键词\n5-10个英文查询', '#9b59b6'),
        (4, '文献检索Agent\n(Literature Searcher)', 'arXiv API\n→ Semantic Scholar\n→ 自主模拟(三级回退)',
         '10-30篇\n结构化文献', '#e74c3c'),
        (5, '文献分析Agent\n(Literature Analyzer)', 'LLM结构化综述\n+ 创新方向提议',
         '研究脉络\n核心发现\n研究空白\n创新方向', '#f39c12'),
        (6, '论文撰写Agent\n(Paper Writer)', 'LLM一次性\n整体生成\n(中英双版)',
         '完整Markdown论文\n(中文+英文)', '#1abc9c'),
        (7, '格式校验Agent\n(Format Checker)', '规则引擎(正则)\n+ LLM语义校验\n(双层检查)',
         '格式评分\n偏差列表\n修正建议', '#e67e22'),
    ]

    y_start = 11.5
    box_h = 1.5
    gap = 0.15

    for i, (num, name, tech, output, color) in enumerate(agents):
        y = y_start - i * (box_h + gap)

        # Agent box
        rect = FancyBboxPatch((0.5, y - box_h/2), 9.0, box_h,
                               boxstyle="round,pad=0.1", facecolor=color, edgecolor='white',
                               alpha=0.15, linewidth=1.5)
        ax.add_patch(rect)
        rect2 = FancyBboxPatch((0.5, y - box_h/2), 9.0, box_h,
                               boxstyle="round,pad=0.1", facecolor='none', edgecolor=color,
                               alpha=0.8, linewidth=2.0)
        ax.add_patch(rect2)

        # Agent number
        ax.text(1.0, y, f'Agent {num}', ha='center', va='center',
                fontsize=11, fontweight='bold', color=color)
        # Name
        ax.text(2.5, y + 0.3, name, ha='center', va='center',
                fontsize=9, fontweight='bold', color='#2c3e50')
        # Tech
        ax.text(5.0, y, tech, ha='center', va='center',
                fontsize=7.5, color='#555555', style='italic')
        # Output
        ax.text(7.8, y, output, ha='center', va='center',
                fontsize=7, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=color, alpha=0.8, linewidth=1.0))

        # Arrow between agents
        if i < len(agents) - 1:
            ax.annotate('', xy=(5, y - box_h/2 - gap), xytext=(5, y - box_h/2),
                        arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

    # 最终输出
    ax.text(5, y_start - len(agents) * (box_h + gap) + 0.3,
            '最终输出：paper.md + paper_en.md + metadata.json',
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2c3e50',
                     edgecolor='#2c3e50', linewidth=1.5, alpha=0.9),
            color='white')

    # 设计理念标注
    ax.text(5, 0.3, '设计理念：格式是人为框定的，创新是AI辅助发现的，内容是两者共同生成的。',
            ha='center', va='center', fontsize=9, color='#888888', style='italic')

    ax.set_title('图1：7-Agent 流水线架构总览', fontsize=14, fontweight='bold',
                 color='#2c3e50', pad=10)

    return save_fig(fig, 'fig1_pipeline_architecture.png')


# ============================================================
# Figure 2: Three-level Fallback Chain
# ============================================================
def fig2_fallback_chain():
    """三级回退链可视化 — 系统核心创新"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    fig.suptitle('图2：渐进式三级回退链（Progressive Three-Level Fallback）',
                 fontsize=13, fontweight='bold', color='#2c3e50', y=0.98)

    # -- Level 1: arXiv API --
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Level 1: arXiv API（主力）', fontsize=12, fontweight='bold', color='#2c3e50')

    # Box
    props = dict(boxstyle='round,pad=0.5', facecolor='#3498db', edgecolor='#2c3e50', alpha=0.15)
    ax.text(5, 7.5, 'arXiv API 检索', ha='center', va='center', fontsize=11,
            fontweight='bold', bbox=props)

    details = [
        '- API: export.arxiv.org/api/query',
        '- 协议: Atom XML over HTTP',
        '- 速率: 3s/request',
        '- 认证: 无需 API Key',
        '- 策略: 按priority遍历查询',
        '- 失败时自动宽泛化',
        '- max: 30条/查询',
    ]
    for j, d in enumerate(details):
        ax.text(1.0, 6.0 - j * 0.65, d, fontsize=8, color='#333333', va='center')

    ax.text(5, 1.5, '[OK] 覆盖 >=90% 场景', ha='center', va='center', fontsize=10,
            fontweight='bold', color='#27ae60')
    ax.annotate('', xy=(9.5, 5), xytext=(10.5, 5),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2.0),
                annotation_clip=False)

    # -- Level 2: Semantic Scholar --
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Level 2: Semantic Scholar（补充）', fontsize=12, fontweight='bold', color='#2c3e50')

    ax.annotate('', xy=(0.5, 5), xytext=(-0.5, 5),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2.0),
                annotation_clip=False)

    props2 = dict(boxstyle='round,pad=0.5', facecolor='#f39c12', edgecolor='#2c3e50', alpha=0.15)
    ax.text(5, 7.5, 'Semantic Scholar API', ha='center', va='center', fontsize=11,
            fontweight='bold', bbox=props2)

    details2 = [
        '> API: api.semanticscholar.org',
        '> 协议: REST JSON over HTTP',
        '> 速率: 5s/request',
        '> 触发条件: arXiv < 10篇',
        '> 追加学术关键词',
        '> 字段: 标题/作者/年份/摘要',
    ]
    for j, d in enumerate(details2):
        ax.text(1.0, 6.0 - j * 0.65, d, fontsize=8, color='#333333', va='center')

    ax.text(5, 1.5, '[WARN] 补充 arXiv 结果不足时', ha='center', va='center', fontsize=10,
            fontweight='bold', color='#e67e22')
    ax.annotate('', xy=(9.5, 5), xytext=(10.5, 5),
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2.5),
                annotation_clip=False)

    # -- Level 3: Mock --
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Level 3: 自主模拟（兜底）', fontsize=12, fontweight='bold', color='#2c3e50')

    ax.annotate('', xy=(0.5, 5), xytext=(-0.5, 5),
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2.5),
                annotation_clip=False)

    props3 = dict(boxstyle='round,pad=0.5', facecolor='#e74c3c', edgecolor='#2c3e50', alpha=0.15)
    ax.text(5, 7.5, '自主模拟生成', ha='center', va='center', fontsize=11,
            fontweight='bold', bbox=props3)

    details3 = [
        '> 触发: 总量 < TARGET_MIN(10)',
        '> 模板库: 8个模板(中英各半)',
        '> 覆盖 Survey/研究/框架/伦理',
        '> 来源标记: source="mock"',
        '> 含随机引用次数',
        '> [注意] 用户需替换真实引文',
    ]
    for j, d in enumerate(details3):
        ax.text(1.0, 6.0 - j * 0.65, d, fontsize=8, color='#333333', va='center')

    ax.text(5, 1.5, '[FALLBACK] 最后手段 (< 5% 场景)', ha='center', va='center', fontsize=10,
            fontweight='bold', color='#c0392b')

    # Flow arrows
    fig.text(0.34, 0.48, '结果 < 10篇', ha='center', fontsize=8, color='#e74c3c', fontweight='bold')
    fig.text(0.67, 0.48, '结果 < 10篇', ha='center', fontsize=8, color='#c0392b', fontweight='bold')

    plt.subplots_adjust(wspace=0.05)
    return save_fig(fig, 'fig2_fallback_chain.png')


# ============================================================
# Figure 3: System Performance Dashboard (Multi-panel)
# ============================================================
def fig3_performance_dashboard():
    """系统性能仪表盘：论文长度、文献数、格式评分、类别分布"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('图3：系统运行性能仪表盘（12次运行统计）', fontsize=13,
                 fontweight='bold', color='#2c3e50')

    papers = RUNS
    dates = [p['date'] for p in papers]
    x = np.arange(len(papers))

    # -- Panel A: Paper Length Distribution --
    ax = axes[0, 0]
    lengths_cn = [p['p_len'] / 1000 for p in papers]
    lengths_en = [p['p_len_en'] / 1000 for p in papers]

    bars_cn = ax.bar(x - 0.15, lengths_cn, 0.3, color='#3498db', alpha=0.8, label='中文论文')
    bars_en = ax.bar(x + 0.15, [l if l > 0 else 0 for l in lengths_en], 0.3,
                     color='#2ecc71', alpha=0.8, label='英文论文')

    # Add value labels
    for i, (cn, en) in enumerate(zip(lengths_cn, lengths_en)):
        if cn > 0:
            ax.text(i - 0.15, cn + 0.5, f'{cn:.0f}k', ha='center', fontsize=6, color='#3498db')
        if en > 0:
            ax.text(i + 0.15, en + 0.5, f'{en:.0f}k', ha='center', fontsize=6, color='#2ecc71')

    ax.axhline(y=np.mean(lengths_cn), color='#e74c3c', linestyle='--', linewidth=0.8,
               label=f'中文均值: {np.mean(lengths_cn):.1f}k')
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('论文字数 (千字)', fontsize=10)
    ax.set_title('A. 论文长度分布', fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(lengths_cn), max(lengths_en)) * 1.2)

    # -- Panel B: Literature Retrieval --
    ax = axes[0, 1]
    lit_total = [p['lit'] for p in papers]
    lit_hq = [p['lit_hq'] for p in papers]

    ax.bar(x - 0.15, lit_total, 0.3, color='#e74c3c', alpha=0.7, label='检索总数')
    ax.bar(x + 0.15, lit_hq, 0.3, color='#2ecc71', alpha=0.7, label='高质量文献')

    for i, (t, hq) in enumerate(zip(lit_total, lit_hq)):
        ax.text(i - 0.15, t + 1, str(t), ha='center', fontsize=6, color='#e74c3c')
        if hq > 0:
            ax.text(i + 0.15, hq + 1, str(hq), ha='center', fontsize=6, color='#2ecc71')

    ax.axhline(y=10, color='#f39c12', linestyle='--', linewidth=1.0, label='最低阈值 (10篇)')
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('文献数量 (篇)', fontsize=10)
    ax.set_title('B. 文献检索效率', fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    # -- Panel C: Format Score Distribution --
    ax = axes[1, 0]
    scores = [p['score'] for p in papers]
    deviations = [p['deviations'] for p in papers]
    colors = ['#27ae60' if s >= 65 else '#f39c12' if s >= 50 else '#e74c3c' for s in scores]

    bars = ax.bar(x, scores, 0.5, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)

    for i, (s, d) in enumerate(zip(scores, deviations)):
        ax.text(i, s + 1, f'{s}分\n({d}处偏差)', ha='center', fontsize=6, color='#333333')

    ax.axhline(y=60, color='#f39c12', linestyle='--', linewidth=1.0, label='及格线 (60分)')
    ax.axhline(y=np.mean(scores), color='#3498db', linestyle='--', linewidth=0.8,
               label=f'平均分: {np.mean(scores):.1f}')
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('格式评分 (满分100)', fontsize=10)
    ax.set_title('C. 格式校验评分', fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)

    # -- Panel D: Category Distribution --
    ax = axes[1, 1]
    from collections import Counter
    cat_counts = Counter(p['category'] for p in papers)
    categories = list(cat_counts.keys())
    cat_values = list(cat_counts.values())
    cat_colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    explode = [0.05] * len(categories)

    wedges, texts, autotexts = ax.pie(cat_values, labels=categories, autopct='%1.1f%%',
                                       colors=cat_colors[:len(categories)],
                                       explode=explode, startangle=90,
                                       textprops={'fontsize': 10})

    for at in autotexts:
        at.set_fontweight('bold')
        at.set_fontsize(9)

    ax.set_title('D. 论文类型分布', fontsize=11, fontweight='bold', loc='left')

    plt.tight_layout()
    return save_fig(fig, 'fig3_performance_dashboard.png')


# ============================================================
# Figure 4: Literature Retrieval Quality Analysis
# ============================================================
def fig4_literature_quality():
    """文献检索质量分析：高质量比率、来源分布"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('图4：文献检索质量分析', fontsize=13, fontweight='bold', color='#2c3e50')

    papers = RUNS
    x = np.arange(len(papers))

    # -- Panel A: High Quality Ratio --
    ax = axes[0]
    hq_ratios = [p['lit_hq'] / max(p['lit'], 1) * 100 for p in papers]
    colors = ['#27ae60' if r >= 70 else '#f39c12' if r >= 50 else '#e74c3c' for r in hq_ratios]

    bars = ax.bar(x, hq_ratios, 0.6, color=colors, alpha=0.8, edgecolor='white')

    for i, (r, p) in enumerate(zip(hq_ratios, papers)):
        ax.text(i, r + 1, f'{r:.0f}%\n({p["lit_hq"]}/{p["lit"]})',
                ha='center', fontsize=7, color='#333333')

    ax.axhline(y=np.mean(hq_ratios), color='#3498db', linestyle='--', linewidth=0.8,
               label=f'平均高质量比: {np.mean(hq_ratios):.1f}%')
    ax.axhline(y=70, color='#27ae60', linestyle=':', linewidth=0.8, label='高质量阈值 (70%)')
    ax.set_xticks(x)
    ax.set_xticklabels([p['date'] for p in papers], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('高质量文献占比 (%)', fontsize=10)
    ax.set_title('A. 文献高质量比率', fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 110)

    # -- Panel B: Literature Count vs Paper Length --
    ax = axes[1]
    lit_counts = [p['lit'] for p in papers]
    p_lengths = [p['p_len'] / 1000 for p in papers]

    # Color by category
    cat_colors_map = {'工科论文': '#3498db', '文科论文': '#2ecc71', '实验报告': '#f39c12', '调研报告': '#e74c3c'}

    for p in papers:
        c = cat_colors_map.get(p['category'], '#95a5a6')
        ax.scatter(p['lit'], p['p_len'] / 1000, s=p['p_len'] / 500, c=c,
                   alpha=0.7, edgecolors='white', linewidth=0.5,
                   label=p['category'] if p['category'] not in [pp['category'] for pp in papers[:papers.index(p)]] else "")

    # Trend line
    z = np.polyfit(lit_counts, p_lengths, 1)
    p_trend = np.poly1d(z)
    x_trend = np.linspace(0, max(lit_counts) * 1.1, 100)
    ax.plot(x_trend, p_trend(x_trend), '--', color='#2c3e50', linewidth=1.0, alpha=0.5,
            label=f'趋势线 (r={np.corrcoef(lit_counts, p_lengths)[0,1]:.2f})')

    ax.set_xlabel('检索文献总量 (篇)', fontsize=10)
    ax.set_ylabel('论文字数 (千字)', fontsize=10)
    ax.set_title('B. 文献量-论文长度关系', fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return save_fig(fig, 'fig4_literature_quality.png')


# ============================================================
# Figure 5: Format Quality & Deviation Analysis
# ============================================================
def fig5_format_quality():
    """格式质量深度分析"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('图5：格式校验深度分析', fontsize=13, fontweight='bold', color='#2c3e50')

    papers = RUNS
    x = np.arange(len(papers))

    # -- Panel A: Score vs Deviations --
    ax = axes[0]
    scores = [p['score'] for p in papers]
    deviations = [p['deviations'] for p in papers]

    ax.scatter(deviations, scores, s=100, c=range(len(papers)), cmap='viridis',
               alpha=0.8, edgecolors='white', linewidth=0.8)

    for i, p in enumerate(papers):
        ax.annotate(p['date'], (deviations[i], scores[i]),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=6, color='#555555')

    # Correlation
    corr = np.corrcoef(deviations, scores)[0, 1]
    ax.text(0.95, 0.05, f'相关系数 r = {corr:.3f}', transform=ax.transAxes,
            fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

    ax.set_xlabel('格式偏差数量', fontsize=10)
    ax.set_ylabel('格式评分', fontsize=10)
    ax.set_title('A. 偏差数量-评分关系', fontsize=11, fontweight='bold', loc='left')
    ax.grid(alpha=0.3)
    ax.set_ylim(30, 85)

    # -- Panel B: Score Distribution Histogram --
    ax = axes[1]
    bins = [35, 45, 55, 65, 75, 85]
    n, bins_out, patches = ax.hist(scores, bins=bins, color='#3498db', alpha=0.7,
                                    edgecolor='white', linewidth=1.5)

    # Color by range
    for i, (patch, left, right) in enumerate(zip(patches, bins_out[:-1], bins_out[1:])):
        mid = (left + right) / 2
        if mid < 50:
            patch.set_facecolor('#e74c3c')
        elif mid < 60:
            patch.set_facecolor('#f39c12')
        elif mid < 70:
            patch.set_facecolor('#3498db')
        else:
            patch.set_facecolor('#27ae60')

    ax.axvline(x=60, color='#f39c12', linestyle='--', linewidth=1.5, label='及格线')
    ax.axvline(x=np.mean(scores), color='#2ecc71', linestyle='-', linewidth=1.5,
               label=f'均值: {np.mean(scores):.1f}')
    ax.axvline(x=np.median(scores), color='#9b59b6', linestyle=':', linewidth=1.5,
               label=f'中位数: {np.median(scores):.0f}')

    ax.set_xlabel('格式评分', fontsize=10)
    ax.set_ylabel('运行次数', fontsize=10)
    ax.set_title('B. 格式评分分布直方图', fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=8)

    # Add stats
    stats_text = f'均值: {np.mean(scores):.1f}  |  标准差: {np.std(scores):.1f}\n最高: {max(scores)}  |  最低: {min(scores)}  |  及格率: {sum(1 for s in scores if s >= 60)/len(scores)*100:.0f}%'
    ax.text(0.5, -0.15, stats_text, transform=ax.transAxes, ha='center',
            fontsize=7, color='#888888')

    plt.tight_layout()
    return save_fig(fig, 'fig5_format_quality.png')


# ============================================================
# Figure 6: Word Count Adaptive Strategy
# ============================================================
def fig6_word_count_strategy():
    """字数自适应策略可视化"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('图6：字数自适应策略（Word Count Tiers）', fontsize=13,
                 fontweight='bold', color='#2c3e50')

    # -- Panel A: Strategy Visualization --
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    tiers = [
        ("< 5,000 字\n(短文)", "10 篇\n文献", "1 轮\n扩展", "2 页\n翻页", '#27ae60', 1.0, 8.5, 8.0),
        ("5,000-20,000\n(中篇)", "15 篇\n文献", "2 轮\n扩展", "2 页\n翻页", '#3498db', 3.3, 8.5, 8.0),
        ("20,000-50,000\n(长文)", "30 篇\n文献", "3 轮\n扩展", "4 页\n翻页", '#f39c12', 5.6, 8.5, 8.0),
        ("> 50,000 字\n(超长篇)", "50 篇\n文献", "5 轮\n扩展", "6 页\n翻页", '#e74c3c', 7.9, 8.5, 8.0),
    ]

    for label, lit, exp, page, color, x, y, w in tiers:
        rect = FancyBboxPatch((x, y - 2.2), w * 0.25, 4.5,
                               boxstyle="round,pad=0.1", facecolor=color, alpha=0.2,
                               edgecolor=color, linewidth=2.0)
        ax.add_patch(rect)
        ax.text(x + w * 0.125, y + 0.2, label, ha='center', fontsize=9, fontweight='bold', color=color)

        # Sub-items
        items = [lit, exp, page]
        for j, item in enumerate(items):
            ax.text(x + w * 0.125, y - 1.0 - j * 0.65, item, ha='center',
                    fontsize=7, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                             edgecolor=color, alpha=0.8, linewidth=0.8))

    ax.set_title('A. 字数级别-策略映射', fontsize=11, fontweight='bold', loc='left')

    # -- Panel B: Actual Paper Length Distribution by Tier --
    ax = axes[1]
    p_lengths = [p['p_len'] for p in RUNS]

    tier_names = ['< 5k\n(短文)', '5k-20k\n(中篇)', '20k-50k\n(长文)', '> 50k\n(超长篇)']
    tier_counts = [
        sum(1 for l in p_lengths if l < 5000),
        sum(1 for l in p_lengths if 5000 <= l < 20000),
        sum(1 for l in p_lengths if 20000 <= l < 50000),
        sum(1 for l in p_lengths if l >= 50000),
    ]
    tier_colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c']

    bars = ax.bar(tier_names, tier_counts, color=tier_colors, alpha=0.8,
                  edgecolor='white', linewidth=1.5, width=0.6)

    for bar, count in zip(bars, tier_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f'{count} 篇', ha='center', fontsize=10, fontweight='bold')

    ax.set_ylabel('论文数量', fontsize=10)
    ax.set_title('B. 实际生成论文字数分布', fontsize=11, fontweight='bold', loc='left')
    ax.set_ylim(0, max(tier_counts) * 1.3)

    # Add scatter overlay of actual lengths
    for l in p_lengths:
        if l < 5000:
            ax.scatter(0 + np.random.uniform(-0.2, 0.2), tier_counts[0] * 0.3 + np.random.uniform(-0.5, 0.5),
                      s=15, c='#27ae60', alpha=0.5)
        elif l < 20000:
            ax.scatter(1 + np.random.uniform(-0.2, 0.2), tier_counts[1] * 0.3 + np.random.uniform(-0.5, 0.5),
                      s=15, c='#3498db', alpha=0.5)
        elif l < 50000:
            ax.scatter(2 + np.random.uniform(-0.2, 0.2), tier_counts[2] * 0.3 + np.random.uniform(-0.5, 0.5),
                      s=15, c='#f39c12', alpha=0.5)
        else:
            ax.scatter(3 + np.random.uniform(-0.2, 0.2), tier_counts[3] * 0.3 + np.random.uniform(-0.5, 0.5),
                      s=15, c='#e74c3c', alpha=0.5)

    plt.tight_layout()
    return save_fig(fig, 'fig6_word_count_strategy.png')


# ============================================================
# Figure 7: Chinese Query Filtering Algorithm Flow
# ============================================================
def fig7_chinese_query_filter():
    """中文查询过滤算法流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('图7：中文查询自动过滤算法（Agent 3 核心创新）', fontsize=13,
                 fontweight='bold', color='#2c3e50', pad=10)

    # Start
    draw_box(ax, 5, 9.5, 4.0, 0.8, '输入：search_query 文本', '#2c3e50', 'white', fontsize=10, fontweight='bold')
    draw_arrow(ax, 5, 9.1, 5, 8.7)

    # Decision: contains Chinese?
    draw_diamond(ax, 5, 8.0, 3.5, 1.4, '包含中文字符？\n(Unicode: 一-鿿)', '#3498db')
    draw_arrow(ax, 5, 7.3, 5, 6.8)
    draw_arrow(ax, 3.25, 8.0, 1.5, 8.0)
    draw_arrow(ax, 1.5, 7.5, 1.5, 6.5)

    # NO branch
    draw_box(ax, 1.5, 6.0, 2.0, 0.8, '纯英文查询\n直接通过', '#27ae60', 'white', fontsize=8)
    draw_arrow(ax, 1.5, 5.5, 1.5, 4.8)

    # YES branch: extract English
    draw_box(ax, 5, 6.3, 3.5, 1.0, '提取英文部分\n(≥3字母的连续英文序列)', '#f39c12', 'white', fontsize=8)
    draw_arrow(ax, 5, 5.8, 5, 5.3)

    # Decision: English extracted?
    draw_diamond(ax, 5, 4.5, 3.0, 1.2, '提取成功？', '#e74c3c')
    draw_arrow(ax, 5, 3.9, 5, 3.3)
    draw_arrow(ax, 3.5, 4.5, 2.0, 4.5)
    draw_arrow(ax, 2.0, 4.0, 2.0, 3.3)

    # Success branch
    draw_box(ax, 5, 2.8, 3.5, 1.0, '用英文部分替换原查询\n标记 note="英文部分提取"', '#2ecc71', 'white', fontsize=8)
    draw_arrow(ax, 5, 2.3, 5, 1.5)
    draw_arrow(ax, 2.0, 3.3, 2.0, 1.5)

    # Failure branch
    draw_box(ax, 2.0, 2.8, 2.5, 1.0, 'source 改为 "web"\n标记 note="中文查询→Web"', '#e67e22', 'white', fontsize=8)
    draw_arrow(ax, 2.0, 2.3, 2.0, 1.5)

    # Merge
    draw_box(ax, 3.5, 1.0, 3.0, 0.8, '输出：处理后的查询', '#2c3e50', 'white', fontsize=10, fontweight='bold')

    # Code snippet
    code_text = (
        "算法伪码：\n"
        "def _filter_chinese_queries(query):\n"
        "    if has_chinese(query):\n"
        "        english_parts = re.findall(r'[a-zA-Z]{3,}', query)\n"
        "        if english_parts:\n"
        "            return ' '.join(english_parts), '英文部分提取'\n"
        "        else:\n"
        "            return query, '中文查询→Web搜索'  # source改为web\n"
        "    return query, None  # 纯英文，无需处理"
    )
    ax.text(7.5, 4.0, code_text, fontsize=6.5, va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2c3e50', edgecolor='#2c3e50',
                     alpha=0.9), color='#ecf0f1')

    return save_fig(fig, 'fig7_chinese_query_filter.png')


def draw_box(ax, x, y, w, h, text, color, text_color='white', fontsize=9, fontweight='normal'):
    """Draw a rounded box with text"""
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.15", facecolor=color,
                           edgecolor='white', alpha=0.9, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight=fontweight)


def draw_diamond(ax, x, y, w, h, text, color, fontsize=8):
    """Draw a diamond (decision) node"""
    diamond_verts = [
        (x, y + h/2),      # top
        (x + w/2, y),      # right
        (x, y - h/2),      # bottom
        (x - w/2, y),      # left
    ]
    diamond = plt.Polygon(diamond_verts, facecolor=color, edgecolor='white',
                          alpha=0.25, linewidth=1.5)
    ax.add_patch(diamond)
    diamond2 = plt.Polygon(diamond_verts, facecolor='none', edgecolor=color,
                           alpha=0.8, linewidth=2.0)
    ax.add_patch(diamond2)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color='#2c3e50', fontweight='bold')


def draw_arrow(ax, x1, y1, x2, y2, color='#555555', lw=1.5):
    """Draw an arrow"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))


# ============================================================
# Figure 8: Agent-Level Strategy Comparison
# ============================================================
def fig8_agent_strategy_comparison():
    """各 Agent 策略与 Fallback 层级对比"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    fig.suptitle('图8：7-Agent 容错策略全景图', fontsize=13, fontweight='bold', color='#2c3e50')

    agents_data = [
        ('Agent 1\n分类器', 'LLM分类\nt=0.1', '规则关键词\n匹配', '默认文科\n论文', 3, '#3498db'),
        ('Agent 2\nRAG检索', '精确匹配\nO(1)', 'ChromaDB\n语义检索', '默认模板', 3, '#2ecc71'),
        ('Agent 3\n关键词', 'LLM层次化\n提取', '正则提取\n术语', '—', 2, '#9b59b6'),
        ('Agent 4\n文献检索', 'arXiv API\n主力', 'Semantic\nScholar', '自主模拟\nMock', 3, '#e74c3c'),
        ('Agent 5\n文献分析', 'LLM结构化\n分析', '简化分析\n模板', '—', 2, '#f39c12'),
        ('Agent 6\n论文撰写', 'LLM一次性\n整体生成', '框架输出', '—', 2, '#1abc9c'),
        ('Agent 7\n格式校验', '规则引擎\n+ LLM双层', '规则评分\nfallback', '—', 2, '#e67e22'),
    ]

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    for i, (name, primary, fb1, fb2, fb_count, color) in enumerate(agents_data):
        y_center = 9.0 - i * 1.3

        # Agent label
        ax.text(0.5, y_center, name, ha='center', va='center',
                fontsize=9, fontweight='bold', color=color)

        # Primary strategy
        rect = FancyBboxPatch((1.5, y_center - 0.45), 2.2, 0.9,
                               boxstyle="round,pad=0.1", facecolor=color, alpha=0.8,
                               edgecolor='white', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(2.6, y_center, primary, ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold')

        # Arrow
        ax.annotate('', xy=(4.0, y_center), xytext=(3.7, y_center),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2))

        # Fallback 1
        rect2 = FancyBboxPatch((4.2, y_center - 0.45), 2.2, 0.9,
                                boxstyle="round,pad=0.1", facecolor=color, alpha=0.3,
                                edgecolor=color, linewidth=1.5)
        ax.add_patch(rect2)
        ax.text(5.3, y_center, fb1, ha='center', va='center',
                fontsize=7.5, color='#2c3e50')

        # Fallback 2 (if exists)
        if fb_count >= 3 and fb2 != '—':
            ax.annotate('', xy=(6.7, y_center), xytext=(6.4, y_center),
                        arrowprops=dict(arrowstyle='->', color='#999999', lw=1.0))
            rect3 = FancyBboxPatch((6.9, y_center - 0.45), 2.2, 0.9,
                                    boxstyle="round,pad=0.1", facecolor=color, alpha=0.15,
                                    edgecolor=color, linewidth=1.2, linestyle='--')
            ax.add_patch(rect3)
            ax.text(8.0, y_center, fb2, ha='center', va='center',
                    fontsize=7.5, color='#666666')

        # Fallback level indicator
        ax.text(9.8, y_center, f'{fb_count}级\n回退', ha='center', va='center',
                fontsize=7, color=color, fontweight='bold')

    # Legend
    ax.text(2.6, 0.3, '■ 主路径', fontsize=8, color='white',
            bbox=dict(facecolor='#3498db', alpha=0.8, boxstyle='round,pad=0.3'))
    ax.text(5.3, 0.3, '■ Fallback 1', fontsize=8, color='#2c3e50',
            bbox=dict(facecolor='#3498db', alpha=0.3, edgecolor='#3498db',
                     boxstyle='round,pad=0.3'))
    ax.text(8.0, 0.3, '■ Fallback 2', fontsize=8, color='#666666',
            bbox=dict(facecolor='#3498db', alpha=0.15, edgecolor='#3498db',
                     boxstyle='round,pad=0.3', linestyle='--'))

    plt.tight_layout()
    return save_fig(fig, 'fig8_agent_strategy_comparison.png')


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  论文配图生成 — 多智能体协作的论文学术写作系统")
    print("=" * 60)
    print(f"  中文字体: {CN_FONT}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  数据来源: {len(RUNS)} 次系统运行")
    print()

    figures = [
        ("图1：7-Agent 流水线架构", fig1_pipeline_architecture),
        ("图2：三级回退链", fig2_fallback_chain),
        ("图3：系统性能仪表盘", fig3_performance_dashboard),
        ("图4：文献检索质量分析", fig4_literature_quality),
        ("图5：格式校验深度分析", fig5_format_quality),
        ("图6：字数自适应策略", fig6_word_count_strategy),
        ("图7：中文查询过滤算法", fig7_chinese_query_filter),
        ("图8：Agent容错策略全景", fig8_agent_strategy_comparison),
    ]

    for name, func in figures:
        print(f"[{name}]")
        func()
        print()

    print(f"\n✓ 全部完成！共生成 {len(figures)} 张配图")
    print(f"  输出目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
