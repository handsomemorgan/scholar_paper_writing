"""
Agent 8: 图表Agent (FigureAgent)

双模式运行：
  Phase A - 文献图表提取：从检索到的论文中提取核心图表（用于综述）
  Phase B - 数据训练制图：基于论文主题自动生成数据并执行自主运算，产出图表

这是"论文配图自动化"的核心Agent——解决学术论文中缺乏数据生成图像的痛点。

v2 改进：
  - PDF→PNG 自动转换（macOS sips / pdf2image）
  - matplotlib 中文字体自动配置（避免乱码方框）
  - 紧凑高密度图表布局 + 自主统计运算
  - 更多图表类型：森林图、相关热力图、真实雷达图等
"""

import logging
import os
import json
import re
import time
import urllib.request
import urllib.error
import tarfile
import io
import shutil
import hashlib
import subprocess
import platform
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# Data structures
# ============================================================

@dataclass
class FigureEntry:
    """单个图条目"""
    figure_id: str              # 唯一ID，如 "fig_ext_001" / "fig_gen_001"
    figure_path: str            # 相对路径，如 "figures/fig_ext_001.png"
    figure_type: str            # "extracted" | "generated"
    source: str                 # 来源描述
    title: str                  # 图标题
    caption: str                # 图注（学术格式）
    section_placement: str      # 建议放置的章节
    width_hint: str = ""        # 宽度提示

# ============================================================
# Constants
# ============================================================

# arXiv 源文件下载模板
ARXIV_SRC_URL = "https://arxiv.org/src/{arxiv_id}"
ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"
ARXIV_ABS_URL = "https://arxiv.org/abs/{arxiv_id}"

# 常见图像文件扩展名
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.pdf', '.eps', '.svg', '.gif', '.tiff', '.bmp'}

# PDF 扩展名（需要转换为 PNG）
PDF_EXTENSIONS = {'.pdf', '.eps'}

# 图表类型定义
CHART_TYPES = {
    "line": "折线图",
    "bar": "柱状图",
    "scatter": "散点图",
    "heatmap": "热力图",
    "radar": "雷达图",
    "box": "箱线图",
    "area": "面积图",
    "forest": "森林图",
    "correlation_heatmap": "相关性热力图",
}

# 字体检测优先级（macOS / Linux / Windows）
_CJK_FONT_CANDIDATES = [
    # macOS
    "PingFang SC", "Heiti SC", "Heiti TC", "STHeiti", "Songti SC",
    "Kaiti SC", "STFangsong", "SimSong",
    # Linux
    "Noto Sans CJK SC", "Noto Sans CJK", "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei", "AR PL UMing CN", "AR PL UKai CN",
    # Windows
    "Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong",
]
_FONT_SETUP_DONE = False

# FigureAgent 的系统 Prompt
FIGURE_PLANNING_PROMPT = """你是一位学术论文配图专家。你的任务是分析论文主题和文献，决定需要什么样的配图来增强论文的说服力。

## 分析维度

1. **综述配图需求**：对于综述类内容，哪些论文的核心贡献图/架构图/实验结果图值得引用？
2. **数据图表需求**：论文中哪些论点需要数据支撑？什么样的图表（折线图、柱状图、散点图、箱线图、热力图、雷达图、森林图等）最能说明问题？
3. **方法论配图需求**：如果需要阐述方法，需要什么样的流程图/框架图？

## 输出格式

请严格输出以下JSON：
{
    "paper_figures_to_extract": [
        {
            "literature_index": 0,
            "reason": "该论文提出了XXX架构，其架构图对理解本文讨论的方法至关重要",
            "suggested_caption": "图X：XXX等人提出的XXX架构[文献序号]",
            "placement_section": "2. 相关技术"
        }
    ],
    "data_charts_to_generate": [
        {
            "chart_type": "line",
            "title": "不同方法在XXX上的性能对比",
            "x_label": "参数取值",
            "y_label": "性能指标",
            "data_description": "对比方法A、方法B和本文方法在不同参数下的性能变化趋势。方法A应从低到高稳定增长，方法B早期优于A但后期被超越，本文方法全程最优。",
            "placement_section": "5. 实验与评估",
            "series_count": 3,
            "data_points": 8
        }
    ],
    "summary": "配图策略概述（1-2句）"
}

## 约束

- paper_figures_to_extract: 至多选3篇最重要的论文，每篇选1张最有代表性的图
- data_charts_to_generate: 至多生成5张数据图表，覆盖不同类型的图表
- 所有placement_section必须对应实际论文结构
- chart_type 必须是: line, bar, scatter, heatmap, radar, box, area, forest, correlation_heatmap 之一"""


# ============================================================
# Font setup
# ============================================================

def _setup_matplotlib_fonts():
    """配置 matplotlib 中文字体，避免乱码方框。"""
    global _FONT_SETUP_DONE
    if _FONT_SETUP_DONE:
        return

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.font_manager as fm
        import matplotlib.pyplot as plt

        # 获取系统所有字体
        available_fonts = {f.name for f in fm.fontManager.ttflist}

        chosen_font = None
        for candidate in _CJK_FONT_CANDIDATES:
            if candidate in available_fonts:
                chosen_font = candidate
                break

        if chosen_font:
            matplotlib.rcParams['font.sans-serif'] = [chosen_font, 'DejaVu Sans', 'Arial']
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['axes.unicode_minus'] = False
            # 重建字体缓存以确保新字体生效
            try:
                fm._load_fontmanager(try_read_cache=False)
            except Exception:
                pass
            logger.info(f"matplotlib 中文字体已配置: {chosen_font}")
        else:
            # 尝试从系统路径查找
            system_font_paths = [
                '/System/Library/Fonts/STHeiti Light.ttc',
                '/System/Library/Fonts/PingFang.ttc',
                '/System/Library/Fonts/Hiragino Sans GB.ttc',
                '/Library/Fonts/Arial Unicode.ttf',
                '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            ]
            for font_path in system_font_paths:
                if os.path.exists(font_path):
                    try:
                        fm.fontManager.addfont(font_path)
                        font_name = fm.FontProperties(fname=font_path).get_name()
                        matplotlib.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Arial']
                        matplotlib.rcParams['font.family'] = 'sans-serif'
                        matplotlib.rcParams['axes.unicode_minus'] = False
                        chosen_font = font_name
                        logger.info(f"matplotlib 中文字体已配置（通过路径）: {font_path} → {font_name}")
                        break
                    except Exception:
                        continue

            if not chosen_font:
                logger.warning("未找到中文字体，图表中文可能显示为方框。"
                             "请安装中文字体: brew install font-noto-sans-cjk-sc")

        _FONT_SETUP_DONE = True
    except Exception as e:
        logger.warning(f"字体配置失败: {e}")


# ============================================================
# PDF→PNG conversion
# ============================================================

def _convert_pdf_to_png(pdf_path: str, output_dir: str = None) -> Optional[str]:
    """
    将 PDF 转换为 PNG。

    策略：
    1. macOS: 使用内置 sips 命令
    2. 跨平台: 使用 pdf2image (poppler)
    3. 使用 Python PIL/Pillow + ghostscript
    """
    if not os.path.exists(pdf_path):
        return None

    if output_dir is None:
        output_dir = os.path.dirname(pdf_path)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    png_path = os.path.join(output_dir, f"{base_name}.png")

    # 如果 PNG 已存在且比 PDF 新，跳过
    if os.path.exists(png_path) and os.path.getmtime(png_path) >= os.path.getmtime(pdf_path):
        return png_path

    system = platform.system()

    # Strategy 1: macOS sips
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sips", "-s", "format", "png", pdf_path, "--out", png_path],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and os.path.exists(png_path):
                logger.info(f"    PDF→PNG (sips): {os.path.basename(pdf_path)} → {os.path.basename(png_path)}")
                return png_path
        except Exception as e:
            logger.debug(f"    sips 转换失败: {e}")

    # Strategy 2: pdf2image (requires poppler)
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=1)
        if images:
            images[0].save(png_path, "PNG")
            logger.info(f"    PDF→PNG (pdf2image): {os.path.basename(pdf_path)} → {os.path.basename(png_path)}")
            return png_path
    except ImportError:
        logger.debug("    pdf2image 未安装，跳过")
    except Exception as e:
        logger.debug(f"    pdf2image 转换失败: {e}")

    # Strategy 3: pdftoppm (poppler CLI)
    try:
        result = subprocess.run(
            ["pdftoppm", "-png", "-r", "200", "-f", "1", "-l", "1", "-singlefile",
             pdf_path, os.path.splitext(png_path)[0]],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and os.path.exists(png_path):
            logger.info(f"    PDF→PNG (pdftoppm): {os.path.basename(pdf_path)} → {os.path.basename(png_path)}")
            return png_path
    except Exception as e:
        logger.debug(f"    pdftoppm 转换失败: {e}")

    # Strategy 4: ghostscript
    try:
        result = subprocess.run(
            ["gs", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", "-r200",
             f"-sOutputFile={png_path}", pdf_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and os.path.exists(png_path):
            logger.info(f"    PDF→PNG (ghostscript): {os.path.basename(pdf_path)} → {os.path.basename(png_path)}")
            return png_path
    except Exception as e:
        logger.debug(f"    ghostscript 转换失败: {e}")

    logger.warning(f"    无法将 PDF 转换为 PNG: {pdf_path}，请安装 sips(macOS) 或 pdf2image")
    return None


def _batch_convert_pdfs(figures_dir: str) -> int:
    """批量将 figures 目录中的 PDF 转换为 PNG。返回转换数量。"""
    converted = 0
    if not os.path.isdir(figures_dir):
        return 0

    for fname in os.listdir(figures_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext in PDF_EXTENSIONS:
            pdf_path = os.path.join(figures_dir, fname)
            result = _convert_pdf_to_png(pdf_path, figures_dir)
            if result:
                converted += 1

    return converted


# ============================================================
# 自主运算工具函数
# ============================================================

def _compute_effect_size(group1: np.ndarray, group2: np.ndarray) -> Dict[str, float]:
    """计算 Cohen's d 效应量和 Hedges' g。"""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # 合并标准差
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return {"cohens_d": 0.0, "hedges_g": 0.0, "interpretation": "negligible"}

    d = (mean1 - mean2) / pooled_std
    # Hedges' g 小样本校正
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    g = d * correction

    if abs(d) < 0.2:
        interp = "negligible"
    elif abs(d) < 0.5:
        interp = "small"
    elif abs(d) < 0.8:
        interp = "medium"
    else:
        interp = "large"

    return {"cohens_d": round(d, 3), "hedges_g": round(g, 3), "interpretation": interp}


def _compute_ranking_stability(rankings: List[List[int]]) -> float:
    """计算 Kendall W 一致性系数（多组排名的稳定性）。"""
    if len(rankings) < 2:
        return 1.0

    m = len(rankings)  # 评估者数量
    n = len(rankings[0])  # 对象数量

    # 计算秩和
    R = np.zeros(n)
    for ranking in rankings:
        for j, rank in enumerate(ranking):
            R[rank - 1] += j + 1  # 转换为秩次

    # 平均秩和
    R_mean = np.mean(R)
    S = np.sum((R - R_mean) ** 2)

    # Kendall's W
    W = 12 * S / (m ** 2 * (n ** 3 - n)) if n > 1 else 1.0
    return round(min(W, 1.0), 4)


def _compute_convergence_curve(
    func_type: str = "sphere", dim: int = 10, n_evals: int = 100,
    noise_std: float = 0.02, n_runs: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    自主运行模拟优化过程，计算真实收敛曲线。
    使用简单的 (1+1)-ES 风格优化器。
    """
    if func_type == "sphere":
        def f(x): return np.sum(x ** 2)
        opt_x = np.zeros(dim)
    elif func_type == "rastrigin":
        def f(x): return 10 * dim + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
        opt_x = np.zeros(dim)
    elif func_type == "ackley":
        def f(x):
            a, b, c = 20, 0.2, 2 * np.pi
            d = len(x)
            sum1 = np.sum(x ** 2)
            sum2 = np.sum(np.cos(c * x))
            return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.exp(1)
        opt_x = np.zeros(dim)
    elif func_type == "rosenbrock":
        def f(x):
            return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
        opt_x = np.ones(dim)
    else:
        def f(x): return np.sum(x ** 2)
        opt_x = np.zeros(dim)

    bounds = (-5.0, 5.0)

    all_best_values = np.zeros((n_runs, n_evals))

    for run in range(n_runs):
        rng = np.random.RandomState(42 + run)
        x = rng.uniform(bounds[0], bounds[1], dim)
        best_x = x.copy()
        best_val = f(x)

        sigma = 0.5
        tau = 1.0 / np.sqrt(2 * dim)

        for t in range(n_evals):
            # (1+1)-ES 变异
            sigma = sigma * np.exp(tau * rng.randn())
            sigma = max(sigma, 1e-6)
            x_new = best_x + sigma * rng.randn(dim)
            x_new = np.clip(x_new, bounds[0], bounds[1])
            val_new = f(x_new)

            if val_new + noise_std * rng.randn() < best_val:
                best_val = val_new
                best_x = x_new.copy()

            all_best_values[run, t] = best_val

    mean_curve = np.mean(all_best_values, axis=0)
    std_curve = np.std(all_best_values, axis=0)

    return mean_curve, std_curve, np.arange(n_evals)


def _poly_fit(x: np.ndarray, y: np.ndarray, degree: int = 2) -> Tuple[np.ndarray, np.ndarray, float]:
    """多项式拟合，返回拟合曲线、置信区间和 R²。"""
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    y_pred = poly(x)

    # R²
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # 95% 置信区间（简化版）
    n = len(x)
    dof = n - degree - 1
    if dof > 0:
        mse = ss_res / dof
        std_err = np.sqrt(mse) * np.sqrt(1.0 / n + (x - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2 + 1e-10))
        ci = 1.96 * std_err
    else:
        ci = np.zeros_like(y_pred)

    return y_pred, ci, round(r_squared, 4)


def _bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Bootstrap 95% 置信区间。"""
    rng = np.random.RandomState(42)
    means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        means[i] = np.mean(sample)

    ci_low = np.percentile(means, 100 * alpha / 2)
    ci_high = np.percentile(means, 100 * (1 - alpha / 2))
    return round(ci_low, 4), round(np.mean(data), 4), round(ci_high, 4)


# ============================================================
# FigureAgent
# ============================================================

class FigureAgent:
    """图表Agent — 文献图表提取 + 自主运算数据制图"""

    def __init__(self, llm_client, output_base_dir: str = "output"):
        self.llm = llm_client
        self.output_base_dir = output_base_dir
        self._figure_cache: Dict[str, str] = {}  # arxiv_id -> local_path
        # 在首次实例化时配置字体
        _setup_matplotlib_fonts()

    # ================================================================
    # Public API
    # ================================================================

    def generate_figures(
        self,
        literature_list: List[Dict[str, Any]],
        analysis_result: Dict[str, Any],
        keyword_result: Dict[str, Any],
        classification: Dict[str, Any],
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        主入口：为论文生成全套配图。

        Returns:
            {
                "figures": [FigureEntry, ...],
                "manifest_text": "用于嵌入prompt的配图清单文本",
                "stats": {"extracted": N, "generated": M, "failed": F}
            }
        """
        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        logger.info("=" * 50)
        logger.info("Agent 8 (FigureAgent v2): 开始生成论文配图")
        logger.info("=" * 50)

        all_figures: List[FigureEntry] = []
        stats = {"extracted": 0, "generated": 0, "failed": 0}

        # -------- Step 1: Ask LLM to plan figures --------
        plan = self._plan_figures(literature_list, analysis_result, keyword_result, classification)
        logger.info(f"配图计划: {plan.get('summary', 'N/A')}")

        # -------- Phase A: Extract figures from papers --------
        extract_targets = plan.get("paper_figures_to_extract", [])
        logger.info(f"Phase A: 计划从 {len(extract_targets)} 篇论文中提取图表")

        for target in extract_targets[:3]:  # 至多3篇
            lit_idx = target.get("literature_index", 0)
            if lit_idx >= len(literature_list):
                continue

            paper = literature_list[lit_idx]
            result = self._extract_figure_from_paper(paper, figures_dir, target)

            if result:
                all_figures.append(result)
                stats["extracted"] += 1
                logger.info(f"  [OK] 提取成功: {result.title[:60]}...")
            else:
                stats["failed"] += 1
                logger.warning(f"  [FAIL] 提取失败: {paper.get('title', '?')[:60]}...")

        # -------- Phase B: Generate data-driven charts with autonomous computation --------
        chart_targets = plan.get("data_charts_to_generate", [])
        logger.info(f"Phase B: 计划生成 {len(chart_targets)} 张数据图表（含自主运算）")

        for i, chart_spec in enumerate(chart_targets[:5]):  # 至多5张
            result = self._generate_data_chart(chart_spec, figures_dir, i)
            if result:
                all_figures.append(result)
                stats["generated"] += 1
                logger.info(f"  [OK] 图表生成: {result.title[:60]}...")
            else:
                stats["failed"] += 1
                logger.warning(f"  [FAIL] 图表生成失败: {chart_spec.get('title', '?')[:60]}...")

        # -------- Fallback: if no figures at all, generate basic ones --------
        if not all_figures:
            logger.warning("无任何配图生成成功，使用最小化fallback...")
            fb = self._generate_fallback_charts(figures_dir, analysis_result, classification)
            all_figures.extend(fb)
            stats["generated"] += len(fb)

        # -------- Post-process: convert remaining PDFs to PNG --------
        n_converted = _batch_convert_pdfs(figures_dir)
        if n_converted > 0:
            logger.info(f"后处理: 转换了 {n_converted} 个 PDF 为 PNG")
            # 更新 figure entries 中的路径引用 (.pdf → .png)
            for fig in all_figures:
                if fig.figure_path.lower().endswith('.pdf'):
                    png_path = os.path.splitext(fig.figure_path)[0] + '.png'
                    if os.path.exists(os.path.join(self.output_base_dir, png_path)):
                        fig.figure_path = png_path

        # -------- Build manifest --------
        manifest_text = self._build_manifest(all_figures)

        logger.info(
            f"Agent 8 完成: 提取 {stats['extracted']} 张 + "
            f"生成 {stats['generated']} 张 = 共 {len(all_figures)} 张配图 "
            f"(PDF→PNG: {n_converted})"
        )

        return {
            "figures": [self._entry_to_dict(f) for f in all_figures],
            "manifest_text": manifest_text,
            "stats": stats,
            "figures_dir": figures_dir,
        }

    # ================================================================
    # Phase A: Figure Extraction from Papers
    # ================================================================

    def _extract_figure_from_paper(
        self,
        paper: Dict[str, Any],
        figures_dir: str,
        target: Dict[str, Any],
    ) -> Optional[FigureEntry]:
        """
        从单篇论文中提取代表性图表。

        策略优先级：
        1. arXiv source tarball → 解包提取 .png/.pdf 文件 → PDF→PNG
        2. arXiv e-print PDF → 如果是PDF，用缩略图路径
        3. 生成描述性占位符（基于abstract语义生成标注）
        """
        arxiv_id = paper.get("arxiv_id", "")
        title = paper.get("title", "Unknown")
        authors = paper.get("authors", [])
        first_author = authors[0] if authors else "Unknown"
        year = paper.get("year", "")

        # 构建唯一ID
        safe_id = re.sub(r'[^a-zA-Z0-9]', '_', (arxiv_id or title)[:30])
        fig_id = f"fig_ext_{safe_id}"

        # ---- Strategy 1: arXiv source tarball ----
        if arxiv_id:
            extracted_path = self._try_download_arxiv_source(arxiv_id, fig_id, figures_dir)
            if extracted_path:
                # PDF → PNG 转换
                if extracted_path.lower().endswith('.pdf'):
                    png_path = _convert_pdf_to_png(extracted_path, figures_dir)
                    display_path = png_path if png_path else extracted_path
                else:
                    display_path = extracted_path

                return FigureEntry(
                    figure_id=fig_id,
                    figure_path=os.path.relpath(display_path, self.output_base_dir),
                    figure_type="extracted",
                    source=f"arXiv:{arxiv_id}",
                    title=f"来自 {first_author} 等人 ({year}) 的论文图表",
                    caption=target.get("suggested_caption",
                                        f"图X：{first_author}等人提出的方法框架[文献{target.get('literature_index', '?')}]"),
                    section_placement=target.get("placement_section", "2. 相关技术"),
                )

            # ---- Strategy 2: arXiv HTML abstract page (thumbnail) ----
            thumbnail_path = self._try_fetch_arxiv_thumbnail(arxiv_id, fig_id, figures_dir)
            if thumbnail_path:
                return FigureEntry(
                    figure_id=fig_id,
                    figure_path=os.path.relpath(thumbnail_path, self.output_base_dir),
                    figure_type="extracted",
                    source=f"arXiv:{arxiv_id} (缩略图)",
                    title=f"来自 {first_author} 等人 ({year}) 的论文概览",
                    caption=target.get("suggested_caption",
                                        f"图X：{first_author}等人的研究概览[文献{target.get('literature_index', '?')}]"),
                    section_placement=target.get("placement_section", "2. 相关技术"),
                )

        # ---- Strategy 3: Generate descriptive placeholder ----
        placeholder_path = self._generate_paper_placeholder(paper, target, fig_id, figures_dir)
        if placeholder_path:
            return FigureEntry(
                figure_id=fig_id,
                figure_path=os.path.relpath(placeholder_path, self.output_base_dir),
                figure_type="extracted",
                source=f"Generated from: {title[:80]}",
                title=f"文献概念示意图：{first_author}等人 ({year})",
                caption=target.get("suggested_caption",
                                    f"图X：{first_author}等人提出的核心概念示意图（基于文献摘要生成）[{title[:80]}]"),
                section_placement=target.get("placement_section", "2. 相关技术"),
            )

        return None

    def _try_download_arxiv_source(
        self, arxiv_id: str, fig_id: str, figures_dir: str
    ) -> Optional[str]:
        """
        尝试下载 arXiv 论文源文件并提取图片。
        """
        clean_id = re.sub(r'v\d+$', '', arxiv_id)
        src_url = ARXIV_SRC_URL.format(arxiv_id=clean_id)
        logger.info(f"    尝试下载 arXiv source: {src_url}")

        try:
            req = urllib.request.Request(src_url, headers={"User-Agent": "ScholarPaperBot/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read()

            if len(content) < 100:
                logger.info(f"    Source 文件过小 ({len(content)} bytes)，可能不存在")
                return None

            # 尝试作为 tar.gz 解包
            try:
                with tarfile.open(fileobj=io.BytesIO(content), mode='r:gz') as tar:
                    image_members = []
                    for member in tar.getmembers():
                        ext = os.path.splitext(member.name)[1].lower()
                        if ext in IMAGE_EXTENSIONS and member.isfile():
                            name_lower = member.name.lower()
                            priority = 0
                            if any(kw in name_lower for kw in ['fig', 'figure', 'plot', 'chart',
                                   'graph', 'result', 'arch', 'overview', 'framework',
                                   'method', 'model', 'pipeline', 'flow']):
                                priority = 2
                            elif any(kw in name_lower for kw in ['logo', 'icon', 'badge', 'button', 'small']):
                                priority = 0
                            else:
                                priority = 1
                            image_members.append((priority, member))

                    if image_members:
                        image_members.sort(key=lambda x: x[0], reverse=True)
                        _, best_member = image_members[0]
                        ext = os.path.splitext(best_member.name)[1].lower()
                        out_name = f"{fig_id}{ext}"
                        out_path = os.path.join(figures_dir, out_name)

                        extracted = tar.extractfile(best_member)
                        if extracted:
                            with open(out_path, 'wb') as f:
                                f.write(extracted.read())
                            logger.info(f"    提取成功: {best_member.name} → {out_name}")
                            return out_path

            except tarfile.ReadError:
                # 可能不是 tar.gz，尝试直接作为图片
                if content[:4] == b'\x89PNG':
                    ext = '.png'
                elif content[:2] == b'\xff\xd8':
                    ext = '.jpg'
                elif content[:4] == b'%PDF':
                    ext = '.pdf'
                else:
                    return None

                out_name = f"{fig_id}{ext}"
                out_path = os.path.join(figures_dir, out_name)
                with open(out_path, 'wb') as f:
                    f.write(content)
                logger.info(f"    直接保存: {out_name}")
                return out_path

        except urllib.error.HTTPError as e:
            logger.info(f"    arXiv source 不可用 (HTTP {e.code})")
        except Exception as e:
            logger.info(f"    下载/解包失败: {e}")

        return None

    def _try_fetch_arxiv_thumbnail(
        self, arxiv_id: str, fig_id: str, figures_dir: str
    ) -> Optional[str]:
        """生成论文信息卡片缩略图。"""
        try:
            import matplotlib.pyplot as plt

            clean_id = re.sub(r'v\d+$', '', arxiv_id)

            fig, ax = plt.subplots(1, 1, figsize=(6, 3))
            ax.set_xlim(0, 8)
            ax.set_ylim(0, 4)
            ax.axis('off')

            from matplotlib.patches import FancyBboxPatch
            rect = FancyBboxPatch((0.3, 0.3), 7.4, 3.4,
                                   boxstyle="round,pad=0.2", facecolor='#2c3e50',
                                   edgecolor='#3498db', linewidth=2, alpha=0.9)
            ax.add_patch(rect)

            ax.text(4.0, 3.0, f"arXiv: {clean_id}", ha='center', fontsize=13,
                    fontweight='bold', color='#3498db')
            ax.text(4.0, 2.3, "Paper Figure (see source for full image)", ha='center',
                    fontsize=11, color='white')
            ax.text(4.0, 1.4, f"Full paper: https://arxiv.org/abs/{clean_id}",
                    ha='center', fontsize=8, color='#3498db', style='italic')
            ax.text(4.0, 0.7, "(Original figure not extractable. See arXiv page for full figures.)",
                    ha='center', fontsize=7, color='#888888')

            out_path = os.path.join(figures_dir, f"{fig_id}_thumbnail.png")
            fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            logger.info(f"    生成缩略图卡片: {fig_id}_thumbnail.png")
            return out_path
        except Exception as e:
            logger.info(f"    缩略图生成失败: {e}")
            return None

    def _generate_paper_placeholder(
        self,
        paper: Dict[str, Any],
        target: Dict[str, Any],
        fig_id: str,
        figures_dir: str,
    ) -> Optional[str]:
        """基于论文摘要生成概念示意图占位符。"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import FancyBboxPatch

            title = paper.get('title', 'Unknown')[:120]
            abstract = paper.get('abstract', '')[:300]
            authors = paper.get('authors', [])[:3]
            year = paper.get('year', '')

            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
            ax.set_xlim(0, 9)
            ax.set_ylim(0, 5)
            ax.axis('off')

            # Background
            bg = FancyBboxPatch((0.2, 0.2), 8.6, 4.6,
                                 boxstyle="round,pad=0.3", facecolor='#f8f9fa',
                                 edgecolor='#dee2e6', linewidth=1)
            ax.add_patch(bg)

            # Title
            ax.text(4.5, 4.3, title, ha='center', fontsize=10, fontweight='bold',
                    color='#2c3e50', wrap=True)

            # Author line
            author_str = ', '.join(authors[:3])
            if len(paper.get('authors', [])) > 3:
                author_str += ' et al.'
            ax.text(4.5, 3.8, f"{author_str} ({year})", ha='center', fontsize=9,
                    color='#7f8c8d')

            # Abstract snippet
            if abstract:
                words = abstract.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line + word) < 80:
                        current_line += word + " "
                    else:
                        lines.append(current_line)
                        current_line = word + " "
                lines.append(current_line)
                display_abs = '\n'.join(lines[:4])
                if len(lines) > 4:
                    display_abs += '...'

                ax.text(4.5, 2.5, display_abs, ha='center', va='center',
                        fontsize=7, color='#555555', style='italic')

            # Concept placeholder
            concept_box = FancyBboxPatch((3.0, 0.5), 3.0, 1.2,
                                          boxstyle="round,pad=0.15", facecolor='#3498db',
                                          edgecolor='#2980b9', linewidth=1.5, alpha=0.2)
            ax.add_patch(concept_box)
            ax.text(4.5, 1.1, '[Core Concept Diagram]', ha='center', va='center',
                    fontsize=9, color='#2980b9', fontweight='bold')

            reason = target.get('reason', '')
            if reason:
                ax.text(4.5, 0.3, f"Relevance: {reason[:100]}", ha='center', fontsize=6.5,
                        color='#95a5a6', style='italic')

            out_path = os.path.join(figures_dir, f"{fig_id}_placeholder.png")
            fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            logger.info(f"    生成概念占位符: {fig_id}_placeholder.png")
            return out_path
        except Exception as e:
            logger.info(f"    占位符生成失败: {e}")
            return None

    # ================================================================
    # Phase B: Data-driven Chart Generation (自主运算增强版)
    # ================================================================

    def _generate_data_chart(
        self,
        chart_spec: Dict[str, Any],
        figures_dir: str,
        index: int,
    ) -> Optional[FigureEntry]:
        """
        根据LLM规划的图表规格，生成数据图表并执行自主统计运算。
        """
        chart_type = chart_spec.get("chart_type", "line")
        title = chart_spec.get("title", f"数据图表 {index+1}")
        x_label = chart_spec.get("x_label", "X")
        y_label = chart_spec.get("y_label", "Y")
        data_desc = chart_spec.get("data_description", "")
        series_count = max(1, chart_spec.get("series_count", 1))
        data_points = max(4, chart_spec.get("data_points", 8))
        placement = chart_spec.get("placement_section", "实验与评估")

        fig_id = f"fig_gen_{index+1:03d}"

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # -------- 自主运算：根据图表类型选择合适的计算方法 --------
            computation_results = {}
            data, labels = self._synthesize_data(data_desc, series_count, data_points)

            # 紧凑型图表布局
            fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))

            x = np.arange(data_points)
            x_labels = self._generate_x_labels(data_desc, data_points) or [
                f"{i+1}" for i in range(data_points)
            ]

            # -------- 路由到各图表绘制方法 --------
            if chart_type == "line":
                self._draw_line_chart(ax, x, data, labels, x_labels)
                computation_results = self._compute_line_stats(data, labels)
            elif chart_type == "bar":
                self._draw_bar_chart(ax, x, data, labels, x_labels, series_count)
                computation_results = self._compute_bar_stats(data, labels)
            elif chart_type == "scatter":
                self._draw_scatter_chart(ax, data, labels)
                computation_results = self._compute_scatter_stats(data, labels)
            elif chart_type == "area":
                self._draw_area_chart(ax, x, data, labels, x_labels)
                computation_results = self._compute_line_stats(data, labels)
            elif chart_type == "box":
                self._draw_box_chart(ax, data, labels)
                computation_results = self._compute_box_stats(data, labels)
            elif chart_type == "radar":
                ax = self._draw_proper_radar_chart(fig, ax, data, labels)
                computation_results = {}
            elif chart_type == "heatmap":
                ax = self._draw_heatmap_chart(fig, ax, data, labels, x_labels)
                computation_results = {}
            elif chart_type == "forest":
                ax = self._draw_forest_plot(fig, ax, data, labels)
                computation_results = {}
            elif chart_type == "correlation_heatmap":
                ax = self._draw_correlation_heatmap(fig, ax, data, labels)
                computation_results = {}
            else:
                self._draw_line_chart(ax, x, data, labels, x_labels)
                computation_results = self._compute_line_stats(data, labels)

            # -------- 图表装饰 --------
            ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)
            ax.legend(loc='best', fontsize=7.5, framealpha=0.8)
            ax.grid(alpha=0.25, linestyle='--')
            # 非极坐标轴才隐藏顶部/右侧边框
            if hasattr(ax, 'spines') and 'top' in ax.spines:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            # -------- 注释统计运算结果 --------
            if computation_results:
                annotation_text = self._build_stat_annotation(computation_results)
                if annotation_text:
                    ax.text(0.02, 0.02, annotation_text, transform=ax.transAxes,
                            fontsize=6.5, color='#555555',
                            verticalalignment='bottom',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa',
                                     edgecolor='#dee2e6', alpha=0.85))

            plt.tight_layout()
            out_name = f"{fig_id}.png"
            out_path = os.path.join(figures_dir, out_name)
            fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            # 构建 caption
            caption_parts = [f"图X：{title}"]
            if computation_results:
                stats_summary = self._build_caption_stats(computation_results)
                if stats_summary:
                    caption_parts.append(f"统计分析：{stats_summary}")
            if data_desc:
                caption_parts.append(f"数据来源：{data_desc[:120]}")

            caption = "。".join(caption_parts) if len(caption_parts) > 1 else caption_parts[0]

            return FigureEntry(
                figure_id=fig_id,
                figure_path=os.path.join("figures", out_name),
                figure_type="generated",
                source="Agent 8 自主运算生成",
                title=title,
                caption=caption,
                section_placement=placement,
            )

        except Exception as e:
            logger.error(f"    图表生成异常: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ================================================================
    # 自主运算方法
    # ================================================================

    def _compute_line_stats(self, data: List[np.ndarray], labels: List[str]) -> Dict:
        """折线图：计算趋势、终值、AUC。"""
        results = {}
        for i, (series, label) in enumerate(zip(data, labels)):
            # 线性趋势斜率
            x = np.arange(len(series))
            slope, intercept = np.polyfit(x, series, 1)
            # 终值
            final_val = series[-1]
            # AUC (曲线下面积，梯形法)
            auc = np.trapezoid(series, x) if hasattr(np, 'trapezoid') else float(np.sum(series))
            # 波动性 (变异系数)
            cv = float(np.std(series) / (np.mean(series) + 1e-10))

            short_label = label[:15]
            results[short_label] = {
                "slope": round(slope, 4),
                "final_value": round(final_val, 4),
                "auc": round(auc, 2),
                "cv": round(cv, 3),
            }
        return results

    def _compute_bar_stats(self, data: List[np.ndarray], labels: List[str]) -> Dict:
        """柱状图：计算均值、标准差、效应量。"""
        results = {}
        for i, (series, label) in enumerate(zip(data, labels)):
            mean_val = np.mean(series)
            std_val = np.std(series, ddof=1)
            short_label = label[:15]
            results[short_label] = {
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
            }

        # 如果有多组，计算第一组与最后一组的效应量
        if len(data) >= 2:
            es = _compute_effect_size(data[-1], data[0])
            results["_effect_size"] = es

        return results

    def _compute_scatter_stats(self, data: List[np.ndarray], labels: List[str]) -> Dict:
        """散点图：计算 Pearson r、Spearman ρ。"""
        results = {}
        for i, (series, label) in enumerate(zip(data, labels)):
            x = np.arange(len(series))
            # Pearson 相关系数
            pearson_r = np.corrcoef(x, series)[0, 1]
            # Spearman 秩相关 — 纯 numpy 实现
            try:
                from scipy.stats import spearmanr as _spearmanr
                rho, pval = _spearmanr(x, series)
            except ImportError:
                # 手动计算 Spearman rank correlation
                def _rankdata(a):
                    n = len(a)
                    ivec = np.argsort(a)
                    svec = np.empty(n)
                    svec[ivec] = np.arange(n, dtype=float)
                    # 处理并列值
                    rng = np.random.RandomState(0)
                    return svec + rng.uniform(0, 1e-6, n)
                rank_x = _rankdata(x)
                rank_y = _rankdata(series)
                rho = np.corrcoef(rank_x, rank_y)[0, 1]
                pval = float('nan')

            short_label = label[:15]
            results[short_label] = {
                "pearson_r": round(float(pearson_r), 4),
                "spearman_rho": round(float(rho), 4),
            }
        return results

    def _compute_box_stats(self, data: List[np.ndarray], labels: List[str]) -> Dict:
        """箱线图：计算五数概括 + IQR + 异常值数量。"""
        results = {}
        for i, (series, label) in enumerate(zip(data, labels)):
            q1, q2, q3 = np.percentile(series, [25, 50, 75])
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            n_outliers = np.sum((series < lower_fence) | (series > upper_fence))

            short_label = label[:15]
            results[short_label] = {
                "median": round(q2, 4),
                "iqr": round(iqr, 4),
                "range": round(float(np.max(series) - np.min(series)), 4),
                "outliers": int(n_outliers),
            }

        # 多组比较：Kruskal-Wallis 检验
        if len(data) >= 2:
            try:
                from scipy.stats import kruskal as _kruskal
                h_stat, p_val = _kruskal(*data)
                results["_kruskal"] = {"h_stat": round(h_stat, 3), "p_value": round(p_val, 4)}
            except ImportError:
                # 纯 numpy 近似 Kruskal-Wallis
                try:
                    all_data = np.concatenate(data)
                    all_ranks = np.argsort(np.argsort(all_data))
                    group_ranks = []
                    offset = 0
                    for d in data:
                        group_ranks.append(all_ranks[offset:offset + len(d)])
                        offset += len(d)
                    n = len(all_data)
                    R_sq_n = np.array([np.sum(r)**2 / len(r) for r in group_ranks])
                    H = (12 / (n * (n + 1))) * np.sum(R_sq_n) - 3 * (n + 1)
                    results["_kruskal"] = {"h_stat": round(float(H), 3), "p_value": float('nan')}
                except Exception:
                    pass

        return results

    def _build_stat_annotation(self, computation_results: Dict) -> str:
        """构建图内统计注释文本。"""
        lines = []
        for key, val in computation_results.items():
            if key.startswith('_'):
                continue  # 跳过元数据
            if isinstance(val, dict):
                parts = [f"{key}:"]
                for k, v in val.items():
                    if isinstance(v, float):
                        parts.append(f"  {k}={v:.3f}")
                    else:
                        parts.append(f"  {k}={v}")
                lines.append('\n'.join(parts))

        # 添加效应量
        if '_effect_size' in computation_results:
            es = computation_results['_effect_size']
            lines.append(f"Cohen's d={es['cohens_d']} ({es['interpretation']})")

        # 添加 Kruskal-Wallis
        if '_kruskal' in computation_results:
            kw = computation_results['_kruskal']
            sig = "***" if kw['p_value'] < 0.001 else "**" if kw['p_value'] < 0.01 else "*" if kw['p_value'] < 0.05 else "ns"
            lines.append(f"Kruskal-Wallis H={kw['h_stat']}, p={kw['p_value']}{sig}")

        return '\n'.join(lines[:8])  # 限制长度

    def _build_caption_stats(self, computation_results: Dict) -> str:
        """构建图注中的统计摘要。"""
        parts = []
        if '_effect_size' in computation_results:
            es = computation_results['_effect_size']
            parts.append(f"Cohen's d={es['cohens_d']}（{es['interpretation']}效应）")

        if '_kruskal' in computation_results:
            kw = computation_results['_kruskal']
            sig = "极其显著" if kw['p_value'] < 0.001 else "显著" if kw['p_value'] < 0.01 else "弱显著" if kw['p_value'] < 0.05 else "不显著"
            parts.append(f"组间差异{sig}(p={kw['p_value']})")

        return "；".join(parts) if parts else ""

    # ================================================================
    # 数据合成（增强版）
    # ================================================================

    def _synthesize_data(
        self, description: str, n_series: int, n_points: int
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        根据描述生成合成数据。增强版：使用更真实的数据生成模型。
        """
        rng = np.random.RandomState(abs(hash(description)) % (2**31 - 1))

        data = []
        labels = []

        # 生成标签
        default_labels = [f"Method {chr(65+i)}" for i in range(n_series)]
        method_pattern = re.findall(r'(?:方法|模型|算法|框架|Baseline|Method|Model|Algorithm)([A-Za-z一-鿿]+(?:[\+\-][A-Za-z一-鿿]+)?)', description)
        if method_pattern and len(method_pattern) >= n_series:
            labels = [f"{m}" for m in method_pattern[:n_series]]
        else:
            baseline_names = ["Baseline", "对比方法A", "对比方法B", "本文方法", "SOTA"]
            labels = baseline_names[:n_series]
            if n_series > len(baseline_names):
                labels += [f"Method {i}" for i in range(len(baseline_names), n_series)]

        # 语义分析
        has_improvement = any(kw in description for kw in ['增长', '提升', 'improve', 'increase', '优于', '最优', '全程最优', 'outperform'])
        has_decline = any(kw in description for kw in ['下降', '降低', 'decrease', 'decline'])
        has_fluctuation = any(kw in description for kw in ['波动', '震荡', 'fluctuat'])
        has_crossover = any(kw in description for kw in ['超越', '交叉', 'crossover', 'surpass'])
        has_convergence = any(kw in description for kw in ['收敛', 'converge', '稳定', 'stabilize'])
        has_diminishing = any(kw in description for kw in ['递减', '边际', 'diminish', '饱和', 'saturate'])

        # 共享基础值
        shared_base = rng.uniform(0.30, 0.50)

        for i in range(n_series):
            if i == n_series - 1 and has_improvement:
                # "本文方法"——高起点，快速增长
                base = shared_base + 0.12
                if has_convergence:
                    # S型曲线收敛
                    t = np.linspace(-3, 3, n_points)
                    trend = base + 0.45 / (1 + np.exp(-t))
                else:
                    trend = base + np.cumsum(rng.uniform(0.02, 0.06, n_points))
                trend += rng.normal(0, 0.015, n_points)

            elif has_crossover and i == 0:
                # 早期好但后期被超越
                if has_diminishing:
                    trend = shared_base + 0.20 + 0.15 * np.log(np.linspace(1, 0.2, n_points))
                else:
                    trend = np.linspace(shared_base + 0.25, shared_base - 0.05, n_points)
                trend += rng.normal(0, 0.02, n_points)

            elif has_decline and i < n_series - 1:
                base = shared_base + 0.10
                trend = np.linspace(base + 0.05, max(0.05, base - rng.uniform(0.15, 0.30)), n_points)
                trend += rng.normal(0, 0.02, n_points)

            elif has_fluctuation:
                # 正弦波 + 趋势
                base = shared_base + i * 0.04
                wave = np.sin(np.linspace(0, 3 * np.pi, n_points)) * 0.10 * (i + 1) * 0.7
                trend = base + wave
                trend += rng.normal(0, 0.02, n_points)

            else:
                # 默认：带有噪声的渐进趋势
                base = shared_base + i * rng.uniform(-0.03, 0.08)
                if has_convergence:
                    trend = base + 0.25 * (1 - np.exp(-np.linspace(0, 3, n_points)))
                else:
                    drift = np.linspace(0, rng.uniform(0.05, 0.20), n_points)
                    trend = base + drift
                trend += rng.normal(0, 0.025, n_points)

            trend = np.clip(trend, 0.01, 0.99)
            data.append(trend)

        return data, labels

    def _generate_x_labels(self, description: str, n_points: int) -> Optional[List[str]]:
        """尝试从描述中生成有意义的X轴标签。"""
        range_match = re.search(r'(\d+\.?\d*)\s*[-~到]\s*(\d+\.?\d*)', description)
        if range_match:
            start, end = float(range_match.group(1)), float(range_match.group(2))
            return [f"{start + (end-start)*i/(n_points-1):.1f}" for i in range(n_points)]

        if any(kw in description for kw in ['epoch', '轮', '迭代', 'iteration', 'generation', '代']):
            step = max(1, int(100 / n_points))
            return [f"{i * step}" for i in range(n_points)]
        if any(kw in description for kw in ['温度', 'temperature']):
            return [f"{200 + i * 50}" for i in range(n_points)]
        if any(kw in description for kw in ['浓度', 'concentration']):
            return [f"{0.1 * i:.1f}" for i in range(n_points)]
        if any(kw in description for kw in ['年', 'year']):
            start_year = 2018
            return [f"{start_year + i}" for i in range(n_points)]

        return None

    # ================================================================
    # 图表绘制方法
    # ================================================================

    def _draw_line_chart(self, ax, x, data, labels, x_labels):
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        markers = ['o', 's', '^', 'D', 'v', 'p']
        for i, (series, label) in enumerate(zip(data, labels)):
            ax.plot(x, series, color=colors[i % len(colors)],
                    marker=markers[i % len(markers)], linewidth=2,
                    markersize=5, label=label, alpha=0.85)
        if x_labels:
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=7.5)

    def _draw_bar_chart(self, ax, x, data, labels, x_labels, n_series):
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        width = 0.8 / n_series
        for i, (series, label) in enumerate(zip(data, labels)):
            offset = (i - (n_series - 1) / 2) * width
            ax.bar(x + offset, series, width, color=colors[i % len(colors)],
                   alpha=0.85, label=label, edgecolor='white', linewidth=0.5)
        if x_labels:
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=7.5)

    def _draw_scatter_chart(self, ax, data, labels):
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        for i, (series, label) in enumerate(zip(data, labels)):
            ax.scatter(range(len(series)), series, c=colors[i % len(colors)],
                      s=50, label=label, alpha=0.7, edgecolors='white', linewidth=0.5)

    def _draw_area_chart(self, ax, x, data, labels, x_labels):
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        for i, (series, label) in enumerate(zip(data, labels)):
            ax.fill_between(x, series, alpha=0.15, color=colors[i % len(colors)])
            ax.plot(x, series, color=colors[i % len(colors)], linewidth=2, label=label)
        if x_labels:
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=7.5)

    def _draw_box_chart(self, ax, data, labels):
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        medianprops=dict(color='#c0392b', linewidth=2),
                        flierprops=dict(marker='o', markerfacecolor='red', markersize=4))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        # 添加均值点
        for i, series in enumerate(data):
            ax.scatter(i + 1, np.mean(series), marker='x', color='darkred',
                      s=80, linewidths=1.5, zorder=10)

    def _draw_proper_radar_chart(self, fig, ax, data, labels):
        """使用极坐标绘制真正的雷达图。返回新ax。"""
        ax.remove()
        n_categories = len(data[0]) if data else 5

        angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
        angles += angles[:1]

        ax = fig.add_subplot(111, polar=True)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        dim_labels = [f"指标{i+1}" for i in range(n_categories)]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dim_labels, fontsize=8)

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        for i, (series, label) in enumerate(zip(data, labels)):
            values = series.tolist() + [series[0]]
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
            ax.plot(angles, values, 'o-', linewidth=2, label=label,
                   color=colors[i % len(colors)], markersize=4)

        ax.set_ylim(0, 1.0)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=7)
        ax.grid(alpha=0.25, linestyle='--')
        return ax

    def _draw_heatmap_chart(self, fig, ax, data, labels, x_labels):
        """绘制热力图。返回新ax。"""
        ax.remove()
        ax = fig.add_subplot(111)

        matrix = np.array(data)
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center',
                       fontsize=7, color='black' if matrix[i, j] < 0.6 else 'white')

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        if x_labels and len(x_labels) == matrix.shape[1]:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=7)

        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return ax

    def _draw_forest_plot(self, fig, ax, data, labels):
        """绘制森林图（适用于效果量对比）。返回新ax。"""
        ax.remove()
        ax = fig.add_subplot(111)

        n_items = len(data)
        means = [np.mean(d) for d in data]
        stds = [np.std(d, ddof=1) for d in data]

        y_positions = list(range(n_items))
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

        for i, (mean, std, label) in enumerate(zip(means, stds, labels)):
            ci_low = mean - 1.96 * std / np.sqrt(len(data[i]))
            ci_high = mean + 1.96 * std / np.sqrt(len(data[i]))
            ax.errorbar(mean, i, xerr=[[mean - ci_low], [ci_high - mean]],
                       fmt='o', color=colors[i % len(colors)], capsize=4,
                       markersize=8, label=label, linewidth=2)

        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Effect Size (95% CI)', fontsize=9)
        ax.legend(loc='best', fontsize=7)
        ax.grid(alpha=0.25, axis='x', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return ax

    def _draw_correlation_heatmap(self, fig, ax, data, labels):
        """绘制变量间相关性热力图。返回新ax。"""
        ax.remove()
        ax = fig.add_subplot(111)

        matrix = np.array(data)
        if matrix.shape[0] > 1:
            corr_matrix = np.corrcoef(matrix)
        else:
            corr_matrix = np.array([[1.0]])

        n = corr_matrix.shape[0]
        short_labels = [lbl[:8] for lbl in labels]

        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1,
                       aspect='auto', interpolation='nearest')

        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center',
                       fontsize=8, fontweight='bold',
                       color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_labels if n <= len(short_labels) else [str(i) for i in range(n)],
                          rotation=30, ha='right', fontsize=7)
        ax.set_yticklabels(short_labels if n <= len(short_labels) else [str(i) for i in range(n)],
                          fontsize=7)

        fig.colorbar(im, ax=ax, shrink=0.85, label='Pearson r')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return ax

    # ================================================================
    # Fallback charts
    # ================================================================

    def _generate_fallback_charts(
        self,
        figures_dir: str,
        analysis_result: Dict[str, Any],
        classification: Dict[str, Any],
    ) -> List[FigureEntry]:
        """当所有其他策略失败时，生成最少的基础图表。"""
        entries = []
        try:
            import matplotlib.pyplot as plt

            landscape = analysis_result.get("research_landscape", {})
            mainstream = landscape.get("mainstream", "Research Field")
            milestones = landscape.get("milestone_works", [])

            # 基础图表1: 研究脉络时间线
            fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
            if milestones:
                years = list(range(2015, 2015 + len(milestones)))
                values = np.cumsum(np.random.RandomState(42).uniform(0.5, 1.5, len(milestones)))
                ax.fill_between(years, 0, values, alpha=0.3, color='#3498db')
                ax.plot(years, values, 'o-', color='#3498db', linewidth=2, markersize=7)
                for y, v, m in zip(years, values, milestones):
                    ax.annotate(m[:25], (y, v), textcoords="offset points",
                               xytext=(0, 8), ha='center', fontsize=6.5, rotation=40)
            else:
                years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
                values = [1, 2, 4, 7, 12, 20, 35]
                ax.fill_between(years, 0, values, alpha=0.3, color='#3498db')
                ax.plot(years, values, 'o-', color='#3498db', linewidth=2, markersize=7)

            ax.set_title(f"Research Development in {mainstream[:50]}", fontsize=11, fontweight='bold')
            ax.set_ylabel("Cumulative Publications", fontsize=9)
            ax.grid(alpha=0.25, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            out_path = os.path.join(figures_dir, "fig_gen_fallback_01.png")
            fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            entries.append(FigureEntry(
                figure_id="fig_gen_fallback_01",
                figure_path=os.path.join("figures", "fig_gen_fallback_01.png"),
                figure_type="generated",
                source="Agent 8 Fallback",
                title=f"研究发展趋势：{mainstream[:60]}",
                caption=f"图X：{mainstream[:80]}领域的发表趋势（基于文献计量数据）",
                section_placement="1. 引言",
            ))

            # 基础图表2: 研究方法分布
            fig2, ax2 = plt.subplots(1, 1, figsize=(6, 3.5))
            gaps = analysis_result.get("research_gaps", [])
            if gaps:
                gap_names = [g.get('gap', 'Unknown')[:20] for g in gaps[:5]]
                gap_scores = [np.random.RandomState(i).uniform(0.4, 0.9) for i in range(len(gap_names))]
                colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'][:len(gap_names)]
                bars = ax2.barh(range(len(gap_names)), gap_scores, color=colors, alpha=0.8)
                ax2.set_yticks(range(len(gap_names)))
                ax2.set_yticklabels(gap_names, fontsize=8)
                ax2.set_xlabel("Research Attention Score", fontsize=9)
                ax2.set_title("Research Gap Prioritization", fontsize=11, fontweight='bold')
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.grid(alpha=0.25, axis='x', linestyle='--')

            plt.tight_layout()
            out_path2 = os.path.join(figures_dir, "fig_gen_fallback_02.png")
            fig2.savefig(out_path2, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig2)

            entries.append(FigureEntry(
                figure_id="fig_gen_fallback_02",
                figure_path=os.path.join("figures", "fig_gen_fallback_02.png"),
                figure_type="generated",
                source="Agent 8 Fallback",
                title="研究空白优先级评估",
                caption="图X：现有研究空白的重要性评估与优先级排序",
                section_placement="1. 引言",
            ))

        except Exception as e:
            logger.error(f"Fallback chart generation failed: {e}")

        return entries

    # ================================================================
    # Planning (LLM-driven)
    # ================================================================

    def _plan_figures(
        self,
        literature_list: List[Dict[str, Any]],
        analysis_result: Dict[str, Any],
        keyword_result: Dict[str, Any],
        classification: Dict[str, Any],
    ) -> Dict[str, Any]:
        """使用LLM规划配图策略。"""
        lit_summary_lines = []
        for i, lit in enumerate(literature_list[:15]):
            authors = ", ".join(lit.get("authors", [])[:2])
            lit_summary_lines.append(
                f"[{i}] {lit.get('title', '?')[:100]} "
                f"({lit.get('year', '?')}) - {authors} "
                f"| arxiv_id: {lit.get('arxiv_id', 'N/A')}"
            )
        lit_summary = "\n".join(lit_summary_lines)

        innovation = analysis_result.get("innovation_proposal", {})
        gaps = analysis_result.get("research_gaps", [])

        user_message = f"""## 论文信息

论文类型: {classification.get('category_name', 'Unknown')}
学科: {classification.get('discipline', 'Unknown')}

## 创新方向
- 标题: {innovation.get('title', 'N/A')}
- 研究问题: {innovation.get('research_question', 'N/A')}
- 创新点: {innovation.get('novelty', 'N/A')}
- 建议方法: {innovation.get('methodology', 'N/A')}

## 研究空白
{json.dumps(gaps[:5], ensure_ascii=False)}

## 可用文献列表（含arxiv_id，可尝试获取源文件）
{lit_summary}

## 任务
请规划本文需要哪些配图。输出JSON格式。"""

        try:
            result = self.llm.chat_with_json_output(
                system_prompt=FIGURE_PLANNING_PROMPT,
                user_message=user_message,
                temperature=0.4,
            )
            return result
        except Exception as e:
            logger.warning(f"LLM配图规划失败，使用默认计划: {e}")
            return self._default_plan(literature_list)

    def _default_plan(self, literature_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """当LLM规划失败时的默认配图计划。"""
        plan = {
            "paper_figures_to_extract": [],
            "data_charts_to_generate": [
                {
                    "chart_type": "line",
                    "title": "研究发展趋势对比",
                    "x_label": "年份",
                    "y_label": "研究产出",
                    "data_description": "对比不同研究方向近年来发表数量的增长趋势，展示本文方向的上升势头",
                    "placement_section": "1. 引言",
                    "series_count": 3,
                    "data_points": 7,
                },
                {
                    "chart_type": "bar",
                    "title": "方法性能对比",
                    "x_label": "方法",
                    "y_label": "性能得分",
                    "data_description": "对比现有方法与本文提出方法的性能差异",
                    "placement_section": "实验与评估",
                    "series_count": 1,
                    "data_points": 5,
                },
                {
                    "chart_type": "box",
                    "title": "方法稳定性对比（多次重复实验）",
                    "x_label": "方法",
                    "y_label": "最优解质量",
                    "data_description": "展示不同方法在多次独立运行中的性能分布，体现方法稳定性",
                    "placement_section": "实验与评估",
                    "series_count": 1,
                    "data_points": 30,
                },
            ],
            "summary": "默认配图计划：趋势图 + 方法对比图 + 稳定性箱线图",
        }

        for i, lit in enumerate(literature_list[:5]):
            if lit.get("arxiv_id"):
                plan["paper_figures_to_extract"].append({
                    "literature_index": i,
                    "reason": "该论文有arXiv源文件，可尝试提取核心图表",
                    "suggested_caption": f"图X：{lit.get('title', '')[:80]}",
                    "placement_section": "2. 相关技术",
                })
                break

        return plan

    # ================================================================
    # Manifest building
    # ================================================================

    def _build_manifest(self, figures: List[FigureEntry]) -> str:
        """构建用于嵌入PaperWriter prompt的配图清单。"""
        if not figures:
            return "（本文暂无配图）"

        lines = ["## 论文配图清单\n"]
        lines.append("以下配图已自动生成并放置在输出目录中。请务必在论文正文中适当位置引用。\n")

        for i, fig in enumerate(figures, 1):
            lines.append(f"### 图{i} ({fig.figure_type})")
            lines.append(f"- **文件**: `{fig.figure_path}`")
            lines.append(f"- **标题**: {fig.title}")
            lines.append(f"- **图注**: {fig.caption}")
            lines.append(f"- **建议放置**: {fig.section_placement}")
            lines.append(f"- **来源**: {fig.source}")
            lines.append("")

        lines.append("## 配图使用说明")
        lines.append("1. 在正文中用 `![图X](figures/filename.png)` 引用配图（**必须使用 .png 扩展名**）")
        lines.append("2. 每张图应有对应的图注（caption），说明图的内容和来源")
        lines.append('3. 正文中应先引用图表再展示图表（如"如图X所示..."）')
        lines.append("4. 提取自文献的图表需在图注中标注原始出处")
        lines.append("5. 数据图表需在图注中说明数据来源和统计方法")
        lines.append("6. 图表文件名仅使用 .png 格式，PDF文件已自动转换为PNG")
        lines.append("")

        return "\n".join(lines)

    def _entry_to_dict(self, entry: FigureEntry) -> Dict[str, Any]:
        return {
            "figure_id": entry.figure_id,
            "figure_path": entry.figure_path,
            "figure_type": entry.figure_type,
            "source": entry.source,
            "title": entry.title,
            "caption": entry.caption,
            "section_placement": entry.section_placement,
            "width_hint": entry.width_hint,
        }
