# coding=utf-8
"""
日报渲染：HTML 邮件正文 + 纯文本 fallback

邮件客户端兼容性约定（针对安卓 QQ 邮箱等）：
- 布局全部使用 ``<table>``，不用 flex / grid
- 样式全部 **inline CSS**，不依赖 ``<style>`` 块（部分客户端会剥离）
- 不使用 JavaScript、不引用任何外部图片 / 脚本 / 字体
- 宽度 ``max-width: 640px`` 并配合 ``width="100%"``，移动端可读
- 所有动态内容经 HTML 转义，避免仓库描述里的尖括号破坏排版
"""

import html as html_module
from dataclasses import dataclass, field
from typing import List, Optional

from .ai.result import SYNTHESIS_UNAVAILABLE_TEXT, AIReportData
from .ai.schemas import RepoAnalysis
from .ai.scoring import ForYouEntry
from .ranking import ScoredRepo
from .timeutils import format_created_display

# 颜色（GitHub Primer 风格，显式指定避免客户端默认样式差异）
COLOR_PAGE_BG = "#f6f8fa"
COLOR_CARD_BG = "#ffffff"
COLOR_BORDER = "#d0d7de"
COLOR_TEXT = "#24292f"
COLOR_MUTED = "#57606a"
COLOR_LINK = "#0969da"
COLOR_UP = "#1a7f37"
COLOR_DOWN = "#cf222e"
COLOR_ACCENT_BG = "#ddf4ff"
# AI 增强内容的底色（与客观数据在视觉上区分开）
COLOR_AI_BG = "#f6f0ff"
COLOR_FORYOU_BG = "#fff8e6"

# AI 枚举值的中文展示
CONFIDENCE_LABELS = {"low": "低", "medium": "中", "high": "高"}
MATURITY_LABELS = {
    "experimental": "实验阶段",
    "growing": "成长期",
    "mature": "成熟",
    "unknown": "",
}

FONT_STACK = (
    "-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',"
    "'Hiragino Sans GB','Microsoft YaHei',sans-serif"
)

NO_DATA = "—"
FIRST_RUN_NOTICE = "首次运行，24h/7d Star 增长将在积累历史数据后启用。"


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------
@dataclass
class ReportSummary:
    """日报概览数据"""

    date: str
    generated_at_display: str = ""
    candidate_count: int = 0
    trending_count: int = 0
    new_repo_count: int = 0
    new_window_days: int = 30
    has_24h_history: bool = False
    has_7d_history: bool = False
    matched_24h: int = 0
    matched_7d: int = 0
    history_days: int = 0
    notes: List[str] = field(default_factory=list)


@dataclass
class ReportContext:
    """渲染日报所需的全部数据"""

    summary: ReportSummary
    hot: List[ScoredRepo] = field(default_factory=list)
    new_rising: List[ScoredRepo] = field(default_factory=list)
    # AI 增强数据；None 或 enabled=False 时整份日报退回基础版本
    ai: Optional[AIReportData] = None

    @property
    def ai_enabled(self) -> bool:
        """是否需要渲染 AI 区块（AI 未启用时整块隐藏）"""
        return bool(self.ai and self.ai.should_render)

    def analysis_for(self, item: ScoredRepo) -> Optional[RepoAnalysis]:
        """取某个仓库的 AI 分析（没有就返回 None，卡片自动退回基础样式）"""
        if not self.ai:
            return None
        return self.ai.analysis_for(item.record.full_name)


# ----------------------------------------------------------------------
# 格式化辅助
# ----------------------------------------------------------------------
def esc(text: Optional[str]) -> str:
    """HTML 转义（None 视为空串）"""
    return html_module.escape(str(text if text is not None else ""), quote=True)


def fmt_int(value: Optional[int]) -> str:
    """整数千分位；None → —"""
    if value is None:
        return NO_DATA
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return NO_DATA


def fmt_delta(value: Optional[int]) -> str:
    """
    增量显示：正增长带 ``+``，无历史数据显示 ``—``

    Examples:
        >>> fmt_delta(1826)
        '+1,826'
        >>> fmt_delta(None)
        '—'
        >>> fmt_delta(-12)
        '-12'
    """
    if value is None:
        return NO_DATA
    try:
        number = int(value)
    except (TypeError, ValueError):
        return NO_DATA
    if number > 0:
        return f"+{number:,}"
    return f"{number:,}"


def fmt_float(value: Optional[float], digits: int = 1) -> str:
    """浮点数显示；None → —"""
    if value is None:
        return NO_DATA
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return NO_DATA


def delta_color(value: Optional[int]) -> str:
    """增量对应的颜色"""
    if value is None:
        return COLOR_MUTED
    if value > 0:
        return COLOR_UP
    if value < 0:
        return COLOR_DOWN
    return COLOR_MUTED


def build_subject(date: str, top_count: int) -> str:
    """
    邮件主题

    Examples:
        >>> build_subject("2026-08-22", 20)
        '🔥 GitHub Daily Radar | 2026-08-22 | Top20'
    """
    return f"🔥 GitHub Daily Radar | {date} | Top{top_count}"


# ----------------------------------------------------------------------
# HTML 渲染
# ----------------------------------------------------------------------
def _overview_rows(summary: ReportSummary) -> List[str]:
    """今日概览的条目（HTML 与纯文本共用文案）"""
    history_24h = (
        f"是（{summary.matched_24h} 个仓库可比）" if summary.has_24h_history else "否（暂无昨日快照）"
    )
    history_7d = (
        f"是（{summary.matched_7d} 个仓库可比）" if summary.has_7d_history else "否（暂无 7 天前快照）"
    )
    return [
        f"候选仓库数量：{summary.candidate_count}",
        f"Trending 仓库数量：{summary.trending_count}",
        f"新项目数量（{summary.new_window_days} 天内创建）：{summary.new_repo_count}",
        f"是否已有 24h 历史：{history_24h}",
        f"是否已有 7d 历史：{history_7d}",
        f"已积累快照天数：{summary.history_days}",
    ]


def _ai_lines(analysis: Optional[RepoAnalysis]) -> List[tuple]:
    """
    榜单卡片上的 AI 增强行（``(标签, 内容)`` 列表）

    AI 数据缺失时返回空列表 —— 卡片就是原来的基础条目，
    这正是「AI 失败时直接显示旧版基础条目」的实现方式。
    """
    if analysis is None:
        return []

    lines: List[tuple] = []
    if analysis.summary_zh:
        lines.append(("一句话", analysis.summary_zh))
    if analysis.category:
        maturity = MATURITY_LABELS.get(analysis.maturity)
        category = analysis.category + (f" · {maturity}" if maturity else "")
        lines.append(("分类", category))
    if analysis.why_hot.summary:
        confidence = CONFIDENCE_LABELS.get(analysis.why_hot.confidence, "")
        text = analysis.why_hot.summary + (f"（可信度：{confidence}）" if confidence else "")
        lines.append(("为什么值得关注", text))
    if analysis.recommended_action:
        lines.append(
            ("推荐", f"{analysis.action_icon} {analysis.action_label}")
        )
    return lines


def _render_ai_block(analysis: Optional[RepoAnalysis]) -> str:
    """把 AI 增强行渲染成卡片里的一小块（全部经过 HTML 转义）"""
    lines = _ai_lines(analysis)
    if not lines:
        return ""
    rows = "".join(
        f'<div style="margin-top:6px;font-size:13px;color:{COLOR_TEXT};'
        f'font-family:{FONT_STACK};line-height:1.6;word-break:break-word;">'
        f'<span style="color:{COLOR_MUTED};">{esc(label)}：</span>{esc(value)}</div>'
        for label, value in lines
    )
    return (
        f'<div style="margin-top:10px;padding:10px 12px;background-color:{COLOR_AI_BG};'
        f'border-radius:6px;">{rows}</div>'
    )


def _render_repo_card(
    item: ScoredRepo,
    rank: int,
    *,
    show_speed: bool = False,
    analysis: Optional[RepoAnalysis] = None,
) -> str:
    """渲染单个仓库卡片（表格实现，邮件客户端兼容）"""
    record = item.record
    delta_24h = item.delta.delta_stars_24h
    delta_7d = item.delta.delta_stars_7d

    metrics = [
        ("⭐ 总 Stars", fmt_int(record.stars), COLOR_TEXT),
        ("24h", fmt_delta(delta_24h), delta_color(delta_24h)),
        ("7d", fmt_delta(delta_7d), delta_color(delta_7d)),
    ]

    if show_speed and item.rising_speed is not None:
        suffix = "（估算）" if item.rising_speed_estimated else ""
        metrics.append(
            ("成长速度", f"{fmt_float(item.rising_speed)} ★/天{suffix}", COLOR_TEXT)
        )

    metric_cells = "".join(
        f'<td style="padding:0 14px 0 0;white-space:nowrap;font-size:14px;'
        f'color:{COLOR_MUTED};font-family:{FONT_STACK};">'
        f'{esc(label)} '
        f'<span style="color:{color};font-weight:600;">{esc(value)}</span>'
        f"</td>"
        for label, value, color in metrics
    )

    meta_line = (
        f"Language: {esc(record.display_language)}"
        f" &nbsp;·&nbsp; Created: {esc(format_created_display(record.created_at))}"
        f" &nbsp;·&nbsp; Forks: {esc(fmt_int(record.forks))}"
        f" &nbsp;·&nbsp; Heat Score: <strong style=\"color:{COLOR_TEXT};\">{esc(fmt_float(item.score))}</strong>"
    )
    if record.trending_rank:
        meta_line += f" &nbsp;·&nbsp; Trending #{int(record.trending_rank)}"

    return f"""
      <tr>
        <td style="padding:0 0 12px 0;">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
                 style="background-color:{COLOR_CARD_BG};border:1px solid {COLOR_BORDER};
                        border-radius:8px;width:100%;">
            <tr>
              <td style="padding:14px 16px;">
                <div style="font-size:16px;font-weight:600;font-family:{FONT_STACK};
                            color:{COLOR_TEXT};line-height:1.4;">
                  <span style="display:inline-block;min-width:34px;color:{COLOR_MUTED};">#{rank}</span>
                  <a href="{esc(record.html_url)}"
                     style="color:{COLOR_LINK};text-decoration:none;word-break:break-all;">{esc(record.full_name)}</a>
                </div>
                <table role="presentation" cellpadding="0" cellspacing="0" border="0"
                       style="margin:8px 0 0 0;">
                  <tr>{metric_cells}</tr>
                </table>
                <div style="margin-top:8px;font-size:13px;color:{COLOR_MUTED};
                            font-family:{FONT_STACK};line-height:1.5;">{meta_line}</div>
                <div style="margin-top:8px;font-size:14px;color:{COLOR_TEXT};
                            font-family:{FONT_STACK};line-height:1.6;word-break:break-word;">
                  {esc(record.display_description)}
                </div>
                {_render_ai_block(analysis)}
                <div style="margin-top:8px;font-size:12px;font-family:{FONT_STACK};">
                  <a href="{esc(record.html_url)}"
                     style="color:{COLOR_LINK};text-decoration:none;word-break:break-all;">{esc(record.html_url)}</a>
                </div>
              </td>
            </tr>
          </table>
        </td>
      </tr>"""


def _render_section(
    title: str,
    items: List[ScoredRepo],
    *,
    empty_text: str,
    show_speed: bool = False,
    context: Optional["ReportContext"] = None,
) -> str:
    """渲染一个榜单区块"""
    if items:
        body = "".join(
            _render_repo_card(
                item,
                index,
                show_speed=show_speed,
                analysis=context.analysis_for(item) if context else None,
            )
            for index, item in enumerate(items, 1)
        )
    else:
        body = f"""
      <tr>
        <td style="padding:0 0 12px 0;">
          <div style="background-color:{COLOR_CARD_BG};border:1px dashed {COLOR_BORDER};
                      border-radius:8px;padding:16px;font-size:14px;color:{COLOR_MUTED};
                      font-family:{FONT_STACK};">{esc(empty_text)}</div>
        </td>
      </tr>"""

    return f"""
    <tr>
      <td style="padding:24px 0 12px 0;font-size:18px;font-weight:600;color:{COLOR_TEXT};
                 font-family:{FONT_STACK};">{esc(title)}</td>
    </tr>
    <tr>
      <td>
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="width:100%;">
          {body}
        </table>
      </td>
    </tr>"""


def _signal_lines(ai: AIReportData) -> List[str]:
    """📡 今日 GitHub 技术信号 的条目（HTML 与纯文本共用）"""
    if not ai.synthesis_available:
        return []
    synthesis = ai.synthesis
    lines = list(synthesis.signals)
    if synthesis.rising_categories:
        lines.append("升温方向：" + "、".join(synthesis.rising_categories))
    if synthesis.watch_tomorrow:
        lines.append("明日关注：" + "、".join(synthesis.watch_tomorrow))
    return lines


def _render_signals_section(ai: Optional[AIReportData]) -> str:
    """📡 今日 GitHub 技术信号（synthesis 失败时显示「不可用」，绝不阻断邮件）"""
    if ai is None or not ai.should_render:
        return ""

    if ai.synthesis_available:
        headline = ai.synthesis.headline or "今日趋势总结"
        items = "".join(
            f'<li style="margin:4px 0;">{esc(line)}</li>' for line in _signal_lines(ai)
        )
        body = (
            f'<div style="font-size:15px;font-weight:600;color:{COLOR_TEXT};'
            f'line-height:1.6;">今日主线：{esc(headline)}</div>'
            + (
                f'<ul style="margin:10px 0 0 0;padding-left:18px;font-size:14px;'
                f'color:{COLOR_TEXT};line-height:1.7;">{items}</ul>'
                if items
                else ""
            )
        )
    else:
        body = (
            f'<div style="font-size:14px;color:{COLOR_MUTED};line-height:1.6;">'
            f"{esc(SYNTHESIS_UNAVAILABLE_TEXT)}</div>"
        )

    return f"""
    <tr>
      <td style="padding:24px 0 12px 0;font-size:18px;font-weight:600;color:{COLOR_TEXT};
                 font-family:{FONT_STACK};">📡 今日 GitHub 技术信号</td>
    </tr>
    <tr>
      <td>
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
               style="background-color:{COLOR_CARD_BG};border:1px solid {COLOR_BORDER};
                      border-radius:8px;width:100%;">
          <tr>
            <td style="padding:14px 16px;font-family:{FONT_STACK};">{body}</td>
          </tr>
        </table>
      </td>
    </tr>"""


def _for_you_rows(entry: ForYouEntry) -> List[tuple]:
    """🎯 For You 卡片的正文行（HTML 与纯文本共用）"""
    analysis = entry.analysis
    rows: List[tuple] = []
    if analysis.summary_zh:
        rows.append(("一句话", analysis.summary_zh))
    if analysis.problem:
        rows.append(("它解决什么问题", analysis.problem))
    rows.append(("为什么与你相关", entry.relevance_explanation))
    if analysis.tech_stack:
        rows.append(("技术栈", "、".join(analysis.tech_stack)))
    if analysis.use_cases:
        rows.append(("应用场景", "、".join(analysis.use_cases)))
    rows.append(("推荐", f"{analysis.action_icon} {analysis.action_label}"))
    if analysis.recommendation_reason:
        rows.append(("推荐原因", analysis.recommendation_reason))
    return rows


def _render_for_you_card(entry: ForYouEntry, rank: int) -> str:
    """渲染一张 🎯 For You 卡片"""
    record = entry.record
    metrics = [
        ("Personal Score", fmt_float(entry.personal_score)),
        ("Heat Score", fmt_float(entry.heat_score)),
        ("相关度", f"{entry.relevance_score}/100"),
        ("关键词匹配", f"{fmt_float(entry.keyword_score, 0)}/100"),
    ]
    metric_cells = "".join(
        f'<td style="padding:0 14px 0 0;white-space:nowrap;font-size:13px;'
        f'color:{COLOR_MUTED};font-family:{FONT_STACK};">{esc(label)} '
        f'<span style="color:{COLOR_TEXT};font-weight:600;">{esc(value)}</span></td>'
        for label, value in metrics
    )
    rows = "".join(
        f'<div style="margin-top:6px;font-size:14px;color:{COLOR_TEXT};'
        f'font-family:{FONT_STACK};line-height:1.6;word-break:break-word;">'
        f'<span style="color:{COLOR_MUTED};">{esc(label)}：</span>{esc(value)}</div>'
        for label, value in _for_you_rows(entry)
    )

    return f"""
      <tr>
        <td style="padding:0 0 12px 0;">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
                 style="background-color:{COLOR_FORYOU_BG};border:1px solid {COLOR_BORDER};
                        border-radius:8px;width:100%;">
            <tr>
              <td style="padding:14px 16px;">
                <div style="font-size:16px;font-weight:600;font-family:{FONT_STACK};
                            color:{COLOR_TEXT};line-height:1.4;">
                  🎯 #{rank}
                  <a href="{esc(record.html_url)}"
                     style="color:{COLOR_LINK};text-decoration:none;word-break:break-all;">{esc(record.full_name)}</a>
                </div>
                <table role="presentation" cellpadding="0" cellspacing="0" border="0"
                       style="margin:8px 0 0 0;">
                  <tr>{metric_cells}</tr>
                </table>
                {rows}
                <div style="margin-top:8px;font-size:12px;font-family:{FONT_STACK};">
                  <a href="{esc(record.html_url)}"
                     style="color:{COLOR_LINK};text-decoration:none;word-break:break-all;">{esc(record.html_url)}</a>
                </div>
              </td>
            </tr>
          </table>
        </td>
      </tr>"""


def _render_for_you_section(ai: Optional[AIReportData]) -> str:
    """🎯 For You Top10"""
    if ai is None or not ai.should_render:
        return ""

    if ai.has_for_you:
        body = "".join(
            _render_for_you_card(entry, index)
            for index, entry in enumerate(ai.for_you, 1)
        )
    else:
        body = f"""
      <tr>
        <td style="padding:0 0 12px 0;">
          <div style="background-color:{COLOR_CARD_BG};border:1px dashed {COLOR_BORDER};
                      border-radius:8px;padding:16px;font-size:14px;color:{COLOR_MUTED};
                      font-family:{FONT_STACK};">今日暂无 AI 个性化推荐（AI 分析不可用或没有匹配项目）。</div>
        </td>
      </tr>"""

    title = f"🎯 For You Top{len(ai.for_you) or 10}"
    return f"""
    <tr>
      <td style="padding:24px 0 12px 0;font-size:18px;font-weight:600;color:{COLOR_TEXT};
                 font-family:{FONT_STACK};">{esc(title)}</td>
    </tr>
    <tr>
      <td>
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="width:100%;">
          {body}
        </table>
      </td>
    </tr>"""


def _usage_rows(ai: AIReportData) -> List[tuple]:
    """
    📊 AI 使用情况 的条目（HTML 与纯文本共用）

    输入 Tokens 下面挂两档缓存明细：命中前缀缓存的输入便宜得多，
    分开列出来才能看懂费用是怎么来的。服务端没返回明细时，
    会全部落在「缓存未命中」那一档（费用同样按更贵的一档估算）。
    """
    usage = ai.usage
    # 用与计价完全相同的口径展示，保证「输入 = 命中 + 未命中」
    cache_hit_tokens, cache_miss_tokens = usage.cache_split()
    rows = [
        ("模型", ai.model or "—"),
        ("分析仓库", str(usage.repositories_analyzed)),
        # 这是**仓库级**的静态分析缓存，和下面 token 级的前缀缓存不是一回事
        ("仓库缓存命中", str(usage.cache_hits)),
        ("API 请求", str(usage.requests)),
        ("输入 Tokens", fmt_int(usage.prompt_tokens)),
        ("├─ 缓存命中", fmt_int(cache_hit_tokens)),
        ("└─ 缓存未命中", fmt_int(cache_miss_tokens)),
        ("输出 Tokens", fmt_int(usage.completion_tokens)),
        ("总 Tokens", fmt_int(usage.total_tokens)),
    ]
    if usage.failed_requests:
        rows.append(("失败请求", str(usage.failed_requests)))
    if ai.cost is not None:
        rows.append(("预估今日费用", ai.cost.format_total()))
    if ai.reused:
        rows.append(("说明", "本次直接复用当天已生成的 AI 结果，未新增 API 调用"))
    return rows


def _render_ai_usage_section(ai: Optional[AIReportData]) -> str:
    """📊 AI 使用情况"""
    if ai is None or not ai.should_render:
        return ""

    items = "".join(
        f'<li style="margin:4px 0;">{esc(label)}：{esc(value)}</li>'
        for label, value in _usage_rows(ai)
    )
    return f"""
    <tr>
      <td style="padding:24px 0 12px 0;font-size:18px;font-weight:600;color:{COLOR_TEXT};
                 font-family:{FONT_STACK};">📊 AI 使用情况</td>
    </tr>
    <tr>
      <td>
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
               style="background-color:{COLOR_CARD_BG};border:1px solid {COLOR_BORDER};
                      border-radius:8px;width:100%;">
          <tr>
            <td style="padding:14px 16px;font-family:{FONT_STACK};font-size:13px;
                       color:{COLOR_TEXT};line-height:1.6;">
              <ul style="margin:0;padding-left:18px;">{items}</ul>
              <div style="margin-top:8px;color:{COLOR_MUTED};">
                费用按 DeepSeek 当前官方定价本地估算（缓存命中的输入更便宜），
                仅供参考，不代表最终账单；服务商价格也可能调整。
              </div>
            </td>
          </tr>
        </table>
      </td>
    </tr>"""


def _ai_explain_lines(ai: Optional[AIReportData]) -> List[str]:
    """数据说明里与 AI 有关的条目"""
    if ai is None or not ai.should_render:
        return []
    lines = [
        f"AI 解读由 {ai.model} 生成，只做增强：不参与 Heat Score、不修改 Star 数据与 24h/7d 增量。",
        "「为什么值得关注」只基于 Star / Trending / 更新时间等客观指标，"
        "没有证据时不会声称具体外部原因，也不代表确定的因果关系。",
        "Personal Score = 0.55×AI 相关度 + 0.25×Heat Score + 0.20×关键词匹配。",
    ]
    lines.extend(ai.notes)
    return lines


def render_html(context: ReportContext) -> str:
    """
    渲染 HTML 邮件正文

    Args:
        context: 报告数据

    Returns:
        完整 HTML 字符串（自包含，无外部资源）
    """
    summary = context.summary

    overview_items = "".join(
        f'<li style="margin:4px 0;">{esc(line)}</li>' for line in _overview_rows(summary)
    )

    notes_html = ""
    if summary.notes:
        note_items = "".join(
            f'<li style="margin:4px 0;">{esc(note)}</li>' for note in summary.notes
        )
        notes_html = f"""
        <div style="margin-top:12px;padding:12px 14px;background-color:{COLOR_ACCENT_BG};
                    border-radius:6px;font-size:13px;color:{COLOR_TEXT};font-family:{FONT_STACK};">
          <ul style="margin:0;padding-left:18px;">{note_items}</ul>
        </div>"""

    signals_section = _render_signals_section(context.ai)
    for_you_section = _render_for_you_section(context.ai)
    hot_section = _render_section(
        f"🔥 Hot Today Top{len(context.hot) or 20}",
        context.hot,
        empty_text="今日没有可用的候选仓库。",
        context=context,
    )
    new_section = _render_section(
        f"🌱 New & Rising Top{len(context.new_rising) or 10}",
        context.new_rising,
        empty_text=(
            f"今日没有符合条件的新项目（{summary.new_window_days} 天内创建且已有一定 Star）。"
        ),
        show_speed=True,
        context=context,
    )
    ai_usage_section = _render_ai_usage_section(context.ai)

    explain = [
        "24h / 7d Star 增长来自每日快照的差值（不是 GitHub 页面上的 stars today）。",
        "首日或缺少对应日期快照时，增量显示为 “—”，不会用总 Star 冒充增量。",
        "Heat Score 由 24h 增长(45%)、7 日均增(20%)、Trending 排名(15%)、"
        "项目新鲜度(10%)、总 Star 规模(10%) 加权得出；缺失指标不按 0 惩罚，"
        "而是对可用指标重新归一化。",
        "数据来源：GitHub Trending 页面 + GitHub REST API 搜索。",
    ]
    explain.extend(_ai_explain_lines(context.ai))
    explain_items = "".join(
        f'<li style="margin:4px 0;">{esc(line)}</li>' for line in explain
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GitHub Daily Radar {esc(summary.date)}</title>
</head>
<body style="margin:0;padding:0;background-color:{COLOR_PAGE_BG};">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
       style="background-color:{COLOR_PAGE_BG};width:100%;padding:16px 0;">
  <tr>
    <td align="center" style="padding:0 12px;">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
             style="max-width:640px;width:100%;">
        <tr>
          <td style="padding:4px 0 0 0;font-size:22px;font-weight:700;color:{COLOR_TEXT};
                     font-family:{FONT_STACK};">🛰️ GitHub Daily Radar</td>
        </tr>
        <tr>
          <td style="padding:4px 0 0 0;font-size:14px;color:{COLOR_MUTED};font-family:{FONT_STACK};">
            {esc(summary.date)}{(" &nbsp;·&nbsp; 生成于 " + esc(summary.generated_at_display)) if summary.generated_at_display else ""}
          </td>
        </tr>
        <tr>
          <td style="padding:16px 0 0 0;">
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
                   style="background-color:{COLOR_CARD_BG};border:1px solid {COLOR_BORDER};
                          border-radius:8px;width:100%;">
              <tr>
                <td style="padding:14px 16px;font-family:{FONT_STACK};">
                  <div style="font-size:16px;font-weight:600;color:{COLOR_TEXT};">今日概览</div>
                  <ul style="margin:8px 0 0 0;padding-left:18px;font-size:14px;
                             color:{COLOR_TEXT};line-height:1.6;">{overview_items}</ul>
                  {notes_html}
                </td>
              </tr>
            </table>
          </td>
        </tr>
        {signals_section}
        {for_you_section}
        {hot_section}
        {new_section}
        {ai_usage_section}
        <tr>
          <td style="padding:24px 0 12px 0;font-size:18px;font-weight:600;color:{COLOR_TEXT};
                     font-family:{FONT_STACK};">📌 数据说明</td>
        </tr>
        <tr>
          <td style="padding:0 0 24px 0;">
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
                   style="background-color:{COLOR_CARD_BG};border:1px solid {COLOR_BORDER};
                          border-radius:8px;width:100%;">
              <tr>
                <td style="padding:14px 16px;font-family:{FONT_STACK};font-size:13px;
                           color:{COLOR_MUTED};line-height:1.6;">
                  <ul style="margin:0;padding-left:18px;">{explain_items}</ul>
                  <div style="margin-top:10px;">
                    由 GitHub Daily Radar 自动生成 · 基于 GitHub Actions 定时运行
                  </div>
                </td>
              </tr>
            </table>
          </td>
        </tr>
      </table>
    </td>
  </tr>
</table>
</body>
</html>
"""


# ----------------------------------------------------------------------
# 纯文本渲染
# ----------------------------------------------------------------------
def _render_text_item(
    item: ScoredRepo,
    rank: int,
    *,
    show_speed: bool = False,
    analysis: Optional[RepoAnalysis] = None,
) -> str:
    record = item.record
    lines = [
        f"#{rank} {record.full_name}",
        f"    ⭐ {fmt_int(record.stars)}   24h: {fmt_delta(item.delta.delta_stars_24h)}"
        f"   7d: {fmt_delta(item.delta.delta_stars_7d)}",
        f"    Language: {record.display_language} | Created: "
        f"{format_created_display(record.created_at)} | Heat Score: {fmt_float(item.score)}",
    ]
    if show_speed and item.rising_speed is not None:
        suffix = "（估算）" if item.rising_speed_estimated else ""
        lines.append(f"    成长速度: {fmt_float(item.rising_speed)} ★/天{suffix}")
    lines.append(f"    {record.display_description}")
    for label, value in _ai_lines(analysis):
        lines.append(f"    {label}: {value}")
    lines.append(f"    {record.html_url}")
    return "\n".join(lines)


def _render_text_for_you(entry: ForYouEntry, rank: int) -> str:
    """🎯 For You 的纯文本卡片"""
    lines = [
        f"🎯 #{rank} {entry.full_name}",
        f"    Personal Score: {fmt_float(entry.personal_score)}"
        f"   Heat Score: {fmt_float(entry.heat_score)}"
        f"   相关度: {entry.relevance_score}/100"
        f"   关键词匹配: {fmt_float(entry.keyword_score, 0)}/100",
    ]
    for label, value in _for_you_rows(entry):
        lines.append(f"    {label}: {value}")
    lines.append(f"    {entry.record.html_url}")
    return "\n".join(lines)


def render_text(context: ReportContext) -> str:
    """渲染纯文本 fallback（用于不支持 HTML 的邮件客户端）"""
    summary = context.summary
    lines: List[str] = [
        "GitHub Daily Radar",
        summary.date,
        "",
        "今日概览",
        "-" * 40,
    ]
    lines.extend(f"- {row}" for row in _overview_rows(summary))

    if summary.notes:
        lines.append("")
        lines.extend(f"* {note}" for note in summary.notes)

    ai = context.ai if context.ai_enabled else None

    if ai is not None:
        lines.extend(["", "📡 今日 GitHub 技术信号", "-" * 40])
        if ai.synthesis_available:
            lines.append(f"今日主线：{ai.synthesis.headline}")
            lines.extend(f"• {line}" for line in _signal_lines(ai))
        else:
            lines.append(SYNTHESIS_UNAVAILABLE_TEXT)

        lines.extend(["", f"🎯 For You Top{len(ai.for_you) or 10}", "-" * 40])
        if ai.has_for_you:
            for index, entry in enumerate(ai.for_you, 1):
                lines.append(_render_text_for_you(entry, index))
                lines.append("")
        else:
            lines.extend(["今日暂无 AI 个性化推荐（AI 分析不可用或没有匹配项目）。", ""])

    lines.extend(["", f"🔥 Hot Today Top{len(context.hot) or 20}", "-" * 40])
    if context.hot:
        for index, item in enumerate(context.hot, 1):
            lines.append(_render_text_item(item, index, analysis=context.analysis_for(item)))
            lines.append("")
    else:
        lines.extend(["今日没有可用的候选仓库。", ""])

    lines.extend([f"🌱 New & Rising Top{len(context.new_rising) or 10}", "-" * 40])
    if context.new_rising:
        for index, item in enumerate(context.new_rising, 1):
            lines.append(
                _render_text_item(
                    item, index, show_speed=True, analysis=context.analysis_for(item)
                )
            )
            lines.append("")
    else:
        lines.extend(
            [
                f"今日没有符合条件的新项目（{summary.new_window_days} 天内创建且已有一定 Star）。",
                "",
            ]
        )

    if ai is not None:
        lines.extend(["📊 AI 使用情况", "-" * 40])
        lines.extend(f"{label}：{value}" for label, value in _usage_rows(ai))
        lines.append("费用按 DeepSeek 当前官方定价本地估算，仅供参考，不代表最终账单。")
        lines.append("")

    lines.extend(
        [
            "数据说明",
            "-" * 40,
            "24h / 7d Star 增长来自每日快照差值，不是 GitHub 页面的 stars today。",
            "缺少对应日期快照时显示 —，不会用总 Star 冒充增量。",
            "Heat Score = 24h增长45% + 7日均增20% + Trending排名15% + 新鲜度10% + 总Star10%，",
            "缺失指标会对剩余指标重新归一化，而不是按 0 惩罚。",
        ]
    )
    lines.extend(_ai_explain_lines(ai))
    lines.extend(
        [
            "",
            "由 GitHub Daily Radar 自动生成 · 基于 GitHub Actions 定时运行",
        ]
    )

    return "\n".join(lines)
