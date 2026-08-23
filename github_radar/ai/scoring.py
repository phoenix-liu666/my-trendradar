# coding=utf-8
"""
Personal Score 与 🎯 For You Top10

公式（规格 §19）::

    PersonalScore = 0.55 × AI relevance
                  + 0.25 × Heat Score
                  + 0.20 × keyword score

三项都是 0~100，因此结果天然落在 0~100。

三项各自的意义
--------------
- **AI relevance**：模型读过 README 之后判断的相关度，理解力最强但会漂移
- **Heat Score**：完全客观的热度，AI 绝对不参与、不修改
- **keyword score**：deterministic，AI 挂掉时仍然有效的个性化锚点

For You **允许出现 Hot20 之外的项目**：
只要它进了 AI 分析（profile 强匹配就会进），
哪怕 Heat Score 只有 50，只要 relevance 与 keyword 足够高，
PersonalScore 依然可以排到最前面。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..models import RepoRecord
from ..ranking import ScoredRepo
from .profile import KeywordMatch
from .schemas import RepoAnalysis

# 权重（规格写死，不做成配置，避免 For You 口径漂移）
WEIGHT_RELEVANCE = 0.55
WEIGHT_HEAT = 0.25
WEIGHT_KEYWORD = 0.20

FOR_YOU_TOP_N = 10


def personal_score(
    relevance: Optional[float], heat: Optional[float], keyword: Optional[float]
) -> float:
    """
    计算 Personal Score（0~100）

    Args:
        relevance: AI relevance score（0~100，缺失按 0）
        heat: Heat Score（0~100，缺失按 0）
        keyword: deterministic keyword score（0~100，缺失按 0）

    Returns:
        保留 1 位小数的 0~100 分数

    Examples:
        >>> personal_score(98, 50, 100)
        86.4
        >>> personal_score(0, 0, 0)
        0.0
    """
    def _clean(value: Optional[float]) -> float:
        if value is None:
            return 0.0
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if number != number:  # NaN
            return 0.0
        return max(0.0, min(100.0, number))

    total = (
        WEIGHT_RELEVANCE * _clean(relevance)
        + WEIGHT_HEAT * _clean(heat)
        + WEIGHT_KEYWORD * _clean(keyword)
    )
    return round(max(0.0, min(100.0, total)), 1)


@dataclass
class ForYouEntry:
    """🎯 For You 榜单里的一项"""

    scored: ScoredRepo
    analysis: RepoAnalysis
    keyword: KeywordMatch = field(default_factory=KeywordMatch)
    personal_score: float = 0.0

    @property
    def record(self) -> RepoRecord:
        return self.scored.record

    @property
    def full_name(self) -> str:
        return self.scored.record.full_name

    @property
    def heat_score(self) -> float:
        return self.scored.score

    @property
    def relevance_score(self) -> int:
        return self.analysis.relevance_score

    @property
    def keyword_score(self) -> float:
        return self.keyword.score if self.keyword else 0.0

    @property
    def relevance_explanation(self) -> str:
        """
        「为什么与你相关」的展示文案

        优先用 AI 给的理由；AI 没给就退回 deterministic 关键词命中，
        保证这一栏永远不会空着。
        """
        if self.analysis.relevance_reason:
            return self.analysis.relevance_reason
        if self.keyword and self.keyword.matched_keywords:
            return f"命中你的兴趣关键词：{self.keyword.describe()}"
        return "暂无明确的相关性说明。"


def build_for_you(
    scored: Sequence[ScoredRepo],
    analyses: Dict[str, RepoAnalysis],
    keyword_scores: Dict[str, KeywordMatch],
    *,
    top_n: int = FOR_YOU_TOP_N,
) -> List[ForYouEntry]:
    """
    生成 🎯 For You 榜单

    只有**拿到 AI 分析**的仓库才会进入 For You —— 因为这一栏要展示
    「为什么与你相关 / 推荐动作」，没有 AI 数据就没有内容可展示。

    Args:
        scored: 当日全部候选（含 Hot20 之外的）
        analyses: ``{full_name_lower: RepoAnalysis}``
        keyword_scores: ``{full_name_lower: KeywordMatch}``
        top_n: 榜单长度

    Returns:
        按 PersonalScore 降序的前 ``top_n`` 项
    """
    if top_n <= 0 or not analyses:
        return []

    analyses = analyses or {}
    keyword_scores = keyword_scores or {}

    entries: List[ForYouEntry] = []
    for item in scored or ():
        key = item.record.full_name.lower()
        analysis = analyses.get(key)
        if analysis is None:
            continue
        keyword = keyword_scores.get(key) or KeywordMatch()
        entries.append(
            ForYouEntry(
                scored=item,
                analysis=analysis,
                keyword=keyword,
                personal_score=personal_score(
                    analysis.relevance_score, item.score, keyword.score
                ),
            )
        )

    entries.sort(
        key=lambda entry: (
            -entry.personal_score,
            -entry.relevance_score,
            -entry.keyword_score,
            -entry.heat_score,
            entry.full_name.lower(),
        )
    )
    return entries[:top_n]
