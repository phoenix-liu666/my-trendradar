# coding=utf-8
"""
AI 候选选择

不能对全部 80~100 个候选调用 AI —— 每天只允许分析
``GITHUB_RADAR_AI_REPO_LIMIT``（默认 30）个 unique repositories。

优先级（规格 §8）
-----------------
====  ========================================================
P1    deterministic keyword 匹配非常高的项目（For You 强候选）
P2    🔥 Hot Today Top20
P3    🌱 New & Rising Top10
P4    其它候选（按 keyword score → Heat Score 排）
====  ========================================================

P1 放在最前面是关键：它保证「**Hot20 之外**但与我高度相关」的项目
一定能进 AI 分析，进而能出现在 🎯 For You Top10 里。

去重后按优先级截断到 limit，**绝不超过**。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..logging_utils import log
from ..models import RepoRecord
from ..ranking import ScoredRepo
from .profile import STRONG_MATCH_THRESHOLD, KeywordMatch

PRIORITY_STRONG_MATCH = 1
PRIORITY_HOT = 2
PRIORITY_RISING = 3
PRIORITY_OTHER = 4

PRIORITY_LABELS: Dict[int, str] = {
    PRIORITY_STRONG_MATCH: "profile 强匹配",
    PRIORITY_HOT: "Hot Today Top20",
    PRIORITY_RISING: "New & Rising Top10",
    PRIORITY_OTHER: "其它候选",
}


@dataclass
class AICandidate:
    """一个进入 AI 分析的候选仓库"""

    scored: ScoredRepo
    keyword: KeywordMatch = field(default_factory=KeywordMatch)
    priority: int = PRIORITY_OTHER
    # README 文本（只有需要完整分析的候选才会去取）
    readme: str = ""
    readme_ok: bool = False
    # 命中缓存时带上的静态字段（命中则无需再让模型重复产出）
    cached_static: Optional[Dict[str, Any]] = None

    @property
    def record(self) -> RepoRecord:
        return self.scored.record

    @property
    def full_name(self) -> str:
        return self.scored.record.full_name

    @property
    def key(self) -> str:
        return self.full_name.lower()

    @property
    def needs_static_analysis(self) -> bool:
        """是否需要完整分析（缓存未命中）"""
        return not self.cached_static

    @property
    def priority_label(self) -> str:
        return PRIORITY_LABELS.get(self.priority, "候选")


def _sorted_by_relevance(
    items: Sequence[ScoredRepo], keyword_scores: Dict[str, KeywordMatch]
) -> List[ScoredRepo]:
    """按 keyword score → Heat Score → 名称 排序（结果稳定可复现）"""
    def sort_key(item: ScoredRepo):
        match = keyword_scores.get(item.record.full_name.lower())
        return (
            -(match.score if match else 0.0),
            -item.score,
            item.record.full_name.lower(),
        )

    return sorted(items, key=sort_key)


def select_ai_candidates(
    scored: Sequence[ScoredRepo],
    hot: Sequence[ScoredRepo],
    rising: Sequence[ScoredRepo],
    keyword_scores: Dict[str, KeywordMatch],
    *,
    limit: int,
    strong_threshold: float = STRONG_MATCH_THRESHOLD,
) -> List[AICandidate]:
    """
    选出本次要送进 DeepSeek 的仓库

    Args:
        scored: 当日全部候选（已按 Heat Score 排序）
        hot: 🔥 Hot Today Top20
        rising: 🌱 New & Rising Top10
        keyword_scores: ``{full_name_lower: KeywordMatch}``
        limit: 硬上限（默认 30，来自 ``GITHUB_RADAR_AI_REPO_LIMIT``）
        strong_threshold: 判定 P1 的 keyword score 门槛

    Returns:
        去重后、按优先级排序、长度 ``<= limit`` 的候选列表
    """
    if limit <= 0:
        return []

    keyword_scores = keyword_scores or {}
    hot_keys = {item.record.full_name.lower() for item in hot or ()}
    rising_keys = {item.record.full_name.lower() for item in rising or ()}

    # ---- P1：profile 强匹配（允许来自 Hot20 之外）----
    strong = [
        item
        for item in scored or ()
        if (keyword_scores.get(item.record.full_name.lower()) or KeywordMatch()).score
        >= strong_threshold
    ]
    strong = _sorted_by_relevance(strong, keyword_scores)

    # ---- P4：其它候选 ----
    others = [
        item
        for item in scored or ()
        if item.record.full_name.lower() not in hot_keys
        and item.record.full_name.lower() not in rising_keys
    ]
    others = _sorted_by_relevance(others, keyword_scores)

    buckets = (
        (PRIORITY_STRONG_MATCH, strong),
        (PRIORITY_HOT, list(hot or ())),
        (PRIORITY_RISING, list(rising or ())),
        (PRIORITY_OTHER, others),
    )

    selected: List[AICandidate] = []
    seen = set()
    for priority, items in buckets:
        for item in items:
            if len(selected) >= limit:
                return selected
            key = item.record.full_name.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            selected.append(
                AICandidate(
                    scored=item,
                    keyword=keyword_scores.get(key) or KeywordMatch(),
                    priority=priority,
                )
            )

    return selected


def describe_selection(candidates: Sequence[AICandidate]) -> str:
    """一行日志：各优先级各选了多少个"""
    counts: Dict[int, int] = {}
    for candidate in candidates:
        counts[candidate.priority] = counts.get(candidate.priority, 0) + 1
    parts = [
        f"{PRIORITY_LABELS[priority]}={counts.get(priority, 0)}"
        for priority in sorted(PRIORITY_LABELS)
        if counts.get(priority)
    ]
    return f"[AI] candidate repos: {len(candidates)}" + (
        f" ({', '.join(parts)})" if parts else ""
    )


def log_selection(candidates: Sequence[AICandidate]) -> None:
    """打印候选选择结果"""
    log(describe_selection(candidates))
