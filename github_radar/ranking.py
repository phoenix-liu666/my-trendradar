# coding=utf-8
"""
GitHub Heat Score 与榜单筛选

设计目标：**可解释**、**可测试**、**对缺失数据友好**。

评分维度与基础权重
------------------
============  ======  ==========================================
维度            权重    含义
============  ======  ==========================================
growth_24h     45%    24h Star 增长（来自快照差值，最能代表“今天很火”）
growth_7d      20%    7 日平均日增（代表“持续在涨”而非一日游）
trending       15%    GitHub Trending 当日排名（排名越靠前越高）
freshness      10%    项目新鲜度（越新越高，指数衰减）
star_scale     10%    总 Star 规模（辅助项，log 压缩避免巨无霸碾压）
============  ======  ==========================================

归一化策略
----------
- Star 类指标先做 ``log1p()`` 压缩，再在**当日候选池内**做 min-max 归一化，
  避免 10 万星项目把小项目全部压成 0。
- Trending 排名直接映射到 (0, 1]，排名 1 为 1.0。
- 新鲜度用指数衰减 ``exp(-age/180)``：新建项目 ≈1.0，半年前 ≈0.37。

缺失指标的处理（关键）
----------------------
**不按 0 惩罚**。若某项指标不存在（例如首日没有 24h 增量、项目不在
Trending 榜上），则把它的权重从分母中剔除，对**剩余可用指标重新归一化**。
因此首日运行时，所有项目都按 trending / 新鲜度 / 总 Star 三项公平比较。
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from .history import StarDelta
from .models import RepoRecord
from .timeutils import age_in_days

# 基础权重（总和为 1.0，实际使用时按可用指标重新归一化）
WEIGHTS: Dict[str, float] = {
    "growth_24h": 0.45,
    "growth_7d": 0.20,
    "trending": 0.15,
    "freshness": 0.10,
    "star_scale": 0.10,
}

METRIC_LABELS: Dict[str, str] = {
    "growth_24h": "24h 增长",
    "growth_7d": "7日均增",
    "trending": "Trending 排名",
    "freshness": "项目新鲜度",
    "star_scale": "总 Star 规模",
}

# 新鲜度指数衰减常数（天）：age=0 → 1.0，age=180 → 0.37，age=365 → 0.13
FRESHNESS_DECAY_DAYS = 180.0

# New & Rising 默认筛选条件
NEW_RISING_MAX_AGE_DAYS = 30
NEW_RISING_MIN_STARS = 50
NEW_RISING_TOP_N = 10
HOT_TODAY_TOP_N = 20


# ----------------------------------------------------------------------
# 评分函数（独立、纯函数、可单测）
# ----------------------------------------------------------------------
def score_with_details(
    components: Dict[str, Optional[float]],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    根据已归一化的分项计算 Heat Score

    Args:
        components: ``{指标名: 0~1 的归一化值或 None}``，None 表示该指标不可用
        weights: 权重表，默认 ``WEIGHTS``

    Returns:
        (0~100 的分数, 实际生效的权重表)
        —— 生效权重已按可用指标重新归一化，总和为 1（无可用指标时为空表）
    """
    weights = weights or WEIGHTS

    available: Dict[str, float] = {}
    for key, weight in weights.items():
        value = components.get(key)
        if value is None or weight <= 0:
            continue
        # 容错：把异常值夹到 [0, 1]
        available[key] = max(0.0, min(1.0, float(value)))

    total_weight = sum(weights[key] for key in available)
    if total_weight <= 0:
        return 0.0, {}

    score = sum(weights[key] * value for key, value in available.items()) / total_weight
    normalized_weights = {key: weights[key] / total_weight for key in available}

    return round(max(0.0, min(1.0, score)) * 100.0, 2), normalized_weights


def score_from_components(
    components: Dict[str, Optional[float]],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Heat Score（0~100）

    这是排名的唯一入口函数：输入归一化后的分项，输出可解释的总分。
    """
    return score_with_details(components, weights)[0]


# ----------------------------------------------------------------------
# 归一化工具
# ----------------------------------------------------------------------
def minmax_normalize(values: Dict[str, float]) -> Dict[str, float]:
    """
    在候选池内做 min-max 归一化

    Args:
        values: ``{key: 原始值}``（只包含该指标可用的仓库）

    Returns:
        ``{key: 0~1}``；当所有值相同时统一返回 0.5（中性，避免全 0 或全 1）
    """
    if not values:
        return {}
    lowest = min(values.values())
    highest = max(values.values())
    span = highest - lowest
    if span <= 0:
        return {key: 0.5 for key in values}
    return {key: (value - lowest) / span for key, value in values.items()}


def freshness_score(age_days: Optional[float]) -> Optional[float]:
    """
    项目新鲜度（0~1），指数衰减；年龄未知返回 None
    """
    if age_days is None:
        return None
    return math.exp(-max(0.0, age_days) / FRESHNESS_DECAY_DAYS)


def trending_score(rank: Optional[int], max_rank: int) -> Optional[float]:
    """
    Trending 排名得分（0~1）

    rank=1 → 1.0；rank=max_rank → 1/max_rank。不在榜上返回 None（不惩罚，走重归一化）。
    """
    if rank is None or rank <= 0 or max_rank <= 0:
        return None
    capped = min(int(rank), max_rank)
    return (max_rank - capped + 1) / float(max_rank)


# ----------------------------------------------------------------------
# 排名
# ----------------------------------------------------------------------
@dataclass
class ScoredRepo:
    """带评分与增量的仓库"""

    record: RepoRecord
    delta: StarDelta = field(default_factory=StarDelta)
    score: float = 0.0
    components: Dict[str, Optional[float]] = field(default_factory=dict)
    weights_used: Dict[str, float] = field(default_factory=dict)
    age_days: Optional[float] = None
    rising_speed: Optional[float] = None
    rising_speed_estimated: bool = False

    @property
    def full_name(self) -> str:
        return self.record.full_name

    def explain(self) -> str:
        """
        生成一行可读的评分解释（调试 / 未来在报告中展示）

        Examples:
            "24h 增长 0.92×45% + Trending 排名 1.00×15% + ..."
        """
        parts = []
        for key, weight in sorted(self.weights_used.items(), key=lambda kv: -kv[1]):
            value = self.components.get(key)
            if value is None:
                continue
            label = METRIC_LABELS.get(key, key)
            parts.append(f"{label} {value:.2f}×{weight * 100:.0f}%")
        return " + ".join(parts) if parts else "无可用指标"


def rank_repositories(
    records: Iterable[RepoRecord],
    deltas: Optional[Dict[str, StarDelta]] = None,
    *,
    reference_time: Optional[datetime] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[ScoredRepo]:
    """
    计算 Heat Score 并按分数降序排序

    Args:
        records: 当日候选仓库
        deltas: ``{full_name_lower: StarDelta}``（首日可为空）
        reference_time: 计算项目年龄的参考时间（默认当前 UTC 时间）
        weights: 自定义权重

    Returns:
        按 Heat Score 降序排列的 ``ScoredRepo`` 列表
    """
    records = [record for record in records if record and record.full_name]
    deltas = deltas or {}

    # ---- 1. 收集各指标原始值 ----
    raw_growth_24h: Dict[str, float] = {}
    raw_growth_7d: Dict[str, float] = {}
    raw_star_scale: Dict[str, float] = {}
    ages: Dict[str, Optional[float]] = {}
    max_rank = 0

    for record in records:
        key = record.full_name.lower()
        delta = deltas.get(key, StarDelta())

        if delta.delta_stars_24h is not None:
            # 负增长（取关）按 0 处理：只在“涨得多”这个维度上比较
            raw_growth_24h[key] = math.log1p(max(0, delta.delta_stars_24h))

        if delta.average_daily_growth_7d is not None:
            raw_growth_7d[key] = math.log1p(max(0.0, delta.average_daily_growth_7d))

        if record.stars is not None:
            raw_star_scale[key] = math.log1p(max(0, record.stars))

        ages[key] = age_in_days(record.created_at, reference_time)

        if record.trending_rank:
            max_rank = max(max_rank, int(record.trending_rank))

    # ---- 2. 池内归一化 ----
    norm_growth_24h = minmax_normalize(raw_growth_24h)
    norm_growth_7d = minmax_normalize(raw_growth_7d)
    norm_star_scale = minmax_normalize(raw_star_scale)

    # ---- 3. 逐仓库评分 ----
    scored: List[ScoredRepo] = []
    for record in records:
        key = record.full_name.lower()
        delta = deltas.get(key, StarDelta())
        age = ages.get(key)

        components: Dict[str, Optional[float]] = {
            "growth_24h": norm_growth_24h.get(key),
            "growth_7d": norm_growth_7d.get(key),
            "trending": trending_score(record.trending_rank, max_rank),
            "freshness": freshness_score(age),
            "star_scale": norm_star_scale.get(key),
        }

        score, weights_used = score_with_details(components, weights)

        speed, estimated = _rising_speed(record, delta, age)
        scored.append(
            ScoredRepo(
                record=record,
                delta=delta,
                score=score,
                components=components,
                weights_used=weights_used,
                age_days=age,
                rising_speed=speed,
                rising_speed_estimated=estimated,
            )
        )

    scored.sort(key=_hot_sort_key)
    return scored


def _hot_sort_key(item: ScoredRepo):
    """Hot 榜排序键：分数 → 24h 增长 → 总 Star → 名称（保证结果稳定可复现）"""
    return (
        -item.score,
        -(item.delta.delta_stars_24h if item.delta.delta_stars_24h is not None else 0),
        -(item.record.stars or 0),
        item.record.full_name.lower(),
    )


def _rising_speed(
    record: RepoRecord, delta: StarDelta, age_days: Optional[float]
) -> Tuple[Optional[float], bool]:
    """
    成长速度（stars/天），用于 New & Rising 排序

    有 24h 快照差值时用真实增量；否则用「总 Star / 项目年龄」近似
    （单位相同，均为 stars/天，但后者是创建至今的平均值，属于估算）。

    Returns:
        (速度, 是否为估算值)
    """
    if delta.delta_stars_24h is not None:
        return float(delta.delta_stars_24h), False
    if record.stars is not None:
        divisor = max(age_days or 0.0, 1.0)
        return record.stars / divisor, True
    return None, True


def select_hot_today(
    scored: List[ScoredRepo], top_n: int = HOT_TODAY_TOP_N
) -> List[ScoredRepo]:
    """取 Heat Score 最高的前 N 个（输入应已由 ``rank_repositories`` 排序）"""
    if top_n <= 0:
        return []
    return list(scored[:top_n])


def select_new_and_rising(
    scored: List[ScoredRepo],
    *,
    max_age_days: int = NEW_RISING_MAX_AGE_DAYS,
    min_stars: int = NEW_RISING_MIN_STARS,
    top_n: int = NEW_RISING_TOP_N,
) -> List[ScoredRepo]:
    """
    New & Rising 榜单

    条件：
    - 创建时间在 ``max_age_days`` 天内（年龄未知的项目一律排除，
      避免 8 年前的老项目混进“新项目”榜）
    - 总 Star >= ``min_stars``（过滤明显无意义的小项目）

    排序：成长速度（stars/天）降序 → 7 日增量 → Heat Score → 名称
    """
    candidates = [
        item
        for item in scored
        if item.age_days is not None
        and item.age_days <= max_age_days
        and item.record.stars is not None
        and item.record.stars >= min_stars
    ]

    candidates.sort(
        key=lambda item: (
            -(item.rising_speed if item.rising_speed is not None else 0.0),
            -(item.delta.delta_stars_7d if item.delta.delta_stars_7d is not None else 0),
            -item.score,
            item.record.full_name.lower(),
        )
    )

    if top_n <= 0:
        return []
    return candidates[:top_n]
