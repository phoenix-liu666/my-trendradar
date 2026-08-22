# coding=utf-8
"""
候选仓库池构建

双数据源：
A. GitHub Trending（daily）—— 页面抓取
B. GitHub API 搜索
   - New & Rising：最近 N 天创建且已有一定 Star 的仓库
   - Popular active：少量最近活跃的热门仓库（补充，稳定不随机，
     这样它们每天都在候选池里，才能积累出真实的 24h / 7d 差值）

去重后目标规模 50~100 个仓库；API 调用量控制在个位数搜索请求 +
少量单仓库详情请求，绝不高并发狂刷。

容错：任何单一数据源失败都只降级，不中断；只有「全部来源都失败、
候选为 0」时才由上层判定为失败。
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .github_api import GitHubAPIClient
from .logging_utils import log, warn
from .models import (
    SOURCE_SEARCH_NEW,
    SOURCE_SEARCH_POPULAR,
    RepoRecord,
    dedupe_records,
)
from .timeutils import DEFAULT_TIMEZONE, age_in_days, now, shift_date_str
from .trending import collect_trending

# --- 默认参数 -----------------------------------------------------------
DEFAULT_NEW_WINDOW_DAYS = 30      # "新项目" 时间窗口
DEFAULT_NEW_MIN_STARS = 50        # 新项目最低 Star（过滤无意义小项目）
DEFAULT_NEW_PER_PAGE = 50         # New & Rising 候选数量
DEFAULT_POPULAR_MIN_STARS = 1000  # 活跃热门项目最低 Star
DEFAULT_POPULAR_WINDOW_DAYS = 2   # 最近 N 天有 push
DEFAULT_POPULAR_PER_PAGE = 25     # 活跃热门候选数量
DEFAULT_MAX_DETAIL_REQUESTS = 30  # 单仓库详情请求上限（控制 API 消耗）
DEFAULT_DETAIL_INTERVAL = 0.2     # 详情请求之间的间隔（秒）


@dataclass
class CollectionResult:
    """候选池构建结果"""

    repositories: List[RepoRecord] = field(default_factory=list)
    trending_count: int = 0
    search_new_count: int = 0
    search_popular_count: int = 0
    detail_fetched: int = 0
    detail_failed: int = 0
    trending_ok: bool = False
    search_ok: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def api_candidate_count(self) -> int:
        """来自 GitHub API 搜索的候选数量（去重前）"""
        return self.search_new_count + self.search_popular_count

    @property
    def unique_count(self) -> int:
        return len(self.repositories)

    def new_repo_count(self, *, max_age_days: int = DEFAULT_NEW_WINDOW_DAYS) -> int:
        """候选池中「新项目」数量（创建时间在窗口内）"""
        count = 0
        for record in self.repositories:
            age = age_in_days(record.created_at)
            if age is not None and age <= max_age_days:
                count += 1
        return count


def _search(
    client: GitHubAPIClient,
    query: str,
    *,
    source: str,
    sort: str,
    per_page: int,
    label: str,
) -> List[RepoRecord]:
    """执行一次搜索并转换为 RepoRecord（失败返回空列表）"""
    try:
        items = client.search_repositories(query, sort=sort, per_page=per_page)
    except Exception as exc:  # 客户端已尽量吞掉异常，这里是最后一道防线
        warn(f"{label} 搜索异常：{type(exc).__name__}: {exc}")
        return []

    records: List[RepoRecord] = []
    for item in items:
        record = RepoRecord.from_api(item, source=source)
        if record is not None:
            records.append(record)

    if not records:
        warn(f"{label} 搜索未返回可用结果（query: {query}）")
    return records


def collect_candidates(
    client: GitHubAPIClient,
    *,
    today: str,
    timezone: str = DEFAULT_TIMEZONE,
    trending_collector: Optional[Callable[..., List[RepoRecord]]] = None,
    new_window_days: int = DEFAULT_NEW_WINDOW_DAYS,
    new_min_stars: int = DEFAULT_NEW_MIN_STARS,
    new_per_page: int = DEFAULT_NEW_PER_PAGE,
    popular_min_stars: int = DEFAULT_POPULAR_MIN_STARS,
    popular_window_days: int = DEFAULT_POPULAR_WINDOW_DAYS,
    popular_per_page: int = DEFAULT_POPULAR_PER_PAGE,
    max_detail_requests: int = DEFAULT_MAX_DETAIL_REQUESTS,
    detail_interval: float = DEFAULT_DETAIL_INTERVAL,
    sleep_func: Callable[[float], None] = time.sleep,
) -> CollectionResult:
    """
    构建当日候选仓库池

    Args:
        client: GitHub API 客户端
        today: 当天日期 "YYYY-MM-DD"（用于动态构造搜索条件）
        timezone: 时区名（用于 collected_at）
        trending_collector: Trending 采集函数（测试可注入），返回 RepoRecord 列表
        new_window_days: 新项目时间窗口（天）
        new_min_stars: 新项目最低 Star
        new_per_page: 新项目候选数量
        popular_min_stars: 活跃热门项目最低 Star
        popular_window_days: 活跃热门项目的 push 时间窗口（天）
        popular_per_page: 活跃热门候选数量
        max_detail_requests: 单仓库详情请求上限
        detail_interval: 详情请求间隔（秒）
        sleep_func: 睡眠函数（测试可注入）

    Returns:
        CollectionResult
    """
    result = CollectionResult()

    # ---------- A. GitHub Trending ----------
    collector = trending_collector or collect_trending
    try:
        trending_records = collector() or []
    except Exception as exc:
        warn(f"Trending 采集失败：{type(exc).__name__}: {exc}")
        trending_records = []
        result.warnings.append("Trending 采集失败")

    result.trending_count = len(trending_records)
    result.trending_ok = bool(trending_records)
    log(f"trending candidates: {result.trending_count}")
    if not result.trending_ok:
        result.warnings.append("Trending 数据不可用，本次仅使用 GitHub API 候选")

    # ---------- B. New & Rising 搜索 ----------
    created_since = shift_date_str(today, -abs(new_window_days)) or today
    new_query = f"created:>={created_since} stars:>{new_min_stars}"
    new_records = _search(
        client,
        new_query,
        source=SOURCE_SEARCH_NEW,
        sort="stars",
        per_page=new_per_page,
        label="New & Rising",
    )
    result.search_new_count = len(new_records)

    # ---------- C. Popular active 搜索（少量补充，保持稳定） ----------
    pushed_since = shift_date_str(today, -abs(popular_window_days)) or today
    popular_query = f"pushed:>={pushed_since} stars:>{popular_min_stars}"
    popular_records = _search(
        client,
        popular_query,
        source=SOURCE_SEARCH_POPULAR,
        sort="stars",
        per_page=popular_per_page,
        label="Popular active",
    )
    result.search_popular_count = len(popular_records)

    result.search_ok = bool(new_records or popular_records)
    if not result.search_ok:
        result.warnings.append("GitHub Search API 不可用，本次仅使用 Trending 候选")
    log(f"API candidates: {result.api_candidate_count}")

    # ---------- 去重合并 ----------
    merged = dedupe_records(list(trending_records) + new_records + popular_records)
    log(f"unique repositories: {len(merged)}")

    # ---------- 补全 Trending-only 仓库的详情 ----------
    pending = [record for record in merged if not record.api_enriched]
    if pending and max_detail_requests > 0:
        log(f"enriching {min(len(pending), max_detail_requests)} trending-only repositories via API...")
    for index, record in enumerate(pending):
        if index >= max_detail_requests:
            warn(
                f"达到详情请求上限（{max_detail_requests}），"
                f"剩余 {len(pending) - index} 个仓库保留 Trending 页面数据"
            )
            break
        if client.rate_limited:
            warn("GitHub API 已限流，剩余仓库保留 Trending 页面数据")
            break

        try:
            payload = client.get_repository(record.full_name)
        except Exception as exc:
            payload = None
            warn(f"获取 {record.full_name} 详情异常：{type(exc).__name__}: {exc}")

        if payload is None:
            # 单个仓库失败不影响整体：保留 Trending HTML 已解析的数据
            result.detail_failed += 1
            continue

        enriched = RepoRecord.from_api(payload)
        if enriched is None:
            result.detail_failed += 1
            continue

        record.merge(enriched)
        result.detail_fetched += 1

        if detail_interval > 0 and index < min(len(pending), max_detail_requests) - 1:
            sleep_func(detail_interval)

    if result.detail_failed:
        warn(f"{result.detail_failed} 个仓库详情获取失败，已使用 Trending 页面数据降级")

    # ---------- 统一打上采集时间 ----------
    collected_at = now(timezone).isoformat(timespec="seconds")
    for record in merged:
        record.collected_at = collected_at

    result.repositories = merged
    return result


def summarize_sources(records: List[RepoRecord]) -> Dict[str, int]:
    """统计各来源命中数量（用于日志与报告概览）"""
    summary: Dict[str, int] = {}
    for record in records:
        for source in record.sources or ["unknown"]:
            summary[source] = summary.get(source, 0) + 1
    return summary
