# coding=utf-8
"""
AI 增强总入口

整条链路::

    profile 加载
        ↓
    deterministic keyword score（全部候选，不花一分钱）
        ↓
    候选选择（P1 强匹配 / P2 Hot20 / P3 Rising10 / P4 其它，硬上限 30）
        ↓
    静态缓存查询（命中的不再读 README、不再重复产出静态字段）
        ↓
    README 获取（只给缓存未命中的候选，单个 ≤6000 字符）
        ↓
    DeepSeek 批量分析（5~8 个/批，thinking disabled）
        ↓
    Personal Score → 🎯 For You Top10
        ↓
    daily synthesis（一次，只吃结构化汇总）
        ↓
    Token 统计 + 费用估算

**唯一的对外承诺**：这个函数永远不抛异常。
任何一步失败都只影响 AI 增强部分，基础日报照常生成、照常发送。
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

from ..logging_utils import log, warn
from ..ranking import ScoredRepo
from .analyzer import analyze_repositories
from .cache import (
    DEFAULT_CACHE_RETENTION_DAYS,
    DailyResultStore,
    StaticAnalysisCache,
    default_cache_dir,
)
from .client import DeepSeekClient
from .config import AIConfig
from .pricing import estimate_cost, get_pricing
from .profile import UserProfile, load_profile, log_profile, score_all
from .readme import fetch_readmes
from .result import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_PARTIAL,
    AIReportData,
    disabled_result,
)
from .schemas import AIUsage
from .scoring import FOR_YOU_TOP_N, build_for_you
from .selector import log_selection, select_ai_candidates
from .synthesis import build_summary_payload, run_synthesis


def run_ai_enhancement(
    scored: Sequence[ScoredRepo],
    hot: Sequence[ScoredRepo],
    rising: Sequence[ScoredRepo],
    *,
    config: AIConfig,
    data_dir: Union[str, Path],
    date: str,
    github_client: Any = None,
    profile: Optional[UserProfile] = None,
    profile_path: Optional[Union[str, Path]] = None,
    deepseek_client: Optional[DeepSeekClient] = None,
    env: Optional[Dict[str, str]] = None,
    force: bool = False,
    now_iso: str = "",
    sleep_func: Callable[[float], None] = time.sleep,
    retention_days: int = DEFAULT_CACHE_RETENTION_DAYS,
) -> AIReportData:
    """
    执行 AI 增强

    Args:
        scored: 当日全部候选（已按 Heat Score 排序，**只读**）
        hot: 🔥 Hot Today Top20
        rising: 🌱 New & Rising Top10
        config: AI 配置
        data_dir: 数据目录（缓存写在 ``<data_dir>/ai_cache/``）
        date: 日期 YYYY-MM-DD
        github_client: GitHub API 客户端（取 README 用；None 则不取 README）
        profile: 兴趣画像（默认从配置文件加载）
        profile_path: 兴趣画像路径（测试注入）
        deepseek_client: DeepSeek 客户端（测试注入）
        env: 环境变量（价格覆盖用）
        force: 是否忽略「当天已算好的 AI 结果」强制重算（对应 ``--force-run``）
        now_iso: 时间戳（写缓存用）
        sleep_func: 睡眠函数（测试注入）
        retention_days: AI 缓存保留天数

    Returns:
        ``AIReportData``（**永远返回**，绝不抛异常）
    """
    if not config.enabled:
        return disabled_result(config.disabled_reason, config.model)

    log(config.describe())

    cache_dir = default_cache_dir(data_dir)
    daily_store = DailyResultStore(cache_dir)

    # ---------- 0. 当天已经算过就直接复用 ----------
    #
    # workflow 每天有 4 次兜底 cron：第一次失败后，后面几次会重跑整条链路。
    # 没有这一层，同一天最多会分析 4×30 个仓库，直接违反
    # 「每天最多 30 个 unique repositories」的硬限制。
    if not force:
        cached = daily_store.load(date, model=config.model)
        if cached is not None:
            analyses, synthesis, usage = cached
            log(f"[AI] reusing today's AI result ({len(analyses)} repositories, no new API call)")
            profile = profile or _load_profile_safely(profile_path)
            keyword_scores = score_all([item.record for item in scored], profile)
            for_you = build_for_you(scored, analyses, keyword_scores, top_n=FOR_YOU_TOP_N)
            return _finish(
                config=config,
                analyses=analyses,
                for_you=for_you,
                synthesis=synthesis,
                usage=usage,
                env=env,
                reused=True,
                notes=[] if synthesis else ["AI 趋势总结今日不可用。"],
                status=STATUS_OK if synthesis else STATUS_PARTIAL,
            )

    # ---------- 1. 画像 + deterministic keyword score ----------
    profile = profile or _load_profile_safely(profile_path)
    log_profile(profile)
    keyword_scores = score_all([item.record for item in scored], profile)

    # ---------- 2. 候选选择（硬上限） ----------
    candidates = select_ai_candidates(
        scored, hot, rising, keyword_scores, limit=config.repo_limit
    )
    log_selection(candidates)
    if not candidates:
        return _finish(
            config=config,
            analyses={},
            for_you=[],
            synthesis=None,
            usage=AIUsage(),
            env=env,
            reused=False,
            notes=["今日没有符合条件的 AI 候选仓库。"],
            status=STATUS_FAILED,
        )

    # ---------- 3. 静态缓存 ----------
    cache = StaticAnalysisCache(cache_dir, model=config.model)
    for candidate in candidates:
        static = cache.get(candidate.record)
        if static:
            candidate.cached_static = static
    log(f"[AI] cache hits: {cache.stats.hits}")

    # ---------- 4. README（只给需要完整分析的候选） ----------
    pending = [c for c in candidates if c.needs_static_analysis]
    if pending and github_client is not None:
        fetch_readmes(
            github_client,
            pending,
            max_chars=config.readme_max_chars,
            total_budget=config.max_input_chars,
            sleep_func=sleep_func,
        )
        # README 拿到之后重算 keyword score：规格 §10 要求 README 也参与
        # deterministic 匹配（选候选时还没有 README，这里补上更准的分数）
        _refresh_keyword_scores(pending, profile, keyword_scores)
    elif pending:
        warn("[AI] 没有可用的 GitHub 客户端，本次不获取 README（分析仍会进行）")

    # ---------- 5. 批量分析 ----------
    client = deepseek_client or DeepSeekClient(
        config.api_key or "",
        model=config.model,
        api_base=config.api_base,
        timeout=config.timeout,
    )

    outcome = analyze_repositories(
        client,
        candidates,
        profile,
        config=config,
        cache=cache,
        sleep_func=sleep_func,
        now_iso=now_iso,
    )

    notes = []
    if outcome.stopped_early:
        notes.append("本次 AI 分析因达到输入上限提前结束，部分项目没有 AI 解读。")
    if outcome.failed_batches and outcome.analyses:
        notes.append("本次有部分 AI 批次失败，对应项目显示的是基础数据。")

    # ---------- 6. Personal Score + For You ----------
    for_you = build_for_you(scored, outcome.analyses, keyword_scores, top_n=FOR_YOU_TOP_N)

    # ---------- 7. 每日趋势总结 ----------
    synthesis = None
    if outcome.analyses:
        payload = build_summary_payload(hot, rising, for_you, outcome.analyses, date=date)
        synthesis_outcome = run_synthesis(client, payload)
        synthesis = synthesis_outcome.synthesis
        if synthesis is None:
            notes.append("AI 趋势总结今日不可用。")
    else:
        notes.append("AI 分析今日不可用，本期为基础日报。")

    # ---------- 8. 统计 + 缓存落盘 ----------
    usage = _collect_usage(client, cache_hits=cache.stats.hits, analyzed=len(outcome.analyses))

    if outcome.analyses:
        daily_store.save(
            date,
            analyses=outcome.analyses,
            synthesis=synthesis,
            usage=usage,
            model=config.model,
            generated_at=now_iso,
        )
    daily_store.prune(retention_days, date)
    cache.prune(retention_days)

    status = STATUS_OK
    if not outcome.analyses:
        status = STATUS_FAILED
    elif outcome.failed_batches or outcome.stopped_early or synthesis is None:
        status = STATUS_PARTIAL

    return _finish(
        config=config,
        analyses=outcome.analyses,
        for_you=for_you,
        synthesis=synthesis,
        usage=usage,
        env=env,
        reused=False,
        notes=notes,
        status=status,
    )


# ----------------------------------------------------------------------
# 内部工具
# ----------------------------------------------------------------------
def _load_profile_safely(profile_path: Optional[Union[str, Path]]) -> UserProfile:
    """加载画像（load_profile 本身已容错，这里再包一层最后防线）"""
    try:
        return load_profile(profile_path)
    except Exception as exc:  # pragma: no cover - load_profile 不应抛异常
        warn(f"[AI] 兴趣画像加载异常（{type(exc).__name__}），使用兜底画像")
        from .profile import fallback_profile

        return fallback_profile()


def _refresh_keyword_scores(candidates, profile, keyword_scores) -> None:
    """README 到手后重算这些候选的 keyword score"""
    from .profile import KeywordIndex, keyword_score

    index = KeywordIndex(profile)
    for candidate in candidates:
        if not candidate.readme:
            continue
        match = keyword_score(
            candidate.record, profile, readme=candidate.readme, index=index
        )
        candidate.keyword = match
        keyword_scores[candidate.key] = match


def _collect_usage(
    client: DeepSeekClient, *, cache_hits: int, analyzed: int
) -> AIUsage:
    """
    汇总用量

    ``client.usage`` 已经累计了本次运行的**所有** API 调用
    （repository analysis + daily synthesis），是唯一权威来源，
    这样不会出现重复计数。
    """
    usage = AIUsage.from_dict(client.usage.to_dict())
    usage.cache_hits = max(0, int(cache_hits))
    usage.repositories_analyzed = max(0, int(analyzed))
    return usage


def _finish(
    *,
    config: AIConfig,
    analyses: Dict[str, Any],
    for_you,
    synthesis,
    usage: AIUsage,
    env: Optional[Dict[str, str]],
    reused: bool,
    notes,
    status: str = STATUS_OK,
) -> AIReportData:
    """统一收尾：估算费用 → 打日志 → 返回结果"""
    pricing = get_pricing(config.model, env)
    cost = estimate_cost(usage.prompt_tokens, usage.completion_tokens, pricing)

    result = AIReportData(
        enabled=True,
        # 一条 AI 分析都没有 = 全部失败，无论上游怎么判断
        status=status if analyses else STATUS_FAILED,
        model=config.model,
        analyses=analyses,
        for_you=list(for_you or []),
        synthesis=synthesis,
        usage=usage,
        cost=cost,
        notes=list(notes or []),
        reused=reused,
    )

    log(
        f"[AI] prompt tokens: {usage.prompt_tokens} | "
        f"completion tokens: {usage.completion_tokens} | "
        f"total tokens: {usage.total_tokens}"
    )
    log(
        f"[AI] requests: {usage.requests} "
        f"(success {usage.successful_requests} / failed {usage.failed_requests}) | "
        f"cache hits: {usage.cache_hits}"
    )
    log(f"[AI] estimated cost: {cost.format_total()}（{pricing.describe()}）")
    log(result.describe())
    return result
