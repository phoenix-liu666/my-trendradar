# coding=utf-8
"""
仓库批量分析编排

为什么必须 batch（规格 §16）
----------------------------
``1 repo = 1 HTTP request`` 意味着 30 个仓库就是 30 次请求 ——
又慢又贵还容易触发限流。这里按 5~8 个一批（默认 6）打包：

    30 repos → 5 batches × 6  +  1 次 daily synthesis  ≈ 6~7 次请求/天

两种批次
--------
``full``    缓存未命中：带 README，模型产出全部字段
``daily``   缓存命中：不带 README，只产出 why_hot / relevance / 推荐动作
            （静态字段直接用缓存，省掉绝大部分 Token）

降级原则
--------
- 单个 batch 失败（超时 / 429 / 5xx / malformed JSON）→ 只有这批没有 AI 数据
- 全部 batch 失败 → 整个日报退回基础版本
- 模型漏掉某个仓库 → 那个仓库没有 AI 数据
- 任何异常都不会向上抛出
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..logging_utils import log, warn
from .cache import StaticAnalysisCache
from .client import DeepSeekClient
from .config import JSON_RETRY_LIMIT, AIConfig
from .profile import UserProfile
from .prompts import (
    RETRY_INSTRUCTION,
    SYSTEM_PROMPT,
    build_daily_analysis_prompt,
    build_full_analysis_prompt,
    build_messages,
    build_repo_facts,
)
from .schemas import AIUsage, RepoAnalysis, apply_hallucination_guard, parse_repo_analysis
from .selector import AICandidate

# batch 之间的间隔（秒），温和一点，避免触发服务端限流
DEFAULT_BATCH_INTERVAL = 0.5


@dataclass
class AnalysisOutcome:
    """一次仓库分析的结果"""

    analyses: Dict[str, RepoAnalysis] = field(default_factory=dict)
    usage: AIUsage = field(default_factory=AIUsage)
    batches: int = 0
    failed_batches: int = 0
    input_chars_used: int = 0
    # 是否因为达到输入字符硬上限而提前停止
    stopped_early: bool = False

    @property
    def analyzed_count(self) -> int:
        return len(self.analyses)

    @property
    def all_failed(self) -> bool:
        return self.batches > 0 and not self.analyses


def chunk(items: Sequence[Any], size: int) -> List[List[Any]]:
    """把序列切成固定大小的块"""
    size = max(1, int(size))
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def _allowed_evidence_keys(candidate: AICandidate) -> List[str]:
    """本仓库真正喂给模型的指标键（why_hot.evidence 只能引用这些）"""
    return sorted(build_repo_facts(candidate).keys())


def _index_response(payload: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    把模型返回的 repositories 数组转成 ``{full_name_lower: item}``

    兼容三种常见形态：
    - ``{"repositories": [...]}``（我们要求的）
    - ``{"results": [...]}`` / ``{"data": [...]}``
    - ``{"owner/repo": {...}}``（直接用仓库名当 key）
    """
    if not isinstance(payload, dict):
        return {}

    items: Any = None
    for key in ("repositories", "results", "data", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            items = value
            break

    indexed: Dict[str, Dict[str, Any]] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("full_name") or item.get("repo") or "").strip().lower()
            if name:
                indexed[name] = item
        return indexed

    # 退化形态：顶层直接是 {"owner/repo": {...}}
    for key, value in payload.items():
        if isinstance(value, dict) and "/" in str(key):
            indexed[str(key).strip().lower()] = value
    return indexed


def _finalize(
    candidate: AICandidate,
    payload: Dict[str, Any],
    *,
    from_cache: bool,
) -> Optional[RepoAnalysis]:
    """把模型返回的单条结果转成经过校验、消毒、幻觉控制的 ``RepoAnalysis``"""
    analysis = parse_repo_analysis(payload, full_name=candidate.full_name)
    if analysis is None:
        return None

    if from_cache and candidate.cached_static:
        # 静态字段以缓存为准（daily 批次本来就没让模型输出这些）
        analysis.apply_static_fields(candidate.cached_static)
        analysis.from_cache = True

    apply_hallucination_guard(analysis, _allowed_evidence_keys(candidate))

    if not analysis.has_content:
        return None
    return analysis


def analyze_repositories(
    client: DeepSeekClient,
    candidates: Sequence[AICandidate],
    profile: UserProfile,
    *,
    config: AIConfig,
    cache: Optional[StaticAnalysisCache] = None,
    batch_interval: float = DEFAULT_BATCH_INTERVAL,
    sleep_func: Callable[[float], None] = time.sleep,
    now_iso: str = "",
) -> AnalysisOutcome:
    """
    批量分析仓库

    Args:
        client: DeepSeek 客户端
        candidates: 已经选好的候选（长度已受 repo limit 约束）
        profile: 用户兴趣画像
        config: AI 配置（batch size / 输入字符上限）
        cache: 静态字段缓存（写入用；读取在 pipeline 里完成）
        batch_interval: batch 之间的间隔（秒）
        sleep_func: 睡眠函数（测试注入）
        now_iso: 写缓存用的时间戳

    Returns:
        ``AnalysisOutcome``（永远返回，绝不抛异常）
    """
    outcome = AnalysisOutcome()
    if not candidates:
        return outcome

    full_batch = [c for c in candidates if c.needs_static_analysis]
    daily_batch = [c for c in candidates if not c.needs_static_analysis]

    log(f"[AI] repositories requiring static analysis: {len(full_batch)}")

    jobs: List[tuple] = []
    for group in chunk(full_batch, config.batch_size):
        jobs.append(("full", group))
    for group in chunk(daily_batch, config.batch_size):
        jobs.append(("daily", group))

    log(f"[AI] batches: {len(jobs)}")

    for index, (kind, group) in enumerate(jobs):
        if kind == "full":
            user_prompt = build_full_analysis_prompt(group, profile)
        else:
            user_prompt = build_daily_analysis_prompt(group, profile)

        # ---- 输入字符硬上限：达到就停，已有结果照常使用 ----
        cost = len(user_prompt) + len(SYSTEM_PROMPT)
        if (
            config.max_input_chars > 0
            and outcome.input_chars_used + cost > config.max_input_chars
        ):
            outcome.stopped_early = True
            warn(
                f"[AI] 已达输入字符硬上限（{config.max_input_chars}），"
                f"停止后续 {len(jobs) - index} 个 batch，使用已有 AI 结果继续生成日报"
            )
            break
        outcome.input_chars_used += cost

        result = client.chat_json(
            build_messages(SYSTEM_PROMPT, user_prompt),
            retry_instruction=RETRY_INSTRUCTION,
            json_retry_limit=JSON_RETRY_LIMIT,
        )
        outcome.batches += 1
        # 逐次累加本批的用量：client.usage 是**跨阶段累计**的，
        # 在这里 merge 会和 synthesis 的用量重复计数
        for chat in result.chat_results:
            outcome.usage.add_request(
                success=chat.ok,
                prompt_tokens=chat.prompt_tokens,
                completion_tokens=chat.completion_tokens,
                total_tokens=chat.total_tokens,
                prompt_cache_hit_tokens=chat.prompt_cache_hit_tokens,
                prompt_cache_miss_tokens=chat.prompt_cache_miss_tokens,
            )

        if not result.ok:
            outcome.failed_batches += 1
            warn(
                f"[AI] batch {index + 1}/{len(jobs)}（{kind}, {len(group)} repos）失败："
                f"{result.error or 'unknown'}，这批仓库使用基础数据"
            )
        else:
            indexed = _index_response(result.data)
            matched = 0
            for candidate in group:
                payload = indexed.get(candidate.key)
                if payload is None:
                    continue
                analysis = _finalize(candidate, payload, from_cache=(kind == "daily"))
                if analysis is None:
                    continue
                outcome.analyses[candidate.key] = analysis
                matched += 1
                if kind == "full" and cache is not None:
                    cache.put(analysis, candidate.record, now_iso=now_iso)

            if matched < len(group):
                warn(
                    f"[AI] batch {index + 1}/{len(jobs)} 只返回了 {matched}/{len(group)} 个仓库，"
                    f"缺失的使用基础数据"
                )

        if batch_interval > 0 and index < len(jobs) - 1:
            sleep_func(batch_interval)

    outcome.usage.repositories_analyzed = len(outcome.analyses)

    if outcome.batches:
        if outcome.all_failed:
            warn("[AI] 全部 batch 均失败，本次日报退回基础版本（不含 AI 增强）")
        else:
            log("[AI] repository analysis completed")

    return outcome
