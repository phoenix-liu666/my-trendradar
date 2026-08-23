# coding=utf-8
"""
每日趋势总结（📡 今日 GitHub 技术信号）

规格 §22 的两条硬要求：
1. 在所有仓库分析完成之后**只调用一次**
2. **不重新提交 README**，输入只有结构化汇总
   （Top20 / Rising10 / For You 的名称、category、Heat Score、24h、relevance）

失败处理：返回 ``None``，报告那一块显示「AI 趋势总结今日不可用」，
**绝不阻止邮件发送**。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..logging_utils import log, warn
from ..ranking import ScoredRepo
from .client import DeepSeekClient
from .config import JSON_RETRY_LIMIT
from .prompts import (
    RETRY_INSTRUCTION,
    SYSTEM_PROMPT,
    build_messages,
    build_synthesis_prompt,
)
from .schemas import AIUsage, DailySynthesis, RepoAnalysis, parse_daily_synthesis
from .scoring import ForYouEntry

# 汇总里每个榜单最多带多少条（控制 synthesis 的输入体积）
MAX_HOT = 20
MAX_RISING = 10
MAX_FOR_YOU = 10


def _repo_summary(
    item: ScoredRepo, analysis: Optional[RepoAnalysis]
) -> Dict[str, Any]:
    """单个仓库的结构化摘要（只放事实，不放 README）"""
    entry: Dict[str, Any] = {
        "full_name": item.record.full_name,
        "heat_score": item.score,
    }
    if item.record.stars is not None:
        entry["stars"] = item.record.stars
    if item.delta.delta_stars_24h is not None:
        entry["delta_24h"] = item.delta.delta_stars_24h
    if item.record.language:
        entry["language"] = item.record.language
    if analysis is not None:
        entry["category"] = analysis.category
        entry["relevance"] = analysis.relevance_score
        if analysis.summary_zh:
            entry["summary_zh"] = analysis.summary_zh
    return entry


def build_summary_payload(
    hot: Sequence[ScoredRepo],
    rising: Sequence[ScoredRepo],
    for_you: Sequence[ForYouEntry],
    analyses: Dict[str, RepoAnalysis],
    *,
    date: str = "",
) -> Dict[str, Any]:
    """构造 synthesis 的输入（纯结构化，体积很小）"""
    analyses = analyses or {}

    def summarize(items: Sequence[ScoredRepo], limit: int) -> List[Dict[str, Any]]:
        return [
            _repo_summary(item, analyses.get(item.record.full_name.lower()))
            for item in list(items or ())[:limit]
        ]

    payload: Dict[str, Any] = {
        "date": date,
        "hot_today": summarize(hot, MAX_HOT),
        "new_and_rising": summarize(rising, MAX_RISING),
        "for_you": [
            {
                "full_name": entry.full_name,
                "personal_score": entry.personal_score,
                "relevance": entry.relevance_score,
                "category": entry.analysis.category,
            }
            for entry in list(for_you or ())[:MAX_FOR_YOU]
        ],
    }

    # 类别分布：让模型能看出「哪一类今天扎堆」，而不是靠单个项目下结论
    distribution: Dict[str, int] = {}
    for analysis in analyses.values():
        distribution[analysis.category] = distribution.get(analysis.category, 0) + 1
    payload["category_distribution"] = dict(
        sorted(distribution.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    return payload


@dataclass
class SynthesisOutcome:
    """synthesis 调用结果"""

    synthesis: Optional[DailySynthesis] = None
    usage: AIUsage = field(default_factory=AIUsage)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.synthesis is not None


def run_synthesis(
    client: DeepSeekClient, payload: Dict[str, Any]
) -> SynthesisOutcome:
    """
    调用一次每日趋势总结

    Returns:
        ``SynthesisOutcome``（失败时 ``synthesis`` 为 None，绝不抛异常）
    """
    outcome = SynthesisOutcome()

    result = client.chat_json(
        build_messages(SYSTEM_PROMPT, build_synthesis_prompt(payload)),
        retry_instruction=RETRY_INSTRUCTION,
        json_retry_limit=JSON_RETRY_LIMIT,
    )
    for chat in result.chat_results:
        outcome.usage.add_request(
            success=chat.ok,
            prompt_tokens=chat.prompt_tokens,
            completion_tokens=chat.completion_tokens,
            total_tokens=chat.total_tokens,
        )

    if not result.ok:
        outcome.error = result.error or "unknown"
        warn(f"[AI] daily synthesis 失败：{outcome.error}（趋势总结今日不可用，日报照常发送）")
        return outcome

    synthesis = parse_daily_synthesis(result.data)
    if synthesis is None:
        outcome.error = "synthesis 内容为空"
        warn("[AI] daily synthesis 返回内容为空，趋势总结今日不可用")
        return outcome

    outcome.synthesis = synthesis
    log("[AI] daily synthesis completed")
    return outcome
