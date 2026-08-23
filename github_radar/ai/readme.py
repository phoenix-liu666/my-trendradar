# coding=utf-8
"""
README 获取（只给最终 AI 候选用）

硬约束（规格 §11 / §24）
------------------------
- **只对最终 AI candidates 取 README**，绝不给 80~100 个候选全取
- 单个 README 最多 ``6000`` 字符，超出截断
- 只给「缓存未命中、需要完整分析」的候选取；命中缓存的不取
- 404 / 编码异常 / 限流 / 网络失败 → 该仓库无 README，继续跑
- 请求之间有间隔，绝不为了 README 高频打 GitHub

总量上限：所有 README 加起来不超过 ``total_budget`` 字符，
防止某天出现一堆超长 README 把 Token 顶穿。
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..logging_utils import log, warn
from .config import README_MAX_CHARS
from .prompts import sanitize_readme
from .selector import AICandidate

# README 请求之间的间隔（秒）
DEFAULT_INTERVAL = 0.2


@dataclass
class ReadmeStats:
    """README 获取统计"""

    requested: int = 0
    fetched: int = 0
    missing: int = 0
    truncated: int = 0
    total_chars: int = 0
    budget_exhausted: bool = False

    def describe(self) -> str:
        suffix = "（已达字符预算上限）" if self.budget_exhausted else ""
        return (
            f"[AI] readme: {self.fetched}/{self.requested} fetched, "
            f"{self.missing} unavailable, {self.truncated} truncated, "
            f"{self.total_chars} chars{suffix}"
        )


def fetch_readmes(
    client: Any,
    candidates: Sequence[AICandidate],
    *,
    max_chars: int = README_MAX_CHARS,
    total_budget: int = 0,
    interval: float = DEFAULT_INTERVAL,
    sleep_func: Callable[[float], None] = time.sleep,
) -> ReadmeStats:
    """
    为候选批量获取 README（就地写入 ``candidate.readme``）

    Args:
        client: GitHub API 客户端（需实现 ``get_readme_text``）
        candidates: 需要完整分析的候选
        max_chars: 单个 README 最大字符数
        total_budget: 所有 README 合计字符上限（<=0 表示不限）
        interval: 请求间隔（秒）
        sleep_func: 睡眠函数（测试注入）

    Returns:
        ``ReadmeStats``
    """
    stats = ReadmeStats()
    if not candidates:
        return stats

    for index, candidate in enumerate(candidates):
        if total_budget > 0 and stats.total_chars >= total_budget:
            stats.budget_exhausted = True
            warn(
                f"[AI] README 字符预算已用尽（{stats.total_chars}/{total_budget}），"
                f"剩余 {len(candidates) - index} 个仓库不再获取 README"
            )
            break

        stats.requested += 1
        text = None
        try:
            text = client.get_readme_text(candidate.full_name)
        except Exception as exc:
            # 客户端已经尽量吞异常，这里是最后一道防线
            warn(f"[AI] 获取 README 异常（{candidate.full_name}）：{type(exc).__name__}: {exc}")
            text = None

        if not text:
            stats.missing += 1
            candidate.readme = ""
            candidate.readme_ok = False
        else:
            original_length = len(text)
            remaining = max_chars
            if total_budget > 0:
                remaining = min(remaining, max(0, total_budget - stats.total_chars))
            cleaned = sanitize_readme(text, remaining)
            candidate.readme = cleaned
            candidate.readme_ok = bool(cleaned)
            if cleaned:
                stats.fetched += 1
                stats.total_chars += len(cleaned)
                if original_length > remaining:
                    stats.truncated += 1
            else:
                stats.missing += 1

        if interval > 0 and index < len(candidates) - 1:
            sleep_func(interval)

    log(stats.describe())
    return stats
