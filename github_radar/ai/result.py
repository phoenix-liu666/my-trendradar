# coding=utf-8
"""
AI 增强结果（报告层唯一需要认识的数据结构）

刻意放在一个**很轻**的模块里：只依赖 dataclass 与同包的纯数据模块，
不 import client / requests。这样 ``report.py`` 可以放心导入它，
而不会因为 AI 相关依赖影响基础日报链路。

``status`` 的三种取值对应规格 §31 的三种最终行为::

    ok       AI 全部成功        → AI 增强日报
    partial  AI 部分失败        → 部分增强日报
    failed   AI 全部失败        → 原始基础日报
    disabled AI 未启用/被跳过   → 原始基础日报（且不展示 AI 区块）
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .pricing import CostEstimate
from .schemas import AIUsage, DailySynthesis, RepoAnalysis
from .scoring import ForYouEntry

STATUS_OK = "ok"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
STATUS_DISABLED = "disabled"

SYNTHESIS_UNAVAILABLE_TEXT = "AI 趋势总结今日不可用。"


@dataclass
class AIReportData:
    """一次运行的 AI 增强结果"""

    enabled: bool = False
    status: str = STATUS_DISABLED
    model: str = ""
    analyses: Dict[str, RepoAnalysis] = field(default_factory=dict)
    for_you: List[ForYouEntry] = field(default_factory=list)
    synthesis: Optional[DailySynthesis] = None
    usage: AIUsage = field(default_factory=AIUsage)
    cost: Optional[CostEstimate] = None
    # 展示在「数据说明」里的提示（例如「AI 部分失败」）
    notes: List[str] = field(default_factory=list)
    # 本次是否直接复用了当天已经算好的 AI 结果（4 次兜底 cron 的第 2~4 次）
    reused: bool = False
    disabled_reason: str = ""

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def analysis_for(self, full_name: str) -> Optional[RepoAnalysis]:
        """按仓库名取分析结果（大小写不敏感）"""
        if not full_name:
            return None
        return self.analyses.get(full_name.lower())

    @property
    def has_analyses(self) -> bool:
        return bool(self.analyses)

    @property
    def has_for_you(self) -> bool:
        return bool(self.for_you)

    @property
    def synthesis_available(self) -> bool:
        return self.synthesis is not None and self.synthesis.has_content

    @property
    def should_render(self) -> bool:
        """
        是否需要在邮件里展示 AI 区块

        AI 未启用时整块隐藏；启用了就要展示（哪怕失败，也要如实说明）。
        """
        return self.enabled

    def describe(self) -> str:
        """一行日志描述"""
        if not self.enabled:
            return f"[AI] status: disabled ({self.disabled_reason or '未启用'})"
        reused = " | reused daily result" if self.reused else ""
        synthesis = "yes" if self.synthesis_available else "no"
        return (
            f"[AI] status: {self.status} | analyzed: {len(self.analyses)} | "
            f"for you: {len(self.for_you)} | synthesis: {synthesis}{reused}"
        )


def disabled_result(reason: str = "", model: str = "") -> AIReportData:
    """构造一个「AI 未启用」的结果"""
    return AIReportData(
        enabled=False,
        status=STATUS_DISABLED,
        model=model,
        disabled_reason=reason,
    )
