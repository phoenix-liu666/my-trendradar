# coding=utf-8
"""
GitHub Daily Radar - AI Intelligence（DeepSeek V4 Flash）

把「GitHub 排行榜」升级成「GitHub 技术情报系统」：中文解读、项目分类、
技术栈、应用场景、基于证据的「为什么值得关注」、AI relevance、
Personal Score、🎯 For You Top10、📡 每日技术信号、Token 与费用统计。

铁律（整个子包的设计前提）
--------------------------
1. **AI 只是 enhancement**：不参与 Heat Score、不碰 Star 数据、
   不影响 24h/7d 增量、不参与 snapshot
2. **AI 绝不能成为单点故障**：DeepSeek 挂掉时基础日报必须照常发送
3. **模型输出永远不可信**：JSON parse → schema validation → sanitize → fallback
4. **README 永远不可信**：明确的 prompt injection 防御 + 沙箱分隔
5. **成本必须有硬上限**：每天最多 30 个仓库、README ≤6000 字符、
   batch ≤8、JSON 重试 ≤1、thinking 明确关闭
6. **Secret 永不进日志**

模块划分
--------
- ``config``     环境变量 → AIConfig（独立开关 GITHUB_RADAR_AI_ENABLED）
- ``profile``    兴趣画像加载 + deterministic keyword score
- ``schemas``    AI 输出的 schema / 校验 / 消毒 / 幻觉控制
- ``prompts``    prompt 构造 + prompt injection 防御
- ``client``     DeepSeek 客户端（thinking disabled / 重试 / usage）
- ``cache``      静态字段缓存 + 当日结果复用
- ``selector``   AI 候选选择（优先级 + 硬上限）
- ``readme``     README 获取与截断
- ``analyzer``   批量仓库分析编排
- ``scoring``    Personal Score + For You Top10
- ``synthesis``  每日趋势总结
- ``pricing``    价格表与费用估算
- ``result``     报告层使用的结果对象
- ``pipeline``   总入口 ``run_ai_enhancement``
"""

from .config import AIConfig, load_ai_config
from .result import (
    STATUS_DISABLED,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_PARTIAL,
    AIReportData,
    disabled_result,
)

__all__ = [
    "AIConfig",
    "AIReportData",
    "STATUS_DISABLED",
    "STATUS_FAILED",
    "STATUS_OK",
    "STATUS_PARTIAL",
    "disabled_result",
    "load_ai_config",
    "run_ai_enhancement",
]


def run_ai_enhancement(*args, **kwargs):
    """
    执行 AI 增强（惰性导入 ``pipeline``）

    惰性导入的原因：``pipeline`` 会拉起 DeepSeek 客户端（依赖 requests），
    而基础日报链路只需要 ``AIReportData`` / ``load_ai_config``。
    AI 没启用时一行多余的 import 都不做。
    """
    from .pipeline import run_ai_enhancement as _run

    return _run(*args, **kwargs)
