# coding=utf-8
"""
AI 配置（全部来自环境变量）

设计要求（见第二阶段规格 §6 / §7 / §24 / §28）：
- GitHub Radar 的 AI 开关**独立**于 TrendRadar 的 ``AI_ANALYSIS_ENABLED``，
  两套系统互不影响
- 只有 ``GITHUB_RADAR_AI_ENABLED`` 明确开启时才会调用 DeepSeek
- 开了开关但没有 ``DEEPSEEK_API_KEY`` → 只 warning，本次禁用 AI，
  **基础日报照常发送**，绝不让 workflow 失败
- 所有成本相关参数都有硬上限，非法值一律回退到默认值

安全约定：本模块持有 API Key，但 ``describe()`` / 任何日志都只输出
"是否存在"，绝不输出 Key 本身。
"""

import os
from dataclasses import dataclass
from typing import Mapping, Optional

from ..logging_utils import warn

# ---- 环境变量名 --------------------------------------------------------
ENV_AI_ENABLED = "GITHUB_RADAR_AI_ENABLED"
ENV_API_KEY = "DEEPSEEK_API_KEY"
ENV_MODEL = "DEEPSEEK_MODEL"
ENV_API_BASE = "DEEPSEEK_API_BASE"
ENV_REPO_LIMIT = "GITHUB_RADAR_AI_REPO_LIMIT"
ENV_MAX_INPUT_CHARS = "GITHUB_RADAR_AI_MAX_INPUT_CHARS"
ENV_BATCH_SIZE = "GITHUB_RADAR_AI_BATCH_SIZE"
ENV_TIMEOUT = "GITHUB_RADAR_AI_TIMEOUT"

# ---- 默认值 ------------------------------------------------------------
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_API_BASE = "https://api.deepseek.com"

# 每天最多送进 DeepSeek 的 unique repositories（硬限制）
DEFAULT_REPO_LIMIT = 30
MAX_REPO_LIMIT = 100

# 单次运行允许拼进 prompt 的总字符数上限
# （防止某天异常 README / 候选数量导致 Token 暴涨）
DEFAULT_MAX_INPUT_CHARS = 120_000
MIN_MAX_INPUT_CHARS = 10_000
MAX_MAX_INPUT_CHARS = 400_000

# 每个 batch 的仓库数量（规格要求 5~8）
DEFAULT_BATCH_SIZE = 6
MAX_BATCH_SIZE = 8

DEFAULT_TIMEOUT = 60
MIN_TIMEOUT = 10
MAX_TIMEOUT = 180

# 单个 README 最大字符数（超出截断）
README_MAX_CHARS = 6000

# JSON 解析失败时最多额外重试次数（规格 §15：最多重试 1 次）
JSON_RETRY_LIMIT = 1

_TRUTHY = {"true", "1", "yes", "on"}


def _env(env: Mapping[str, str], name: str) -> str:
    """读取环境变量并去掉首尾空白（缺失返回空串）"""
    try:
        return (env.get(name) or "").strip()
    except Exception:  # pragma: no cover - Mapping 实现异常时的最后防线
        return ""


def _int_env(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """
    读取整数型环境变量

    非法值 → warning + 默认值；超出范围 → 夹紧到 [minimum, maximum]。
    绝不因为一个配置错误让整个 Radar 失败。
    """
    raw = _env(env, name)
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        warn(f"[AI] {name}='{raw}' 不是合法整数，使用默认值 {default}")
        return default
    if value < minimum:
        warn(f"[AI] {name}={value} 小于下限 {minimum}，按 {minimum} 处理")
        return minimum
    if value > maximum:
        warn(f"[AI] {name}={value} 超过上限 {maximum}，按 {maximum} 处理")
        return maximum
    return value


@dataclass
class AIConfig:
    """一次运行的 AI 配置"""

    enabled: bool = False
    api_key: Optional[str] = None
    model: str = DEFAULT_MODEL
    api_base: str = DEFAULT_API_BASE
    repo_limit: int = DEFAULT_REPO_LIMIT
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS
    batch_size: int = DEFAULT_BATCH_SIZE
    timeout: int = DEFAULT_TIMEOUT
    readme_max_chars: int = README_MAX_CHARS
    # 关闭原因（仅用于日志与报告说明，绝不含敏感信息）
    disabled_reason: str = ""

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    def describe(self) -> str:
        """一行可安全打印的描述（不含 API Key）"""
        if not self.enabled:
            return f"[AI] enabled: false{(' (' + self.disabled_reason + ')') if self.disabled_reason else ''}"
        return (
            f"[AI] enabled: true | model: {self.model} | repo limit: {self.repo_limit} "
            f"| batch size: {self.batch_size} | max input chars: {self.max_input_chars}"
        )


def load_ai_config(
    env: Optional[Mapping[str, str]] = None, *, skip_ai: bool = False
) -> AIConfig:
    """
    从环境变量构造 AI 配置

    Args:
        env: 环境变量映射（默认 ``os.environ``）
        skip_ai: 命令行 ``--skip-ai``（优先级最高，直接关闭 AI）

    Returns:
        ``AIConfig``；任何异常情况都返回一个 ``enabled=False`` 的配置，
        调用方据此走「基础日报」路径。
    """
    env = env if env is not None else os.environ

    model = _env(env, ENV_MODEL) or DEFAULT_MODEL
    api_base = _env(env, ENV_API_BASE) or DEFAULT_API_BASE

    if skip_ai:
        return AIConfig(enabled=False, model=model, disabled_reason="skip_ai")

    flag = _env(env, ENV_AI_ENABLED).lower()
    if flag not in _TRUTHY:
        return AIConfig(
            enabled=False,
            model=model,
            disabled_reason=f"{ENV_AI_ENABLED} 未开启",
        )

    api_key = _env(env, ENV_API_KEY) or None
    if not api_key:
        # 规格 §6：开了开关但没有 Key → warning + 本次禁用 AI + 继续基础日报
        warn(
            f"[AI] {ENV_AI_ENABLED}=true 但未配置 {ENV_API_KEY}，"
            f"本次运行禁用 AI，基础日报照常发送"
        )
        return AIConfig(
            enabled=False,
            model=model,
            disabled_reason=f"缺少 {ENV_API_KEY}",
        )

    return AIConfig(
        enabled=True,
        api_key=api_key,
        model=model,
        api_base=api_base,
        repo_limit=_int_env(
            env, ENV_REPO_LIMIT, DEFAULT_REPO_LIMIT, minimum=0, maximum=MAX_REPO_LIMIT
        ),
        max_input_chars=_int_env(
            env,
            ENV_MAX_INPUT_CHARS,
            DEFAULT_MAX_INPUT_CHARS,
            minimum=MIN_MAX_INPUT_CHARS,
            maximum=MAX_MAX_INPUT_CHARS,
        ),
        batch_size=_int_env(
            env, ENV_BATCH_SIZE, DEFAULT_BATCH_SIZE, minimum=1, maximum=MAX_BATCH_SIZE
        ),
        timeout=_int_env(
            env, ENV_TIMEOUT, DEFAULT_TIMEOUT, minimum=MIN_TIMEOUT, maximum=MAX_TIMEOUT
        ),
    )
