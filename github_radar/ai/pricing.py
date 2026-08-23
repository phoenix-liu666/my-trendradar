# coding=utf-8
"""
费用估算（规格 §26）

价格**只在这里维护**，绝不散落到业务代码里。

重要声明
--------
这里算出来的永远是 **estimated cost（预估费用）**，不是真实账单：
- 服务商可能有缓存命中价、阶梯价、折扣时段、赠送额度
- 汇率、税费、账户类型都会影响最终金额
- 模型价格随时可能调整

因此报告里必须写「约 / 预估」，绝不能声称是实际花费。
默认价格可以随时用环境变量覆盖，不需要改代码::

    DEEPSEEK_CACHE_HIT_INPUT_PRICE_PER_1M=0.02   # 命中前缀缓存的输入
    DEEPSEEK_INPUT_PRICE_PER_1M=1.00             # 未命中的输入
    DEEPSEEK_OUTPUT_PRICE_PER_1M=2.00            # 输出
    DEEPSEEK_PRICE_CURRENCY=CNY
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from ..logging_utils import warn

ENV_INPUT_PRICE = "DEEPSEEK_INPUT_PRICE_PER_1M"
ENV_CACHE_HIT_PRICE = "DEEPSEEK_CACHE_HIT_INPUT_PRICE_PER_1M"
ENV_OUTPUT_PRICE = "DEEPSEEK_OUTPUT_PRICE_PER_1M"
ENV_CURRENCY = "DEEPSEEK_PRICE_CURRENCY"

CURRENCY_SYMBOLS: Dict[str, str] = {
    "CNY": "¥",
    "RMB": "¥",
    "USD": "$",
    "EUR": "€",
}


@dataclass(frozen=True)
class ModelPricing:
    """
    每百万 token 的价格

    输入分两档（DeepSeek 的前缀缓存机制）::

        cache hit   命中前缀缓存的输入，极其便宜
        cache miss  未命中的输入，正常价格

    ``cache_hit_input_price_per_1m=None`` 表示「不知道该模型的缓存价」，
    此时缓存命中的 token 一律按 **cache miss 价**估算 —— 宁可高估，不要低估。
    """

    input_price_per_1m: float
    output_price_per_1m: float
    currency: str = "CNY"
    cache_hit_input_price_per_1m: Optional[float] = None

    @property
    def symbol(self) -> str:
        return CURRENCY_SYMBOLS.get(self.currency.upper(), "")

    @property
    def cache_miss_input_price_per_1m(self) -> float:
        """未命中缓存的输入价（即 ``input_price_per_1m`` 的语义别名）"""
        return self.input_price_per_1m

    @property
    def effective_cache_hit_price_per_1m(self) -> float:
        """实际用于估算的缓存命中价（未配置时退回 cache miss 价）"""
        if self.cache_hit_input_price_per_1m is None:
            return self.input_price_per_1m
        return self.cache_hit_input_price_per_1m

    def describe(self) -> str:
        return (
            f"{self.symbol}{self.effective_cache_hit_price_per_1m}/1M cached input, "
            f"{self.symbol}{self.input_price_per_1m}/1M input, "
            f"{self.symbol}{self.output_price_per_1m}/1M output ({self.currency})"
        )


# 默认价格表（CNY / 每百万 token）
#
# ⚠️ deepseek-v4-flash 一档来自 DeepSeek 当前公布的官方定价；
#    其余型号只是保守占位。无论哪一档，算出来的都只是**本地估算**，
#    不代表最终账单，且 DeepSeek 随时可能调整价格 ——
#    需要精确统计时用上面的环境变量覆盖即可，不必改代码。
DEFAULT_PRICING = ModelPricing(
    input_price_per_1m=1.0,
    output_price_per_1m=2.0,
    currency="CNY",
    cache_hit_input_price_per_1m=0.02,
)

PRICING_TABLE: Dict[str, ModelPricing] = {
    # DeepSeek V4 Flash 当前官方定价（人民币 / 每百万 token）
    "deepseek-v4-flash": ModelPricing(
        input_price_per_1m=1.00,        # cache miss 输入
        output_price_per_1m=2.00,       # 输出
        currency="CNY",
        cache_hit_input_price_per_1m=0.02,   # cache hit 输入
    ),
    # 以下两档没有确认过的缓存价，缓存命中按 cache miss 价保守估算
    "deepseek-v4-pro": ModelPricing(
        input_price_per_1m=4.0, output_price_per_1m=12.0, currency="CNY"
    ),
    "deepseek-chat": ModelPricing(
        input_price_per_1m=2.0, output_price_per_1m=8.0, currency="CNY"
    ),
}


def _float_env(env: Mapping[str, str], name: str) -> Optional[float]:
    """读取浮点型环境变量（非法值只 warning）"""
    raw = (env.get(name) or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        warn(f"[AI] {name}='{raw}' 不是合法数字，忽略")
        return None
    if value < 0:
        warn(f"[AI] {name}={value} 为负数，忽略")
        return None
    return value


def get_pricing(model: str, env: Optional[Mapping[str, str]] = None) -> ModelPricing:
    """
    获取某个模型的价格

    顺序：环境变量覆盖 > 价格表 > 默认值。

    Args:
        model: 模型名（如 deepseek-v4-flash）
        env: 环境变量（默认 ``os.environ``）
    """
    env = env if env is not None else os.environ

    base = PRICING_TABLE.get((model or "").strip().lower(), DEFAULT_PRICING)

    input_price = _float_env(env, ENV_INPUT_PRICE)
    cache_hit_price = _float_env(env, ENV_CACHE_HIT_PRICE)
    output_price = _float_env(env, ENV_OUTPUT_PRICE)
    currency = (env.get(ENV_CURRENCY) or "").strip().upper() or base.currency

    return ModelPricing(
        input_price_per_1m=base.input_price_per_1m if input_price is None else input_price,
        output_price_per_1m=base.output_price_per_1m if output_price is None else output_price,
        currency=currency,
        cache_hit_input_price_per_1m=(
            base.cache_hit_input_price_per_1m if cache_hit_price is None else cache_hit_price
        ),
    )


@dataclass
class CostEstimate:
    """一次运行的预估费用"""

    input_cost: float = 0.0
    output_cost: float = 0.0
    currency: str = "CNY"
    symbol: str = "¥"
    # 输入费用的两档明细（``input_cost`` = 两者之和）
    cache_hit_cost: float = 0.0
    cache_miss_cost: float = 0.0
    # 实际参与计价的 token（缺字段降级后的口径）
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0

    @property
    def total_cost(self) -> float:
        return round(self.input_cost + self.output_cost, 6)

    def format_total(self) -> str:
        """
        展示用文本，**永远带「约」字**

        Examples:
            >>> CostEstimate(0.05, 0.05).format_total()
            '约 ¥0.1000（预估）'
        """
        return f"约 {self.symbol}{self.total_cost:.4f}（预估）"

    def to_dict(self) -> Dict[str, float]:
        return {
            "input_cost": round(self.input_cost, 6),
            "cache_hit_cost": round(self.cache_hit_cost, 6),
            "cache_miss_cost": round(self.cache_miss_cost, 6),
            "output_cost": round(self.output_cost, 6),
            "total_cost": self.total_cost,
        }


def _tokens(value: Any) -> int:
    """把 token 数转成 >= 0 的整数（非法输入按 0 处理）"""
    if value is None or isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    pricing: ModelPricing,
    *,
    cache_hit_tokens: int = 0,
    cache_miss_tokens: int = 0,
) -> CostEstimate:
    """
    估算费用（输入分缓存命中 / 未命中两档）

    ::

        input_cost  = cache_hit_tokens   / 1_000_000 × cache_hit_price
                    + cache_miss_tokens  / 1_000_000 × cache_miss_price
        output_cost = completion_tokens  / 1_000_000 × output_price

    安全降级（规格要求）：服务端没返回缓存明细时，
    **全部 prompt_tokens 按 cache miss（更贵的一档）估算**，绝不报错；
    明细之和小于 prompt_tokens 时，差额同样计入 cache miss。

    Args:
        prompt_tokens: 输入 token 总数
        completion_tokens: 输出 token 总数
        pricing: 价格
        cache_hit_tokens: 命中前缀缓存的输入 token（可选）
        cache_miss_tokens: 未命中的输入 token（可选）

    Returns:
        ``CostEstimate``（负数 / 非法输入按 0 处理）
    """
    prompt = _tokens(prompt_tokens)
    hit = _tokens(cache_hit_tokens)
    miss = _tokens(cache_miss_tokens)

    if hit + miss < prompt:
        # 含「完全没有缓存字段」的情况：差额一律按未命中计价
        miss += prompt - hit - miss

    cache_hit_cost = hit / 1_000_000.0 * pricing.effective_cache_hit_price_per_1m
    cache_miss_cost = miss / 1_000_000.0 * pricing.cache_miss_input_price_per_1m

    return CostEstimate(
        input_cost=cache_hit_cost + cache_miss_cost,
        output_cost=_tokens(completion_tokens) / 1_000_000.0 * pricing.output_price_per_1m,
        currency=pricing.currency,
        symbol=pricing.symbol,
        cache_hit_cost=cache_hit_cost,
        cache_miss_cost=cache_miss_cost,
        cache_hit_tokens=hit,
        cache_miss_tokens=miss,
    )
