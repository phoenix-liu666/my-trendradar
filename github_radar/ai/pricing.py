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

    DEEPSEEK_INPUT_PRICE_PER_1M=1.0
    DEEPSEEK_OUTPUT_PRICE_PER_1M=2.0
    DEEPSEEK_PRICE_CURRENCY=CNY
"""

import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from ..logging_utils import warn

ENV_INPUT_PRICE = "DEEPSEEK_INPUT_PRICE_PER_1M"
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
    """每百万 token 的价格"""

    input_price_per_1m: float
    output_price_per_1m: float
    currency: str = "CNY"

    @property
    def symbol(self) -> str:
        return CURRENCY_SYMBOLS.get(self.currency.upper(), "")

    def describe(self) -> str:
        return (
            f"{self.symbol}{self.input_price_per_1m}/1M input, "
            f"{self.symbol}{self.output_price_per_1m}/1M output ({self.currency})"
        )


# 默认价格表（CNY / 每百万 token）
#
# ⚠️ 这是**保守的默认估算值**，不是官方报价单。
#    真实价格以服务商账单为准；要精确统计请用上面的环境变量覆盖。
DEFAULT_PRICING = ModelPricing(
    input_price_per_1m=1.0, output_price_per_1m=2.0, currency="CNY"
)

PRICING_TABLE: Dict[str, ModelPricing] = {
    "deepseek-v4-flash": ModelPricing(
        input_price_per_1m=1.0, output_price_per_1m=2.0, currency="CNY"
    ),
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
    output_price = _float_env(env, ENV_OUTPUT_PRICE)
    currency = (env.get(ENV_CURRENCY) or "").strip().upper() or base.currency

    return ModelPricing(
        input_price_per_1m=base.input_price_per_1m if input_price is None else input_price,
        output_price_per_1m=base.output_price_per_1m if output_price is None else output_price,
        currency=currency,
    )


@dataclass
class CostEstimate:
    """一次运行的预估费用"""

    input_cost: float = 0.0
    output_cost: float = 0.0
    currency: str = "CNY"
    symbol: str = "¥"

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
            "output_cost": round(self.output_cost, 6),
            "total_cost": self.total_cost,
        }


def estimate_cost(
    prompt_tokens: int, completion_tokens: int, pricing: ModelPricing
) -> CostEstimate:
    """
    估算费用

    ::

        input_cost  = prompt_tokens     / 1_000_000 × input_price
        output_cost = completion_tokens / 1_000_000 × output_price

    Args:
        prompt_tokens: 输入 token 总数
        completion_tokens: 输出 token 总数
        pricing: 价格

    Returns:
        ``CostEstimate``（负数 / 非法输入按 0 处理）
    """
    def _tokens(value: int) -> float:
        try:
            return max(0.0, float(value or 0))
        except (TypeError, ValueError):
            return 0.0

    return CostEstimate(
        input_cost=_tokens(prompt_tokens) / 1_000_000.0 * pricing.input_price_per_1m,
        output_cost=_tokens(completion_tokens) / 1_000_000.0 * pricing.output_price_per_1m,
        currency=pricing.currency,
        symbol=pricing.symbol,
    )
