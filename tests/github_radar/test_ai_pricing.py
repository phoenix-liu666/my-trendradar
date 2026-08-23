# coding=utf-8
"""费用估算测试"""

import unittest

from github_radar.ai.pricing import (
    DEFAULT_PRICING,
    PRICING_TABLE,
    ModelPricing,
    estimate_cost,
    get_pricing,
)


class PricingTableTest(unittest.TestCase):
    def test_default_model_is_in_the_table(self):
        self.assertIn("deepseek-v4-flash", PRICING_TABLE)

    def test_unknown_model_falls_back_to_default(self):
        pricing = get_pricing("some-unknown-model", {})
        self.assertEqual(pricing.input_price_per_1m, DEFAULT_PRICING.input_price_per_1m)

    def test_model_name_is_case_insensitive(self):
        self.assertEqual(
            get_pricing("DeepSeek-V4-Flash", {}).input_price_per_1m,
            PRICING_TABLE["deepseek-v4-flash"].input_price_per_1m,
        )

    def test_env_overrides_the_table(self):
        pricing = get_pricing(
            "deepseek-v4-flash",
            {"DEEPSEEK_INPUT_PRICE_PER_1M": "0.5", "DEEPSEEK_OUTPUT_PRICE_PER_1M": "1.5"},
        )
        self.assertEqual(pricing.input_price_per_1m, 0.5)
        self.assertEqual(pricing.output_price_per_1m, 1.5)

    def test_invalid_env_is_ignored(self):
        pricing = get_pricing("deepseek-v4-flash", {"DEEPSEEK_INPUT_PRICE_PER_1M": "abc"})
        self.assertEqual(
            pricing.input_price_per_1m, PRICING_TABLE["deepseek-v4-flash"].input_price_per_1m
        )

    def test_negative_env_is_ignored(self):
        pricing = get_pricing("deepseek-v4-flash", {"DEEPSEEK_OUTPUT_PRICE_PER_1M": "-3"})
        self.assertEqual(
            pricing.output_price_per_1m, PRICING_TABLE["deepseek-v4-flash"].output_price_per_1m
        )

    def test_currency_override(self):
        pricing = get_pricing("deepseek-v4-flash", {"DEEPSEEK_PRICE_CURRENCY": "usd"})
        self.assertEqual(pricing.currency, "USD")
        self.assertEqual(pricing.symbol, "$")

    def test_prices_are_not_hardcoded_in_business_code(self):
        """价格只在 pricing.py 里维护"""
        import pathlib

        ai_dir = pathlib.Path(__file__).resolve().parents[2] / "github_radar" / "ai"
        for path in ai_dir.glob("*.py"):
            if path.name == "pricing.py":
                continue
            with self.subTest(module=path.name):
                self.assertNotIn("per_1m=", path.read_text(encoding="utf-8"))


class EstimateCostTest(unittest.TestCase):
    """34. estimated cost"""

    def setUp(self):
        self.pricing = ModelPricing(input_price_per_1m=1.0, output_price_per_1m=2.0)

    def test_formula(self):
        cost = estimate_cost(1_000_000, 500_000, self.pricing)
        self.assertAlmostEqual(cost.input_cost, 1.0)
        self.assertAlmostEqual(cost.output_cost, 1.0)
        self.assertAlmostEqual(cost.total_cost, 2.0)

    def test_realistic_daily_usage(self):
        # 82,431 输入 + 11,283 输出 ≈ ¥0.105
        cost = estimate_cost(82_431, 11_283, self.pricing)
        self.assertAlmostEqual(cost.total_cost, 0.104997, places=5)

    def test_zero_tokens(self):
        cost = estimate_cost(0, 0, self.pricing)
        self.assertEqual(cost.total_cost, 0.0)

    def test_negative_and_garbage_tokens_are_safe(self):
        self.assertEqual(estimate_cost(-100, -100, self.pricing).total_cost, 0.0)
        self.assertEqual(estimate_cost(None, None, self.pricing).total_cost, 0.0)

    def test_display_always_says_estimated(self):
        text = estimate_cost(100_000, 10_000, self.pricing).format_total()
        self.assertIn("约", text)
        self.assertIn("预估", text)
        self.assertIn("¥", text)

    def test_to_dict(self):
        payload = estimate_cost(1_000_000, 0, self.pricing).to_dict()
        self.assertEqual(payload["input_cost"], 1.0)
        self.assertEqual(payload["output_cost"], 0.0)
        self.assertEqual(payload["total_cost"], 1.0)


if __name__ == "__main__":
    unittest.main()
