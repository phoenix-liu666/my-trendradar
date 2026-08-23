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


class FlashPricingTest(unittest.TestCase):
    """deepseek-v4-flash 的三档价格（人民币 / 每百万 token）"""

    def setUp(self):
        self.pricing = get_pricing("deepseek-v4-flash", {})

    def test_three_tier_prices(self):
        self.assertEqual(self.pricing.cache_hit_input_price_per_1m, 0.02)
        self.assertEqual(self.pricing.cache_miss_input_price_per_1m, 1.00)
        self.assertEqual(self.pricing.output_price_per_1m, 2.00)
        self.assertEqual(self.pricing.currency, "CNY")

    def test_cache_miss_price_is_the_input_price_alias(self):
        self.assertEqual(
            self.pricing.cache_miss_input_price_per_1m, self.pricing.input_price_per_1m
        )

    def test_effective_cache_hit_price(self):
        self.assertEqual(self.pricing.effective_cache_hit_price_per_1m, 0.02)

    def test_unknown_cache_price_falls_back_to_miss_price(self):
        """没有确认过缓存价的模型：缓存命中按更贵的 miss 价保守估算"""
        pricing = get_pricing("deepseek-v4-pro", {})
        self.assertIsNone(pricing.cache_hit_input_price_per_1m)
        self.assertEqual(
            pricing.effective_cache_hit_price_per_1m, pricing.input_price_per_1m
        )

    def test_cache_hit_price_can_be_overridden(self):
        pricing = get_pricing(
            "deepseek-v4-flash", {"DEEPSEEK_CACHE_HIT_INPUT_PRICE_PER_1M": "0.05"}
        )
        self.assertEqual(pricing.cache_hit_input_price_per_1m, 0.05)
        self.assertEqual(pricing.input_price_per_1m, 1.00)

    def test_describe_lists_all_three_tiers(self):
        text = self.pricing.describe()
        self.assertIn("0.02", text)
        self.assertIn("1.0", text)
        self.assertIn("2.0", text)


class MixedCacheCostTest(unittest.TestCase):
    """混合缓存费用"""

    def setUp(self):
        self.pricing = get_pricing("deepseek-v4-flash", {})

    def test_mixed_hit_and_miss(self):
        # 600,000 命中 ×0.02 + 400,000 未命中 ×1.00 + 100,000 输出 ×2.00
        cost = estimate_cost(
            1_000_000, 100_000, self.pricing,
            cache_hit_tokens=600_000, cache_miss_tokens=400_000,
        )
        self.assertAlmostEqual(cost.cache_hit_cost, 0.012)
        self.assertAlmostEqual(cost.cache_miss_cost, 0.400)
        self.assertAlmostEqual(cost.input_cost, 0.412)
        self.assertAlmostEqual(cost.output_cost, 0.200)
        self.assertAlmostEqual(cost.total_cost, 0.612)

    def test_all_cached_is_much_cheaper_than_all_missed(self):
        cached = estimate_cost(1_000_000, 0, self.pricing, cache_hit_tokens=1_000_000)
        missed = estimate_cost(1_000_000, 0, self.pricing, cache_miss_tokens=1_000_000)
        self.assertAlmostEqual(cached.total_cost, 0.02)
        self.assertAlmostEqual(missed.total_cost, 1.00)
        self.assertLess(cached.total_cost, missed.total_cost)

    def test_realistic_daily_usage(self):
        # 82,431 输入（其中 60,000 命中）+ 11,283 输出
        cost = estimate_cost(
            82_431, 11_283, self.pricing,
            cache_hit_tokens=60_000, cache_miss_tokens=22_431,
        )
        expected = 60_000 / 1e6 * 0.02 + 22_431 / 1e6 * 1.00 + 11_283 / 1e6 * 2.00
        self.assertAlmostEqual(cost.total_cost, round(expected, 6))

    def test_breakdown_is_exposed(self):
        cost = estimate_cost(
            1000, 100, self.pricing, cache_hit_tokens=700, cache_miss_tokens=300
        )
        payload = cost.to_dict()
        self.assertIn("cache_hit_cost", payload)
        self.assertIn("cache_miss_cost", payload)
        self.assertAlmostEqual(
            payload["cache_hit_cost"] + payload["cache_miss_cost"], payload["input_cost"]
        )
        self.assertEqual(cost.cache_hit_tokens, 700)
        self.assertEqual(cost.cache_miss_tokens, 300)


class MissingCacheFieldFallbackTest(unittest.TestCase):
    """无缓存字段 fallback：全部按 cache miss 估算，绝不报错"""

    def setUp(self):
        self.pricing = get_pricing("deepseek-v4-flash", {})

    def test_no_detail_prices_everything_as_miss(self):
        cost = estimate_cost(1_000_000, 0, self.pricing)
        self.assertEqual(cost.cache_hit_tokens, 0)
        self.assertEqual(cost.cache_miss_tokens, 1_000_000)
        self.assertAlmostEqual(cost.total_cost, 1.00)

    def test_partial_detail_puts_the_remainder_in_miss(self):
        cost = estimate_cost(1_000_000, 0, self.pricing, cache_hit_tokens=200_000)
        self.assertEqual(cost.cache_miss_tokens, 800_000)
        self.assertAlmostEqual(cost.total_cost, 200_000 / 1e6 * 0.02 + 800_000 / 1e6 * 1.00)

    def test_fallback_never_underestimates(self):
        with_detail = estimate_cost(
            1_000_000, 0, self.pricing, cache_hit_tokens=900_000, cache_miss_tokens=100_000
        )
        without_detail = estimate_cost(1_000_000, 0, self.pricing)
        self.assertGreater(without_detail.total_cost, with_detail.total_cost)

    def test_garbage_detail_is_safe(self):
        cost = estimate_cost(1000, 100, self.pricing, cache_hit_tokens=None, cache_miss_tokens="x")
        self.assertEqual(cost.cache_miss_tokens, 1000)
        self.assertGreater(cost.total_cost, 0)

    def test_detail_larger_than_prompt_is_kept(self):
        """明细比总数还大（服务端口径不一致）时按明细算，不做缩水"""
        cost = estimate_cost(100, 0, self.pricing, cache_hit_tokens=80, cache_miss_tokens=80)
        self.assertEqual(cost.cache_hit_tokens, 80)
        self.assertEqual(cost.cache_miss_tokens, 80)


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
