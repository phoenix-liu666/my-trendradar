# coding=utf-8
"""AI 输出 schema 校验、消毒与幻觉控制测试"""

import unittest

from github_radar.ai.schemas import (
    ACTIONS,
    MAX_TECH_STACK,
    MAX_USE_CASES,
    NO_EXTERNAL_EVIDENCE_TEXT,
    AIUsage,
    apply_hallucination_guard,
    clamp_score,
    clean_list,
    clean_text,
    filter_evidence,
    looks_like_external_claim,
    parse_daily_synthesis,
    parse_repo_analysis,
)

from .ai_helpers import analysis_payload

ALLOWED = ["stars", "delta_24h", "delta_7d", "trending_rank", "pushed_at", "heat_score"]


class SchemaParsingTest(unittest.TestCase):
    """9. JSON schema parsing"""

    def test_parses_a_complete_payload(self):
        analysis = parse_repo_analysis(analysis_payload("owner/repo"), full_name="owner/repo")
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.full_name, "owner/repo")
        self.assertEqual(analysis.category, "Developer Tool")
        self.assertEqual(analysis.maturity, "growing")
        self.assertEqual(analysis.recommended_action, "watch")
        self.assertEqual(analysis.why_hot.confidence, "medium")

    def test_full_name_comes_from_our_input_not_the_model(self):
        """模型张冠李戴时以我们的输入为准"""
        payload = analysis_payload("owner/repo")
        payload["full_name"] = "evil/other"
        analysis = parse_repo_analysis(payload, full_name="owner/repo")
        self.assertEqual(analysis.full_name, "owner/repo")

    def test_missing_fields_get_safe_defaults(self):
        analysis = parse_repo_analysis({"summary_zh": "只有一句话"}, full_name="a/b")
        self.assertEqual(analysis.problem, "")
        self.assertEqual(analysis.category, "Other")
        self.assertEqual(analysis.maturity, "unknown")
        self.assertEqual(analysis.recommended_action, "watch")
        self.assertEqual(analysis.why_hot.confidence, "low")
        self.assertEqual(analysis.relevance_score, 0)

    def test_non_dict_payload_returns_none(self):
        for payload in ("string", 42, None, ["list"]):
            with self.subTest(payload=payload):
                self.assertIsNone(parse_repo_analysis(payload, full_name="a/b"))

    def test_missing_full_name_returns_none(self):
        self.assertIsNone(parse_repo_analysis({"summary_zh": "x"}, full_name=""))

    def test_why_hot_as_plain_string_is_tolerated(self):
        analysis = parse_repo_analysis(
            {"summary_zh": "x", "why_hot": "只是一个字符串"}, full_name="a/b"
        )
        self.assertEqual(analysis.why_hot.summary, "只是一个字符串")
        self.assertEqual(analysis.why_hot.confidence, "low")

    def test_enum_values_are_normalised(self):
        analysis = parse_repo_analysis(
            analysis_payload("a/b", maturity="  MATURE  ", recommended_action="STUDY"),
            full_name="a/b",
        )
        self.assertEqual(analysis.maturity, "mature")
        self.assertEqual(analysis.recommended_action, "study")

    def test_invalid_enum_falls_back_to_default(self):
        analysis = parse_repo_analysis(
            analysis_payload("a/b", maturity="超级成熟", recommended_action="explode"),
            full_name="a/b",
        )
        self.assertEqual(analysis.maturity, "unknown")
        self.assertEqual(analysis.recommended_action, "watch")
        self.assertIn(analysis.recommended_action, ACTIONS)

    def test_control_characters_are_stripped(self):
        analysis = parse_repo_analysis(
            {"summary_zh": "第一行\n第二行\t制表符"}, full_name="a/b"
        )
        self.assertNotIn("\n", analysis.summary_zh)
        self.assertNotIn("\t", analysis.summary_zh)

    def test_nested_object_in_string_field_is_dropped(self):
        analysis = parse_repo_analysis(
            {"summary_zh": {"nested": "object"}, "problem": "ok"}, full_name="a/b"
        )
        self.assertEqual(analysis.summary_zh, "")
        self.assertEqual(analysis.problem, "ok")


class ClampTest(unittest.TestCase):
    """12. relevance score clamp"""

    def test_in_range_values_pass_through(self):
        for value in (0, 50, 100):
            self.assertEqual(clamp_score(value), value)

    def test_over_range_is_clamped_to_100(self):
        self.assertEqual(clamp_score(1000), 100)
        self.assertEqual(clamp_score(101), 100)

    def test_below_range_is_clamped_to_zero(self):
        self.assertEqual(clamp_score(-50), 0)

    def test_strings_are_parsed(self):
        self.assertEqual(clamp_score("87"), 87)
        self.assertEqual(clamp_score("87 分"), 87)

    def test_garbage_becomes_zero(self):
        for value in (None, True, "", "abc", [], {}):
            with self.subTest(value=value):
                self.assertEqual(clamp_score(value), 0)

    def test_float_is_rounded(self):
        self.assertEqual(clamp_score(87.6), 88)

    def test_clamped_through_parse(self):
        analysis = parse_repo_analysis(
            analysis_payload("a/b", relevance_score=9999), full_name="a/b"
        )
        self.assertEqual(analysis.relevance_score, 100)


class ListLimitTest(unittest.TestCase):
    """13. tech_stack 长度限制 / 14. use_cases 长度限制"""

    def test_tech_stack_is_capped_at_eight(self):
        payload = analysis_payload("a/b", tech_stack=[f"tech{i}" for i in range(30)])
        analysis = parse_repo_analysis(payload, full_name="a/b")
        self.assertEqual(len(analysis.tech_stack), MAX_TECH_STACK)
        self.assertEqual(analysis.tech_stack[0], "tech0")

    def test_use_cases_are_capped_at_five(self):
        payload = analysis_payload("a/b", use_cases=[f"case{i}" for i in range(20)])
        analysis = parse_repo_analysis(payload, full_name="a/b")
        self.assertEqual(len(analysis.use_cases), MAX_USE_CASES)

    def test_duplicates_are_removed(self):
        payload = analysis_payload("a/b", tech_stack=["Python", "python", "PYTHON", "Rust"])
        analysis = parse_repo_analysis(payload, full_name="a/b")
        self.assertEqual(analysis.tech_stack, ["Python", "Rust"])

    def test_string_instead_of_list_is_tolerated(self):
        analysis = parse_repo_analysis(
            analysis_payload("a/b", tech_stack="Python"), full_name="a/b"
        )
        self.assertEqual(analysis.tech_stack, ["Python"])

    def test_non_list_non_string_becomes_empty(self):
        analysis = parse_repo_analysis(
            analysis_payload("a/b", use_cases={"a": 1}), full_name="a/b"
        )
        self.assertEqual(analysis.use_cases, [])

    def test_long_items_are_truncated(self):
        analysis = parse_repo_analysis(
            analysis_payload("a/b", tech_stack=["x" * 500]), full_name="a/b"
        )
        self.assertLessEqual(len(analysis.tech_stack[0]), 60)

    def test_clean_list_helpers(self):
        self.assertEqual(clean_list(None, max_items=3, max_length=10), [])
        self.assertEqual(clean_list(["a", "", None, "b"], max_items=3, max_length=10), ["a", "b"])

    def test_clean_text_truncates(self):
        self.assertLessEqual(len(clean_text("x" * 300, 100)), 100)


class HallucinationGuardTest(unittest.TestCase):
    """28. hallucination guard"""

    def analyse(self, **overrides):
        payload = analysis_payload("a/b", **overrides)
        analysis = parse_repo_analysis(payload, full_name="a/b")
        return apply_hallucination_guard(analysis, ALLOWED)

    def test_funding_claim_is_replaced(self):
        analysis = self.analyse(
            why_hot={
                "summary": "该项目刚刚完成 A 轮融资，因此 Star 暴涨。",
                "confidence": "high",
                "evidence": ["delta_24h"],
            }
        )
        self.assertEqual(analysis.why_hot.summary, NO_EXTERNAL_EVIDENCE_TEXT)
        self.assertEqual(analysis.why_hot.confidence, "low")

    def test_big_company_adoption_claim_is_replaced(self):
        analysis = self.analyse(
            why_hot={
                "summary": "谷歌宣布采用该项目，社区随之爆发。",
                "confidence": "high",
                "evidence": ["stars"],
            }
        )
        self.assertEqual(analysis.why_hot.summary, NO_EXTERNAL_EVIDENCE_TEXT)

    def test_media_claim_is_replaced(self):
        analysis = self.analyse(
            why_hot={
                "summary": "登上 Hacker News 首页并被大量媒体报道。",
                "confidence": "medium",
                "evidence": ["trending_rank"],
            }
        )
        self.assertEqual(analysis.why_hot.summary, NO_EXTERNAL_EVIDENCE_TEXT)

    def test_celebrity_claim_is_replaced(self):
        analysis = self.analyse(
            why_hot={"summary": "马斯克转发推荐了这个项目。", "confidence": "high", "evidence": []}
        )
        self.assertEqual(analysis.why_hot.summary, NO_EXTERNAL_EVIDENCE_TEXT)

    def test_evidence_based_summary_is_kept(self):
        text = "24h Star 增长明显，Trending 排名靠前，近期仍在更新。"
        analysis = self.analyse(
            why_hot={
                "summary": text,
                "confidence": "medium",
                "evidence": ["delta_24h=1200", "trending_rank=3"],
            }
        )
        self.assertEqual(analysis.why_hot.summary, text)
        self.assertEqual(analysis.why_hot.confidence, "medium")

    def test_unsupported_evidence_items_are_dropped(self):
        analysis = self.analyse(
            why_hot={
                "summary": "近期关注度上升。",
                "confidence": "high",
                "evidence": ["delta_24h=100", "某位大V的推荐", "公司内部消息"],
            }
        )
        self.assertEqual(analysis.why_hot.evidence, ["delta_24h=100"])

    def test_confidence_downgrades_when_no_evidence_survives(self):
        analysis = self.analyse(
            why_hot={
                "summary": "近期关注度上升。",
                "confidence": "high",
                "evidence": ["朋友圈都在传"],
            }
        )
        self.assertEqual(analysis.why_hot.evidence, [])
        self.assertEqual(analysis.why_hot.confidence, "low")

    def test_reasons_with_external_claims_are_cleared(self):
        analysis = self.analyse(
            relevance_reason="因为被字节跳动采用所以值得关注",
            recommendation_reason="上了 Product Hunt 首页",
        )
        self.assertEqual(analysis.relevance_reason, "")
        self.assertEqual(analysis.recommendation_reason, "")

    def test_filter_evidence_accepts_chinese_aliases(self):
        kept = filter_evidence(["24h 增长 1200", "无关内容"], ALLOWED)
        self.assertEqual(kept, ["24h 增长 1200"])

    def test_filter_evidence_with_no_allowed_keys(self):
        self.assertEqual(filter_evidence(["stars=1"], []), [])

    def test_detector_recognises_claims(self):
        self.assertTrue(looks_like_external_claim("完成了 B 轮融资"))
        self.assertTrue(looks_like_external_claim("被微软采用"))
        self.assertFalse(looks_like_external_claim("24h Star 增长明显"))


class SynthesisSchemaTest(unittest.TestCase):
    """27. daily synthesis 的 schema"""

    def test_parses_and_caps_lists(self):
        synthesis = parse_daily_synthesis(
            {
                "headline": "今天很热闹",
                "signals": [f"signal{i}" for i in range(20)],
                "rising_categories": [f"cat{i}" for i in range(20)],
                "watch_tomorrow": [f"w{i}" for i in range(20)],
            }
        )
        self.assertEqual(synthesis.headline, "今天很热闹")
        self.assertEqual(len(synthesis.signals), 5)
        self.assertEqual(len(synthesis.rising_categories), 5)
        self.assertEqual(len(synthesis.watch_tomorrow), 5)

    def test_empty_payload_returns_none(self):
        self.assertIsNone(parse_daily_synthesis({}))
        self.assertIsNone(parse_daily_synthesis("not a dict"))
        self.assertIsNone(parse_daily_synthesis(None))


class UsageAggregationTest(unittest.TestCase):
    """33. usage token aggregation"""

    def test_add_request_accumulates(self):
        usage = AIUsage()
        usage.add_request(success=True, prompt_tokens=100, completion_tokens=20, total_tokens=120)
        usage.add_request(success=True, prompt_tokens=200, completion_tokens=30, total_tokens=230)
        self.assertEqual(usage.prompt_tokens, 300)
        self.assertEqual(usage.completion_tokens, 50)
        self.assertEqual(usage.total_tokens, 350)
        self.assertEqual(usage.requests, 2)
        self.assertEqual(usage.successful_requests, 2)
        self.assertEqual(usage.failed_requests, 0)

    def test_failed_requests_are_counted_separately(self):
        usage = AIUsage()
        usage.add_request(success=False)
        self.assertEqual(usage.requests, 1)
        self.assertEqual(usage.failed_requests, 1)
        self.assertEqual(usage.total_tokens, 0)

    def test_total_is_derived_when_missing(self):
        usage = AIUsage()
        usage.add_request(success=True, prompt_tokens=100, completion_tokens=25)
        self.assertEqual(usage.total_tokens, 125)

    def test_merge(self):
        a = AIUsage(prompt_tokens=10, requests=1, cache_hits=2)
        b = AIUsage(prompt_tokens=5, requests=1, cache_hits=3)
        a.merge(b)
        self.assertEqual(a.prompt_tokens, 15)
        self.assertEqual(a.requests, 2)
        self.assertEqual(a.cache_hits, 5)

    def test_round_trip_dict(self):
        usage = AIUsage(prompt_tokens=11, completion_tokens=2, total_tokens=13, cache_hits=4)
        restored = AIUsage.from_dict(usage.to_dict())
        self.assertEqual(restored.to_dict(), usage.to_dict())

    def test_from_dict_ignores_garbage(self):
        usage = AIUsage.from_dict({"prompt_tokens": "abc", "cache_hits": 3})
        self.assertEqual(usage.prompt_tokens, 0)
        self.assertEqual(usage.cache_hits, 3)


if __name__ == "__main__":
    unittest.main()
