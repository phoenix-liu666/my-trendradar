# coding=utf-8
"""Personal Score 与 For You Top10 测试"""

import unittest

from github_radar.ai.profile import KeywordMatch
from github_radar.ai.schemas import parse_repo_analysis
from github_radar.ai.scoring import (
    WEIGHT_HEAT,
    WEIGHT_KEYWORD,
    WEIGHT_RELEVANCE,
    build_for_you,
    personal_score,
)

from .ai_helpers import analysis_payload, make_scored


def analysis(full_name, relevance=50, **overrides):
    return parse_repo_analysis(
        analysis_payload(full_name, relevance_score=relevance, **overrides),
        full_name=full_name,
    )


class PersonalScoreTest(unittest.TestCase):
    """15. Personal Score"""

    def test_weights_match_the_specification(self):
        self.assertEqual(WEIGHT_RELEVANCE, 0.55)
        self.assertEqual(WEIGHT_HEAT, 0.25)
        self.assertEqual(WEIGHT_KEYWORD, 0.20)
        self.assertAlmostEqual(WEIGHT_RELEVANCE + WEIGHT_HEAT + WEIGHT_KEYWORD, 1.0)

    def test_formula(self):
        # 0.55*98 + 0.25*50 + 0.20*100 = 53.9 + 12.5 + 20 = 86.4
        self.assertEqual(personal_score(98, 50, 100), 86.4)

    def test_all_hundred_gives_hundred(self):
        self.assertEqual(personal_score(100, 100, 100), 100.0)

    def test_all_zero_gives_zero(self):
        self.assertEqual(personal_score(0, 0, 0), 0.0)

    def test_missing_components_count_as_zero(self):
        self.assertEqual(personal_score(None, None, None), 0.0)
        self.assertEqual(personal_score(100, None, None), 55.0)

    def test_out_of_range_inputs_are_clamped(self):
        self.assertEqual(personal_score(500, 500, 500), 100.0)
        self.assertEqual(personal_score(-100, 0, 0), 0.0)

    def test_garbage_inputs_are_safe(self):
        self.assertEqual(personal_score("abc", None, []), 0.0)

    def test_result_is_within_range(self):
        for relevance in (0, 33, 77, 100):
            for heat in (0, 50, 100):
                score = personal_score(relevance, heat, 50)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 100.0)


class ForYouTest(unittest.TestCase):
    """16. For You Top10"""

    def setUp(self):
        self.scored = [make_scored(f"owner/repo{i:02d}", score=90.0 - i) for i in range(30)]

    def build(self, analyses, keywords=None, top_n=10):
        return build_for_you(self.scored, analyses, keywords or {}, top_n=top_n)

    def test_only_analysed_repos_appear(self):
        analyses = {"owner/repo05": analysis("owner/repo05", relevance=80)}
        entries = self.build(analyses)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].full_name, "owner/repo05")

    def test_top_n_is_respected(self):
        analyses = {
            item.record.full_name.lower(): analysis(item.record.full_name, relevance=50)
            for item in self.scored
        }
        self.assertEqual(len(self.build(analyses)), 10)
        self.assertEqual(len(self.build(analyses, top_n=3)), 3)

    def test_sorted_by_personal_score(self):
        analyses = {
            "owner/repo00": analysis("owner/repo00", relevance=10),
            "owner/repo01": analysis("owner/repo01", relevance=95),
        }
        entries = self.build(analyses)
        self.assertEqual(entries[0].full_name, "owner/repo01")
        self.assertGreater(entries[0].personal_score, entries[1].personal_score)

    def test_repo_outside_hot20_can_top_the_list(self):
        """Heat Score 50、relevance 98、keyword 100 仍然可以排第一"""
        low_heat = make_scored("niche/gem", score=50.0)
        pool = self.scored + [low_heat]
        analyses = {
            "niche/gem": analysis("niche/gem", relevance=98),
            "owner/repo00": analysis("owner/repo00", relevance=40),
        }
        keywords = {"niche/gem": KeywordMatch(score=100.0, top_category="ai_agents")}

        entries = build_for_you(pool, analyses, keywords, top_n=10)
        self.assertEqual(entries[0].full_name, "niche/gem")
        self.assertEqual(entries[0].personal_score, personal_score(98, 50.0, 100.0))
        # 而且它确实不在 Heat Score 前 20 里
        top20 = [item.record.full_name for item in sorted(pool, key=lambda i: -i.score)[:20]]
        self.assertNotIn("niche/gem", top20)

    def test_entry_exposes_display_fields(self):
        analyses = {"owner/repo00": analysis("owner/repo00", relevance=88)}
        keywords = {"owner/repo00": KeywordMatch(score=70.0, matched_keywords=["MCP"], top_category="ai_agents")}
        entry = build_for_you(self.scored, analyses, keywords)[0]

        self.assertEqual(entry.relevance_score, 88)
        self.assertEqual(entry.keyword_score, 70.0)
        self.assertEqual(entry.heat_score, 90.0)
        self.assertEqual(entry.analysis.action_label, "关注")

    def test_relevance_explanation_falls_back_to_keywords(self):
        item = analysis("owner/repo00", relevance=88)
        item.relevance_reason = ""
        keywords = {"owner/repo00": KeywordMatch(score=70.0, matched_keywords=["MCP"], top_category="ai_agents")}
        entry = build_for_you(self.scored, {"owner/repo00": item}, keywords)[0]
        self.assertIn("MCP", entry.relevance_explanation)

    def test_relevance_explanation_never_empty(self):
        item = analysis("owner/repo00")
        item.relevance_reason = ""
        entry = build_for_you(self.scored, {"owner/repo00": item}, {})[0]
        self.assertTrue(entry.relevance_explanation)

    def test_empty_analyses_gives_empty_list(self):
        self.assertEqual(self.build({}), [])

    def test_zero_top_n_gives_empty_list(self):
        analyses = {"owner/repo00": analysis("owner/repo00")}
        self.assertEqual(self.build(analyses, top_n=0), [])

    def test_ordering_is_deterministic(self):
        analyses = {
            "owner/repo00": analysis("owner/repo00", relevance=50),
            "owner/repo01": analysis("owner/repo01", relevance=50),
        }
        first = [e.full_name for e in self.build(analyses)]
        second = [e.full_name for e in self.build(analyses)]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
