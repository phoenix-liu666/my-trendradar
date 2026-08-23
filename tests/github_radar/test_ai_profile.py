# coding=utf-8
"""兴趣画像加载与 deterministic keyword score 测试"""

import tempfile
import unittest
from pathlib import Path

from github_radar.ai.profile import (
    FALLBACK_PROFILE_NAME,
    STRONG_MATCH_THRESHOLD,
    keyword_score,
    load_profile,
    score_all,
)
from github_radar.models import RepoRecord

from .ai_helpers import make_record

REAL_PROFILE = Path(__file__).resolve().parents[2] / "config" / "github_radar_profile.yaml"


def write_yaml(directory: Path, text: str) -> Path:
    path = directory / "profile.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class ProfileLoadingTest(unittest.TestCase):
    """1. profile loading"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_loads_repository_profile(self):
        profile = load_profile(REAL_PROFILE)
        self.assertFalse(profile.is_fallback)
        self.assertEqual(profile.name, "personal-tech-radar")
        names = [c.name for c in profile.interests]
        self.assertIn("ai_agents", names)
        self.assertIn("computational_optics", names)

    def test_weights_are_read_from_yaml(self):
        profile = load_profile(REAL_PROFILE)
        weights = {c.name: c.weight for c in profile.interests}
        self.assertEqual(weights["ai_agents"], 1.0)
        self.assertEqual(weights["deep_learning"], 0.8)
        self.assertEqual(weights["productivity"], 0.7)

    def test_custom_profile(self):
        path = write_yaml(
            self.dir,
            "profile:\n  name: mine\ninterests:\n  optics:\n    weight: 2.0\n    keywords:\n      - metalens\n",
        )
        profile = load_profile(path)
        self.assertEqual(profile.name, "mine")
        self.assertEqual(len(profile.interests), 1)
        self.assertEqual(profile.interests[0].weight, 2.0)


class ProfileFallbackTest(unittest.TestCase):
    """2. profile missing fallback"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_file_falls_back(self):
        profile = load_profile(self.dir / "does-not-exist.yaml")
        self.assertTrue(profile.is_fallback)
        self.assertEqual(profile.name, FALLBACK_PROFILE_NAME)
        self.assertTrue(profile.interests)

    def test_broken_yaml_falls_back(self):
        path = write_yaml(self.dir, "interests: [unclosed\n  - broken:")
        profile = load_profile(path)
        self.assertTrue(profile.is_fallback)

    def test_non_mapping_root_falls_back(self):
        path = write_yaml(self.dir, "- just\n- a\n- list\n")
        self.assertTrue(load_profile(path).is_fallback)

    def test_missing_interests_section_falls_back(self):
        path = write_yaml(self.dir, "profile:\n  name: empty\n")
        self.assertTrue(load_profile(path).is_fallback)

    def test_empty_keywords_are_dropped_and_fall_back(self):
        path = write_yaml(
            self.dir, "interests:\n  a:\n    weight: 1.0\n    keywords: []\n"
        )
        self.assertTrue(load_profile(path).is_fallback)

    def test_invalid_weight_becomes_default(self):
        path = write_yaml(
            self.dir,
            "interests:\n  a:\n    weight: not-a-number\n    keywords:\n      - metalens\n",
        )
        profile = load_profile(path)
        self.assertFalse(profile.is_fallback)
        self.assertEqual(profile.interests[0].weight, 1.0)

    def test_negative_weight_becomes_default(self):
        path = write_yaml(
            self.dir, "interests:\n  a:\n    weight: -5\n    keywords:\n      - metalens\n"
        )
        self.assertEqual(load_profile(path).interests[0].weight, 1.0)

    def test_partially_broken_profile_keeps_valid_categories(self):
        path = write_yaml(
            self.dir,
            "interests:\n"
            "  good:\n    weight: 1.0\n    keywords:\n      - metalens\n"
            "  empty:\n    weight: 1.0\n    keywords: []\n"
            "  broken: 'not a mapping'\n",
        )
        profile = load_profile(path)
        self.assertFalse(profile.is_fallback)
        self.assertEqual([c.name for c in profile.interests], ["good"])


class KeywordScoreTest(unittest.TestCase):
    """3. keyword score"""

    def setUp(self):
        self.profile = load_profile(REAL_PROFILE)

    def score(self, record: RepoRecord, readme: str = "") -> float:
        return keyword_score(record, self.profile, readme=readme).score

    def test_score_is_within_range(self):
        for record in (
            make_record("a/b", description="nothing relevant here", topics=[]),
            make_record("x/ai-agent", description="AI Agent MCP", topics=["mcp"]),
        ):
            score = self.score(record)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)

    def test_irrelevant_repo_scores_zero(self):
        record = make_record("some/crud-app", description="A simple CRUD web app", topics=["web"])
        self.assertEqual(self.score(record), 0.0)

    def test_strong_match_reaches_threshold(self):
        record = make_record(
            "acme/coding-agent",
            description="An autonomous agent framework with MCP tool use",
            topics=["ai-agent", "agentic-coding"],
        )
        self.assertGreaterEqual(self.score(record), STRONG_MATCH_THRESHOLD)

    def test_category_weight_caps_the_score(self):
        """只命中 productivity（权重 0.7）时上限就是 70 分"""
        record = make_record(
            "me/notes",
            description="Obsidian knowledge management automation workflow productivity",
            topics=["developer-tools"],
        )
        match = keyword_score(record, self.profile)
        self.assertEqual(match.top_category, "productivity")
        self.assertLessEqual(match.score, 70.0)

    def test_name_match_weighs_more_than_readme(self):
        by_name = self.score(make_record("lab/metalens-designer", description="", topics=[]))
        by_readme = self.score(
            make_record("lab/xyz", description="", topics=[]), readme="metalens research"
        )
        self.assertGreater(by_name, by_readme)

    def test_readme_contributes_to_the_score(self):
        record = make_record("lab/xyz", description="", topics=[])
        self.assertEqual(self.score(record), 0.0)
        self.assertGreater(self.score(record, readme="FDTD and RCWA simulation"), 0.0)

    def test_word_boundary_prevents_false_positives(self):
        """MCP 不应该命中 mcphersons"""
        record = make_record("x/mcphersons", description="about the mcphersons family", topics=[])
        self.assertEqual(self.score(record), 0.0)

    def test_multi_word_keyword_tolerates_separators(self):
        for text in ("ai-agent", "AI_Agent", "AI  Agent"):
            with self.subTest(text=text):
                record = make_record("x/y", description=text, topics=[])
                self.assertGreater(self.score(record), 0.0)

    def test_matched_keywords_are_reported(self):
        record = make_record("x/y", description="metalens and metasurface design", topics=[])
        match = keyword_score(record, self.profile)
        self.assertEqual(match.top_category, "computational_optics")
        self.assertIn("metalens", match.matched_keywords)
        self.assertIn("metasurface", match.matched_keywords)

    def test_score_is_deterministic(self):
        record = make_record("x/ai-agent", description="AI Agent with MCP", topics=["mcp"])
        self.assertEqual(self.score(record), self.score(record))

    def test_score_all_returns_lowercase_keys(self):
        records = [make_record("Owner/Repo", description="AI Agent")]
        scores = score_all(records, self.profile)
        self.assertIn("owner/repo", scores)

    def test_empty_profile_scores_zero(self):
        from github_radar.ai.profile import UserProfile

        match = keyword_score(make_record("a/b", description="AI Agent"), UserProfile())
        self.assertEqual(match.score, 0.0)


if __name__ == "__main__":
    unittest.main()
