# coding=utf-8
"""AI 候选选择测试（优先级 / 硬上限 / Hot20 之外的强匹配）"""

import unittest

from github_radar.ai.profile import KeywordMatch
from github_radar.ai.selector import (
    PRIORITY_HOT,
    PRIORITY_OTHER,
    PRIORITY_RISING,
    PRIORITY_STRONG_MATCH,
    select_ai_candidates,
)

from .ai_helpers import make_scored


def build_pool(count: int = 100):
    """构造一个 100 个候选的池子（Heat Score 递减）"""
    return [
        make_scored(f"owner/repo{index:03d}", score=100.0 - index * 0.5)
        for index in range(count)
    ]


def keywords(**scores) -> dict:
    return {name.lower(): KeywordMatch(score=value) for name, value in scores.items()}


class CandidatePrefilterTest(unittest.TestCase):
    """4. candidate prefilter"""

    def setUp(self):
        self.pool = build_pool()
        self.hot = self.pool[:20]
        self.rising = self.pool[20:30]

    def test_hot_and_rising_are_selected_first_without_keywords(self):
        selected = select_ai_candidates(self.pool, self.hot, self.rising, {}, limit=30)
        names = [c.full_name for c in selected]
        self.assertEqual(names[:20], [item.record.full_name for item in self.hot])
        self.assertEqual(names[20:30], [item.record.full_name for item in self.rising])

    def test_priorities_are_labelled(self):
        selected = select_ai_candidates(self.pool, self.hot, self.rising, {}, limit=30)
        self.assertEqual(selected[0].priority, PRIORITY_HOT)
        self.assertEqual(selected[25].priority, PRIORITY_RISING)

    def test_strong_match_takes_priority_one(self):
        scores = {"owner/repo099": KeywordMatch(score=95.0)}
        selected = select_ai_candidates(self.pool, self.hot, self.rising, scores, limit=30)
        self.assertEqual(selected[0].full_name, "owner/repo099")
        self.assertEqual(selected[0].priority, PRIORITY_STRONG_MATCH)

    def test_others_are_ordered_by_keyword_then_heat(self):
        scores = {
            "owner/repo080": KeywordMatch(score=40.0),
            "owner/repo090": KeywordMatch(score=20.0),
        }
        selected = select_ai_candidates(self.pool, self.hot, self.rising, scores, limit=33)
        tail = [c.full_name for c in selected if c.priority == PRIORITY_OTHER]
        self.assertEqual(tail[0], "owner/repo080")
        self.assertEqual(tail[1], "owner/repo090")

    def test_no_duplicates(self):
        scores = {"owner/repo000": KeywordMatch(score=99.0)}
        selected = select_ai_candidates(self.pool, self.hot, self.rising, scores, limit=30)
        names = [c.full_name for c in selected]
        self.assertEqual(len(names), len(set(names)))

    def test_empty_pool_returns_nothing(self):
        self.assertEqual(select_ai_candidates([], [], [], {}, limit=30), [])


class RepoLimitTest(unittest.TestCase):
    """5. AI repo limit 30 / 32. repo limit"""

    def setUp(self):
        self.pool = build_pool(300)
        self.hot = self.pool[:20]
        self.rising = self.pool[20:30]

    def test_default_limit_is_respected_with_300_candidates(self):
        selected = select_ai_candidates(self.pool, self.hot, self.rising, {}, limit=30)
        self.assertEqual(len(selected), 30)

    def test_limit_can_be_lowered(self):
        selected = select_ai_candidates(self.pool, self.hot, self.rising, {}, limit=20)
        self.assertEqual(len(selected), 20)

    def test_limit_can_be_raised(self):
        selected = select_ai_candidates(self.pool, self.hot, self.rising, {}, limit=40)
        self.assertEqual(len(selected), 40)

    def test_zero_limit_disables_ai_candidates(self):
        self.assertEqual(select_ai_candidates(self.pool, self.hot, self.rising, {}, limit=0), [])

    def test_limit_holds_even_when_everything_is_a_strong_match(self):
        scores = {item.record.full_name.lower(): KeywordMatch(score=100.0) for item in self.pool}
        selected = select_ai_candidates(self.pool, self.hot, self.rising, scores, limit=30)
        self.assertEqual(len(selected), 30)


class OutsideHotTwentyTest(unittest.TestCase):
    """6. Hot20 之外的项目也要能进 For You 候选"""

    def setUp(self):
        self.pool = build_pool()
        self.hot = self.pool[:20]
        self.rising = self.pool[20:30]

    def test_low_heat_but_high_keyword_repo_is_selected(self):
        target = "owner/repo095"          # Heat Score 很低，肯定不在 Hot20
        scores = {target: KeywordMatch(score=100.0, top_category="ai_agents")}
        selected = select_ai_candidates(self.pool, self.hot, self.rising, scores, limit=30)

        names = [c.full_name for c in selected]
        self.assertIn(target, names)
        self.assertNotIn(target, [item.record.full_name for item in self.hot])
        self.assertEqual(names[0], target)   # 而且排在最前面

    def test_multiple_strong_matches_are_ordered_by_keyword_score(self):
        scores = {
            "owner/repo095": KeywordMatch(score=80.0),
            "owner/repo096": KeywordMatch(score=99.0),
        }
        selected = select_ai_candidates(self.pool, self.hot, self.rising, scores, limit=30)
        self.assertEqual(
            [c.full_name for c in selected[:2]], ["owner/repo096", "owner/repo095"]
        )

    def test_strong_matches_still_leave_room_for_hot20(self):
        scores = {f"owner/repo{index:03d}": KeywordMatch(score=90.0) for index in range(95, 100)}
        selected = select_ai_candidates(self.pool, self.hot, self.rising, scores, limit=30)
        names = [c.full_name for c in selected]
        self.assertEqual(len(selected), 30)
        for item in self.hot[:10]:
            self.assertIn(item.record.full_name, names)

    def test_below_threshold_is_not_priority_one(self):
        scores = {"owner/repo095": KeywordMatch(score=30.0)}
        selected = select_ai_candidates(self.pool, self.hot, self.rising, scores, limit=30)
        self.assertNotEqual(selected[0].full_name, "owner/repo095")


if __name__ == "__main__":
    unittest.main()
