# coding=utf-8
"""Heat Score 与榜单筛选测试"""

import unittest
from datetime import datetime, timedelta, timezone

from github_radar.history import StarDelta
from github_radar.models import RepoRecord
from github_radar.ranking import (
    WEIGHTS,
    freshness_score,
    minmax_normalize,
    rank_repositories,
    score_from_components,
    score_with_details,
    select_hot_today,
    select_new_and_rising,
    trending_score,
)

NOW = datetime(2026, 8, 22, 8, 10, tzinfo=timezone.utc)


def iso_days_ago(days: float) -> str:
    return (NOW - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def record(full_name, **kwargs):
    return RepoRecord(full_name=full_name, **kwargs)


class ScoreFunctionTest(unittest.TestCase):
    """评分函数本身（纯函数，独立可测）"""

    def test_all_metrics_max_gives_100(self):
        components = {key: 1.0 for key in WEIGHTS}
        self.assertEqual(score_from_components(components), 100.0)

    def test_all_metrics_zero_gives_0(self):
        components = {key: 0.0 for key in WEIGHTS}
        self.assertEqual(score_from_components(components), 0.0)

    def test_score_is_bounded(self):
        components = {key: 5.0 for key in WEIGHTS}  # 越界值应被夹住
        self.assertEqual(score_from_components(components), 100.0)
        components = {key: -3.0 for key in WEIGHTS}
        self.assertEqual(score_from_components(components), 0.0)

    def test_missing_metrics_are_renormalized_not_zero_penalized(self):
        """缺失指标不按 0 惩罚，而是对可用指标重新归一化"""
        only_stars = {"star_scale": 1.0}
        self.assertEqual(score_from_components(only_stars), 100.0)

        partial = {"growth_24h": 0.5, "star_scale": 1.0}
        # (0.45*0.5 + 0.10*1.0) / (0.45 + 0.10) = 0.5909...
        self.assertAlmostEqual(score_from_components(partial), 59.09, places=2)

    def test_weights_used_sum_to_one(self):
        score, weights_used = score_with_details({"growth_24h": 0.5, "freshness": 1.0})
        self.assertGreater(score, 0)
        self.assertAlmostEqual(sum(weights_used.values()), 1.0)
        self.assertEqual(set(weights_used), {"growth_24h", "freshness"})

    def test_no_available_metric_gives_zero(self):
        self.assertEqual(score_from_components({}), 0.0)
        self.assertEqual(score_from_components({key: None for key in WEIGHTS}), 0.0)

    def test_growth_dominates_star_scale(self):
        """24h 增长权重最高：高增长小项目应超过零增长巨无霸"""
        fast_small = {"growth_24h": 1.0, "star_scale": 0.1}
        slow_huge = {"growth_24h": 0.0, "star_scale": 1.0}
        self.assertGreater(
            score_from_components(fast_small), score_from_components(slow_huge)
        )

    def test_custom_weights_are_respected(self):
        components = {"growth_24h": 0.0, "star_scale": 1.0}
        custom = {"growth_24h": 0.1, "star_scale": 0.9}
        self.assertAlmostEqual(score_from_components(components, custom), 90.0)


class NormalizationTest(unittest.TestCase):
    def test_minmax_basic(self):
        result = minmax_normalize({"a": 0.0, "b": 5.0, "c": 10.0})
        self.assertAlmostEqual(result["a"], 0.0)
        self.assertAlmostEqual(result["b"], 0.5)
        self.assertAlmostEqual(result["c"], 1.0)

    def test_identical_values_are_neutral(self):
        result = minmax_normalize({"a": 3.0, "b": 3.0})
        self.assertEqual(result, {"a": 0.5, "b": 0.5})

    def test_empty_input(self):
        self.assertEqual(minmax_normalize({}), {})

    def test_freshness_decays_with_age(self):
        self.assertAlmostEqual(freshness_score(0), 1.0)
        self.assertLess(freshness_score(365), freshness_score(30))
        self.assertGreater(freshness_score(30), 0.0)
        self.assertIsNone(freshness_score(None))

    def test_trending_score_mapping(self):
        self.assertAlmostEqual(trending_score(1, 25), 1.0)
        self.assertAlmostEqual(trending_score(25, 25), 1 / 25)
        self.assertGreater(trending_score(1, 25), trending_score(10, 25))
        self.assertIsNone(trending_score(None, 25))
        self.assertIsNone(trending_score(3, 0))


class RankRepositoriesTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            record(
                "hot/rocket",
                stars=8000,
                created_at=iso_days_ago(10),
                trending_rank=1,
            ),
            record(
                "mega/oldproject",
                stars=200000,
                created_at=iso_days_ago(3000),
            ),
            record(
                "quiet/newbie",
                stars=120,
                created_at=iso_days_ago(5),
            ),
        ]
        self.deltas = {
            "hot/rocket": StarDelta(delta_stars_24h=1800, delta_stars_7d=6300, average_daily_growth_7d=900.0),
            "mega/oldproject": StarDelta(delta_stars_24h=30, delta_stars_7d=210, average_daily_growth_7d=30.0),
            "quiet/newbie": StarDelta(delta_stars_24h=5, delta_stars_7d=20, average_daily_growth_7d=20 / 7),
        }

    def test_high_growth_repo_ranks_first(self):
        scored = rank_repositories(self.records, self.deltas, reference_time=NOW)
        self.assertEqual(scored[0].full_name, "hot/rocket")
        self.assertTrue(0.0 <= scored[0].score <= 100.0)

    def test_all_scores_within_bounds(self):
        scored = rank_repositories(self.records, self.deltas, reference_time=NOW)
        for item in scored:
            self.assertGreaterEqual(item.score, 0.0)
            self.assertLessEqual(item.score, 100.0)

    def test_first_day_without_deltas_still_ranks(self):
        """首日无历史：不报错，退化为 Trending 排名 / 新鲜度 / 总 Star"""
        scored = rank_repositories(self.records, {}, reference_time=NOW)
        self.assertEqual(len(scored), 3)
        for item in scored:
            self.assertIsNone(item.components["growth_24h"])
            self.assertIsNone(item.components["growth_7d"])
            self.assertNotIn("growth_24h", item.weights_used)
        self.assertEqual(scored[0].full_name, "hot/rocket")  # Trending #1 + 新项目

    def test_components_recorded_for_explainability(self):
        scored = rank_repositories(self.records, self.deltas, reference_time=NOW)
        top = scored[0]
        self.assertAlmostEqual(top.components["trending"], 1.0)
        self.assertIsNotNone(top.components["freshness"])
        self.assertIn("24h", top.explain())

    def test_missing_stars_does_not_crash(self):
        records = [record("no/stars", created_at=iso_days_ago(1)), *self.records]
        scored = rank_repositories(records, self.deltas, reference_time=NOW)
        self.assertEqual(len(scored), 4)
        no_stars = [item for item in scored if item.full_name == "no/stars"][0]
        self.assertIsNone(no_stars.components["star_scale"])

    def test_ordering_is_deterministic_for_ties(self):
        tied = [record("b/b", stars=100, created_at=iso_days_ago(1)),
                record("a/a", stars=100, created_at=iso_days_ago(1))]
        scored = rank_repositories(tied, {}, reference_time=NOW)
        self.assertEqual([item.full_name for item in scored], ["a/a", "b/b"])

    def test_select_hot_today_limits_count(self):
        scored = rank_repositories(self.records, self.deltas, reference_time=NOW)
        self.assertEqual(len(select_hot_today(scored, top_n=2)), 2)
        self.assertEqual(select_hot_today(scored, top_n=0), [])
        self.assertEqual(len(select_hot_today(scored, top_n=99)), 3)

    def test_empty_input(self):
        self.assertEqual(rank_repositories([], {}), [])


class NewAndRisingTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            record("new/fast", stars=900, created_at=iso_days_ago(6)),
            record("new/slow", stars=200, created_at=iso_days_ago(25)),
            record("old/legend", stars=90000, created_at=iso_days_ago(2900)),
            record("new/tiny", stars=12, created_at=iso_days_ago(2)),
            record("unknown/age", stars=5000, created_at=None),
        ]

    def test_excludes_old_projects(self):
        scored = rank_repositories(self.records, {}, reference_time=NOW)
        picks = select_new_and_rising(scored)
        names = [item.full_name for item in picks]
        self.assertNotIn("old/legend", names)

    def test_excludes_projects_below_min_stars(self):
        scored = rank_repositories(self.records, {}, reference_time=NOW)
        names = [item.full_name for item in select_new_and_rising(scored)]
        self.assertNotIn("new/tiny", names)

    def test_excludes_unknown_age(self):
        scored = rank_repositories(self.records, {}, reference_time=NOW)
        names = [item.full_name for item in select_new_and_rising(scored)]
        self.assertNotIn("unknown/age", names)

    def test_uses_stars_per_day_estimate_without_history(self):
        scored = rank_repositories(self.records, {}, reference_time=NOW)
        picks = select_new_and_rising(scored)
        self.assertEqual(picks[0].full_name, "new/fast")  # 900/6 > 200/25
        self.assertTrue(picks[0].rising_speed_estimated)
        self.assertAlmostEqual(picks[0].rising_speed, 900 / 6, places=1)

    def test_prefers_real_24h_delta_when_available(self):
        deltas = {
            "new/fast": StarDelta(delta_stars_24h=10),
            "new/slow": StarDelta(delta_stars_24h=400),
        }
        scored = rank_repositories(self.records, deltas, reference_time=NOW)
        picks = select_new_and_rising(scored)
        self.assertEqual(picks[0].full_name, "new/slow")
        self.assertFalse(picks[0].rising_speed_estimated)
        self.assertEqual(picks[0].rising_speed, 400.0)

    def test_top_n_limit(self):
        scored = rank_repositories(self.records, {}, reference_time=NOW)
        self.assertEqual(len(select_new_and_rising(scored, top_n=1)), 1)
        self.assertEqual(select_new_and_rising(scored, top_n=0), [])

    def test_custom_window_and_min_stars(self):
        scored = rank_repositories(self.records, {}, reference_time=NOW)
        picks = select_new_and_rising(scored, max_age_days=7, min_stars=100)
        self.assertEqual([item.full_name for item in picks], ["new/fast"])


if __name__ == "__main__":
    unittest.main()
