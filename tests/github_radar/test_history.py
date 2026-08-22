# coding=utf-8
"""快照读写、保留策略与 Star 增量计算测试"""

import json
import tempfile
import unittest
from pathlib import Path

from github_radar.history import (
    LATEST_FILENAME,
    SnapshotStore,
    compute_deltas,
    load_history,
)
from github_radar.models import RepoRecord


def make_record(full_name, stars=None, **kwargs):
    return RepoRecord(full_name=full_name, stars=stars, **kwargs)


class SnapshotIOTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name) / "github_radar"
        self.store = SnapshotStore(self.data_dir)

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_creates_daily_file_with_expected_shape(self):
        records = [
            make_record(
                "owner/repo",
                stars=1234,
                forks=56,
                language="Python",
                created_at="2026-08-01T00:00:00Z",
                trending_rank=3,
                topics=["ai", "agent"],
            )
        ]
        path = self.store.save(
            "2026-08-22",
            records,
            generated_at="2026-08-22T08:10:00+08:00",
            timezone="Asia/Shanghai",
        )

        self.assertEqual(path.name, "2026-08-22.json")
        self.assertTrue(path.is_file())

        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["date"], "2026-08-22")
        self.assertEqual(payload["generated_at"], "2026-08-22T08:10:00+08:00")
        self.assertEqual(payload["repository_count"], 1)

        entry = payload["repositories"]["owner/repo"]
        self.assertEqual(entry["stars"], 1234)
        self.assertEqual(entry["forks"], 56)
        self.assertEqual(entry["language"], "Python")
        self.assertEqual(entry["created_at"], "2026-08-01T00:00:00Z")
        self.assertEqual(entry["trending_rank"], 3)
        # 描述与 topics 刻意不入快照，控制 git 体积增长
        self.assertNotIn("description", entry)
        self.assertNotIn("topics", entry)

    def test_save_writes_latest_alongside_daily_file(self):
        self.store.save("2026-08-22", [make_record("a/b", stars=1)], generated_at="x")
        self.assertTrue((self.data_dir / LATEST_FILENAME).is_file())
        # latest.json 不能取代每日文件
        self.assertTrue((self.data_dir / "2026-08-22.json").is_file())

    def test_snapshot_json_is_stable_for_small_git_diffs(self):
        records = [make_record("b/b", stars=2), make_record("a/a", stars=1)]
        self.store.save("2026-08-22", records, generated_at="x")
        text = (self.data_dir / "2026-08-22.json").read_text(encoding="utf-8")
        self.assertLess(text.index('"a/a"'), text.index('"b/b"'))  # sort_keys
        self.assertTrue(text.endswith("\n"))

    def test_load_missing_snapshot_returns_empty(self):
        self.assertIsNone(self.store.load("2026-01-01"))
        self.assertEqual(self.store.load_repositories("2026-01-01"), {})

    def test_corrupt_snapshot_degrades_gracefully(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "2026-08-21.json").write_text("{not json", encoding="utf-8")
        self.assertIsNone(self.store.load("2026-08-21"))
        self.assertEqual(self.store.load_repositories("2026-08-21"), {})

    def test_repository_keys_are_case_insensitive(self):
        self.store.save("2026-08-21", [make_record("Owner/Repo", stars=10)], generated_at="x")
        repos = self.store.load_repositories("2026-08-21")
        self.assertIn("owner/repo", repos)

    def test_available_dates_ignores_latest_and_other_files(self):
        self.store.save("2026-08-21", [make_record("a/b", stars=1)], generated_at="x")
        self.store.save("2026-08-22", [make_record("a/b", stars=2)], generated_at="x")
        (self.data_dir / "notes.txt").write_text("hi", encoding="utf-8")
        self.assertEqual(self.store.available_dates(), ["2026-08-21", "2026-08-22"])


class RetentionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = SnapshotStore(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def _seed(self, dates):
        for date in dates:
            self.store.save(date, [make_record("a/b", stars=1)], generated_at="x")

    def test_prune_removes_snapshots_older_than_retention(self):
        self._seed(["2026-01-01", "2026-05-24", "2026-05-25", "2026-08-22"])
        removed = self.store.prune(90, "2026-08-22")
        self.assertEqual(removed, ["2026-01-01"])
        remaining = self.store.available_dates()
        self.assertNotIn("2026-01-01", remaining)
        self.assertIn("2026-08-22", remaining)

    def test_prune_keeps_exactly_90_days(self):
        # 2026-05-24 距 2026-08-22 正好 90 天 → 保留；再早一天 → 删除
        self._seed(["2026-05-23", "2026-05-24"])
        removed = self.store.prune(90, "2026-08-22")
        self.assertEqual(removed, ["2026-05-23"])
        self.assertEqual(self.store.available_dates(), ["2026-05-24"])

    def test_prune_disabled_when_retention_is_zero(self):
        self._seed(["2020-01-01"])
        self.assertEqual(self.store.prune(0, "2026-08-22"), [])
        self.assertEqual(self.store.available_dates(), ["2020-01-01"])

    def test_prune_never_touches_latest_json(self):
        self._seed(["2020-01-01"])
        self.store.prune(90, "2026-08-22")
        self.assertTrue((self.store.data_dir / LATEST_FILENAME).is_file())


class DeltaTest(unittest.TestCase):
    def test_24h_delta_uses_yesterday_snapshot(self):
        records = [make_record("a/b", stars=1200)]
        deltas = compute_deltas(records, {"a/b": {"stars": 1000}}, {})
        delta = deltas["a/b"]
        self.assertEqual(delta.delta_stars_24h, 200)
        self.assertIsNone(delta.delta_stars_7d)
        self.assertIsNone(delta.average_daily_growth_7d)

    def test_7d_delta_and_average(self):
        records = [make_record("a/b", stars=1700)]
        deltas = compute_deltas(records, {"a/b": {"stars": 1600}}, {"a/b": {"stars": 1000}})
        delta = deltas["a/b"]
        self.assertEqual(delta.delta_stars_24h, 100)
        self.assertEqual(delta.delta_stars_7d, 700)
        self.assertAlmostEqual(delta.average_daily_growth_7d, 100.0)

    def test_first_day_without_history_yields_none(self):
        deltas = compute_deltas([make_record("a/b", stars=500)], {}, {})
        delta = deltas["a/b"]
        self.assertIsNone(delta.delta_stars_24h)
        self.assertIsNone(delta.delta_stars_7d)
        self.assertFalse(delta.has_24h)
        self.assertFalse(delta.has_7d)

    def test_total_stars_are_never_used_as_delta(self):
        """核心约定：没有历史时 24h 增量必须是 None，绝不能等于总 Star"""
        deltas = compute_deltas([make_record("a/b", stars=99999)], {}, {})
        self.assertIsNone(deltas["a/b"].delta_stars_24h)

    def test_repo_absent_from_history_yields_none(self):
        deltas = compute_deltas(
            [make_record("new/repo", stars=300)], {"other/repo": {"stars": 10}}, {}
        )
        self.assertIsNone(deltas["new/repo"].delta_stars_24h)

    def test_missing_today_stars_yields_none(self):
        deltas = compute_deltas([make_record("a/b", stars=None)], {"a/b": {"stars": 10}}, {})
        self.assertIsNone(deltas["a/b"].delta_stars_24h)

    def test_negative_delta_is_kept_as_is(self):
        deltas = compute_deltas([make_record("a/b", stars=90)], {"a/b": {"stars": 100}}, {})
        self.assertEqual(deltas["a/b"].delta_stars_24h, -10)

    def test_broken_history_entry_is_ignored(self):
        deltas = compute_deltas(
            [make_record("a/b", stars=100)], {"a/b": {"stars": "oops"}}, {"a/b": None}
        )
        self.assertIsNone(deltas["a/b"].delta_stars_24h)
        self.assertIsNone(deltas["a/b"].delta_stars_7d)

    def test_case_insensitive_matching(self):
        deltas = compute_deltas([make_record("Owner/Repo", stars=120)], {"owner/repo": {"stars": 100}}, {})
        self.assertEqual(deltas["owner/repo"].delta_stars_24h, 20)


class LoadHistoryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = SnapshotStore(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_first_run_status(self):
        records = [make_record("a/b", stars=100)]
        yesterday, week_ago, status = load_history(self.store, "2026-08-22", records)
        self.assertEqual(yesterday, {})
        self.assertEqual(week_ago, {})
        self.assertTrue(status.is_first_run)
        self.assertFalse(status.has_yesterday)
        self.assertFalse(status.has_week_ago)
        self.assertEqual(status.available_days, 0)
        self.assertIn("first run", status.describe())

    def test_second_day_status(self):
        self.store.save("2026-08-21", [make_record("a/b", stars=100)], generated_at="x")
        records = [make_record("a/b", stars=150)]
        yesterday, week_ago, status = load_history(self.store, "2026-08-22", records)
        self.assertEqual(yesterday["a/b"]["stars"], 100)
        self.assertEqual(week_ago, {})
        self.assertTrue(status.has_yesterday)
        self.assertFalse(status.has_week_ago)
        self.assertEqual(status.matched_24h, 1)
        self.assertEqual(status.matched_7d, 0)
        self.assertEqual(status.available_days, 1)
        self.assertIn("1 day available", status.describe())

    def test_eighth_day_status_has_both_windows(self):
        for offset, date in enumerate(
            ["2026-08-15", "2026-08-16", "2026-08-21"]
        ):
            self.store.save(date, [make_record("a/b", stars=100 + offset)], generated_at="x")
        records = [make_record("a/b", stars=500)]
        yesterday, week_ago, status = load_history(self.store, "2026-08-22", records)
        self.assertTrue(status.has_yesterday)
        self.assertTrue(status.has_week_ago)
        self.assertEqual(status.matched_24h, 1)
        self.assertEqual(status.matched_7d, 1)
        self.assertFalse(status.is_first_run)
        self.assertEqual(week_ago["a/b"]["stars"], 100)

    def test_gap_day_does_not_fake_24h_delta(self):
        """昨天的快照缺失时，绝不用前天的数据冒充 24h 增量"""
        self.store.save("2026-08-20", [make_record("a/b", stars=100)], generated_at="x")
        records = [make_record("a/b", stars=400)]
        yesterday, _week, status = load_history(self.store, "2026-08-22", records)
        self.assertEqual(yesterday, {})
        self.assertFalse(status.has_yesterday)
        deltas = compute_deltas(records, yesterday, {})
        self.assertIsNone(deltas["a/b"].delta_stars_24h)

    def test_today_snapshot_not_counted_as_history(self):
        self.store.save("2026-08-22", [make_record("a/b", stars=100)], generated_at="x")
        _y, _w, status = load_history(self.store, "2026-08-22", [make_record("a/b", stars=100)])
        self.assertEqual(status.available_days, 0)


if __name__ == "__main__":
    unittest.main()
