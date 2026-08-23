# coding=utf-8
"""每日运行状态（幂等）测试

重点验证两条不能出错的规则：
- 「快照 + 邮件」都完成 → 后续触发跳过
- 「快照完成但邮件没发」 → 后续触发**必须**继续跑（否则邮件永远补不上）

以及一条安全兜底：状态文件损坏时按「今天什么都没做」处理，宁可多发不可漏发。
"""

import json
import tempfile
import unittest
from pathlib import Path

from github_radar.state import (
    DEFAULT_STATE_DIRNAME,
    STATE_SCHEMA_VERSION,
    DailyState,
    StateStore,
)

DATE = "2026-08-23"


class DailyStateSkipTest(unittest.TestCase):
    """should_skip 的判定矩阵"""

    def test_fresh_state_never_skips(self):
        self.assertFalse(DailyState(date=DATE).should_skip())

    def test_snapshot_only_does_not_skip_when_email_needed(self):
        """核心场景：快照成功但邮件失败，后续兜底触发必须继续跑"""
        state = DailyState(date=DATE)
        state.mark_snapshot_completed("2026-08-23T08:17:00+08:00")
        self.assertFalse(state.should_skip(needs_email=True))

    def test_snapshot_only_skips_when_email_not_needed(self):
        """--no-email 的运行本来就不发信，快照完成即无事可做"""
        state = DailyState(date=DATE)
        state.mark_snapshot_completed("2026-08-23T08:17:00+08:00")
        self.assertTrue(state.should_skip(needs_email=False))

    def test_fully_completed_day_skips(self):
        state = DailyState(date=DATE)
        state.mark_snapshot_completed("2026-08-23T08:17:00+08:00")
        state.mark_email_sent("2026-08-23T08:17:30+08:00")
        self.assertTrue(state.should_skip())
        self.assertTrue(state.is_complete)

    def test_email_sent_without_snapshot_still_runs(self):
        """罕见但要兜住：邮件发了、快照没写成功 → 还得回来补快照"""
        state = DailyState(date=DATE)
        state.mark_email_sent("2026-08-23T08:17:30+08:00")
        self.assertFalse(state.should_skip())
        self.assertIsNone(state.completed_at)


class DailyStateMarkTest(unittest.TestCase):
    """状态更新语义"""

    def test_completed_at_only_set_when_both_done(self):
        state = DailyState(date=DATE)
        state.mark_snapshot_completed("t-snapshot")
        self.assertIsNone(state.completed_at)

        state.mark_email_sent("t-email")
        self.assertEqual(state.completed_at, "t-email")

    def test_completed_at_keeps_first_value(self):
        state = DailyState(date=DATE)
        state.mark_snapshot_completed("t1")
        state.mark_email_sent("t2")
        state.mark_email_sent("t3")
        self.assertEqual(state.completed_at, "t2")
        self.assertEqual(state.email_sent_at, "t2")

    def test_mark_run_started_counts_runs(self):
        state = DailyState(date=DATE)
        state.mark_run_started("t1")
        state.mark_run_started("t2")
        self.assertEqual(state.runs, 2)
        self.assertEqual(state.last_run_at, "t2")

    def test_describe_mentions_both_flags(self):
        state = DailyState(date=DATE)
        self.assertIn("snapshot=pending", state.describe())
        self.assertIn("email=pending", state.describe())


class DailyStateSerializationTest(unittest.TestCase):
    """序列化与反序列化"""

    def test_round_trip(self):
        state = DailyState(date=DATE, timezone="Asia/Shanghai")
        state.mark_run_started("2026-08-23T08:17:00+08:00")
        state.mark_snapshot_completed("2026-08-23T08:17:00+08:00")
        state.mark_email_sent("2026-08-23T08:17:00+08:00")
        state.last_status = "ok"

        restored = DailyState.from_dict(state.to_dict(), date=DATE)

        self.assertEqual(restored, state)

    def test_to_dict_contains_required_fields(self):
        payload = DailyState(date=DATE).to_dict()
        for key in ("snapshot_completed", "email_sent", "completed_at"):
            self.assertIn(key, payload)
        self.assertEqual(payload["schema_version"], STATE_SCHEMA_VERSION)
        self.assertEqual(payload["date"], DATE)

    def test_from_dict_uses_filename_date_not_content(self):
        restored = DailyState.from_dict({"date": "1999-01-01"}, date=DATE)
        self.assertEqual(restored.date, DATE)

    def test_non_boolean_flags_are_treated_as_not_done(self):
        """fail-open：任何非 true 的值都当作「没做」，只会多跑不会漏发"""
        for value in ("true", 1, "yes", None, [], {}):
            with self.subTest(value=value):
                restored = DailyState.from_dict(
                    {"snapshot_completed": value, "email_sent": value}, date=DATE
                )
                self.assertFalse(restored.snapshot_completed)
                self.assertFalse(restored.email_sent)
                self.assertFalse(restored.should_skip())

    def test_broken_runs_field_degrades_to_zero(self):
        restored = DailyState.from_dict({"runs": "many"}, date=DATE)
        self.assertEqual(restored.runs, 0)


class StateStoreIOTest(unittest.TestCase):
    """状态文件读写"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.state_dir = Path(self._tmp.name) / "github_radar" / DEFAULT_STATE_DIRNAME
        self.store = StateStore(self.state_dir)

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_creates_directory_and_file(self):
        state = DailyState(date=DATE)
        state.mark_snapshot_completed("2026-08-23T08:17:00+08:00")

        path = self.store.save(state)

        self.assertIsNotNone(path)
        self.assertEqual(path, self.state_dir / f"{DATE}.json")
        self.assertTrue(path.is_file())

        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(payload["snapshot_completed"])
        self.assertFalse(payload["email_sent"])
        self.assertIsNone(payload["completed_at"])

    def test_saved_file_is_stable_for_git(self):
        """sort_keys + 末尾换行：每天的 diff 才不会无意义地抖动"""
        self.store.save(DailyState(date=DATE))
        text = (self.state_dir / f"{DATE}.json").read_text(encoding="utf-8")
        self.assertTrue(text.endswith("}\n"))
        keys = [line.split('"')[1] for line in text.splitlines() if line.startswith('  "')]
        self.assertEqual(keys, sorted(keys))

    def test_save_leaves_no_temp_file(self):
        self.store.save(DailyState(date=DATE))
        leftovers = [p.name for p in self.state_dir.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_load_round_trip(self):
        state = DailyState(date=DATE)
        state.mark_run_started("2026-08-23T08:17:00+08:00")
        state.mark_snapshot_completed("2026-08-23T08:17:00+08:00")
        state.mark_email_sent("2026-08-23T08:57:00+08:00")
        state.last_status = "ok"
        self.store.save(state)

        loaded = self.store.load(DATE)

        self.assertTrue(loaded.snapshot_completed)
        self.assertTrue(loaded.email_sent)
        self.assertEqual(loaded.completed_at, "2026-08-23T08:57:00+08:00")
        self.assertEqual(loaded.runs, 1)
        self.assertEqual(loaded.last_status, "ok")
        self.assertTrue(loaded.should_skip())

    def test_load_missing_file_returns_empty_state(self):
        loaded = self.store.load(DATE)
        self.assertEqual(loaded.date, DATE)
        self.assertFalse(loaded.snapshot_completed)
        self.assertFalse(loaded.email_sent)
        self.assertFalse(self.store.exists(DATE))

    def test_load_corrupted_file_returns_empty_state(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / f"{DATE}.json").write_text("{not json", encoding="utf-8")

        loaded = self.store.load(DATE)

        self.assertFalse(loaded.should_skip())

    def test_load_non_object_file_returns_empty_state(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / f"{DATE}.json").write_text("[1, 2, 3]", encoding="utf-8")

        self.assertFalse(self.store.load(DATE).should_skip())

    def test_save_failure_returns_none_instead_of_raising(self):
        """状态写不进去也绝不能让日报流程崩掉"""
        self.state_dir.parent.mkdir(parents=True, exist_ok=True)
        self.state_dir.write_text("I am a file, not a directory", encoding="utf-8")

        self.assertIsNone(self.store.save(DailyState(date=DATE)))


class StateStorePruneTest(unittest.TestCase):
    """保留策略"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.state_dir = Path(self._tmp.name) / DEFAULT_STATE_DIRNAME
        self.store = StateStore(self.state_dir)

    def tearDown(self):
        self._tmp.cleanup()

    def _seed(self, *dates):
        for date_str in dates:
            self.store.save(DailyState(date=date_str))

    def test_prunes_only_expired_dates(self):
        self._seed("2026-01-01", "2026-08-20", "2026-08-23")

        removed = self.store.prune(90, "2026-08-23")

        self.assertEqual(removed, ["2026-01-01"])
        self.assertEqual(self.store.available_dates(), ["2026-08-20", "2026-08-23"])

    def test_zero_retention_keeps_everything(self):
        self._seed("2020-01-01", "2026-08-23")
        self.assertEqual(self.store.prune(0, "2026-08-23"), [])
        self.assertEqual(len(self.store.available_dates()), 2)

    def test_prune_on_missing_directory_is_noop(self):
        self.assertEqual(StateStore(self.state_dir / "nope").prune(90, "2026-08-23"), [])

    def test_available_dates_ignores_other_files(self):
        self._seed("2026-08-23")
        (self.state_dir / ".gitkeep").write_text("keep", encoding="utf-8")
        (self.state_dir / "notes.txt").write_text("hi", encoding="utf-8")
        (self.state_dir / "sub").mkdir()

        self.assertEqual(self.store.available_dates(), ["2026-08-23"])
        self.assertEqual(self.store.prune(90, "2026-08-23"), [])


if __name__ == "__main__":
    unittest.main()
