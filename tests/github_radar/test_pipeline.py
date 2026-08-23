# coding=utf-8
"""
端到端流程测试（CLI）

用假的采集层替换网络访问，验证：
- 首日无历史也能正常产出日报与快照
- 第二天能算出 24h 增量
- 第七天之后能算出 7d 增量
- 0 个仓库时明确失败且不发邮件
- 邮件失败时退出码为 1
- 每日幂等：一天 4 次兜底触发只发一封信、只留一个正式快照
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from github_radar import cli
from github_radar.collector import CollectionResult
from github_radar.models import SOURCE_SEARCH_NEW, SOURCE_TRENDING, RepoRecord
from github_radar.state import DEFAULT_STATE_DIRNAME

EMAIL_ENV = {
    "EMAIL_FROM": "a@qq.com",
    "EMAIL_PASSWORD": "secret",
    "EMAIL_TO": "b@qq.com",
}


def make_records(stars_map, created_at="2026-08-01T00:00:00Z"):
    records = []
    for index, (full_name, stars) in enumerate(stars_map.items(), 1):
        records.append(
            RepoRecord(
                full_name=full_name,
                stars=stars,
                forks=stars // 10,
                language="Python",
                description=f"{full_name} description",
                created_at=created_at,
                trending_rank=index if index <= 2 else None,
                sources=[SOURCE_TRENDING if index <= 2 else SOURCE_SEARCH_NEW],
                api_enriched=True,
            )
        )
    return records


def fake_collection(records):
    return CollectionResult(
        repositories=records,
        trending_count=min(2, len(records)),
        search_new_count=len(records),
        search_popular_count=0,
        trending_ok=bool(records),
        search_ok=bool(records),
    )


class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.data_dir = root / "data"
        self.output_dir = root / "output"

    def tearDown(self):
        self._tmp.cleanup()

    def run_cli(self, records, date, extra_args=(), env=None, send_result=True):
        collection = fake_collection(records)
        args = [
            "--date", date,
            "--timezone", "Asia/Shanghai",
            "--data-dir", str(self.data_dir),
            "--output-dir", str(self.output_dir),
            *extra_args,
        ]
        env = env if env is not None else {}
        with mock.patch.object(cli, "GitHubAPIClient") as client_cls, mock.patch.object(
            cli, "collect_candidates", return_value=collection
        ) as collect, mock.patch.object(
            cli, "_send", return_value=send_result
        ) as send, mock.patch.dict(
            cli.os.environ, env, clear=True
        ):
            client_cls.return_value.authenticated = True
            client_cls.return_value.rate_limited = False
            client_cls.return_value.describe.return_value = "fake client"
            code = cli.main(args)
        self.sent = send
        self.collected = collect
        return code

    def snapshot(self, date):
        return json.loads((self.data_dir / f"{date}.json").read_text(encoding="utf-8"))

    def report_html(self, date):
        return (self.output_dir / f"{date}.html").read_text(encoding="utf-8")

    def state_path(self, date):
        return self.data_dir / DEFAULT_STATE_DIRNAME / f"{date}.json"

    def state(self, date):
        return json.loads(self.state_path(date).read_text(encoding="utf-8"))

    def write_state(self, date, **fields):
        """手工写一个当天状态文件（模拟上一次运行留下的现场）"""
        path = self.state_path(date)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "date": date}
        payload.update(fields)
        path.write_text(json.dumps(payload), encoding="utf-8")


class FirstRunTest(PipelineTestCase):
    def test_first_run_generates_snapshot_and_report(self):
        records = make_records({"hot/one": 5000, "hot/two": 3000, "new/three": 800})
        code = self.run_cli(records, "2026-08-22", ["--no-email"])

        self.assertEqual(code, 0)

        snapshot = self.snapshot("2026-08-22")
        self.assertEqual(snapshot["date"], "2026-08-22")
        self.assertEqual(snapshot["repository_count"], 3)
        self.assertEqual(snapshot["repositories"]["hot/one"]["stars"], 5000)
        self.assertTrue((self.data_dir / "latest.json").is_file())

        html = self.report_html("2026-08-22")
        self.assertIn("Hot Today Top", html)
        self.assertIn("New &amp; Rising Top", html)
        self.assertIn("首次运行", html)
        self.assertIn("—", html)             # 24h / 7d 显示占位符
        self.assertNotIn("+5,000", html)     # 绝不把总 Star 当增量

    def test_first_run_does_not_send_email_when_disabled(self):
        self.run_cli(make_records({"a/b": 100}), "2026-08-22", ["--no-email"])
        self.sent.assert_not_called()

    def test_no_snapshot_flag(self):
        self.run_cli(make_records({"a/b": 100}), "2026-08-22", ["--no-email", "--no-snapshot"])
        self.assertFalse((self.data_dir / "2026-08-22.json").exists())


class DeltaAcrossDaysTest(PipelineTestCase):
    def test_second_day_computes_24h_delta(self):
        self.run_cli(make_records({"hot/one": 5000}), "2026-08-21", ["--no-email"])
        self.run_cli(make_records({"hot/one": 5300}), "2026-08-22", ["--no-email"])

        html = self.report_html("2026-08-22")
        self.assertIn("+300", html)
        self.assertNotIn("首次运行", html)
        self.assertIn("是否已有 24h 历史：是", html)
        self.assertIn("是否已有 7d 历史：否", html)

    def test_eighth_day_computes_7d_delta(self):
        self.run_cli(make_records({"hot/one": 1000}), "2026-08-15", ["--no-email"])
        self.run_cli(make_records({"hot/one": 1600}), "2026-08-21", ["--no-email"])
        self.run_cli(make_records({"hot/one": 1700}), "2026-08-22", ["--no-email"])

        html = self.report_html("2026-08-22")
        self.assertIn("+100", html)   # 24h: 1700 - 1600
        self.assertIn("+700", html)   # 7d : 1700 - 1000
        self.assertIn("是否已有 7d 历史：是", html)

    def test_missing_yesterday_snapshot_keeps_24h_empty(self):
        self.run_cli(make_records({"hot/one": 1000}), "2026-08-20", ["--no-email"])
        self.run_cli(make_records({"hot/one": 1900}), "2026-08-22", ["--no-email"])
        html = self.report_html("2026-08-22")
        self.assertNotIn("+900", html)
        self.assertIn("是否已有 24h 历史：否", html)

    def test_retention_prunes_old_snapshots(self):
        self.run_cli(make_records({"a/b": 10}), "2026-01-01", ["--no-email"])
        self.run_cli(make_records({"a/b": 20}), "2026-08-22", ["--no-email", "--retention-days", "90"])
        self.assertFalse((self.data_dir / "2026-01-01.json").exists())
        self.assertTrue((self.data_dir / "2026-08-22.json").exists())


class FailureModeTest(PipelineTestCase):
    def test_zero_repositories_fails_without_email(self):
        code = self.run_cli([], "2026-08-22")
        self.assertEqual(code, 1)
        self.sent.assert_not_called()
        self.assertFalse((self.data_dir / "2026-08-22.json").exists())

    def test_zero_repositories_prints_expected_message(self):
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.run_cli([], "2026-08-22")
        self.assertIn("No GitHub repository data could be collected.", buffer.getvalue())

    def test_missing_email_config_fails(self):
        code = self.run_cli(make_records({"a/b": 10}), "2026-08-22", env={})
        self.assertEqual(code, 1)
        self.sent.assert_not_called()
        # 快照仍然已经落盘，明天的增量不受影响
        self.assertTrue((self.data_dir / "2026-08-22.json").is_file())

    def test_email_failure_returns_exit_code_1(self):
        code = self.run_cli(
            make_records({"a/b": 10}), "2026-08-22", env=EMAIL_ENV, send_result=False
        )
        self.assertEqual(code, 1)
        self.sent.assert_called_once()
        self.assertTrue((self.data_dir / "2026-08-22.json").is_file())

    def test_email_success_returns_zero(self):
        code = self.run_cli(
            make_records({"a/b": 10}), "2026-08-22", env=EMAIL_ENV, send_result=True
        )
        self.assertEqual(code, 0)
        self.sent.assert_called_once()


class DailyIdempotencyTest(PipelineTestCase):
    """
    每日幂等：workflow 每天有 4 次兜底 cron，这里验证它们不会互相打架

    对应 data/github_radar/state/YYYY-MM-DD.json（见 github_radar/state.py）
    """

    DATE = "2026-08-23"

    def setUp(self):
        super().setUp()
        self.gh_output_file = Path(self._tmp.name) / "gh_output.txt"
        self.env = dict(EMAIL_ENV, GITHUB_OUTPUT=str(self.gh_output_file))

    def run_day(self, stars, *, send_result=True, extra_args=(), date=None):
        """模拟一次触发；每次清空 GITHUB_OUTPUT，便于只断言本次结果"""
        self.gh_output_file.write_text("", encoding="utf-8")
        return self.run_cli(
            make_records({"hot/one": stars}),
            date or self.DATE,
            extra_args,
            env=self.env,
            send_result=send_result,
        )

    def gh_output(self):
        return self.gh_output_file.read_text(encoding="utf-8")

    def stars_in_snapshot(self, date=None):
        return self.snapshot(date or self.DATE)["repositories"]["hot/one"]["stars"]

    # ------------------------------------------------------------------
    # 当天第一次运行
    # ------------------------------------------------------------------
    def test_first_run_of_day_completes_and_records_state(self):
        code = self.run_day(1000)

        self.assertEqual(code, 0)
        self.sent.assert_called_once()
        self.assertTrue((self.data_dir / f"{self.DATE}.json").is_file())

        state = self.state(self.DATE)
        self.assertTrue(state["snapshot_completed"])
        self.assertTrue(state["email_sent"])
        self.assertIsNotNone(state["completed_at"])
        self.assertEqual(state["runs"], 1)
        self.assertEqual(state["last_status"], "ok")

        output = self.gh_output()
        self.assertIn("status=ok", output)
        self.assertIn("snapshot=written", output)
        self.assertIn("email=sent", output)

    def test_state_file_lives_under_data_dir_state(self):
        self.run_day(1000)
        expected = self.data_dir / DEFAULT_STATE_DIRNAME / f"{self.DATE}.json"
        self.assertTrue(expected.is_file())

    # ------------------------------------------------------------------
    # 当天第二次自动运行应跳过
    # ------------------------------------------------------------------
    def test_second_scheduled_run_same_day_skips(self):
        self.assertEqual(self.run_day(1000), 0)

        code = self.run_day(9999)

        self.assertEqual(code, 0)           # 正常退出，不是失败
        self.sent.assert_not_called()       # 不重复发信
        self.collected.assert_not_called()  # 连采集都不做，省 API 配额和时间

        output = self.gh_output()
        self.assertIn("status=skipped", output)
        self.assertIn("email=already_sent", output)
        self.assertIn(f"snapshot_date={self.DATE}", output)

    def test_skipped_run_changes_nothing_on_disk(self):
        """跳过的触发不写任何文件——workflow 那边才会是 nothing to commit"""
        self.run_day(1000)
        snapshot_before = (self.data_dir / f"{self.DATE}.json").read_bytes()
        state_before = self.state_path(self.DATE).read_bytes()

        self.run_day(9999)

        self.assertEqual((self.data_dir / f"{self.DATE}.json").read_bytes(), snapshot_before)
        self.assertEqual(self.state_path(self.DATE).read_bytes(), state_before)

    def test_four_triggers_send_exactly_one_email(self):
        """一天 4 次兜底 cron 全部跑一遍：只发一封信、只留一个正式快照"""
        total_sends = 0
        for index in range(4):
            self.assertEqual(self.run_day(1000 + index * 100), 0)
            total_sends += self.sent.call_count

        self.assertEqual(total_sends, 1)
        self.assertEqual(self.state(self.DATE)["runs"], 1)
        self.assertEqual(self.stars_in_snapshot(), 1000)

    # ------------------------------------------------------------------
    # 首次邮件失败、第二次成功
    # ------------------------------------------------------------------
    def test_email_retried_on_next_trigger_after_failure(self):
        self.assertEqual(self.run_day(1000, send_result=False), 1)

        state = self.state(self.DATE)
        self.assertTrue(state["snapshot_completed"])
        self.assertFalse(state["email_sent"])
        self.assertIsNone(state["completed_at"])
        self.assertEqual(state["last_status"], "email_failed")

        self.assertEqual(self.run_day(1500, send_result=True), 0)
        # 关键：不能因为快照已经存在就误判为「今天做完了」
        self.sent.assert_called_once()

        state = self.state(self.DATE)
        self.assertTrue(state["email_sent"])
        self.assertIsNotNone(state["completed_at"])
        self.assertEqual(state["runs"], 2)

    def test_retry_keeps_the_official_snapshot(self):
        """补发邮件的那次不覆盖快照：一天最终只有一个正式版本"""
        self.run_day(1000, send_result=False)
        self.run_day(1500, send_result=True)

        self.assertEqual(self.stars_in_snapshot(), 1000)
        self.assertEqual(
            sorted(p.name for p in self.data_dir.glob("2026-*.json")),
            [f"{self.DATE}.json"],
        )
        self.assertIn("snapshot=kept", self.gh_output())

    def test_third_trigger_skips_once_retry_succeeded(self):
        self.run_day(1000, send_result=False)
        self.run_day(1500, send_result=True)

        self.assertEqual(self.run_day(2000), 0)
        self.sent.assert_not_called()
        self.assertIn("status=skipped", self.gh_output())

    def test_repeated_email_failures_keep_retrying(self):
        for _ in range(3):
            self.assertEqual(self.run_day(1000, send_result=False), 1)
            self.sent.assert_called_once()

        self.assertEqual(self.state(self.DATE)["runs"], 3)
        self.assertEqual(self.stars_in_snapshot(), 1000)

    # ------------------------------------------------------------------
    # 状态文件读写（CLI 侧）
    # ------------------------------------------------------------------
    def test_hand_written_state_is_honoured(self):
        """外部写入的状态文件也能正确驱动决策"""
        self.run_day(1000, send_result=False)
        self.write_state(self.DATE, snapshot_completed=True, email_sent=False, runs=1)

        self.assertEqual(self.run_day(1200), 0)

        self.sent.assert_called_once()
        self.assertEqual(self.stars_in_snapshot(), 1000)   # 快照仍是正式版本

    def test_corrupted_state_falls_back_to_running(self):
        """状态文件损坏时宁可多发一封，也不要让当天彻底收不到日报"""
        self.run_day(1000)
        self.state_path(self.DATE).write_text("{broken", encoding="utf-8")

        self.assertEqual(self.run_day(1100), 0)
        self.sent.assert_called_once()

    def test_old_state_files_are_pruned_with_snapshots(self):
        self.run_day(10, date="2026-01-01")
        self.run_day(20, extra_args=["--retention-days", "90"])

        self.assertFalse(self.state_path("2026-01-01").exists())
        self.assertTrue(self.state_path(self.DATE).exists())

    def test_collection_failure_does_not_block_next_trigger(self):
        self.assertEqual(self.run_cli([], self.DATE, env=self.env), 1)

        state = self.state(self.DATE)
        self.assertFalse(state["snapshot_completed"])
        self.assertFalse(state["email_sent"])
        self.assertEqual(state["last_status"], "failed")

        self.assertEqual(self.run_day(1000), 0)
        self.sent.assert_called_once()

    # ------------------------------------------------------------------
    # force_run
    # ------------------------------------------------------------------
    def test_force_run_reruns_a_completed_day(self):
        self.run_day(1000)

        code = self.run_day(1800, extra_args=["--force-run"])

        self.assertEqual(code, 0)
        self.sent.assert_called_once()
        self.assertEqual(self.stars_in_snapshot(), 1800)   # 快照被刷新
        self.assertEqual(self.state(self.DATE)["runs"], 2)

        output = self.gh_output()
        self.assertIn("status=ok", output)
        self.assertIn("snapshot=written", output)
        self.assertIn("email=sent", output)

    def test_force_run_still_leaves_one_snapshot_per_day(self):
        self.run_day(1000)
        self.run_day(1800, extra_args=["--force-run"])

        self.assertEqual(
            sorted(p.name for p in self.data_dir.glob("2026-*.json")),
            [f"{self.DATE}.json"],
        )

    def test_next_scheduled_trigger_skips_again_after_force_run(self):
        self.run_day(1000)
        self.run_day(1800, extra_args=["--force-run"])

        self.assertEqual(self.run_day(2500), 0)
        self.sent.assert_not_called()
        self.assertIn("status=skipped", self.gh_output())

    # ------------------------------------------------------------------
    # --no-state / --no-email 的交互
    # ------------------------------------------------------------------
    def test_no_state_disables_idempotency(self):
        self.run_day(1000, extra_args=["--no-state"])
        self.assertFalse(self.state_path(self.DATE).exists())

        code = self.run_day(1700, extra_args=["--no-state"])

        self.assertEqual(code, 0)
        self.sent.assert_called_once()
        self.assertEqual(self.stars_in_snapshot(), 1700)

    def test_no_email_run_skips_once_snapshot_is_done(self):
        self.run_day(1000, extra_args=["--no-email"])
        state = self.state(self.DATE)
        self.assertTrue(state["snapshot_completed"])
        self.assertFalse(state["email_sent"])

        self.assertEqual(self.run_day(2000, extra_args=["--no-email"]), 0)
        self.collected.assert_not_called()

    def test_email_still_sent_after_a_no_email_run(self):
        """调试用的 --no-email 运行不能把当天的日报「吃掉」"""
        self.run_day(1000, extra_args=["--no-email"])

        self.assertEqual(self.run_day(1000), 0)

        self.sent.assert_called_once()
        self.assertTrue(self.state(self.DATE)["email_sent"])


class GithubOutputTest(PipelineTestCase):
    def test_writes_github_output_when_available(self):
        output_file = Path(self._tmp.name) / "gh_output.txt"
        env = {"GITHUB_OUTPUT": str(output_file)}
        code = self.run_cli(make_records({"a/b": 10}), "2026-08-22", ["--no-email"], env=env)
        self.assertEqual(code, 0)
        content = output_file.read_text(encoding="utf-8")
        self.assertIn("status=ok", content)
        self.assertIn("snapshot_date=2026-08-22", content)
        self.assertIn("repo_count=1", content)


if __name__ == "__main__":
    unittest.main()
