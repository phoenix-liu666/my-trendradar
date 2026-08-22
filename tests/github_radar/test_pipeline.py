# coding=utf-8
"""
端到端流程测试（CLI）

用假的采集层替换网络访问，验证：
- 首日无历史也能正常产出日报与快照
- 第二天能算出 24h 增量
- 第七天之后能算出 7d 增量
- 0 个仓库时明确失败且不发邮件
- 邮件失败时退出码为 1
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from github_radar import cli
from github_radar.collector import CollectionResult
from github_radar.models import SOURCE_SEARCH_NEW, SOURCE_TRENDING, RepoRecord


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
        ), mock.patch.object(cli, "_send", return_value=send_result) as send, mock.patch.dict(
            cli.os.environ, env, clear=True
        ):
            client_cls.return_value.authenticated = True
            client_cls.return_value.rate_limited = False
            client_cls.return_value.describe.return_value = "fake client"
            code = cli.main(args)
        self.sent = send
        return code

    def snapshot(self, date):
        return json.loads((self.data_dir / f"{date}.json").read_text(encoding="utf-8"))

    def report_html(self, date):
        return (self.output_dir / f"{date}.html").read_text(encoding="utf-8")


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
        env = {
            "EMAIL_FROM": "a@qq.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_TO": "b@qq.com",
        }
        code = self.run_cli(make_records({"a/b": 10}), "2026-08-22", env=env, send_result=False)
        self.assertEqual(code, 1)
        self.sent.assert_called_once()
        self.assertTrue((self.data_dir / "2026-08-22.json").is_file())

    def test_email_success_returns_zero(self):
        env = {
            "EMAIL_FROM": "a@qq.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_TO": "b@qq.com",
        }
        code = self.run_cli(make_records({"a/b": 10}), "2026-08-22", env=env, send_result=True)
        self.assertEqual(code, 0)
        self.sent.assert_called_once()


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
