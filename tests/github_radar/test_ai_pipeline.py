# coding=utf-8
"""
AI 端到端测试（全部 mock，不消耗真实 DeepSeek 配额）

覆盖规格 §35 的 Mock E2E::

    100 candidates → Hot20 / Rising10 / Profile matches
    → 30 AI candidates → 部分 cache → DeepSeek batch
    → For You Top10 → daily synthesis → HTML email

以及 §31 的降级矩阵：AI 全部失败 / 部分失败 / 未启用 / skip_ai
时，基础日报都必须照常生成。
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from github_radar import cli
from github_radar.ai.cache import StaticAnalysisCache, default_cache_dir
from github_radar.ai.client import DeepSeekClient
from github_radar.ai.config import load_ai_config
from github_radar.ai.pipeline import run_ai_enhancement
from github_radar.ai.profile import load_profile
from github_radar.ai.result import STATUS_FAILED, STATUS_OK, STATUS_PARTIAL
from github_radar.ai.schemas import parse_repo_analysis
from github_radar.collector import CollectionResult
from github_radar.models import SOURCE_SEARCH_NEW, SOURCE_TRENDING, RepoRecord
from github_radar.report import ReportContext, ReportSummary, render_html

from .ai_helpers import (
    FakeChatSession,
    FakeReadmeClient,
    analysis_payload,
    chat_response,
    error_response,
    make_scored,
    no_sleep,
    synthesis_payload,
)
from .test_ai_profile import REAL_PROFILE

SECRET = "sk-e2e-test-key-0123456789"

AI_ENV = {
    "GITHUB_RADAR_AI_ENABLED": "true",
    "DEEPSEEK_API_KEY": SECRET,
    "DEEPSEEK_MODEL": "deepseek-v4-flash",
}

# CLI 集成测试用的邮箱配置（_send 已被打桩，这里只是让配置检查通过）
EMAIL_ENV = {
    "EMAIL_FROM": "a@qq.com",
    "EMAIL_PASSWORD": "app-password",
    "EMAIL_TO": "b@qq.com",
}


def build_pool(count=100):
    """100 个候选：前 20 热度最高，另外埋 3 个 profile 强相关但热度很低的项目"""
    pool = [make_scored(f"owner/repo{i:03d}", score=100.0 - i * 0.6) for i in range(count)]
    for index, name in enumerate(("lab/metalens-designer", "acme/coding-agent", "who/mcp-toolkit")):
        pool.append(
            make_scored(
                name,
                score=12.0 - index,
                description={
                    "lab/metalens-designer": "metasurface inverse design with FDTD and RCWA",
                    "acme/coding-agent": "an autonomous coding agent with MCP tool use",
                    "who/mcp-toolkit": "agent framework for computer use and tool use",
                }[name],
                topics=["ai-agent"] if index else ["photonics"],
            )
        )
    return pool


def responder(names_by_call=None, *, synthesis=True):
    """
    构造一个「按请求内容回答」的假会话

    仓库分析请求：把 prompt 里出现的 full_name 都回一遍
    synthesis 请求：返回趋势总结
    """
    def handler(url, payload):
        text = "\n".join(str(m.get("content") or "") for m in payload.get("messages", []))
        if "今天 GitHub Daily Radar 的结构化汇总" in text or "hot_today" in text:
            if not synthesis:
                return error_response(500)
            return chat_response(synthesis_payload(), prompt_tokens=800, completion_tokens=120)

        names = [line.split("full_name: ", 1)[1].strip()
                 for line in text.splitlines() if line.startswith("full_name: ")]
        return chat_response(
            {"repositories": [analysis_payload(name, relevance_score=_relevance(name))
                              for name in names]},
            prompt_tokens=5000,
            completion_tokens=600,
        )

    return handler


def _relevance(name: str) -> int:
    """让 profile 强相关的项目拿到高 relevance，便于验证 For You 排序"""
    if name in ("lab/metalens-designer", "acme/coding-agent", "who/mcp-toolkit"):
        return 96
    return 40


class PipelineBaseTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name) / "data"
        self.pool = build_pool()
        self.hot = self.pool[:20]
        self.rising = self.pool[20:30]

    def tearDown(self):
        self._tmp.cleanup()

    def client(self, session, **kwargs):
        return DeepSeekClient(
            SECRET,
            model="deepseek-v4-flash",
            api_base="https://api.deepseek.com",
            session=session,
            sleep_func=no_sleep,
            **kwargs,
        )

    def run_pipeline(self, session, *, github_client=None, force=False, env=None, **kwargs):
        config = load_ai_config(env or AI_ENV)
        return run_ai_enhancement(
            self.pool,
            self.hot,
            self.rising,
            config=config,
            data_dir=self.data_dir,
            date="2026-08-23",
            github_client=github_client or FakeReadmeClient(
                {item.record.full_name: "# docs\nSome documentation." for item in self.pool}
            ),
            profile_path=REAL_PROFILE,
            deepseek_client=self.client(session) if session is not None else None,
            env={},
            force=force,
            sleep_func=no_sleep,
            **kwargs,
        )


class MockEndToEndTest(PipelineBaseTest):
    """35. Mock E2E"""

    def test_full_flow(self):
        session = FakeChatSession(responder())
        result = self.run_pipeline(session)

        # --- candidate limit ---
        self.assertEqual(result.usage.repositories_analyzed, 30)
        self.assertEqual(result.status, STATUS_OK)

        # --- 请求次数：5 批分析 + 1 次 synthesis ---
        self.assertEqual(session.call_count, 6)
        self.assertEqual(result.usage.requests, 6)

        # --- For You：榜首必须是 profile 强相关的项目，而不是热度最高的项目 ---
        self.assertEqual(len(result.for_you), 10)
        self.assertIn(
            result.for_you[0].full_name,
            {"lab/metalens-designer", "acme/coding-agent", "who/mcp-toolkit"},
        )
        self.assertNotIn(result.for_you[0].full_name, {i.record.full_name for i in self.hot})

        # --- synthesis ---
        self.assertTrue(result.synthesis_available)
        self.assertIn("Coding Agent", result.synthesis.headline)

        # --- token / cost ---
        self.assertEqual(result.usage.prompt_tokens, 5 * 5000 + 800)
        self.assertEqual(result.usage.completion_tokens, 5 * 600 + 120)
        self.assertGreater(result.cost.total_cost, 0)

        # --- HTML ---
        html = render_html(
            ReportContext(
                summary=ReportSummary(date="2026-08-23"),
                hot=self.hot,
                new_rising=self.rising,
                ai=result,
            )
        )
        self.assertIn("📡 今日 GitHub 技术信号", html)
        self.assertIn("🎯 For You Top10", html)
        self.assertIn("🔥 Hot Today Top", html)
        self.assertIn("📊 AI 使用情况", html)
        self.assertIn("acme/coding-agent", html)

    def test_low_heat_profile_match_reaches_for_you(self):
        """Hot20 之外的强相关项目必须能进 AI 分析并出现在 For You"""
        result = self.run_pipeline(FakeChatSession(responder()))
        names = [entry.full_name for entry in result.for_you]

        for name in ("lab/metalens-designer", "acme/coding-agent", "who/mcp-toolkit"):
            with self.subTest(repo=name):
                self.assertIn(name, names)
                self.assertNotIn(name, [item.record.full_name for item in self.hot])

    def test_readme_is_only_fetched_for_ai_candidates(self):
        github = FakeReadmeClient({item.record.full_name: "# docs" for item in self.pool})
        self.run_pipeline(FakeChatSession(responder()), github_client=github)
        self.assertEqual(len(github.requests), 30)
        self.assertLess(len(github.requests), len(self.pool))

    def test_partial_cache_reduces_work(self):
        cache = StaticAnalysisCache(default_cache_dir(self.data_dir), model="deepseek-v4-flash")
        for item in self.hot[:8]:
            analysis = parse_repo_analysis(
                analysis_payload(item.record.full_name), full_name=item.record.full_name
            )
            cache.put(analysis, item.record)

        github = FakeReadmeClient({item.record.full_name: "# docs" for item in self.pool})
        result = self.run_pipeline(FakeChatSession(responder()), github_client=github)

        self.assertEqual(result.usage.cache_hits, 8)
        # 缓存命中的仓库不再读 README
        self.assertEqual(len(github.requests), 22)
        self.assertEqual(result.usage.repositories_analyzed, 30)

    def test_daily_result_is_reused_by_the_next_trigger(self):
        first = FakeChatSession(responder())
        self.run_pipeline(first)

        second = FakeChatSession(responder())
        result = self.run_pipeline(second)

        self.assertEqual(second.call_count, 0)      # 一次 API 都不再打
        self.assertTrue(result.reused)
        self.assertEqual(result.usage.repositories_analyzed, 30)

    def test_force_recomputes(self):
        self.run_pipeline(FakeChatSession(responder()))
        second = FakeChatSession(responder())
        result = self.run_pipeline(second, force=True)

        self.assertGreater(second.call_count, 0)
        self.assertFalse(result.reused)


class DegradationTest(PipelineBaseTest):
    """31. AI 失败必须降级"""

    def test_all_batches_failing_yields_failed_status(self):
        session = FakeChatSession([error_response(503)])
        result = self.run_pipeline(session)

        self.assertEqual(result.status, STATUS_FAILED)
        self.assertEqual(result.analyses, {})
        self.assertEqual(result.for_you, [])
        self.assertFalse(result.synthesis_available)

    def test_synthesis_failure_is_partial_not_fatal(self):
        session = FakeChatSession(responder(synthesis=False))
        result = self.run_pipeline(session)

        self.assertEqual(result.status, STATUS_PARTIAL)
        self.assertTrue(result.has_analyses)
        self.assertFalse(result.synthesis_available)
        self.assertIn("AI 趋势总结今日不可用。", result.notes)

    def test_disabled_without_api_key(self):
        result = self.run_pipeline(
            None, env={"GITHUB_RADAR_AI_ENABLED": "true"}
        )
        self.assertFalse(result.enabled)
        self.assertIn("DEEPSEEK_API_KEY", result.disabled_reason)

    def test_disabled_when_flag_is_off(self):
        result = self.run_pipeline(None, env={"DEEPSEEK_API_KEY": SECRET})
        self.assertFalse(result.enabled)

    def test_no_github_client_still_analyses(self):
        session = FakeChatSession(responder())
        result = self.run_pipeline(session, github_client=None)
        self.assertTrue(result.has_analyses)


# ----------------------------------------------------------------------
# CLI 级别：AI 失败时基础邮件仍然发送
# ----------------------------------------------------------------------
def make_records(names, stars=1000):
    records = []
    for index, name in enumerate(names, 1):
        records.append(
            RepoRecord(
                full_name=name,
                stars=stars + index,
                forks=10,
                language="Python",
                description=f"{name} description",
                created_at="2026-08-01T00:00:00Z",
                pushed_at="2026-08-22T00:00:00Z",
                trending_rank=index if index <= 2 else None,
                sources=[SOURCE_TRENDING if index <= 2 else SOURCE_SEARCH_NEW],
                api_enriched=True,
            )
        )
    return records


class CliIntegrationTest(unittest.TestCase):
    """26. AI 失败基础邮件仍生成 / 30. skip_ai"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.data_dir = root / "data"
        self.output_dir = root / "output"
        self.gh_output = root / "gh_output.txt"
        self.gh_output.write_text("", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def run_cli(self, *, env, session=None, extra_args=(), records=None, date="2026-08-23"):
        records = records or make_records(["hot/one", "hot/two", "acme/coding-agent"])
        collection = CollectionResult(
            repositories=records,
            trending_count=2,
            search_new_count=len(records),
            trending_ok=True,
            search_ok=True,
        )
        args = [
            "--date", date,
            "--timezone", "Asia/Shanghai",
            "--data-dir", str(self.data_dir),
            "--output-dir", str(self.output_dir),
            *extra_args,
        ]
        full_env = dict(EMAIL_ENV, **env)
        full_env["GITHUB_OUTPUT"] = str(self.gh_output)

        def client_factory(*a, **kw):
            return DeepSeekClient(
                SECRET,
                model=kw.get("model", "deepseek-v4-flash"),
                api_base="https://api.deepseek.com",
                session=session,
                sleep_func=no_sleep,
            )

        with mock.patch.object(cli, "GitHubAPIClient") as client_cls, mock.patch.object(
            cli, "collect_candidates", return_value=collection
        ), mock.patch.object(cli, "_send", return_value=True) as send, mock.patch.dict(
            cli.os.environ, full_env, clear=True
        ), mock.patch(
            "github_radar.ai.pipeline.DeepSeekClient", side_effect=client_factory
        ):
            client_cls.return_value.authenticated = True
            client_cls.return_value.rate_limited = False
            client_cls.return_value.describe.return_value = "fake client"
            client_cls.return_value.get_readme_text.return_value = "# docs"
            code = cli.main(args)

        self.sent = send
        return code

    def html(self, date="2026-08-23"):
        return (self.output_dir / f"{date}.html").read_text(encoding="utf-8")

    def outputs(self):
        return self.gh_output.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    def test_ai_enabled_produces_enhanced_report(self):
        code = self.run_cli(env=dict(AI_ENV), session=FakeChatSession(responder()))
        self.assertEqual(code, 0)

        html = self.html()
        self.assertIn("📊 AI 使用情况", html)
        self.assertIn("🎯 For You", html)
        self.assertIn("ai=ok", self.outputs())

    def test_ai_total_failure_still_sends_the_base_email(self):
        code = self.run_cli(env=dict(AI_ENV), session=FakeChatSession([error_response(500)]))

        self.assertEqual(code, 0)
        self.sent.assert_called_once()          # 邮件照常发出
        html = self.html()
        self.assertIn("🔥 Hot Today Top", html)  # 基础榜单完好
        self.assertIn("hot/one", html)
        self.assertIn("ai=failed", self.outputs())

    def test_ai_failure_does_not_break_the_snapshot(self):
        self.run_cli(env=dict(AI_ENV), session=FakeChatSession([error_response(500)]))
        snapshot = json.loads(
            (self.data_dir / "2026-08-23.json").read_text(encoding="utf-8")
        )
        self.assertEqual(snapshot["repository_count"], 3)
        self.assertEqual(snapshot["repositories"]["hot/one"]["stars"], 1001)

    def test_missing_api_key_disables_ai_without_failing(self):
        code = self.run_cli(env={"GITHUB_RADAR_AI_ENABLED": "true"})

        self.assertEqual(code, 0)
        self.sent.assert_called_once()
        self.assertNotIn("AI 使用情况", self.html())
        self.assertIn("ai=disabled", self.outputs())

    def test_skip_ai_makes_no_api_calls(self):
        session = FakeChatSession(responder())
        code = self.run_cli(env=dict(AI_ENV), session=session, extra_args=["--skip-ai"])

        self.assertEqual(code, 0)
        self.assertEqual(session.call_count, 0)
        self.assertNotIn("AI 使用情况", self.html())
        self.assertIn("ai=disabled", self.outputs())

    def test_ai_off_by_default(self):
        session = FakeChatSession(responder())
        code = self.run_cli(env={"DEEPSEEK_API_KEY": SECRET}, session=session)

        self.assertEqual(code, 0)
        self.assertEqual(session.call_count, 0)
        self.assertNotIn("AI 使用情况", self.html())

    def test_unexpected_ai_exception_is_contained(self):
        with mock.patch.object(cli, "run_ai_enhancement", side_effect=RuntimeError("boom")):
            code = self.run_cli(env=dict(AI_ENV), session=FakeChatSession(responder()))

        self.assertEqual(code, 0)
        self.sent.assert_called_once()
        self.assertIn("🔥 Hot Today Top", self.html())

    def test_heat_score_is_untouched_by_ai(self):
        """AI 开与关，Hot 榜的顺序与分数必须完全一致"""
        self.run_cli(env=dict(AI_ENV), session=FakeChatSession(responder()), extra_args=["--skip-ai"])
        base = self.html()

        self.setUp()
        self.run_cli(env=dict(AI_ENV), session=FakeChatSession(responder()))
        enhanced = self.html()

        for name in ("hot/one", "hot/two", "acme/coding-agent"):
            with self.subTest(repo=name):
                self.assertIn(name, base)
                self.assertIn(name, enhanced)
        self.assertEqual(
            _heat_scores(base), _heat_scores(enhanced), "AI 不允许改变 Heat Score"
        )


def _heat_scores(html: str):
    import re

    return re.findall(r"Heat Score: <strong[^>]*>([\d.]+)</strong>", html)


if __name__ == "__main__":
    unittest.main()
