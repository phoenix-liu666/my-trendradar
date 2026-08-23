# coding=utf-8
"""批量分析编排测试（batch / 部分失败 / 全部失败 / README / 输入上限）"""

import tempfile
import unittest
from pathlib import Path

from github_radar.ai.analyzer import analyze_repositories, chunk
from github_radar.ai.cache import StaticAnalysisCache
from github_radar.ai.client import DeepSeekClient
from github_radar.ai.config import AIConfig
from github_radar.ai.profile import KeywordMatch, load_profile
from github_radar.ai.readme import fetch_readmes
from github_radar.ai.schemas import NO_EXTERNAL_EVIDENCE_TEXT, parse_repo_analysis
from github_radar.ai.selector import AICandidate

from .ai_helpers import (
    FakeChatSession,
    FakeReadmeClient,
    analysis_payload,
    batch_payload,
    chat_response,
    error_response,
    make_scored,
    no_sleep,
)
from .test_ai_profile import REAL_PROFILE

SECRET = "sk-test-key-abcdefghijklmnop"


def make_candidates(count: int, *, cached: int = 0):
    """前 ``cached`` 个带缓存（走 daily 批次），其余走 full 批次"""
    candidates = []
    for index in range(count):
        candidate = AICandidate(scored=make_scored(f"owner/repo{index:02d}"))
        candidate.keyword = KeywordMatch(score=50.0)
        if index < cached:
            candidate.cached_static = {
                "summary_zh": "缓存的一句话",
                "problem": "缓存的问题",
                "category": "Coding Agent",
                "tech_stack": ["Python"],
                "use_cases": ["演示"],
                "maturity": "growing",
            }
        candidates.append(candidate)
    return candidates


def make_client(session, **kwargs):
    return DeepSeekClient(
        SECRET,
        model="deepseek-v4-flash",
        api_base="https://api.deepseek.com",
        session=session,
        sleep_func=no_sleep,
        **kwargs,
    )


def config(**overrides) -> AIConfig:
    base = dict(enabled=True, api_key=SECRET, model="deepseek-v4-flash", batch_size=6)
    base.update(overrides)
    return AIConfig(**base)


class ChunkTest(unittest.TestCase):
    def test_chunks_are_bounded(self):
        self.assertEqual(len(chunk(list(range(30)), 6)), 5)
        self.assertEqual(len(chunk(list(range(30)), 8)), 4)
        self.assertEqual(chunk([], 6), [])
        self.assertEqual(chunk([1], 0), [[1]])


class BatchingTest(unittest.TestCase):
    """16. batch 调用：30 个仓库 → 5 个请求，而不是 30 个"""

    def setUp(self):
        self.profile = load_profile(REAL_PROFILE)

    def run_batches(self, candidates, batch_size=6):
        session = FakeChatSession(
            lambda url, payload: chat_response(
                batch_payload(_names_in(payload, candidates))
            )
        )
        client = make_client(session)
        outcome = analyze_repositories(
            client,
            candidates,
            self.profile,
            config=config(batch_size=batch_size),
            sleep_func=no_sleep,
        )
        return outcome, session

    def test_thirty_repos_take_five_requests(self):
        outcome, session = self.run_batches(make_candidates(30))
        self.assertEqual(session.call_count, 5)
        self.assertEqual(outcome.batches, 5)
        self.assertEqual(outcome.analyzed_count, 30)

    def test_batch_size_eight_takes_four_requests(self):
        _, session = self.run_batches(make_candidates(30), batch_size=8)
        self.assertEqual(session.call_count, 4)

    def test_cached_and_uncached_are_batched_separately(self):
        outcome, session = self.run_batches(make_candidates(30, cached=6))
        # 24 full → 4 批；6 daily → 1 批
        self.assertEqual(session.call_count, 5)
        self.assertEqual(outcome.analyzed_count, 30)

    def test_daily_batch_uses_cached_static_fields(self):
        candidates = make_candidates(2, cached=2)
        outcome, _ = self.run_batches(candidates)
        analysis = outcome.analyses["owner/repo00"]
        self.assertEqual(analysis.summary_zh, "缓存的一句话")
        self.assertEqual(analysis.category, "Coding Agent")
        self.assertTrue(analysis.from_cache)

    def test_usage_is_accumulated(self):
        outcome, _ = self.run_batches(make_candidates(12))
        self.assertEqual(outcome.usage.requests, 2)
        self.assertEqual(outcome.usage.prompt_tokens, 2000)
        self.assertEqual(outcome.usage.repositories_analyzed, 12)

    def test_empty_candidates_make_no_requests(self):
        session = FakeChatSession([chat_response({})])
        outcome = analyze_repositories(
            make_client(session), [], self.profile, config=config()
        )
        self.assertEqual(session.call_count, 0)
        self.assertEqual(outcome.batches, 0)


def _names_in(payload, candidates):
    """从请求 payload 里找出这一批包含哪些仓库"""
    text = "\n".join(str(m.get("content") or "") for m in payload.get("messages", []))
    return [c.full_name for c in candidates if f"full_name: {c.full_name}" in text]


class PartialFailureTest(unittest.TestCase):
    """24. batch partial failure"""

    def setUp(self):
        self.profile = load_profile(REAL_PROFILE)
        self.candidates = make_candidates(12)

    def test_one_failed_batch_keeps_the_others(self):
        state = {"calls": 0}

        def handler(url, payload):
            state["calls"] += 1
            if state["calls"] == 1:
                return error_response(500)
            return chat_response(batch_payload(_names_in(payload, self.candidates)))

        session = FakeChatSession(handler)
        outcome = analyze_repositories(
            make_client(session, max_retries=0),
            self.candidates,
            self.profile,
            config=config(),
            sleep_func=no_sleep,
        )

        self.assertEqual(outcome.batches, 2)
        self.assertEqual(outcome.failed_batches, 1)
        self.assertEqual(outcome.analyzed_count, 6)
        self.assertFalse(outcome.all_failed)

    def test_repos_missing_from_the_response_are_skipped(self):
        session = FakeChatSession(
            lambda url, payload: chat_response(batch_payload(["owner/repo00"]))
        )
        outcome = analyze_repositories(
            make_client(session), self.candidates, self.profile, config=config(), sleep_func=no_sleep
        )
        # 两个 batch 都只回了 owner/repo00，第二批里根本没有这个仓库 → 只有 1 条结果，
        # 其余 11 个仓库退回基础数据
        self.assertEqual(outcome.analyzed_count, 1)
        self.assertEqual(outcome.failed_batches, 0)

    def test_malformed_batch_falls_back_to_base_data(self):
        session = FakeChatSession([chat_response("not json at all")])
        outcome = analyze_repositories(
            make_client(session),
            make_candidates(6),
            self.profile,
            config=config(),
            sleep_func=no_sleep,
        )
        self.assertEqual(outcome.failed_batches, 1)
        self.assertEqual(outcome.analyzed_count, 0)


class AllFailureTest(unittest.TestCase):
    """25. AI 全部失败"""

    def setUp(self):
        self.profile = load_profile(REAL_PROFILE)

    def test_every_batch_failing_yields_no_analyses(self):
        session = FakeChatSession([error_response(503)])
        outcome = analyze_repositories(
            make_client(session, max_retries=0),
            make_candidates(12),
            self.profile,
            config=config(),
            sleep_func=no_sleep,
        )
        self.assertTrue(outcome.all_failed)
        self.assertEqual(outcome.analyzed_count, 0)
        self.assertEqual(outcome.failed_batches, 2)

    def test_timeout_on_every_batch_is_safe(self):
        from .ai_helpers import RaisingChatSession

        outcome = analyze_repositories(
            make_client(RaisingChatSession(), max_retries=0),
            make_candidates(6),
            self.profile,
            config=config(),
            sleep_func=no_sleep,
        )
        self.assertEqual(outcome.analyzed_count, 0)


class HallucinationGuardIntegrationTest(unittest.TestCase):
    """28. hallucination guard（编排层）"""

    def test_external_claims_are_scrubbed_before_reaching_the_report(self):
        candidates = make_candidates(1)
        payload = {
            "repositories": [
                analysis_payload(
                    "owner/repo00",
                    why_hot={
                        "summary": "被谷歌采用并获得大额融资，因此爆火。",
                        "confidence": "high",
                        "evidence": ["某内部人士透露"],
                    },
                )
            ]
        }
        session = FakeChatSession([chat_response(payload)])
        outcome = analyze_repositories(
            make_client(session),
            candidates,
            load_profile(REAL_PROFILE),
            config=config(),
            sleep_func=no_sleep,
        )
        analysis = outcome.analyses["owner/repo00"]
        self.assertEqual(analysis.why_hot.summary, NO_EXTERNAL_EVIDENCE_TEXT)
        self.assertEqual(analysis.why_hot.confidence, "low")
        self.assertEqual(analysis.why_hot.evidence, [])


class InputBudgetTest(unittest.TestCase):
    """28/24. 输入字符硬上限：达到就停，已有结果继续用"""

    def test_analysis_stops_when_the_budget_is_exhausted(self):
        candidates = make_candidates(18)
        session = FakeChatSession(
            lambda url, payload: chat_response(batch_payload(_names_in(payload, candidates)))
        )
        outcome = analyze_repositories(
            make_client(session),
            candidates,
            load_profile(REAL_PROFILE),
            config=config(max_input_chars=6000),
            sleep_func=no_sleep,
        )
        self.assertTrue(outcome.stopped_early)
        self.assertLess(outcome.batches, 3)
        self.assertGreater(outcome.analyzed_count, 0)   # 已有结果照常使用


class CacheWriteTest(unittest.TestCase):
    """完整分析的结果会写进静态缓存；daily 批次不会重复写"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = StaticAnalysisCache(Path(self._tmp.name), model="deepseek-v4-flash")

    def tearDown(self):
        self._tmp.cleanup()

    def test_full_batch_results_are_cached(self):
        candidates = make_candidates(2)
        session = FakeChatSession(
            lambda url, payload: chat_response(batch_payload(_names_in(payload, candidates)))
        )
        analyze_repositories(
            make_client(session),
            candidates,
            load_profile(REAL_PROFILE),
            config=config(),
            cache=self.cache,
            sleep_func=no_sleep,
        )
        self.assertEqual(self.cache.stats.writes, 2)
        self.assertTrue(self.cache.path_for("owner/repo00").is_file())


class ReadmeFetchTest(unittest.TestCase):
    """7. README truncation / 只给需要完整分析的候选取 README"""

    def test_readme_is_fetched_only_for_given_candidates(self):
        candidates = make_candidates(3)
        client = FakeReadmeClient({c.full_name: "# docs" for c in candidates})
        stats = fetch_readmes(client, candidates[:2], sleep_func=no_sleep)

        self.assertEqual(client.requests, ["owner/repo00", "owner/repo01"])
        self.assertEqual(stats.fetched, 2)
        self.assertEqual(candidates[2].readme, "")

    def test_long_readme_is_truncated_to_the_limit(self):
        candidates = make_candidates(1)
        client = FakeReadmeClient({"owner/repo00": "x" * 50_000})
        stats = fetch_readmes(client, candidates, max_chars=6000, sleep_func=no_sleep)

        self.assertEqual(stats.truncated, 1)
        self.assertLessEqual(len(candidates[0].readme), 6000)

    def test_missing_readme_is_not_fatal(self):
        candidates = make_candidates(2)
        client = FakeReadmeClient({"owner/repo00": "# docs"}, failing=["owner/repo01"])
        stats = fetch_readmes(client, candidates, sleep_func=no_sleep)

        self.assertEqual(stats.fetched, 1)
        self.assertEqual(stats.missing, 1)
        self.assertTrue(candidates[0].readme_ok)
        self.assertFalse(candidates[1].readme_ok)

    def test_exception_is_swallowed(self):
        candidates = make_candidates(1)
        client = FakeReadmeClient(raising=["owner/repo00"])
        stats = fetch_readmes(client, candidates, sleep_func=no_sleep)
        self.assertEqual(stats.missing, 1)

    def test_total_budget_stops_further_requests(self):
        candidates = make_candidates(5)
        client = FakeReadmeClient({c.full_name: "y" * 5000 for c in candidates})
        stats = fetch_readmes(
            client, candidates, max_chars=6000, total_budget=8000, sleep_func=no_sleep
        )
        self.assertTrue(stats.budget_exhausted)
        self.assertLessEqual(stats.total_chars, 8000)
        self.assertLess(len(client.requests), 5)


if __name__ == "__main__":
    unittest.main()
