# coding=utf-8
"""候选池构建、去重与容错测试"""

import unittest

from github_radar.collector import collect_candidates, summarize_sources
from github_radar.models import (
    SOURCE_SEARCH_NEW,
    SOURCE_SEARCH_POPULAR,
    SOURCE_TRENDING,
    RepoRecord,
    dedupe_records,
)

from .helpers import FakeAPIClient, no_sleep, repo_payload

TODAY = "2026-08-22"


def trending_record(full_name, rank, **kwargs):
    """模拟 Trending 解析出的（未经 API 补全的）记录"""
    kwargs.setdefault("stars", 1000)
    return RepoRecord(
        full_name=full_name,
        trending_rank=rank,
        trending_stars_today=500,
        sources=[SOURCE_TRENDING],
        **kwargs,
    )


class DedupeTest(unittest.TestCase):
    def test_merges_same_repo_from_multiple_sources(self):
        html_record = trending_record("kortix-ai/suna", 1, stars=18000, language=None)
        api_record = RepoRecord.from_api(
            repo_payload("kortix-ai/suna", stars=18342, language="Python"),
            source=SOURCE_SEARCH_NEW,
        )
        merged = dedupe_records([html_record, api_record])

        self.assertEqual(len(merged), 1)
        record = merged[0]
        self.assertEqual(record.stars, 18342)          # API 数据优先
        self.assertEqual(record.language, "Python")
        self.assertEqual(record.trending_rank, 1)      # Trending 独有字段保留
        self.assertEqual(record.trending_stars_today, 500)
        self.assertEqual(record.sources, [SOURCE_TRENDING, SOURCE_SEARCH_NEW])
        self.assertTrue(record.api_enriched)

    def test_case_insensitive_dedupe(self):
        merged = dedupe_records(
            [RepoRecord(full_name="Owner/Repo"), RepoRecord(full_name="owner/repo")]
        )
        self.assertEqual(len(merged), 1)

    def test_invalid_records_are_dropped(self):
        merged = dedupe_records(
            [RepoRecord(full_name=""), RepoRecord(full_name="no-slash"), None]
        )
        self.assertEqual(merged, [])

    def test_order_is_first_seen(self):
        merged = dedupe_records(
            [RepoRecord(full_name="b/b"), RepoRecord(full_name="a/a"), RepoRecord(full_name="b/b")]
        )
        self.assertEqual([r.full_name for r in merged], ["b/b", "a/a"])

    def test_summarize_sources(self):
        records = dedupe_records(
            [
                trending_record("a/b", 1),
                RepoRecord.from_api(repo_payload("a/b"), source=SOURCE_SEARCH_NEW),
                RepoRecord.from_api(repo_payload("c/d"), source=SOURCE_SEARCH_POPULAR),
            ]
        )
        summary = summarize_sources(records)
        self.assertEqual(summary[SOURCE_TRENDING], 1)
        self.assertEqual(summary[SOURCE_SEARCH_NEW], 1)
        self.assertEqual(summary[SOURCE_SEARCH_POPULAR], 1)


class CollectCandidatesTest(unittest.TestCase):
    def _collect(self, client, trending, **kwargs):
        kwargs.setdefault("sleep_func", no_sleep)
        return collect_candidates(
            client,
            today=TODAY,
            trending_collector=lambda: trending,
            **kwargs,
        )

    def test_combines_and_dedupes_all_sources(self):
        client = FakeAPIClient(
            search_results=[
                [repo_payload("kortix-ai/suna", stars=18342), repo_payload("new/one")],
                [repo_payload("mega/repo", stars=90000)],
            ],
            repo_details={"only/trending": repo_payload("only/trending", stars=700)},
        )
        trending = [trending_record("kortix-ai/suna", 1), trending_record("only/trending", 2)]

        result = self._collect(client, trending)

        names = sorted(record.full_name for record in result.repositories)
        self.assertEqual(names, ["kortix-ai/suna", "mega/repo", "new/one", "only/trending"])
        self.assertEqual(result.trending_count, 2)
        self.assertEqual(result.search_new_count, 2)
        self.assertEqual(result.search_popular_count, 1)
        self.assertEqual(result.api_candidate_count, 3)
        self.assertEqual(result.unique_count, 4)
        self.assertTrue(result.trending_ok)
        self.assertTrue(result.search_ok)

    def test_search_queries_are_built_from_today(self):
        client = FakeAPIClient(search_results=[[], []])
        self._collect(client, [trending_record("a/b", 1)], new_window_days=30, new_min_stars=50)
        self.assertEqual(client.search_queries[0], "created:>=2026-07-23 stars:>50")
        self.assertTrue(client.search_queries[1].startswith("pushed:>=2026-08-20 stars:>"))

    def test_trending_failure_still_produces_candidates(self):
        client = FakeAPIClient(search_results=[[repo_payload("a/b")], [repo_payload("c/d")]])

        def broken_trending():
            raise RuntimeError("trending down")

        result = collect_candidates(
            client, today=TODAY, trending_collector=broken_trending, sleep_func=no_sleep
        )
        self.assertEqual(result.unique_count, 2)
        self.assertFalse(result.trending_ok)
        self.assertTrue(result.search_ok)
        self.assertTrue(any("Trending" in note for note in result.warnings))

    def test_search_failure_still_produces_candidates(self):
        client = FakeAPIClient(
            search_results=[[], []],
            repo_details={"only/trending": repo_payload("only/trending", stars=700)},
        )
        result = self._collect(client, [trending_record("only/trending", 1)])
        self.assertEqual(result.unique_count, 1)
        self.assertTrue(result.trending_ok)
        self.assertFalse(result.search_ok)
        self.assertEqual(result.repositories[0].stars, 700)

    def test_all_sources_failing_yields_empty_pool(self):
        client = FakeAPIClient(search_results=[[], []])
        result = self._collect(client, [])
        self.assertEqual(result.unique_count, 0)
        self.assertFalse(result.trending_ok)
        self.assertFalse(result.search_ok)

    def test_single_repository_failure_does_not_break_others(self):
        """核心容错要求：一个仓库的 API 请求失败不能影响整体"""
        client = FakeAPIClient(
            search_results=[[], []],
            repo_details={"good/repo": repo_payload("good/repo", stars=555)},
            failing_repos=["bad/repo"],
        )
        trending = [
            trending_record("bad/repo", 1, stars=321, description="from html"),
            trending_record("good/repo", 2),
        ]
        result = self._collect(client, trending)

        self.assertEqual(result.unique_count, 2)
        self.assertEqual(result.detail_failed, 1)
        self.assertEqual(result.detail_fetched, 1)

        by_name = {record.full_name: record for record in result.repositories}
        # 失败的仓库保留 Trending HTML 数据，而不是被丢弃
        self.assertEqual(by_name["bad/repo"].stars, 321)
        self.assertEqual(by_name["bad/repo"].description, "from html")
        self.assertFalse(by_name["bad/repo"].api_enriched)
        self.assertEqual(by_name["good/repo"].stars, 555)
        self.assertTrue(by_name["good/repo"].api_enriched)

    def test_detail_requests_are_capped(self):
        details = {f"o/r{i}": repo_payload(f"o/r{i}") for i in range(10)}
        client = FakeAPIClient(search_results=[[], []], repo_details=details)
        trending = [trending_record(f"o/r{i}", i + 1) for i in range(10)]

        result = self._collect(client, trending, max_detail_requests=3)
        self.assertEqual(len(client.detail_requests), 3)
        self.assertEqual(result.detail_fetched, 3)
        self.assertEqual(result.unique_count, 10)  # 其余仍保留 Trending 数据

    def test_rate_limited_client_skips_enrichment(self):
        client = FakeAPIClient(search_results=[[], []], rate_limited=True)
        result = self._collect(client, [trending_record("a/b", 1)])
        self.assertEqual(client.detail_requests, [])
        self.assertEqual(result.unique_count, 1)

    def test_collected_at_is_stamped(self):
        client = FakeAPIClient(search_results=[[repo_payload("a/b")], []])
        result = self._collect(client, [])
        self.assertTrue(result.repositories[0].collected_at)

    def test_new_repo_count(self):
        client = FakeAPIClient(
            search_results=[
                [
                    repo_payload("new/one", created_at="2026-08-10T00:00:00Z"),
                    repo_payload("old/one", created_at="2018-01-01T00:00:00Z"),
                ],
                [],
            ]
        )
        result = self._collect(client, [])
        self.assertEqual(result.new_repo_count(max_age_days=30), 1)


if __name__ == "__main__":
    unittest.main()
