# coding=utf-8
"""Trending HTML 解析测试（不访问网络）"""

import unittest

from github_radar import trending
from github_radar.models import SOURCE_TRENDING

from .helpers import FakeResponse, FakeSession, RaisingSession, load_fixture, no_sleep


class ParseTrendingHTMLTest(unittest.TestCase):
    """解析真实结构的 Trending 页面"""

    @classmethod
    def setUpClass(cls):
        cls.html = load_fixture("trending_sample.html")
        cls.records = trending.parse_trending_html(cls.html)

    def test_parses_all_articles_in_order(self):
        self.assertEqual(
            [record.full_name for record in self.records],
            ["kortix-ai/suna", "openclaw/clawbench", "torvalds/linux"],
        )
        self.assertEqual([r.trending_rank for r in self.records], [1, 2, 3])

    def test_extracts_full_metadata(self):
        first = self.records[0]
        self.assertEqual(first.stars, 18342)
        self.assertEqual(first.forks, 2411)
        self.assertEqual(first.language, "Python")
        self.assertEqual(first.trending_stars_today, 1826)
        self.assertIn("Open Source Generalist AI Agent", first.description)
        self.assertEqual(first.sources, [SOURCE_TRENDING])
        self.assertEqual(first.owner, "kortix-ai")
        self.assertEqual(first.name, "suna")
        self.assertEqual(first.html_url, "https://github.com/kortix-ai/suna")

    def test_html_entities_are_unescaped(self):
        self.assertIn("Suna & friends", self.records[0].description)
        self.assertIn("<self-hosted>", self.records[0].description)

    def test_missing_fields_stay_none_not_faked(self):
        second = self.records[1]
        self.assertIsNone(second.description)
        self.assertIsNone(second.language)
        self.assertEqual(second.display_description, "No description provided.")
        self.assertEqual(second.display_language, "Unknown")

    def test_missing_stars_today_is_none(self):
        third = self.records[2]
        self.assertIsNone(third.trending_stars_today)
        self.assertEqual(third.stars, 201502)

    def test_login_links_are_not_parsed_as_repositories(self):
        names = [record.full_name for record in self.records]
        self.assertNotIn("login/return_to", names)
        for name in names:
            self.assertFalse(name.startswith("login"))

    def test_api_enriched_is_false_for_html_only_records(self):
        self.assertFalse(any(record.api_enriched for record in self.records))


class ParseTrendingEdgeCaseTest(unittest.TestCase):
    """结构变化 / 空内容时的降级行为"""

    def test_changed_structure_returns_empty_without_crash(self):
        html = load_fixture("trending_changed_structure.html")
        self.assertEqual(trending.parse_trending_html(html), [])

    def test_empty_html_returns_empty(self):
        self.assertEqual(trending.parse_trending_html(""), [])
        self.assertEqual(trending.parse_trending_html(None), [])

    def test_article_without_repo_link_is_skipped(self):
        html = "<article class='Box-row'><p>nothing here</p></article>"
        self.assertEqual(trending.parse_trending_html(html), [])

    def test_duplicate_repositories_are_deduped(self):
        html = load_fixture("trending_sample.html")
        records = trending.parse_trending_html(html + html)
        self.assertEqual(len(records), 3)

    def test_parse_count_handles_suffixes(self):
        self.assertEqual(trending._parse_count("1,234"), 1234)
        self.assertEqual(trending._parse_count(" 12k "), 12000)
        self.assertIsNone(trending._parse_count(""))
        self.assertIsNone(trending._parse_count("no digits"))


class FetchTrendingTest(unittest.TestCase):
    """抓取层的容错"""

    def test_fetch_returns_text_on_200(self):
        session = FakeSession([FakeResponse(200, text="<html>ok</html>")])
        html = trending.fetch_trending_html(session=session, sleep_func=no_sleep)
        self.assertEqual(html, "<html>ok</html>")
        self.assertEqual(session.calls[0]["params"], {"since": "daily"})
        self.assertIn("User-Agent", session.calls[0]["headers"])

    def test_fetch_retries_then_gives_up_on_error_status(self):
        session = FakeSession([FakeResponse(503, text="")])
        html = trending.fetch_trending_html(
            session=session, max_retries=2, sleep_func=no_sleep
        )
        self.assertIsNone(html)
        self.assertEqual(len(session.calls), 3)  # 1 次首发 + 2 次重试，绝不无限重试

    def test_fetch_handles_network_exception(self):
        session = RaisingSession()
        html = trending.fetch_trending_html(
            session=session, max_retries=1, sleep_func=no_sleep
        )
        self.assertIsNone(html)
        self.assertEqual(session.call_count, 2)

    def test_collect_trending_degrades_to_empty_list(self):
        def broken_fetcher(**_kwargs):
            raise RuntimeError("fetcher exploded")

        self.assertEqual(trending.collect_trending(fetcher=broken_fetcher), [])

    def test_collect_trending_parses_with_injected_fetcher(self):
        html = load_fixture("trending_sample.html")
        records = trending.collect_trending(fetcher=lambda **_kwargs: html)
        self.assertEqual(len(records), 3)
        self.assertEqual(trending.summarize_trending(records)["with_stars_today"], 2)


if __name__ == "__main__":
    unittest.main()
