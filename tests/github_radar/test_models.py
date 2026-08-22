# coding=utf-8
"""数据模型测试"""

import unittest

from github_radar.models import SOURCE_SEARCH_NEW, RepoRecord

from .helpers import repo_payload


class FromAPITest(unittest.TestCase):
    def test_builds_record_from_payload(self):
        record = RepoRecord.from_api(
            repo_payload("owner/repo", stars=123, topics=["ai"]), source=SOURCE_SEARCH_NEW
        )
        self.assertEqual(record.full_name, "owner/repo")
        self.assertEqual(record.owner, "owner")
        self.assertEqual(record.name, "repo")
        self.assertEqual(record.stars, 123)
        self.assertEqual(record.topics, ["ai"])
        self.assertEqual(record.sources, [SOURCE_SEARCH_NEW])
        self.assertTrue(record.api_enriched)

    def test_rejects_invalid_payload(self):
        self.assertIsNone(RepoRecord.from_api(None))
        self.assertIsNone(RepoRecord.from_api({}))
        self.assertIsNone(RepoRecord.from_api({"full_name": "no-slash"}))
        self.assertIsNone(RepoRecord.from_api("not a dict"))

    def test_missing_fields_are_none_not_faked(self):
        record = RepoRecord.from_api(
            {"full_name": "a/b", "stargazers_count": None, "description": "", "topics": None}
        )
        self.assertIsNone(record.stars)
        self.assertIsNone(record.description)
        self.assertIsNone(record.language)
        self.assertEqual(record.topics, [])

    def test_non_numeric_counts_are_dropped(self):
        record = RepoRecord.from_api({"full_name": "a/b", "stargazers_count": "many"})
        self.assertIsNone(record.stars)


class DerivedFieldsTest(unittest.TestCase):
    def test_owner_name_and_url_are_derived(self):
        record = RepoRecord(full_name="owner/repo")
        self.assertEqual(record.owner, "owner")
        self.assertEqual(record.name, "repo")
        self.assertEqual(record.html_url, "https://github.com/owner/repo")

    def test_display_placeholders(self):
        record = RepoRecord(full_name="a/b")
        self.assertEqual(record.display_description, "No description provided.")
        self.assertEqual(record.display_language, "Unknown")


class MergeTest(unittest.TestCase):
    def test_api_data_wins_over_html_data(self):
        html = RepoRecord(full_name="a/b", stars=1000, language=None, trending_rank=4)
        api = RepoRecord.from_api(repo_payload("a/b", stars=1042, language="Rust"))
        html.merge(api)
        self.assertEqual(html.stars, 1042)
        self.assertEqual(html.language, "Rust")
        self.assertEqual(html.trending_rank, 4)
        self.assertTrue(html.api_enriched)

    def test_html_data_is_kept_when_api_field_missing(self):
        html = RepoRecord(full_name="a/b", stars=1000, description="from html")
        api = RepoRecord.from_api({"full_name": "a/b"})
        html.merge(api)
        self.assertEqual(html.description, "from html")
        self.assertEqual(html.stars, 1000)

    def test_merge_none_is_safe(self):
        record = RepoRecord(full_name="a/b", stars=5)
        record.merge(None)
        self.assertEqual(record.stars, 5)

    def test_sources_are_unioned(self):
        first = RepoRecord(full_name="a/b", sources=["trending"])
        second = RepoRecord(full_name="a/b", sources=["trending", "search:new"])
        first.merge(second)
        self.assertEqual(first.sources, ["trending", "search:new"])


class SnapshotDictTest(unittest.TestCase):
    def test_only_known_fields_are_stored(self):
        record = RepoRecord(
            full_name="a/b",
            stars=10,
            forks=2,
            language="Go",
            created_at="2026-08-01T00:00:00Z",
            trending_rank=1,
            trending_stars_today=50,
            collected_at="2026-08-22T08:10:00+08:00",
        )
        data = record.to_snapshot_dict()
        self.assertEqual(data["stars"], 10)
        self.assertEqual(data["trending_rank"], 1)
        self.assertNotIn("collected_at", data)   # 每日文件已有 generated_at
        self.assertNotIn("html_url", data)       # 可由 full_name 推导
        self.assertNotIn("sources", data)
        self.assertNotIn("description", data)    # 体积控制：历史快照只为算差值服务

    def test_none_fields_are_omitted(self):
        data = RepoRecord(full_name="a/b").to_snapshot_dict()
        self.assertEqual(data, {})

    def test_to_dict_is_complete(self):
        data = RepoRecord(full_name="a/b", stars=1).to_dict()
        for key in ("full_name", "html_url", "stars", "topics", "sources", "trending_rank"):
            self.assertIn(key, data)


if __name__ == "__main__":
    unittest.main()
