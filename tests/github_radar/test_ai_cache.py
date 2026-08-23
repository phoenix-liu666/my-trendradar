# coding=utf-8
"""AI 缓存测试（命中 / 未命中 / 失效 / 损坏 fail-open / 当日结果复用）"""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from github_radar.ai.cache import (
    CACHE_SCHEMA_VERSION,
    DailyResultStore,
    StaticAnalysisCache,
    cache_key,
)
from github_radar.ai.schemas import AIUsage, DailySynthesis, parse_repo_analysis

from .ai_helpers import analysis_payload, make_record


def iso_days_ago(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")


class CacheKeyTest(unittest.TestCase):
    def test_owner_repo_becomes_a_safe_filename(self):
        self.assertEqual(cache_key("Owner/Repo"), "owner__repo")

    def test_special_characters_are_replaced(self):
        self.assertEqual(cache_key("a b/c:d"), "a-b__c-d")

    def test_empty_name(self):
        self.assertEqual(cache_key(""), "unknown")


class CacheBaseTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "ai_cache"
        self.cache = StaticAnalysisCache(self.dir, model="deepseek-v4-flash")

    def tearDown(self):
        self._tmp.cleanup()

    def analysis(self, full_name="owner/repo", **overrides):
        return parse_repo_analysis(analysis_payload(full_name, **overrides), full_name=full_name)


class CacheHitTest(CacheBaseTest):
    """17. cache hit"""

    def test_put_then_get(self):
        record = make_record("owner/repo")
        self.assertTrue(self.cache.put(self.analysis(), record))

        static = self.cache.get(record)
        self.assertIsNotNone(static)
        self.assertEqual(static["category"], "Developer Tool")
        self.assertEqual(self.cache.stats.hits, 1)

    def test_only_static_fields_are_cached(self):
        record = make_record("owner/repo")
        self.cache.put(self.analysis(), record)

        static = self.cache.get(record)
        for key in ("summary_zh", "problem", "category", "tech_stack", "use_cases", "maturity"):
            self.assertIn(key, static)
        # 每日上下文绝不缓存
        for key in ("why_hot", "relevance_score", "relevance_reason", "recommended_action"):
            self.assertNotIn(key, static)

    def test_case_insensitive_lookup(self):
        self.cache.put(self.analysis("Owner/Repo"), make_record("Owner/Repo"))
        self.assertIsNotNone(self.cache.get(make_record("owner/repo")))

    def test_cache_file_contains_metadata(self):
        record = make_record("owner/repo", pushed_at="2026-08-20T00:00:00Z")
        self.cache.put(self.analysis(), record)

        payload = json.loads(self.cache.path_for("owner/repo").read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], CACHE_SCHEMA_VERSION)
        self.assertEqual(payload["full_name"], "owner/repo")
        self.assertEqual(payload["repo_pushed_at"], "2026-08-20T00:00:00Z")
        self.assertEqual(payload["model"], "deepseek-v4-flash")
        self.assertTrue(payload["cached_at"])


class CacheMissTest(CacheBaseTest):
    """18. cache miss"""

    def test_unknown_repo_is_a_miss(self):
        self.assertIsNone(self.cache.get(make_record("nobody/here")))
        self.assertEqual(self.cache.stats.misses, 1)

    def test_empty_analysis_is_not_cached(self):
        empty = parse_repo_analysis({"summary_zh": ""}, full_name="a/b")
        self.assertFalse(self.cache.put(empty, make_record("a/b")))
        self.assertIsNone(self.cache.get(make_record("a/b")))

    def test_missing_record_is_a_miss(self):
        self.assertIsNone(self.cache.get(None))


class CacheInvalidationTest(CacheBaseTest):
    """19. cache invalidation"""

    def test_changed_pushed_at_invalidates(self):
        self.cache.put(self.analysis(), make_record("owner/repo", pushed_at="2026-08-20T00:00:00Z"))
        stale = self.cache.get(make_record("owner/repo", pushed_at="2026-08-23T00:00:00Z"))
        self.assertIsNone(stale)
        self.assertEqual(self.cache.stats.stale, 1)

    def test_same_pushed_at_still_hits(self):
        self.cache.put(self.analysis(), make_record("owner/repo", pushed_at="2026-08-20T00:00:00Z"))
        self.assertIsNotNone(self.cache.get(make_record("owner/repo", pushed_at="2026-08-20T00:00:00Z")))

    def test_ttl_expiry_invalidates(self):
        record = make_record("owner/repo")
        self.cache.put(self.analysis(), record, now_iso=iso_days_ago(30))
        self.assertIsNone(self.cache.get(record))

    def test_within_ttl_still_hits(self):
        record = make_record("owner/repo")
        self.cache.put(self.analysis(), record, now_iso=iso_days_ago(3))
        self.assertIsNotNone(self.cache.get(record))

    def test_schema_version_change_invalidates(self):
        record = make_record("owner/repo")
        self.cache.put(self.analysis(), record)
        path = self.cache.path_for("owner/repo")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version"] = 999
        path.write_text(json.dumps(payload), encoding="utf-8")

        self.assertIsNone(self.cache.get(record))

    def test_name_mismatch_invalidates(self):
        record = make_record("owner/repo")
        self.cache.put(self.analysis(), record)
        path = self.cache.path_for("owner/repo")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["full_name"] = "someone/else"
        path.write_text(json.dumps(payload), encoding="utf-8")

        self.assertIsNone(self.cache.get(record))

    def test_prune_removes_old_entries(self):
        self.cache.put(self.analysis("old/one"), make_record("old/one"), now_iso=iso_days_ago(100))
        self.cache.put(self.analysis("new/one"), make_record("new/one"))

        removed = self.cache.prune(30)
        self.assertEqual(removed, 1)
        self.assertFalse(self.cache.path_for("old/one").exists())
        self.assertTrue(self.cache.path_for("new/one").exists())


class CacheCorruptionTest(CacheBaseTest):
    """20. cache corruption fail-open"""

    def test_broken_json_is_treated_as_a_miss(self):
        record = make_record("owner/repo")
        self.cache.put(self.analysis(), record)
        self.cache.path_for("owner/repo").write_text("{not json", encoding="utf-8")

        self.assertIsNone(self.cache.get(record))
        self.assertEqual(self.cache.stats.corrupted, 1)

    def test_non_object_root_is_treated_as_a_miss(self):
        record = make_record("owner/repo")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cache.path_for("owner/repo").write_text("[1, 2, 3]", encoding="utf-8")
        self.assertIsNone(self.cache.get(record))

    def test_missing_static_section_is_a_miss(self):
        record = make_record("owner/repo")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cache.path_for("owner/repo").write_text(
            json.dumps({"schema_version": CACHE_SCHEMA_VERSION, "full_name": "owner/repo"}),
            encoding="utf-8",
        )
        self.assertIsNone(self.cache.get(record))

    def test_corruption_never_raises(self):
        record = make_record("owner/repo")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cache.path_for("owner/repo").write_bytes(b"\xff\xfe\x00broken")
        self.assertIsNone(self.cache.get(record))   # 不抛异常就算通过


class DailyResultStoreTest(unittest.TestCase):
    """当天 AI 结果复用（配合 4 次兜底 cron 保证每天只分析 30 个仓库）"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = DailyResultStore(Path(self._tmp.name) / "ai_cache")

    def tearDown(self):
        self._tmp.cleanup()

    def save(self, date="2026-08-23", model="deepseek-v4-flash"):
        analyses = {
            "owner/repo": parse_repo_analysis(analysis_payload("owner/repo"), full_name="owner/repo")
        }
        self.store.save(
            date,
            analyses=analyses,
            synthesis=DailySynthesis(headline="今天很热闹", signals=["a"]),
            usage=AIUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120, requests=2),
            model=model,
        )

    def test_round_trip(self):
        self.save()
        loaded = self.store.load("2026-08-23", model="deepseek-v4-flash")
        self.assertIsNotNone(loaded)

        analyses, synthesis, usage = loaded
        self.assertIn("owner/repo", analyses)
        self.assertEqual(analyses["owner/repo"].category, "Developer Tool")
        self.assertEqual(synthesis.headline, "今天很热闹")
        self.assertEqual(usage.prompt_tokens, 100)

    def test_missing_date_returns_none(self):
        self.assertIsNone(self.store.load("2026-01-01"))

    def test_model_change_invalidates(self):
        self.save(model="deepseek-v4-flash")
        self.assertIsNone(self.store.load("2026-08-23", model="deepseek-v4-pro"))

    def test_corrupted_file_returns_none(self):
        self.save()
        self.store.path_for("2026-08-23").write_text("{broken", encoding="utf-8")
        self.assertIsNone(self.store.load("2026-08-23"))

    def test_prune_removes_old_days(self):
        self.save("2026-01-01")
        self.save("2026-08-23")
        removed = self.store.prune(30, "2026-08-23")
        self.assertEqual(removed, ["2026-01-01"])
        self.assertTrue(self.store.path_for("2026-08-23").exists())


if __name__ == "__main__":
    unittest.main()
