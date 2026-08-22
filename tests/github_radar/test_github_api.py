# coding=utf-8
"""GitHub API 客户端测试（全部使用假会话，不访问网络）"""

import time
import unittest

from github_radar.github_api import GitHubAPIClient

from .helpers import FakeResponse, FakeSession, RaisingSession, repo_payload


class RecordingSleep:
    """记录 sleep 调用，避免测试真的等待"""

    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)

    @property
    def total(self):
        return sum(self.calls)


def make_client(session, **kwargs):
    kwargs.setdefault("sleep_func", RecordingSleep())
    return GitHubAPIClient(token="dummy-token", session=session, **kwargs)


class RequestTest(unittest.TestCase):
    def test_success_returns_json(self):
        session = FakeSession([FakeResponse(200, json_data={"ok": True})])
        client = make_client(session)
        self.assertEqual(client._request("https://api.github.com/x"), {"ok": True})
        self.assertEqual(client.request_count, 1)

    def test_token_is_sent_but_never_returned_in_describe(self):
        session = FakeSession([FakeResponse(200, json_data={})])
        client = make_client(session)
        client._request("https://api.github.com/x")
        self.assertEqual(
            session.calls[0]["headers"]["Authorization"], "Bearer dummy-token"
        )
        self.assertNotIn("dummy-token", client.describe())
        self.assertTrue(client.authenticated)

    def test_anonymous_client_sends_no_auth_header(self):
        session = FakeSession([FakeResponse(200, json_data={})])
        client = GitHubAPIClient(token=None, session=session)
        client._request("https://api.github.com/x")
        self.assertNotIn("Authorization", session.calls[0]["headers"])
        self.assertFalse(client.authenticated)

    def test_404_is_not_retried(self):
        session = FakeSession([FakeResponse(404)])
        client = make_client(session)
        self.assertIsNone(client._request("https://api.github.com/repos/a/b"))
        self.assertEqual(len(session.calls), 1)

    def test_422_is_not_retried(self):
        session = FakeSession([FakeResponse(422)])
        client = make_client(session)
        self.assertIsNone(client._request("https://api.github.com/search/repositories"))
        self.assertEqual(len(session.calls), 1)

    def test_5xx_is_retried_then_gives_up(self):
        sleep = RecordingSleep()
        session = FakeSession([FakeResponse(502)])
        client = make_client(session, max_retries=2, sleep_func=sleep)
        self.assertIsNone(client._request("https://api.github.com/x"))
        self.assertEqual(len(session.calls), 3)  # 有限重试
        self.assertEqual(len(sleep.calls), 2)

    def test_network_exception_is_retried_then_gives_up(self):
        sleep = RecordingSleep()
        session = RaisingSession()
        client = make_client(session, max_retries=1, sleep_func=sleep)
        self.assertIsNone(client._request("https://api.github.com/x"))
        self.assertEqual(session.call_count, 2)

    def test_invalid_json_returns_none(self):
        session = FakeSession([FakeResponse(200, json_data=None, text="<html>")])
        client = make_client(session)
        self.assertIsNone(client._request("https://api.github.com/x"))


class RateLimitTest(unittest.TestCase):
    def test_403_with_exhausted_quota_marks_rate_limited(self):
        sleep = RecordingSleep()
        session = FakeSession(
            [
                FakeResponse(
                    403,
                    headers={
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(time.time()) + 3600),
                    },
                )
            ]
        )
        client = make_client(session, sleep_func=sleep)
        self.assertIsNone(client._request("https://api.github.com/x"))
        self.assertTrue(client.rate_limited)
        self.assertEqual(sleep.total, 0)  # 等待过长直接放弃，不挂住 workflow

    def test_429_with_short_retry_after_waits_and_retries(self):
        sleep = RecordingSleep()
        session = FakeSession(
            [
                FakeResponse(429, headers={"Retry-After": "2"}),
                FakeResponse(200, json_data={"ok": 1}),
            ]
        )
        client = make_client(session, sleep_func=sleep)
        self.assertEqual(client._request("https://api.github.com/x"), {"ok": 1})
        self.assertIn(2.0, sleep.calls)
        self.assertFalse(client.rate_limited)

    def test_403_without_quota_header_is_not_retried(self):
        session = FakeSession([FakeResponse(403, headers={})])
        client = make_client(session)
        self.assertIsNone(client._request("https://api.github.com/x"))
        self.assertEqual(len(session.calls), 1)
        self.assertFalse(client.rate_limited)

    def test_rate_limited_client_skips_further_calls(self):
        session = FakeSession([FakeResponse(200, json_data={"items": []})])
        client = make_client(session)
        client.rate_limited = True
        self.assertEqual(client.search_repositories("q"), [])
        self.assertIsNone(client.get_repository("a/b"))
        self.assertEqual(len(session.calls), 0)


class BusinessMethodTest(unittest.TestCase):
    def test_search_returns_items(self):
        payload = {"items": [repo_payload("a/b"), repo_payload("c/d"), "garbage"]}
        session = FakeSession([FakeResponse(200, json_data=payload)])
        client = make_client(session)
        items = client.search_repositories("created:>=2026-07-23 stars:>50")
        self.assertEqual(len(items), 2)
        self.assertEqual(session.calls[0]["params"]["q"], "created:>=2026-07-23 stars:>50")
        self.assertEqual(session.calls[0]["params"]["sort"], "stars")

    def test_search_per_page_is_capped(self):
        session = FakeSession([FakeResponse(200, json_data={"items": []})])
        client = make_client(session)
        client.search_repositories("q", per_page=500)
        self.assertEqual(session.calls[0]["params"]["per_page"], 100)

    def test_search_failure_returns_empty_list(self):
        session = FakeSession([FakeResponse(500)])
        client = make_client(session, max_retries=0)
        self.assertEqual(client.search_repositories("q"), [])

    def test_get_repository_validates_input(self):
        session = FakeSession([FakeResponse(200, json_data=repo_payload("a/b"))])
        client = make_client(session)
        self.assertIsNone(client.get_repository(""))
        self.assertIsNone(client.get_repository("no-slash"))
        self.assertEqual(len(session.calls), 0)

    def test_get_repository_returns_payload(self):
        session = FakeSession([FakeResponse(200, json_data=repo_payload("a/b", stars=42))])
        client = make_client(session)
        data = client.get_repository("a/b")
        self.assertEqual(data["stargazers_count"], 42)

    def test_get_repository_failure_returns_none(self):
        session = FakeSession([FakeResponse(404)])
        client = make_client(session)
        self.assertIsNone(client.get_repository("a/b"))


if __name__ == "__main__":
    unittest.main()
