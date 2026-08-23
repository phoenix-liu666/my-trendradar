# coding=utf-8
"""DeepSeek 客户端测试（thinking / 重试 / 限流 / JSON / Secret）"""

import io
import unittest
from contextlib import redirect_stdout

from github_radar.ai.client import (
    MAX_RETRY_WAIT,
    THINKING_DISABLED,
    DeepSeekClient,
    extract_json,
)
from github_radar.ai.config import load_ai_config

from .ai_helpers import (
    FakeChatResponse,
    FakeChatSession,
    RaisingChatSession,
    chat_response,
    error_response,
    no_sleep,
)

SECRET = "sk-super-secret-deepseek-key-1234567890"


def make_client(session, **kwargs):
    return DeepSeekClient(
        SECRET,
        model=kwargs.pop("model", "deepseek-v4-flash"),
        api_base=kwargs.pop("api_base", "https://api.deepseek.com"),
        session=session,
        sleep_func=kwargs.pop("sleep_func", no_sleep),
        **kwargs,
    )


class ThinkingDisabledTest(unittest.TestCase):
    """31. thinking disabled"""

    def test_request_body_disables_thinking(self):
        session = FakeChatSession([chat_response({"ok": True})])
        client = make_client(session)
        client.chat([{"role": "user", "content": "hi"}])

        payload = session.payload()
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(THINKING_DISABLED, {"type": "disabled"})

    def test_thinking_is_disabled_on_every_request_including_retries(self):
        session = FakeChatSession([error_response(500), chat_response({"ok": True})])
        client = make_client(session)
        client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(session.call_count, 2)
        for index in range(session.call_count):
            with self.subTest(call=index):
                self.assertEqual(session.payload(index)["thinking"], {"type": "disabled"})

    def test_json_mode_and_model_are_set(self):
        session = FakeChatSession([chat_response({"ok": True})])
        client = make_client(session, model="deepseek-v4-flash")
        client.chat([{"role": "user", "content": "hi"}])

        payload = session.payload()
        self.assertEqual(payload["model"], "deepseek-v4-flash")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertIs(payload["stream"], False)

    def test_model_can_be_overridden(self):
        config = load_ai_config(
            {
                "GITHUB_RADAR_AI_ENABLED": "true",
                "DEEPSEEK_API_KEY": SECRET,
                "DEEPSEEK_MODEL": "deepseek-v4-pro",
            }
        )
        self.assertEqual(config.model, "deepseek-v4-pro")

    def test_default_model_is_flash(self):
        config = load_ai_config(
            {"GITHUB_RADAR_AI_ENABLED": "true", "DEEPSEEK_API_KEY": SECRET}
        )
        self.assertEqual(config.model, "deepseek-v4-flash")


class TimeoutTest(unittest.TestCase):
    """21. timeout"""

    def test_timeout_is_passed_to_the_session(self):
        session = FakeChatSession([chat_response({"ok": True})])
        make_client(session, timeout=42).chat([{"role": "user", "content": "hi"}])
        self.assertEqual(session.calls[0]["timeout"], 42)

    def test_timeout_exception_degrades_gracefully(self):
        session = RaisingChatSession(TimeoutError("timed out"))
        result = make_client(session).chat([{"role": "user", "content": "hi"}])

        self.assertFalse(result.ok)
        self.assertIn("TimeoutError", result.error)
        self.assertEqual(session.call_count, 3)   # 1 次 + 2 次重试

    def test_network_error_degrades_gracefully(self):
        session = RaisingChatSession(ConnectionError("boom"))
        result = make_client(session, max_retries=1).chat([{"role": "user", "content": "hi"}])
        self.assertFalse(result.ok)
        self.assertEqual(session.call_count, 2)

    def test_retries_are_bounded(self):
        session = RaisingChatSession()
        make_client(session, max_retries=0).chat([{"role": "user", "content": "hi"}])
        self.assertEqual(session.call_count, 1)


class RateLimitTest(unittest.TestCase):
    """22. 429"""

    def test_429_then_success(self):
        session = FakeChatSession([error_response(429), chat_response({"ok": True})])
        result = make_client(session).chat([{"role": "user", "content": "hi"}])
        self.assertTrue(result.ok)
        self.assertEqual(session.call_count, 2)

    def test_repeated_429_gives_up(self):
        session = FakeChatSession([error_response(429)])
        result = make_client(session, max_retries=2).chat([{"role": "user", "content": "hi"}])
        self.assertFalse(result.ok)
        self.assertIn("429", result.error)
        self.assertEqual(session.call_count, 3)

    def test_long_retry_after_is_not_waited_for(self):
        """限流要等一小时？直接放弃这一批，绝不让 workflow 挂在那里"""
        waits = []
        session = FakeChatSession([error_response(429, headers={"Retry-After": "3600"})])
        result = make_client(session, sleep_func=waits.append).chat(
            [{"role": "user", "content": "hi"}]
        )
        self.assertFalse(result.ok)
        self.assertEqual(session.call_count, 1)
        self.assertEqual(waits, [])
        self.assertLess(MAX_RETRY_WAIT, 3600)

    def test_short_retry_after_is_honoured(self):
        waits = []
        session = FakeChatSession([error_response(429, headers={"Retry-After": "2"}),
                                   chat_response({"ok": True})])
        client = make_client(session, sleep_func=waits.append)
        result = client.chat([{"role": "user", "content": "hi"}])
        self.assertTrue(result.ok)
        self.assertEqual(waits, [2.0])


class ServerErrorTest(unittest.TestCase):
    """23. 5xx"""

    def test_500_is_retried(self):
        session = FakeChatSession([error_response(500), chat_response({"ok": True})])
        self.assertTrue(make_client(session).chat([{"role": "user", "content": "x"}]).ok)
        self.assertEqual(session.call_count, 2)

    def test_persistent_5xx_degrades(self):
        for status in (500, 502, 503):
            with self.subTest(status=status):
                session = FakeChatSession([error_response(status)])
                result = make_client(session, max_retries=1).chat([{"role": "user", "content": "x"}])
                self.assertFalse(result.ok)
                self.assertEqual(session.call_count, 2)

    def test_4xx_is_not_retried(self):
        for status in (400, 401, 402, 403, 404):
            with self.subTest(status=status):
                session = FakeChatSession([error_response(status)])
                result = make_client(session).chat([{"role": "user", "content": "x"}])
                self.assertFalse(result.ok)
                self.assertEqual(session.call_count, 1)

    def test_error_hint_never_leaks_the_key(self):
        buffer = io.StringIO()
        session = FakeChatSession([error_response(401)])
        with redirect_stdout(buffer):
            make_client(session).chat([{"role": "user", "content": "x"}])
        output = buffer.getvalue()
        self.assertIn("API Key 无效或已过期", output)
        self.assertNotIn(SECRET, output)


class MalformedJsonTest(unittest.TestCase):
    """10. malformed JSON / 11. JSON retry"""

    def test_extract_json_handles_plain_json(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_extract_json_strips_code_fences(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_extract_json_ignores_surrounding_prose(self):
        self.assertEqual(extract_json('好的，结果如下：\n{"a": 1}\n希望有帮助'), {"a": 1})

    def test_extract_json_wraps_bare_arrays(self):
        self.assertEqual(extract_json('[{"full_name": "a/b"}]'),
                         {"repositories": [{"full_name": "a/b"}]})

    def test_extract_json_returns_none_for_garbage(self):
        for text in ("", "not json at all", "{unclosed"):
            with self.subTest(text=text):
                self.assertIsNone(extract_json(text))

    def test_malformed_json_triggers_exactly_one_retry(self):
        session = FakeChatSession([
            chat_response("这不是 JSON"),
            chat_response({"repositories": []}),
        ])
        result = make_client(session).chat_json(
            [{"role": "user", "content": "x"}], retry_instruction="只输出 JSON"
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(session.call_count, 2)

    def test_retry_message_is_appended(self):
        session = FakeChatSession([
            chat_response("nope"),
            chat_response({"repositories": []}),
        ])
        make_client(session).chat_json(
            [{"role": "user", "content": "x"}], retry_instruction="只输出 JSON"
        )
        messages = session.payload(1)["messages"]
        self.assertEqual(messages[-1]["content"], "只输出 JSON")
        self.assertEqual(messages[-2]["role"], "assistant")

    def test_second_malformed_response_gives_up(self):
        session = FakeChatSession([chat_response("still not json")])
        result = make_client(session).chat_json(
            [{"role": "user", "content": "x"}], retry_instruction="只输出 JSON"
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "malformed JSON")
        self.assertEqual(session.call_count, 2)   # 首次 + 1 次 JSON 重试，绝不无限重试

    def test_json_retry_can_be_disabled(self):
        session = FakeChatSession([chat_response("nope")])
        result = make_client(session).chat_json(
            [{"role": "user", "content": "x"}],
            retry_instruction="只输出 JSON",
            json_retry_limit=0,
        )
        self.assertFalse(result.ok)
        self.assertEqual(session.call_count, 1)

    def test_transport_failure_does_not_add_json_retries(self):
        session = FakeChatSession([error_response(500)])
        result = make_client(session, max_retries=1).chat_json(
            [{"role": "user", "content": "x"}], retry_instruction="只输出 JSON"
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(session.call_count, 2)   # 只有传输层的 1 次重试

    def test_response_without_choices_fails_safely(self):
        session = FakeChatSession(
            [FakeChatResponse(status_code=200, json_data={"usage": {"prompt_tokens": 1}})]
        )
        result = make_client(session).chat([{"role": "user", "content": "x"}])
        self.assertFalse(result.ok)
        self.assertIn("没有可用内容", result.error)


class UsageTest(unittest.TestCase):
    """33. usage token aggregation（客户端侧）"""

    def test_usage_is_accumulated_across_calls(self):
        session = FakeChatSession([
            chat_response({"a": 1}, prompt_tokens=100, completion_tokens=10),
            chat_response({"a": 2}, prompt_tokens=200, completion_tokens=20),
        ])
        client = make_client(session)
        client.chat([{"role": "user", "content": "x"}])
        client.chat([{"role": "user", "content": "y"}])

        self.assertEqual(client.usage.prompt_tokens, 300)
        self.assertEqual(client.usage.completion_tokens, 30)
        self.assertEqual(client.usage.total_tokens, 330)
        self.assertEqual(client.usage.requests, 2)
        self.assertEqual(client.usage.successful_requests, 2)

    def test_failed_requests_are_counted(self):
        session = FakeChatSession([error_response(400)])
        client = make_client(session)
        client.chat([{"role": "user", "content": "x"}])
        self.assertEqual(client.usage.failed_requests, 1)
        self.assertEqual(client.usage.total_tokens, 0)

    def test_missing_usage_field_is_tolerated(self):
        session = FakeChatSession([FakeChatResponse(
            status_code=200,
            json_data={"choices": [{"message": {"content": '{"a": 1}'}}]},
        )])
        client = make_client(session)
        result = client.chat([{"role": "user", "content": "x"}])
        self.assertTrue(result.ok)
        self.assertEqual(client.usage.total_tokens, 0)


class SecretSafetyTest(unittest.TestCase):
    """35. Secret 不出日志"""

    def test_key_is_only_in_the_authorization_header(self):
        session = FakeChatSession([chat_response({"a": 1})])
        make_client(session).chat([{"role": "user", "content": "x"}])

        headers = session.calls[0]["headers"]
        self.assertEqual(headers["Authorization"], f"Bearer {SECRET}")
        self.assertNotIn(SECRET, str(session.calls[0]["payload"]))

    def test_key_never_appears_in_logs_on_failure(self):
        buffer = io.StringIO()
        session = RaisingChatSession(RuntimeError(f"connection failed with {SECRET}"))
        with redirect_stdout(buffer):
            make_client(session).chat([{"role": "user", "content": "x"}])

        output = buffer.getvalue()
        self.assertNotIn(SECRET, output)
        self.assertIn("***", output)

    def test_config_describe_hides_the_key(self):
        config = load_ai_config(
            {"GITHUB_RADAR_AI_ENABLED": "true", "DEEPSEEK_API_KEY": SECRET}
        )
        self.assertNotIn(SECRET, config.describe())
        self.assertTrue(config.has_api_key)


if __name__ == "__main__":
    unittest.main()
