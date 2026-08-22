# coding=utf-8
"""邮件适配器测试（不连接真实 SMTP）"""

import io
import smtplib
import unittest
from contextlib import redirect_stdout

from github_radar import mailer
from github_radar.logging_utils import mask_email, redact
from github_radar.mailer import (
    FALLBACK_SMTP_CONFIGS,
    EmailConfig,
    build_message,
    load_email_config,
    resolve_smtp,
    send_report_email,
)

PASSWORD = "super-secret-auth-code"


class FakeSMTP:
    """记录调用的假 SMTP 客户端"""

    def __init__(self, fail_with=None):
        self.fail_with = fail_with
        self.logged_in = None
        self.sent_messages = []
        self.quit_called = False

    def login(self, user, password):
        if isinstance(self.fail_with, Exception):
            raise self.fail_with
        self.logged_in = (user, password)

    def send_message(self, message):
        self.sent_messages.append(message)

    def quit(self):
        self.quit_called = True


def make_config(**kwargs):
    defaults = dict(
        from_email="sender@qq.com",
        password=PASSWORD,
        to_email="receiver@qq.com",
    )
    defaults.update(kwargs)
    return EmailConfig(**defaults)


class ResolveSMTPTest(unittest.TestCase):
    def test_qq_uses_ssl_465(self):
        server, port, use_tls = resolve_smtp("someone@qq.com", presets=FALLBACK_SMTP_CONFIGS)
        self.assertEqual((server, port, use_tls), ("smtp.qq.com", 465, False))

    def test_gmail_uses_starttls_587(self):
        server, port, use_tls = resolve_smtp("someone@gmail.com", presets=FALLBACK_SMTP_CONFIGS)
        self.assertEqual((server, port, use_tls), ("smtp.gmail.com", 587, True))

    def test_custom_server_port_465_is_ssl(self):
        self.assertEqual(
            resolve_smtp("a@example.com", "smtp.example.com", 465, FALLBACK_SMTP_CONFIGS),
            ("smtp.example.com", 465, False),
        )

    def test_custom_server_port_587_is_tls(self):
        self.assertEqual(
            resolve_smtp("a@example.com", "smtp.example.com", 587, FALLBACK_SMTP_CONFIGS),
            ("smtp.example.com", 587, True),
        )

    def test_unknown_domain_falls_back(self):
        server, port, use_tls = resolve_smtp("a@weird-domain.io", presets=FALLBACK_SMTP_CONFIGS)
        self.assertEqual((server, port, use_tls), ("smtp.weird-domain.io", 587, True))

    def test_reuses_trendradar_presets_when_importable(self):
        """优先复用 TrendRadar 的 SMTP_CONFIGS；不可用时降级到内置预设"""
        presets = mailer.load_smtp_presets()
        self.assertIn("qq.com", presets)
        self.assertEqual(presets["qq.com"]["server"], "smtp.qq.com")
        self.assertEqual(presets["qq.com"]["port"], 465)


class LoadEmailConfigTest(unittest.TestCase):
    def test_reads_all_env_vars(self):
        config, missing = load_email_config(
            {
                "EMAIL_FROM": "a@qq.com",
                "EMAIL_PASSWORD": PASSWORD,
                "EMAIL_TO": "b@qq.com, c@163.com",
                "EMAIL_SMTP_SERVER": "smtp.custom.com",
                "EMAIL_SMTP_PORT": "465",
            }
        )
        self.assertEqual(missing, [])
        self.assertEqual(config.recipients, ["b@qq.com", "c@163.com"])
        self.assertEqual(config.smtp_server, "smtp.custom.com")
        self.assertEqual(config.smtp_port, 465)

    def test_reports_missing_variables(self):
        config, missing = load_email_config({"EMAIL_FROM": "a@qq.com"})
        self.assertIsNone(config)
        self.assertEqual(missing, ["EMAIL_PASSWORD", "EMAIL_TO"])

    def test_invalid_port_is_ignored(self):
        config, _ = load_email_config(
            {"EMAIL_FROM": "a@qq.com", "EMAIL_PASSWORD": "x", "EMAIL_TO": "b@qq.com",
             "EMAIL_SMTP_PORT": "not-a-port"}
        )
        self.assertIsNone(config.smtp_port)


class BuildMessageTest(unittest.TestCase):
    def setUp(self):
        self.message = build_message(
            subject="🔥 GitHub Daily Radar | 2026-08-22 | Top20",
            html_body="<p>hello</p>",
            text_body="hello",
            from_email="sender@qq.com",
            recipients=["a@qq.com", "b@qq.com"],
        )

    def test_headers(self):
        self.assertIn("GitHub Daily Radar", str(self.message["Subject"]))
        self.assertIn("sender@qq.com", self.message["From"])
        self.assertEqual(self.message["To"], "a@qq.com, b@qq.com")
        self.assertTrue(self.message["Message-ID"])
        self.assertTrue(self.message["Date"])

    def test_multipart_alternative_order(self):
        parts = self.message.get_payload()
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].get_content_type(), "text/plain")
        self.assertEqual(parts[1].get_content_type(), "text/html")


class SendReportEmailTest(unittest.TestCase):
    def _send(self, fake, config=None, **kwargs):
        def factory(server, port, use_tls, timeout):
            fake.connection = (server, port, use_tls, timeout)
            return fake

        return send_report_email(
            config or make_config(),
            subject="subject",
            html_body="<p>x</p>",
            text_body="x",
            smtp_factory=factory,
            presets=FALLBACK_SMTP_CONFIGS,
            **kwargs,
        )

    def test_successful_send(self):
        fake = FakeSMTP()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertTrue(self._send(fake))
        self.assertEqual(fake.logged_in, ("sender@qq.com", PASSWORD))
        self.assertEqual(len(fake.sent_messages), 1)
        self.assertTrue(fake.quit_called)
        self.assertEqual(fake.connection, ("smtp.qq.com", 465, False, 30))

    def test_password_never_appears_in_logs(self):
        fake = FakeSMTP()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self._send(fake)
        output = buffer.getvalue()
        self.assertNotIn(PASSWORD, output)

    def test_recipient_is_masked_in_logs(self):
        fake = FakeSMTP()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self._send(fake, make_config(to_email="receiver@qq.com"))
        output = buffer.getvalue()
        self.assertNotIn("receiver@qq.com", output)
        self.assertIn("r*******@qq.com", output)

    def test_auth_failure_returns_false_without_leaking(self):
        fake = FakeSMTP(fail_with=smtplib.SMTPAuthenticationError(535, b"auth failed"))
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertFalse(self._send(fake))
        output = buffer.getvalue()
        self.assertNotIn(PASSWORD, output)
        self.assertIn("认证失败", output)

    def test_connection_failure_returns_false(self):
        fake = FakeSMTP(fail_with=smtplib.SMTPConnectError(421, b"nope"))
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertFalse(self._send(fake))

    def test_unexpected_exception_is_redacted(self):
        fake = FakeSMTP(fail_with=RuntimeError(f"boom {PASSWORD}"))
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertFalse(self._send(fake))
        output = buffer.getvalue()
        self.assertNotIn(PASSWORD, output)
        self.assertIn("***", output)

    def test_empty_recipients_fail_fast(self):
        fake = FakeSMTP()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertFalse(self._send(fake, make_config(to_email="  ")))
        self.assertIsNone(fake.logged_in)


class MaskingTest(unittest.TestCase):
    def test_mask_email(self):
        self.assertEqual(mask_email("abcdef@qq.com"), "a*****@qq.com")
        self.assertEqual(mask_email("a@qq.com"), "*@qq.com")
        self.assertEqual(mask_email(""), "(未设置)")
        self.assertEqual(mask_email("a@qq.com,bb@163.com"), "*@qq.com, b*@163.com")

    def test_redact(self):
        self.assertEqual(redact("token=abcd1234", "abcd1234"), "token=***")
        self.assertEqual(redact("nothing", None), "nothing")
        self.assertEqual(redact("short", "ab"), "short")  # 过短的值不参与替换


if __name__ == "__main__":
    unittest.main()
