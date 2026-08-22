# coding=utf-8
"""
轻量邮件适配器

为什么不直接调用 ``trendradar.notification.senders.send_to_email``：
该函数与新闻报告耦合（必须传入 HTML **文件路径**、主题固定为
“TrendRadar 热点分析报告 - …”、发件人名固定为 TrendRadar），
而且 ``import trendradar.*`` 会连带加载 litellm / boto3 等 AI 依赖。

因此这里只做一件事：**复用它的 SMTP 服务商自动识别表**（惰性导入，
导入失败自动降级到内置精简预设），其余按 GitHub Radar 的需要重新组装。

安全约定：
- 授权码只在 ``smtp.login()`` 时使用，绝不打印、绝不写入任何文件
- 日志中的邮箱一律脱敏
- 打印第三方异常前先用 ``redact`` 抹掉可能出现的敏感值
"""

import os
import smtplib
from dataclasses import dataclass
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate, make_msgid
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .logging_utils import error, log, mask_email, redact, warn

DEFAULT_SENDER_NAME = "GitHub Daily Radar"
DEFAULT_TIMEOUT = 30

# 兜底 SMTP 预设（仅当无法复用 TrendRadar 的 SMTP_CONFIGS 时使用）
FALLBACK_SMTP_CONFIGS: Dict[str, Dict[str, Any]] = {
    "qq.com": {"server": "smtp.qq.com", "port": 465, "encryption": "SSL"},
    "foxmail.com": {"server": "smtp.qq.com", "port": 465, "encryption": "SSL"},
    "163.com": {"server": "smtp.163.com", "port": 465, "encryption": "SSL"},
    "126.com": {"server": "smtp.126.com", "port": 465, "encryption": "SSL"},
    "gmail.com": {"server": "smtp.gmail.com", "port": 587, "encryption": "TLS"},
}

ENV_FROM = "EMAIL_FROM"
ENV_PASSWORD = "EMAIL_PASSWORD"
ENV_TO = "EMAIL_TO"
ENV_SMTP_SERVER = "EMAIL_SMTP_SERVER"
ENV_SMTP_PORT = "EMAIL_SMTP_PORT"


@dataclass
class EmailConfig:
    """邮件配置（来自环境变量，绝不落盘）"""

    from_email: str
    password: str
    to_email: str
    smtp_server: Optional[str] = None
    smtp_port: Optional[int] = None

    @property
    def recipients(self) -> List[str]:
        return [item.strip() for item in self.to_email.split(",") if item.strip()]


def load_smtp_presets() -> Dict[str, Dict[str, Any]]:
    """
    加载 SMTP 服务商预设

    优先复用 TrendRadar 已有的 ``SMTP_CONFIGS``（惰性导入，避免在不需要
    发邮件时拖入其依赖链）；不可用时降级到内置精简预设。
    """
    try:
        from trendradar.notification.senders import SMTP_CONFIGS  # 惰性导入

        if isinstance(SMTP_CONFIGS, dict) and SMTP_CONFIGS:
            return dict(SMTP_CONFIGS)
    except Exception as exc:
        warn(
            f"无法复用 TrendRadar 的 SMTP 预设（{type(exc).__name__}），"
            f"使用 GitHub Radar 内置精简预设"
        )
    return dict(FALLBACK_SMTP_CONFIGS)


def resolve_smtp(
    from_email: str,
    custom_server: Optional[str] = None,
    custom_port: Optional[int] = None,
    presets: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> Tuple[str, int, bool]:
    """
    解析 SMTP 服务器配置（与 TrendRadar 的判定规则保持一致）

    Args:
        from_email: 发件人邮箱
        custom_server: 自定义 SMTP 服务器
        custom_port: 自定义端口
        presets: 服务商预设表，默认调用 ``load_smtp_presets()``

    Returns:
        (服务器, 端口, 是否使用 STARTTLS)
        —— ``use_tls=False`` 表示直接 SSL（如 QQ 邮箱 465）

    Examples:
        >>> resolve_smtp("someone@qq.com", presets=FALLBACK_SMTP_CONFIGS)
        ('smtp.qq.com', 465, False)
    """
    presets = presets if presets is not None else load_smtp_presets()
    domain = (from_email or "").split("@")[-1].lower().strip()

    if custom_server and custom_port:
        port = int(custom_port)
        # 端口约定：465 = SSL，587 = STARTTLS，其它端口优先尝试 STARTTLS
        use_tls = port != 465
        return custom_server, port, use_tls

    preset = presets.get(domain)
    if preset:
        return (
            str(preset.get("server")),
            int(preset.get("port", 465)),
            str(preset.get("encryption", "SSL")).upper() == "TLS",
        )

    warn(f"未识别的邮箱服务商（{domain or '未知'}），使用通用 SMTP 配置 smtp.{domain}:587")
    return f"smtp.{domain}", 587, True


def load_email_config(
    env: Optional[Mapping[str, str]] = None,
) -> Tuple[Optional[EmailConfig], List[str]]:
    """
    从环境变量读取邮件配置

    Returns:
        (配置或 None, 缺失的环境变量名列表)
    """
    env = env if env is not None else os.environ

    from_email = (env.get(ENV_FROM) or "").strip()
    password = env.get(ENV_PASSWORD) or ""
    to_email = (env.get(ENV_TO) or "").strip()

    missing = [
        name
        for name, value in (
            (ENV_FROM, from_email),
            (ENV_PASSWORD, password),
            (ENV_TO, to_email),
        )
        if not value
    ]
    if missing:
        return None, missing

    port_raw = (env.get(ENV_SMTP_PORT) or "").strip()
    port: Optional[int] = None
    if port_raw:
        try:
            port = int(port_raw)
        except ValueError:
            warn(f"{ENV_SMTP_PORT} 不是合法端口，已忽略")

    server = (env.get(ENV_SMTP_SERVER) or "").strip() or None

    return (
        EmailConfig(
            from_email=from_email,
            password=password,
            to_email=to_email,
            smtp_server=server,
            smtp_port=port,
        ),
        [],
    )


def build_message(
    *,
    subject: str,
    html_body: str,
    text_body: str,
    from_email: str,
    recipients: List[str],
    sender_name: str = DEFAULT_SENDER_NAME,
) -> MIMEMultipart:
    """
    构造 multipart/alternative 邮件（纯文本在前，HTML 在后）

    Returns:
        可直接交给 ``smtplib`` 发送的邮件对象
    """
    message = MIMEMultipart("alternative")
    # formataddr 默认 charset="utf-8"，非 ASCII 发件人名会自动做 RFC 2047 编码
    message["From"] = formataddr((sender_name, from_email))
    message["To"] = ", ".join(recipients)
    message["Subject"] = Header(subject, "utf-8")
    # MIME-Version 由 MIMEMultipart 自动写入，这里不再重复设置
    message["Date"] = formatdate(localtime=True)
    # Message-ID 的域名与发件域保持一致：既符合惯例、有利于投递，
    # 也避免把运行机器的主机名写进邮件头
    domain = from_email.split("@")[-1].strip() if "@" in (from_email or "") else ""
    message["Message-ID"] = make_msgid(domain=domain) if domain else make_msgid()

    # 顺序很重要：邮件客户端会优先展示最后一个可渲染的部分
    message.attach(MIMEText(text_body or "", "plain", "utf-8"))
    message.attach(MIMEText(html_body or "", "html", "utf-8"))
    return message


def _default_smtp_factory(server: str, port: int, use_tls: bool, timeout: int) -> Any:
    """建立 SMTP 连接（SSL 或 STARTTLS）"""
    if use_tls:
        client = smtplib.SMTP(server, port, timeout=timeout)
        client.ehlo()
        client.starttls()
        client.ehlo()
        return client
    client = smtplib.SMTP_SSL(server, port, timeout=timeout)
    client.ehlo()
    return client


def send_report_email(
    config: EmailConfig,
    *,
    subject: str,
    html_body: str,
    text_body: str,
    sender_name: str = DEFAULT_SENDER_NAME,
    timeout: int = DEFAULT_TIMEOUT,
    smtp_factory: Optional[Callable[[str, int, bool, int], Any]] = None,
    presets: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> bool:
    """
    发送日报邮件

    Args:
        config: 邮件配置
        subject: 主题
        html_body: HTML 正文
        text_body: 纯文本 fallback
        sender_name: 发件人显示名
        timeout: SMTP 超时（秒）
        smtp_factory: SMTP 连接工厂（测试可注入）
        presets: SMTP 预设表（测试可注入）

    Returns:
        True 表示发送成功；False 表示失败（错误原因已打印，且不含敏感信息）
    """
    recipients = config.recipients
    if not recipients:
        error("收件人为空，无法发送邮件")
        return False

    server, port, use_tls = resolve_smtp(
        config.from_email, config.smtp_server, config.smtp_port, presets
    )
    message = build_message(
        subject=subject,
        html_body=html_body,
        text_body=text_body,
        from_email=config.from_email,
        recipients=recipients,
        sender_name=sender_name,
    )

    mode = "STARTTLS" if use_tls else "SSL"
    log(
        f"sending email via {server}:{port} ({mode}) "
        f"from {mask_email(config.from_email)} to {mask_email(config.to_email)}"
    )

    factory = smtp_factory or _default_smtp_factory
    client = None
    try:
        client = factory(server, port, use_tls, timeout)
        client.login(config.from_email, config.password)
        client.send_message(message)
        return True
    except smtplib.SMTPAuthenticationError:
        # 不打印异常详情：部分服务商会在错误信息中回显认证内容
        error("邮件发送失败：SMTP 认证失败，请检查 EMAIL_FROM 与 EMAIL_PASSWORD（授权码）")
    except smtplib.SMTPRecipientsRefused:
        error("邮件发送失败：收件人被拒绝，请检查 EMAIL_TO")
    except smtplib.SMTPSenderRefused:
        error("邮件发送失败：发件人被拒绝，请检查 EMAIL_FROM 是否与授权码匹配")
    except smtplib.SMTPConnectError:
        error(f"邮件发送失败：无法连接 SMTP 服务器 {server}:{port}")
    except smtplib.SMTPServerDisconnected:
        error("邮件发送失败：SMTP 服务器意外断开连接")
    except smtplib.SMTPException as exc:
        error(f"邮件发送失败：{type(exc).__name__}: {redact(str(exc), config.password)}")
    except Exception as exc:
        error(f"邮件发送失败：{type(exc).__name__}: {redact(str(exc), config.password)}")
    finally:
        if client is not None:
            try:
                client.quit()
            except Exception:
                pass

    return False
