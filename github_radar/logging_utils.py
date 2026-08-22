# coding=utf-8
"""
日志工具

统一日志前缀，并提供邮箱脱敏函数。

安全约定（必须遵守）：
- 绝不打印 GITHUB_TOKEN / EMAIL_PASSWORD / 授权码
- 邮箱地址一律经 ``mask_email`` 脱敏后再输出（Actions 日志对公开仓库可见）
"""

import sys
from typing import Optional

LOG_PREFIX = "[GitHubRadar]"


def _emit(text: str) -> None:
    """
    输出一行日志

    Windows 控制台在被重定向时使用本地代码页（如 cp936），emoji 等字符
    会触发 UnicodeEncodeError。这里做兜底降级，保证日志本身不会让任务失败。
    """
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe, flush=True)


def log(message: str) -> None:
    """普通信息日志"""
    _emit(f"{LOG_PREFIX} {message}")


def warn(message: str) -> None:
    """警告日志（不中断流程）"""
    _emit(f"{LOG_PREFIX} [warning] {message}")


def error(message: str) -> None:
    """错误日志（通常伴随降级或退出）"""
    _emit(f"{LOG_PREFIX} [error] {message}")


def mask_email(address: Optional[str]) -> str:
    """
    邮箱脱敏，用于日志输出

    支持逗号分隔的多个收件人。

    Args:
        address: 原始邮箱字符串，如 "abcdef@qq.com,x@163.com"

    Returns:
        脱敏后的字符串，如 "a*****@qq.com, *@163.com"

    Examples:
        >>> mask_email("abcdef@qq.com")
        'a*****@qq.com'
        >>> mask_email("")
        '(未设置)'
    """
    if not address:
        return "(未设置)"

    masked = []
    for item in str(address).split(","):
        item = item.strip()
        if not item:
            continue
        if "@" not in item:
            # 不是邮箱格式，整体打码，避免意外泄漏
            masked.append("*" * len(item))
            continue
        local, _, domain = item.partition("@")
        if len(local) <= 1:
            masked_local = "*"
        else:
            masked_local = local[0] + "*" * (len(local) - 1)
        masked.append(f"{masked_local}@{domain}")

    return ", ".join(masked) if masked else "(未设置)"


def redact(text: str, *secrets: Optional[str]) -> str:
    """
    从文本中抹去敏感值（防御性处理，用于打印第三方异常信息）

    Args:
        text: 原始文本
        *secrets: 需要抹去的敏感值（None / 空值会被忽略）

    Returns:
        抹去敏感值后的文本
    """
    result = str(text)
    for secret in secrets:
        if secret and len(str(secret)) >= 4:
            result = result.replace(str(secret), "***")
    return result
