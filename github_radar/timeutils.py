# coding=utf-8
"""
时间/日期工具

与 ``trendradar.utils.time`` 功能相近，但**不 import trendradar**
（见 ``github_radar/__init__.py`` 的解耦说明）。

时区解析顺序：pytz（项目已有依赖）→ 标准库 zoneinfo → 固定 UTC+8 兜底，
保证在任何环境下都不会因为时区数据缺失而崩溃。
"""

from datetime import datetime, date, timedelta, timezone as dt_timezone, tzinfo
from typing import Optional

from .logging_utils import warn

DEFAULT_TIMEZONE = "Asia/Shanghai"

# 兜底时区：北京时间 UTC+8（仅当 pytz 与 zoneinfo 都不可用时使用）
_FALLBACK_TZ = dt_timezone(timedelta(hours=8), "UTC+8")

DATE_FORMAT = "%Y-%m-%d"


def get_timezone(name: str = DEFAULT_TIMEZONE) -> tzinfo:
    """
    获取时区对象

    Args:
        name: 时区名，如 "Asia/Shanghai"

    Returns:
        tzinfo 对象；解析失败时返回 UTC+8 兜底时区
    """
    try:
        import pytz  # 项目已有依赖

        return pytz.timezone(name)
    except Exception:
        pass

    try:
        from zoneinfo import ZoneInfo  # Python 3.9+ 标准库

        return ZoneInfo(name)
    except Exception:
        warn(f"无法解析时区 '{name}'，回退到固定 UTC+8")
        return _FALLBACK_TZ


def now(tz_name: str = DEFAULT_TIMEZONE) -> datetime:
    """获取指定时区的当前时间（带时区信息）"""
    return datetime.now(get_timezone(tz_name))


def today_str(tz_name: str = DEFAULT_TIMEZONE) -> str:
    """获取指定时区的今天日期字符串（YYYY-MM-DD）"""
    return now(tz_name).strftime(DATE_FORMAT)


def parse_date_str(value: str) -> Optional[date]:
    """
    解析 YYYY-MM-DD 字符串

    Args:
        value: 日期字符串

    Returns:
        date 对象，解析失败返回 None
    """
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip(), DATE_FORMAT).date()
    except (ValueError, TypeError):
        return None


def shift_date_str(date_str: str, days: int) -> Optional[str]:
    """
    日期字符串偏移

    Args:
        date_str: 基准日期 "YYYY-MM-DD"
        days: 偏移天数（可为负）

    Returns:
        偏移后的日期字符串；输入非法时返回 None

    Examples:
        >>> shift_date_str("2026-08-22", -1)
        '2026-08-21'
        >>> shift_date_str("2026-08-22", -7)
        '2026-08-15'
    """
    base = parse_date_str(date_str)
    if base is None:
        return None
    return (base + timedelta(days=days)).strftime(DATE_FORMAT)


def days_between(earlier: str, later: str) -> Optional[int]:
    """计算两个 YYYY-MM-DD 日期之间相差的天数（later - earlier）"""
    a = parse_date_str(earlier)
    b = parse_date_str(later)
    if a is None or b is None:
        return None
    return (b - a).days


def parse_github_time(value: Optional[str]) -> Optional[datetime]:
    """
    解析 GitHub API 返回的时间（ISO 8601，通常形如 2026-08-01T12:34:56Z）

    Args:
        value: 时间字符串

    Returns:
        带时区信息的 datetime；无法解析返回 None
    """
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        # 退化处理：只取日期部分
        try:
            parsed = datetime.strptime(text[:10], DATE_FORMAT)
        except (ValueError, TypeError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt_timezone.utc)
    return parsed


def age_in_days(
    created_at: Optional[str], reference: Optional[datetime] = None
) -> Optional[float]:
    """
    计算项目年龄（天）

    Args:
        created_at: 创建时间（ISO 字符串）
        reference: 参考时间点，默认取当前时间（UTC）

    Returns:
        年龄天数（float，>= 0）；无法解析时返回 None
    """
    created = parse_github_time(created_at)
    if created is None:
        return None

    ref = reference or datetime.now(dt_timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=dt_timezone.utc)

    delta = ref - created
    days = delta.total_seconds() / 86400.0
    # 未来时间（时钟偏差）按 0 处理，避免出现负年龄
    return max(0.0, days)


def format_created_display(created_at: Optional[str]) -> str:
    """把创建时间格式化为 YYYY-MM-DD 用于展示；无法解析返回 '—'"""
    parsed = parse_github_time(created_at)
    if parsed is None:
        return "—"
    return parsed.strftime(DATE_FORMAT)
