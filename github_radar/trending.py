# coding=utf-8
"""
GitHub Trending 抓取与解析

数据源：https://github.com/trending?since=daily

重要约定：
- Trending 页面上的 ``stars today`` **只作为辅助信息记录**，
  不作为 24h Star 增长的数据源（真实增量一律来自每日快照差值）。
- Trending 的 HTML 结构随时可能变化：解析失败只记 warning 并返回空列表，
  绝不抛异常中断整个日报（GitHub API 候选仍可继续工作）。

解析实现使用标准库 ``re`` + ``html``，不引入 BeautifulSoup 等新依赖。
"""

import html as html_module
import re
import time
from typing import Any, Callable, Dict, List, Optional

from .github_api import DEFAULT_USER_AGENT
from .logging_utils import log, warn
from .models import SOURCE_TRENDING, RepoRecord

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

TRENDING_URL = "https://github.com/trending"

DEFAULT_TIMEOUT = 15
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BASE_DELAY = 3.0

# --- 解析用正则（全部带容错回退） ---------------------------------------
_ARTICLE_RE = re.compile(r"<article\b[^>]*>(.*?)</article>", re.IGNORECASE | re.DOTALL)
_H2_RE = re.compile(r"<h2\b[^>]*>(.*?)</h2>", re.IGNORECASE | re.DOTALL)
# 仅匹配 /owner/repo 形式（排除 /login?return_to=... 这类链接）
_REPO_HREF_RE = re.compile(r'href="/([^/"?#\s]+)/([^/"?#\s]+)"', re.IGNORECASE)
_DESC_RE = re.compile(
    r'<p\b[^>]*class="[^"]*\bcol-9\b[^"]*"[^>]*>(.*?)</p>', re.IGNORECASE | re.DOTALL
)
_ANY_P_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)
_STARS_RE = re.compile(
    r'href="/[^"]+/stargazers"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL
)
_FORKS_RE = re.compile(r'href="/[^"]+/forks"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
_LANGUAGE_RE = re.compile(
    r'itemprop="programmingLanguage"[^>]*>(.*?)</span>', re.IGNORECASE | re.DOTALL
)
_STARS_TODAY_RE = re.compile(
    r"([\d,\.]+)\s*stars?\s+(?:today|this\s+week|this\s+month)", re.IGNORECASE
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# 非仓库路径（防御性排除，避免把导航链接误判为仓库）
_NON_REPO_OWNERS = {
    "login",
    "signup",
    "sponsors",
    "topics",
    "collections",
    "trending",
    "orgs",
    "settings",
    "notifications",
    "search",
}


def _strip_tags(fragment: str) -> str:
    """去掉 HTML 标签、反转义实体并压缩空白"""
    if not fragment:
        return ""
    text = _TAG_RE.sub(" ", fragment)
    text = html_module.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def _parse_count(fragment: str) -> Optional[int]:
    """
    从 HTML 片段中解析数字（如 "13,096" / "1.2k"）

    Returns:
        int；解析失败返回 None（不伪造 0）
    """
    text = _strip_tags(fragment)
    if not text:
        return None
    match = re.search(r"([\d][\d,\.]*)\s*([kKmM])?", text)
    if not match:
        return None
    raw, suffix = match.group(1), match.group(2)
    raw = raw.replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    if suffix:
        value *= 1000 if suffix.lower() == "k" else 1_000_000
    return int(value)


def _extract_full_name(article_html: str) -> Optional[str]:
    """从单个 article 片段中提取 owner/repo"""
    candidates: List[str] = []

    h2_match = _H2_RE.search(article_html)
    if h2_match:
        candidates.append(h2_match.group(1))
    candidates.append(article_html)

    for candidate in candidates:
        for owner, name in _REPO_HREF_RE.findall(candidate):
            if owner.lower() in _NON_REPO_OWNERS:
                continue
            if name.lower() in ("stargazers", "forks"):
                continue
            return f"{owner}/{name}"
    return None


def parse_trending_html(html_text: str) -> List[RepoRecord]:
    """
    解析 GitHub Trending 页面 HTML

    Args:
        html_text: 页面 HTML

    Returns:
        RepoRecord 列表（含 trending_rank，从 1 开始）；
        解析不到任何条目时返回空列表并记录 warning
    """
    if not html_text:
        warn("Trending 页面内容为空，跳过解析")
        return []

    articles = _ARTICLE_RE.findall(html_text)
    if not articles:
        warn("Trending 页面结构可能已变化：未找到任何 <article> 条目")
        return []

    records: List[RepoRecord] = []
    seen = set()
    failed = 0

    for article_html in articles:
        full_name = _extract_full_name(article_html)
        if not full_name:
            failed += 1
            continue

        key = full_name.lower()
        if key in seen:
            continue
        seen.add(key)

        desc_match = _DESC_RE.search(article_html) or _ANY_P_RE.search(article_html)
        description = _strip_tags(desc_match.group(1)) if desc_match else ""

        language_match = _LANGUAGE_RE.search(article_html)
        language = _strip_tags(language_match.group(1)) if language_match else ""

        stars_match = _STARS_RE.search(article_html)
        stars = _parse_count(stars_match.group(1)) if stars_match else None

        forks_match = _FORKS_RE.search(article_html)
        forks = _parse_count(forks_match.group(1)) if forks_match else None

        today_match = _STARS_TODAY_RE.search(_strip_tags(article_html))
        stars_today = None
        if today_match:
            try:
                stars_today = int(today_match.group(1).replace(",", "").split(".")[0])
            except ValueError:
                stars_today = None

        records.append(
            RepoRecord(
                full_name=full_name,
                description=description or None,
                language=language or None,
                stars=stars,
                forks=forks,
                trending_rank=len(records) + 1,
                trending_stars_today=stars_today,
                sources=[SOURCE_TRENDING],
            )
        )

    if failed:
        warn(f"Trending 解析：{failed} 个条目未能提取仓库名（页面结构可能有变化）")
    if not records:
        warn("Trending 解析未得到任何仓库，将仅依赖 GitHub API 候选")

    return records


def fetch_trending_html(
    *,
    since: str = "daily",
    session: Any = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    sleep_func: Callable[[float], None] = time.sleep,
    user_agent: str = DEFAULT_USER_AGENT,
) -> Optional[str]:
    """
    抓取 GitHub Trending 页面 HTML

    Args:
        since: daily / weekly / monthly
        session: 注入的 HTTP 会话（测试用），默认使用 requests
        timeout: 超时（秒）
        max_retries: 重试次数上限
        retry_base_delay: 指数退避基数（秒）
        sleep_func: 睡眠函数
        user_agent: User-Agent

    Returns:
        HTML 文本；失败返回 None
    """
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    params = {"since": since}

    http = session
    if http is None:
        if requests is None:  # pragma: no cover
            warn("requests 未安装，无法抓取 Trending 页面")
            return None
        http = requests

    attempt = 0
    while attempt <= max_retries:
        try:
            response = http.get(
                TRENDING_URL, params=params, headers=headers, timeout=timeout
            )
            status = getattr(response, "status_code", None)
            if status == 200:
                return getattr(response, "text", "") or ""
            warn(f"抓取 Trending 失败：HTTP {status}")
        except Exception as exc:
            warn(f"抓取 Trending 异常：{type(exc).__name__}: {exc}")

        attempt += 1
        if attempt > max_retries:
            break
        delay = retry_base_delay * (2 ** (attempt - 1))
        warn(f"{delay:.1f}s 后重试 Trending（{attempt}/{max_retries}）")
        sleep_func(delay)

    warn("Trending 抓取最终失败，改用 GitHub API 候选继续")
    return None


def collect_trending(
    *,
    since: str = "daily",
    session: Any = None,
    fetcher: Optional[Callable[..., Optional[str]]] = None,
    **fetch_kwargs: Any,
) -> List[RepoRecord]:
    """
    抓取并解析 Trending，返回候选仓库

    Args:
        since: daily / weekly / monthly
        session: 注入的 HTTP 会话
        fetcher: 自定义抓取函数（测试可注入，返回 HTML 文本或 None）
        **fetch_kwargs: 透传给抓取函数

    Returns:
        RepoRecord 列表；任何失败都返回空列表（不抛异常）
    """
    log("collecting GitHub Trending...")
    try:
        if fetcher is not None:
            html_text = fetcher(since=since, session=session, **fetch_kwargs)
        else:
            html_text = fetch_trending_html(since=since, session=session, **fetch_kwargs)
    except Exception as exc:  # 抓取器本身异常也必须降级
        warn(f"Trending 抓取器异常：{type(exc).__name__}: {exc}")
        return []

    if not html_text:
        return []

    try:
        return parse_trending_html(html_text)
    except Exception as exc:  # 解析异常同样降级
        warn(f"Trending 解析异常（页面结构可能已变化）：{type(exc).__name__}: {exc}")
        return []


def summarize_trending(records: List[RepoRecord]) -> Dict[str, Any]:
    """统计 Trending 解析结果（用于日志与报告概览）"""
    with_stars_today = sum(1 for r in records if r.trending_stars_today is not None)
    return {
        "count": len(records),
        "with_stars_today": with_stars_today,
    }
