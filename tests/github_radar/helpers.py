# coding=utf-8
"""
测试公共工具：假响应 / 假会话 / 假 API 客户端 / 数据工厂

所有测试都不访问真实网络。
"""

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name: str) -> str:
    """读取 fixtures 目录下的文本文件"""
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


class FakeResponse:
    """模拟 requests 的响应对象"""

    def __init__(
        self,
        status_code: int = 200,
        json_data: Any = None,
        text: str = "",
        headers: Optional[Dict[str, str]] = None,
    ):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.headers = headers or {}

    def json(self) -> Any:
        if self._json_data is None:
            raise ValueError("no json body")
        return self._json_data


class FakeSession:
    """
    模拟 requests.Session

    Args:
        responses: 响应序列（按调用顺序返回；用尽后重复最后一个），
                   或者一个 ``handler(url, params) -> FakeResponse`` 函数
    """

    def __init__(self, responses: Any):
        self.calls: List[Dict[str, Any]] = []
        if callable(responses):
            self._handler: Optional[Callable[..., FakeResponse]] = responses
            self._queue: List[FakeResponse] = []
        else:
            self._handler = None
            self._queue = list(responses)

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(
            {"url": url, "params": params, "headers": headers, "timeout": timeout}
        )
        if self._handler is not None:
            return self._handler(url, params)
        if not self._queue:
            raise AssertionError(f"FakeSession: 没有为 {url} 准备响应")
        if len(self._queue) == 1:
            return self._queue[0]
        return self._queue.pop(0)


class RaisingSession:
    """每次请求都抛异常的会话（用于测试网络失败降级）"""

    def __init__(self, exc: Optional[Exception] = None):
        self.exc = exc or ConnectionError("boom")
        self.call_count = 0

    def get(self, *args, **kwargs):
        self.call_count += 1
        raise self.exc


def repo_payload(
    full_name: str,
    *,
    stars: int = 100,
    forks: int = 10,
    language: Optional[str] = "Python",
    created_at: str = "2026-08-01T00:00:00Z",
    pushed_at: str = "2026-08-22T00:00:00Z",
    description: Optional[str] = "demo repository",
    topics: Optional[Sequence[str]] = None,
    open_issues: int = 3,
) -> Dict[str, Any]:
    """构造一个 GitHub API 风格的仓库 JSON"""
    owner, _, name = full_name.partition("/")
    return {
        "full_name": full_name,
        "name": name,
        "owner": {"login": owner},
        "html_url": f"https://github.com/{full_name}",
        "description": description,
        "language": language,
        "topics": list(topics or []),
        "stargazers_count": stars,
        "forks_count": forks,
        "open_issues_count": open_issues,
        "created_at": created_at,
        "updated_at": pushed_at,
        "pushed_at": pushed_at,
    }


class FakeAPIClient:
    """
    模拟 GitHubAPIClient（只实现 collector 用到的接口）

    Args:
        search_results: ``[第一次搜索结果, 第二次搜索结果, ...]``
        repo_details: ``{full_name: payload}``
        failing_repos: 这些仓库的详情请求返回 None（模拟单点失败）
        rate_limited: 是否处于限流状态
    """

    def __init__(
        self,
        search_results: Optional[Iterable[List[Dict[str, Any]]]] = None,
        repo_details: Optional[Dict[str, Dict[str, Any]]] = None,
        failing_repos: Iterable[str] = (),
        rate_limited: bool = False,
    ):
        self._search_results = [list(item) for item in (search_results or [])]
        self._repo_details = dict(repo_details or {})
        self._failing = {name.lower() for name in failing_repos}
        self.rate_limited = rate_limited
        self.search_queries: List[str] = []
        self.detail_requests: List[str] = []
        self.request_count = 0
        self.authenticated = True

    def search_repositories(self, query, *, sort="stars", order="desc", per_page=50):
        self.search_queries.append(query)
        self.request_count += 1
        if not self._search_results:
            return []
        return self._search_results.pop(0)

    def get_repository(self, full_name):
        self.detail_requests.append(full_name)
        self.request_count += 1
        if full_name.lower() in self._failing:
            return None
        return self._repo_details.get(full_name)

    def describe(self):
        return "FakeAPIClient"


def no_sleep(_seconds: float) -> None:
    """替代 time.sleep，避免测试真的等待"""
    return None
