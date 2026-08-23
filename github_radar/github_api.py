# coding=utf-8
"""
GitHub REST API 客户端

特性：
- 统一 User-Agent、超时、有限次数重试（绝不无限重试）
- 处理 403 / 429（含 secondary rate limit）、5xx、网络异常
- 尊重 ``X-RateLimit-Remaining`` / ``X-RateLimit-Reset`` / ``Retry-After``
- 任何失败都返回 ``None``（调用方降级），不抛异常打断整个日报
- 只读操作，token 只出现在请求头，**绝不写日志**

这是一个每天运行一次的轻量任务：串行请求 + 请求间隔，不做并发。
"""

import base64
import binascii
import json
import time
from typing import Any, Callable, Dict, List, Optional

from .logging_utils import log, warn

try:  # requests 是项目已有依赖；缺失时仍允许 import 本模块（便于纯离线测试）
    import requests
except ImportError:  # pragma: no cover - 正常环境不会走到
    requests = None

API_ROOT = "https://api.github.com"

# 明确表明身份，便于 GitHub 侧排查；不包含任何用户隐私
DEFAULT_USER_AGENT = "TrendRadar-GitHubRadar/1.0 (+https://github.com/sansan0/TrendRadar)"

DEFAULT_TIMEOUT = 15
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 2.0
# 触发限流时最多愿意等待的秒数，超过则直接放弃（避免 workflow 长时间挂住）
DEFAULT_MAX_RATE_LIMIT_WAIT = 60.0


def _header(response: Any, name: str) -> Optional[str]:
    """
    大小写不敏感地读取响应头（兼容 requests 与测试用的简单 dict）
    """
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        value = headers.get(name)
        if value is not None:
            return str(value)
    except AttributeError:
        return None
    lowered = name.lower()
    try:
        for key, value in headers.items():
            if str(key).lower() == lowered:
                return str(value)
    except AttributeError:
        return None
    return None


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


class GitHubAPIClient:
    """GitHub REST API 只读客户端"""

    def __init__(
        self,
        token: Optional[str] = None,
        *,
        session: Any = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
        max_rate_limit_wait: float = DEFAULT_MAX_RATE_LIMIT_WAIT,
        request_interval: float = 0.0,
        sleep_func: Callable[[float], None] = time.sleep,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        """
        Args:
            token: GitHub token（Actions 自动提供的 GITHUB_TOKEN 即可，无需 PAT）
            session: 注入的 HTTP 会话对象（需实现 ``get``），默认使用 requests.Session
            timeout: 单次请求超时（秒）
            max_retries: 失败重试次数上限（不含首次请求）
            retry_base_delay: 指数退避基数（秒）
            max_rate_limit_wait: 遇到限流时最多等待的秒数
            request_interval: 每次请求之间的最小间隔（秒），避免过快请求
            sleep_func: 睡眠函数（测试可注入以避免真实等待）
            user_agent: User-Agent
        """
        self._token = token or None
        self._session = session
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.retry_base_delay = retry_base_delay
        self.max_rate_limit_wait = max_rate_limit_wait
        self.request_interval = request_interval
        self._sleep = sleep_func
        self.user_agent = user_agent

        # 运行状态（供调用方决策是否继续消耗 API）
        self.request_count = 0
        self.rate_limited = False
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # 基础设施
    # ------------------------------------------------------------------
    @property
    def authenticated(self) -> bool:
        """是否携带 token（仅返回布尔值，绝不暴露 token 内容）"""
        return bool(self._token)

    def _get_session(self) -> Any:
        if self._session is None:
            if requests is None:  # pragma: no cover
                raise RuntimeError("requests 未安装，无法发起 HTTP 请求")
            self._session = requests.Session()
        return self._session

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": self.user_agent,
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _rate_limit_wait_seconds(self, response: Any) -> Optional[float]:
        """
        根据响应头推算需要等待的秒数

        优先 ``Retry-After``（秒），其次 ``X-RateLimit-Reset``（Unix 时间戳）。
        """
        retry_after = _to_float(_header(response, "Retry-After"))
        if retry_after is not None and retry_after >= 0:
            return retry_after

        reset_at = _to_float(_header(response, "X-RateLimit-Reset"))
        if reset_at is not None:
            wait = reset_at - time.time()
            if wait > 0:
                return wait
            return 0.0
        return None

    def _request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        发起 GET 请求并返回 JSON

        Returns:
            解析后的 JSON（dict/list）；任何失败均返回 None
        """
        attempt = 0
        while attempt <= self.max_retries:
            if attempt > 0 or self.request_count > 0:
                if self.request_interval > 0:
                    self._sleep(self.request_interval)

            try:
                self.request_count += 1
                response = self._get_session().get(
                    url,
                    params=params,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
            except Exception as exc:  # 网络异常统一降级重试
                self.last_error = f"{type(exc).__name__}: {exc}"
                attempt += 1
                if attempt > self.max_retries:
                    warn(f"请求失败（网络异常，已重试 {self.max_retries} 次）: {url} -> {self.last_error}")
                    return None
                delay = self.retry_base_delay * (2 ** (attempt - 1))
                warn(f"请求异常 {self.last_error}，{delay:.1f}s 后重试（{attempt}/{self.max_retries}）")
                self._sleep(delay)
                continue

            status = getattr(response, "status_code", None)

            if status == 200:
                return self._parse_json(response, url)

            if status in (301, 302, 307, 308):
                # requests 默认已跟随重定向；到这里说明未跟随，直接放弃
                warn(f"请求被重定向且未跟随，跳过: {url}")
                return None

            if status == 404:
                warn(f"资源不存在（404），跳过: {url}")
                return None

            if status == 422:
                # 搜索语法错误等，重试没有意义
                warn(f"请求被拒绝（422，通常为查询语法问题），跳过: {url}")
                return None

            if status in (403, 429):
                handled = self._handle_rate_limit(response, url, attempt, status)
                if handled:
                    attempt += 1
                    continue
                return None

            if status is not None and 500 <= int(status) < 600:
                attempt += 1
                if attempt > self.max_retries:
                    warn(f"GitHub 服务端错误（{status}），已达重试上限，跳过: {url}")
                    return None
                delay = self.retry_base_delay * (2 ** (attempt - 1))
                warn(f"GitHub 服务端错误（{status}），{delay:.1f}s 后重试（{attempt}/{self.max_retries}）")
                self._sleep(delay)
                continue

            warn(f"未预期的响应状态 {status}，跳过: {url}")
            return None

        return None

    def _handle_rate_limit(self, response: Any, url: str, attempt: int, status: Any) -> bool:
        """
        处理 403 / 429

        Returns:
            True 表示已等待、可以重试；False 表示放弃本次请求
        """
        remaining = _header(response, "X-RateLimit-Remaining")
        is_rate_limit = status == 429 or (remaining is not None and remaining.strip() == "0")

        if not is_rate_limit:
            # 403 也可能是权限问题（例如未授权访问），重试无意义
            warn(f"请求被拒绝（{status}，非配额耗尽，可能是权限或滥用检测），跳过: {url}")
            return False

        wait = self._rate_limit_wait_seconds(response)
        if wait is None:
            wait = self.retry_base_delay * (2 ** attempt)

        if attempt >= self.max_retries or wait > self.max_rate_limit_wait:
            self.rate_limited = True
            warn(
                f"GitHub API 触发限流（{status}），需等待约 {wait:.0f}s，"
                f"超出可接受范围，后续将改用已有数据继续生成日报"
            )
            return False

        warn(f"GitHub API 限流（{status}），等待 {wait:.0f}s 后重试（{attempt + 1}/{self.max_retries}）")
        self._sleep(wait)
        return True

    @staticmethod
    def _parse_json(response: Any, url: str) -> Optional[Any]:
        """解析响应 JSON，失败时降级为 None"""
        try:
            return response.json()
        except Exception:
            pass
        try:
            return json.loads(getattr(response, "text", "") or "")
        except Exception:
            warn(f"响应不是合法 JSON，跳过: {url}")
            return None

    # ------------------------------------------------------------------
    # 业务接口
    # ------------------------------------------------------------------
    def search_repositories(
        self,
        query: str,
        *,
        sort: str = "stars",
        order: str = "desc",
        per_page: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        搜索仓库

        Args:
            query: GitHub 搜索语法，如 "created:>=2026-07-23 stars:>50"
            sort: 排序字段（stars / forks / updated）
            order: 排序方向
            per_page: 单页数量（GitHub 上限 100），只取一页以控制 API 消耗

        Returns:
            仓库 JSON 列表；失败返回空列表
        """
        if self.rate_limited:
            warn("已处于限流状态，跳过搜索请求")
            return []

        params = {
            "q": query,
            "sort": sort,
            "order": order,
            "per_page": max(1, min(int(per_page), 100)),
        }
        data = self._request(f"{API_ROOT}/search/repositories", params=params)
        if not isinstance(data, dict):
            return []
        items = data.get("items")
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

    def get_repository(self, full_name: str) -> Optional[Dict[str, Any]]:
        """
        获取单个仓库详情

        Args:
            full_name: "owner/repo"

        Returns:
            仓库 JSON；失败返回 None（调用方应保留已有数据继续运行）
        """
        if not full_name or "/" not in full_name:
            return None
        if self.rate_limited:
            return None
        data = self._request(f"{API_ROOT}/repos/{full_name}")
        if isinstance(data, dict) and data.get("full_name"):
            return data
        return None

    def get_readme_text(self, full_name: str) -> Optional[str]:
        """
        获取仓库 README 的纯文本内容

        只在 AI 增强阶段被调用，且只对最终 AI 候选调用（绝不给所有候选读 README）。

        Args:
            full_name: "owner/repo"

        Returns:
            README 文本；仓库没有 README（404）、编码异常、限流等一律返回 None
            —— 调用方必须能在没有 README 的情况下继续工作。
        """
        if not full_name or "/" not in full_name:
            return None
        if self.rate_limited:
            return None

        data = self._request(f"{API_ROOT}/repos/{full_name}/readme")
        if not isinstance(data, dict):
            return None

        content = data.get("content")
        encoding = str(data.get("encoding") or "").lower()
        if not isinstance(content, str) or not content:
            return None

        if encoding == "base64":
            try:
                raw = base64.b64decode(content, validate=False)
            except (binascii.Error, ValueError) as exc:
                warn(f"README 解码失败（{full_name}）：{type(exc).__name__}")
                return None
            # GitHub 上的 README 编码五花八门：UTF-8 解不出来就替换非法字节，
            # 绝不因为一个字符让整个流程失败
            text = raw.decode("utf-8", errors="replace")
        else:
            text = content

        text = text.strip()
        return text or None

    def describe(self) -> str:
        """返回一行可安全打印的客户端状态描述（不含 token）"""
        mode = "authenticated (GITHUB_TOKEN)" if self.authenticated else "anonymous (60 req/h)"
        return f"GitHub API {mode}, requests={self.request_count}, rate_limited={self.rate_limited}"


def log_client_status(client: GitHubAPIClient) -> None:
    """打印客户端状态（便于 Actions 日志排查）"""
    log(client.describe())
