# coding=utf-8
"""
AI 测试公共工具

所有 AI 测试都不访问真实网络、不消耗真实 DeepSeek 配额：
DeepSeek 的 HTTP 会话一律用 ``FakeChatSession`` 打桩。
"""

import json
from typing import Any, Callable, Dict, List, Optional, Sequence

from github_radar.history import StarDelta
from github_radar.models import RepoRecord
from github_radar.ranking import ScoredRepo


class FakeChatResponse:
    """模拟 DeepSeek 的 HTTP 响应"""

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


def chat_response(
    content: Any,
    *,
    prompt_tokens: int = 1000,
    completion_tokens: int = 200,
    total_tokens: Optional[int] = None,
    status_code: int = 200,
) -> FakeChatResponse:
    """
    构造一个成功的 chat/completions 响应

    Args:
        content: 字符串直接用作 content；dict/list 会被 json.dumps
    """
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return FakeChatResponse(
        status_code=status_code,
        json_data={
            "id": "fake",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": (
                    total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
                ),
            },
        },
    )


def error_response(status_code: int, headers: Optional[Dict[str, str]] = None) -> FakeChatResponse:
    """构造一个错误响应（429 / 5xx / 4xx）"""
    return FakeChatResponse(status_code=status_code, json_data={"error": "boom"}, headers=headers)


class FakeChatSession:
    """
    模拟 requests.Session 的 ``post``

    Args:
        responses: 响应序列（按顺序返回，用尽后重复最后一个），
                   或 ``handler(url, payload) -> FakeChatResponse``
    """

    def __init__(self, responses: Any):
        self.calls: List[Dict[str, Any]] = []
        if callable(responses):
            self._handler: Optional[Callable[..., FakeChatResponse]] = responses
            self._queue: List[FakeChatResponse] = []
        else:
            self._handler = None
            self._queue = list(responses)

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(
            {"url": url, "payload": json, "headers": headers, "timeout": timeout}
        )
        if self._handler is not None:
            return self._handler(url, json)
        if not self._queue:
            raise AssertionError(f"FakeChatSession: 没有为 {url} 准备响应")
        if len(self._queue) == 1:
            return self._queue[0]
        return self._queue.pop(0)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def payload(self, index: int = 0) -> Dict[str, Any]:
        return self.calls[index]["payload"]

    def prompts(self, index: int = 0) -> str:
        """把某次调用的 messages 拼成一个字符串，便于断言"""
        messages = self.payload(index).get("messages") or []
        return "\n".join(str(m.get("content") or "") for m in messages)


class RaisingChatSession:
    """每次都抛异常的会话（超时 / 网络错误）"""

    def __init__(self, exc: Optional[Exception] = None):
        self.exc = exc or TimeoutError("timed out")
        self.call_count = 0

    def post(self, *args, **kwargs):
        self.call_count += 1
        raise self.exc


class FakeReadmeClient:
    """模拟 GitHubAPIClient 的 README 接口"""

    def __init__(
        self,
        readmes: Optional[Dict[str, str]] = None,
        *,
        failing: Sequence[str] = (),
        raising: Sequence[str] = (),
    ):
        self.readmes = dict(readmes or {})
        self.failing = {name.lower() for name in failing}
        self.raising = {name.lower() for name in raising}
        self.requests: List[str] = []

    def get_readme_text(self, full_name: str) -> Optional[str]:
        self.requests.append(full_name)
        key = full_name.lower()
        if key in self.raising:
            raise RuntimeError("boom")
        if key in self.failing:
            return None
        return self.readmes.get(full_name) or self.readmes.get(key)


# ----------------------------------------------------------------------
# 数据工厂
# ----------------------------------------------------------------------
def make_record(
    full_name: str,
    *,
    stars: int = 1000,
    description: str = "a demo repository",
    topics: Optional[Sequence[str]] = None,
    language: str = "Python",
    created_at: str = "2026-08-01T00:00:00Z",
    pushed_at: str = "2026-08-22T00:00:00Z",
    trending_rank: Optional[int] = None,
    forks: int = 10,
) -> RepoRecord:
    """构造一个 RepoRecord"""
    return RepoRecord(
        full_name=full_name,
        stars=stars,
        forks=forks,
        language=language,
        description=description,
        topics=list(topics or []),
        created_at=created_at,
        updated_at=pushed_at,
        pushed_at=pushed_at,
        trending_rank=trending_rank,
        api_enriched=True,
    )


def make_scored(
    full_name: str,
    *,
    score: float = 50.0,
    delta_24h: Optional[int] = 100,
    delta_7d: Optional[int] = 500,
    **record_kwargs,
) -> ScoredRepo:
    """构造一个 ScoredRepo（不经过 rank_repositories，便于精确控制分数）"""
    return ScoredRepo(
        record=make_record(full_name, **record_kwargs),
        delta=StarDelta(
            delta_stars_24h=delta_24h,
            delta_stars_7d=delta_7d,
            average_daily_growth_7d=(delta_7d / 7.0) if delta_7d is not None else None,
        ),
        score=score,
    )


def analysis_payload(full_name: str, **overrides) -> Dict[str, Any]:
    """构造一条符合 schema 的模型输出"""
    payload = {
        "full_name": full_name,
        "summary_zh": f"{full_name} 是一个用于演示的项目。",
        "problem": "解决演示场景下的问题。",
        "category": "Developer Tool",
        "tech_stack": ["Python", "FastAPI"],
        "use_cases": ["本地开发", "CI 集成"],
        "why_hot": {
            "summary": "24h Star 增长明显，且近期仍在更新。",
            "confidence": "medium",
            "evidence": ["delta_24h=100", "pushed_at 近期"],
        },
        "maturity": "growing",
        "relevance_score": 70,
        "relevance_reason": "与 AI Agent 方向相关。",
        "recommended_action": "watch",
        "recommendation_reason": "先观察一段时间。",
    }
    payload.update(overrides)
    return payload


def batch_payload(full_names: Sequence[str], **overrides) -> Dict[str, Any]:
    """构造一批仓库的模型输出"""
    return {
        "repositories": [analysis_payload(name, **overrides) for name in full_names]
    }


def synthesis_payload(**overrides) -> Dict[str, Any]:
    """构造一份 daily synthesis 输出"""
    payload = {
        "headline": "Coding Agent 与 MCP 工具链继续保持较高热度。",
        "signals": [
            "Agent 开发工具占 Hot20 中较高比例",
            "两个 Scientific AI 新项目出现较快 Star 增长",
        ],
        "rising_categories": ["Coding Agent", "Scientific AI"],
        "watch_tomorrow": ["本地部署类 AI 工具"],
    }
    payload.update(overrides)
    return payload


def no_sleep(_seconds: float) -> None:
    """替代 time.sleep"""
    return None
