# coding=utf-8
"""
DeepSeek 客户端（OpenAI 兼容的 chat/completions 接口）

设计要求
--------
- **Thinking 明确关闭**：请求体固定携带 ``{"thinking": {"type": "disabled"}}``，
  不依赖模型默认行为（规格 §4）——降 Token、降费用、提速、稳 JSON
- 超时 / 有限重试 / 429 / 5xx / malformed JSON 全部安全降级，
  任何失败都只返回 ``ChatResult(ok=False)``，**绝不抛异常打断日报**（规格 §31）
- 逐次累计 ``usage.prompt_tokens`` / ``completion_tokens`` / ``total_tokens``
- API Key 只出现在请求头；日志、异常文本一律经 ``redact`` 脱敏（规格 §17）

与 ``github_radar/github_api.py`` 保持同样的风格：串行、限次重试、不并发。
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..logging_utils import redact, warn
from .schemas import AIUsage

try:  # requests 是项目已有依赖；缺失时仍允许 import 本模块（便于纯离线测试）
    import requests
except ImportError:  # pragma: no cover - 正常环境不会走到
    requests = None

CHAT_COMPLETIONS_PATH = "/chat/completions"

DEFAULT_USER_AGENT = "TrendRadar-GitHubRadar-AI/1.0"

# 有限重试：网络异常 / 429 / 5xx
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BASE_DELAY = 2.0
# 限流时最多愿意等待的秒数，超过直接放弃（不让 workflow 挂住）
MAX_RETRY_WAIT = 20.0

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 4000

# 明确关闭 thinking（规格 §4：不要依赖模型默认行为）
THINKING_DISABLED: Dict[str, Any] = {"type": "disabled"}

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


@dataclass
class ChatResult:
    """一次 chat 调用的结果（永远不抛异常，用字段表达失败）"""

    ok: bool = False
    content: str = ""
    error: str = ""
    status_code: Optional[int] = None
    prompt_tokens: int = 0
    # 命中 / 未命中前缀缓存的输入 token（服务端不返回时为 0，费用按未命中估算）
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # 本次调用实际发生的 HTTP 请求次数（含重试）
    http_requests: int = 0

    @property
    def failed(self) -> bool:
        return not self.ok


def _int_field(payload: Any, key: str) -> int:
    """从 usage 字典里安全读取整数"""
    if not isinstance(payload, dict):
        return 0
    value = payload.get(key)
    if value is None or isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def extract_json(content: str) -> Optional[Dict[str, Any]]:
    """
    从模型回复中提取 JSON 对象

    依次尝试：
    1. 直接 ``json.loads``
    2. 剥掉 ```json ... ``` 代码块
    3. 截取第一个 ``{`` 到最后一个 ``}``（模型常在 JSON 前后带一句废话）

    Returns:
        dict；解析失败返回 None（**绝不抛异常**）
    """
    if not content:
        return None
    text = str(content).strip()

    for candidate in _json_candidates(text):
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            # 模型偶尔直接返回数组，统一包成 {"repositories": [...]}
            return {"repositories": data}
    return None


def _json_candidates(text: str) -> List[str]:
    """生成可能是 JSON 的片段（按可信度排序）"""
    candidates = [text]

    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    seen = set()
    unique: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


class DeepSeekClient:
    """DeepSeek chat/completions 客户端（只做一件事：把 messages 换成文本）"""

    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        api_base: str,
        timeout: int = 60,
        session: Any = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
        sleep_func: Callable[[float], None] = time.sleep,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        """
        Args:
            api_key: DeepSeek API Key（**绝不会被打印**）
            model: 模型名（默认 deepseek-v4-flash，可由 DEEPSEEK_MODEL 覆盖）
            api_base: API 根地址
            timeout: 单次请求超时（秒）
            session: 注入的 HTTP 会话（需实现 ``post``），默认 requests.Session
            max_retries: 失败重试次数上限（不含首次请求）
            retry_base_delay: 指数退避基数（秒）
            sleep_func: 睡眠函数（测试注入以避免真实等待）
            temperature: 采样温度（结构化抽取任务用低温）
            max_tokens: 单次最大生成长度
        """
        self._api_key = api_key or ""
        self.model = model
        self.api_base = (api_base or "").rstrip("/")
        self.timeout = timeout
        self._session = session
        self.max_retries = max(0, int(max_retries))
        self.retry_base_delay = retry_base_delay
        self._sleep = sleep_func
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.user_agent = user_agent

        self.usage = AIUsage()
        self.last_error: str = ""

    # ------------------------------------------------------------------
    # 基础设施
    # ------------------------------------------------------------------
    @property
    def endpoint(self) -> str:
        return f"{self.api_base}{CHAT_COMPLETIONS_PATH}"

    def _get_session(self) -> Any:
        if self._session is None:
            if requests is None:  # pragma: no cover
                raise RuntimeError("requests 未安装，无法调用 DeepSeek")
            self._session = requests.Session()
        return self._session

    def _headers(self) -> Dict[str, str]:
        """请求头（Authorization 只在这里出现，绝不写日志）"""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }

    def build_payload(
        self, messages: List[Dict[str, str]], *, json_mode: bool = True
    ) -> Dict[str, Any]:
        """
        构造请求体

        ``thinking: {"type": "disabled"}`` 是硬性要求，
        不允许被 kwargs 覆盖（见规格 §4）。
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "thinking": dict(THINKING_DISABLED),
            "temperature": self.temperature,
            "stream": False,
        }
        if self.max_tokens and self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _safe(self, text: Any) -> str:
        """脱敏任意文本（防止第三方异常里回显 Key）"""
        return redact(str(text), self._api_key)

    # ------------------------------------------------------------------
    # 调用
    # ------------------------------------------------------------------
    def chat(
        self, messages: List[Dict[str, str]], *, json_mode: bool = True
    ) -> ChatResult:
        """
        发起一次 chat 调用

        Returns:
            ``ChatResult``；**任何失败都以 ok=False 返回**，不抛异常。
            usage 会同时累加到 ``self.usage``（供日报统计）。
        """
        payload = self.build_payload(messages, json_mode=json_mode)
        result = ChatResult()
        attempt = 0

        while attempt <= self.max_retries:
            try:
                result.http_requests += 1
                response = self._get_session().post(
                    self.endpoint,
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
            except Exception as exc:  # 网络异常 / 超时统一降级重试
                message = f"{type(exc).__name__}: {self._safe(exc)}"
                self.last_error = message
                attempt += 1
                if attempt > self.max_retries:
                    warn(f"[AI] DeepSeek 请求失败（已重试 {self.max_retries} 次）：{message}")
                    result.error = message
                    self._record(result)
                    return result
                delay = self.retry_base_delay * (2 ** (attempt - 1))
                warn(f"[AI] DeepSeek 请求异常（{message}），{delay:.1f}s 后重试（{attempt}/{self.max_retries}）")
                self._sleep(delay)
                continue

            status = getattr(response, "status_code", None)
            result.status_code = status

            if status == 200:
                return self._handle_success(response, result)

            if status in (429,) or (status is not None and 500 <= int(status) < 600):
                attempt += 1
                if attempt > self.max_retries:
                    result.error = f"HTTP {status}"
                    warn(f"[AI] DeepSeek 返回 {status}，已达重试上限，本批降级")
                    self._record(result)
                    return result
                delay = self._retry_delay(response, attempt)
                if delay > MAX_RETRY_WAIT:
                    result.error = f"HTTP {status}（需等待 {delay:.0f}s，超出可接受范围）"
                    warn(f"[AI] DeepSeek 限流需等待 {delay:.0f}s，超出上限，本批降级")
                    self._record(result)
                    return result
                warn(f"[AI] DeepSeek 返回 {status}，{delay:.1f}s 后重试（{attempt}/{self.max_retries}）")
                self._sleep(delay)
                continue

            # 4xx（401/403/400 等）重试没有意义
            result.error = f"HTTP {status}"
            warn(f"[AI] DeepSeek 返回 {status}，不重试，本批降级（{self._error_hint(status)}）")
            self._record(result)
            return result

        result.error = result.error or "unknown"
        self._record(result)
        return result

    @staticmethod
    def _error_hint(status: Any) -> str:
        """给出无敏感信息的排查提示"""
        hints = {
            400: "请求体被拒绝",
            401: "API Key 无效或已过期",
            402: "账户余额不足",
            403: "无权访问该模型",
            404: "接口地址或模型名不存在",
            422: "参数不被接受",
        }
        try:
            return hints.get(int(status), "未预期的响应状态")
        except (TypeError, ValueError):
            return "未预期的响应状态"

    def _retry_delay(self, response: Any, attempt: int) -> float:
        """计算重试等待时间（优先 Retry-After）"""
        headers = getattr(response, "headers", None) or {}
        raw = None
        try:
            raw = headers.get("Retry-After")
        except AttributeError:
            raw = None
        if raw is not None:
            try:
                return max(0.0, float(str(raw).strip()))
            except (TypeError, ValueError):
                pass
        return self.retry_base_delay * (2 ** (attempt - 1))

    def _handle_success(self, response: Any, result: ChatResult) -> ChatResult:
        """解析 200 响应"""
        data: Any = None
        try:
            data = response.json()
        except Exception:
            try:
                data = json.loads(getattr(response, "text", "") or "")
            except Exception:
                data = None

        if not isinstance(data, dict):
            result.error = "响应不是合法 JSON"
            warn("[AI] DeepSeek 响应不是合法 JSON，本批降级")
            self._record(result)
            return result

        usage = data.get("usage")
        result.prompt_tokens = _int_field(usage, "prompt_tokens")
        # DeepSeek 的前缀缓存明细；老接口 / 其它服务商没有这两个字段时保持 0，
        # 由 AIUsage 统一按「全部未命中」安全降级
        result.prompt_cache_hit_tokens = _int_field(usage, "prompt_cache_hit_tokens")
        result.prompt_cache_miss_tokens = _int_field(usage, "prompt_cache_miss_tokens")
        result.completion_tokens = _int_field(usage, "completion_tokens")
        result.total_tokens = _int_field(usage, "total_tokens")

        content = self._extract_content(data)
        if not content:
            result.error = "响应中没有可用内容"
            warn("[AI] DeepSeek 响应中没有可用内容，本批降级")
            self._record(result)
            return result

        result.ok = True
        result.content = content
        self._record(result)
        return result

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        """从 choices[0].message.content 取文本（兼容 content 为块数组的情况）"""
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        message = first.get("message")
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(str(item))
            content = "\n".join(part for part in parts if part)
        if content is None:
            return ""
        return str(content).strip()

    def _record(self, result: ChatResult) -> None:
        """把一次调用计入 usage 统计"""
        self.usage.add_request(
            success=result.ok,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            prompt_cache_hit_tokens=result.prompt_cache_hit_tokens,
            prompt_cache_miss_tokens=result.prompt_cache_miss_tokens,
        )
        if not result.ok and result.error:
            self.last_error = result.error

    # ------------------------------------------------------------------
    # 结构化 JSON 调用
    # ------------------------------------------------------------------
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        retry_instruction: str,
        json_retry_limit: int = 1,
    ) -> "JsonResult":
        """
        调用模型并要求返回 JSON

        malformed JSON 时最多再试 ``json_retry_limit`` 次（规格 §15：最多 1 次），
        仍然失败就返回 ``ok=False``，由调用方走「基础数据」降级。

        Returns:
            ``JsonResult``
        """
        outcome = JsonResult()
        current = list(messages)

        for attempt in range(max(0, int(json_retry_limit)) + 1):
            result = self.chat(current)
            outcome.attempts += 1
            outcome.http_requests += result.http_requests
            outcome.chat_results.append(result)

            if not result.ok:
                outcome.error = result.error
                # 传输层失败：客户端内部已经重试过，这里不再叠加 JSON 重试
                return outcome

            data = extract_json(result.content)
            if data is not None:
                outcome.ok = True
                outcome.data = data
                return outcome

            outcome.error = "malformed JSON"
            if attempt >= max(0, int(json_retry_limit)):
                warn("[AI] DeepSeek 返回的不是合法 JSON，且已达 JSON 重试上限，本批降级")
                return outcome

            warn("[AI] DeepSeek 返回的不是合法 JSON，按规格再重试 1 次")
            current = list(messages) + [
                {"role": "assistant", "content": (result.content or "")[:500]},
                {"role": "user", "content": retry_instruction},
            ]

        return outcome


@dataclass
class JsonResult:
    """``chat_json`` 的结果"""

    ok: bool = False
    data: Optional[Dict[str, Any]] = None
    error: str = ""
    attempts: int = 0
    http_requests: int = 0
    chat_results: List[ChatResult] = field(default_factory=list)
