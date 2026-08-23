# coding=utf-8
"""
AI 输出的 Schema、校验与消毒

**模型的输出永远当成不可信数据**：
JSON parse → schema validation → sanitize → fallback，四步缺一不可。

本模块只做纯函数式的数据处理，不发起任何网络请求，因此可以被完整单测。

关键约束（规格 §13 / §14）
--------------------------
- ``tech_stack`` ≤ 8 项，``use_cases`` ≤ 5 项
- ``relevance_score`` 必须 clamp 到 0~100
- ``why_hot.confidence`` ∈ {low, medium, high}
- ``maturity`` ∈ {experimental, growing, mature, unknown}
- ``recommended_action`` ∈ {skip, watch, star, study, try}
- ``why_hot.evidence`` 只能引用**我们实际喂进去的**指标；
  引用不到的证据一律丢弃，并在必要时把 confidence 降级
- 出现「融资 / 大厂采用 / 名人推荐 / 媒体报道」等外部事件断言但没有证据时，
  直接替换成中性表述（见 ``NO_EXTERNAL_EVIDENCE_TEXT``）
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# ---- 枚举 --------------------------------------------------------------
CONFIDENCE_LEVELS = ("low", "medium", "high")
DEFAULT_CONFIDENCE = "low"

MATURITY_LEVELS = ("experimental", "growing", "mature", "unknown")
DEFAULT_MATURITY = "unknown"

ACTIONS = ("skip", "watch", "star", "study", "try")
DEFAULT_ACTION = "watch"

ACTION_LABELS: Dict[str, str] = {
    "skip": "跳过",
    "watch": "关注",
    "star": "收藏",
    "study": "阅读源码/架构",
    "try": "值得亲自试用",
}

ACTION_ICONS: Dict[str, str] = {
    "skip": "⏭️",
    "watch": "👀",
    "star": "⭐",
    "study": "📖",
    "try": "🧪",
}

CATEGORIES = (
    "Coding Agent",
    "Developer Tool",
    "Scientific AI",
    "LLM Infrastructure",
    "Research Tool",
    "Knowledge Management",
    "Computer Vision",
    "Data Tool",
    "Security",
    "Productivity",
    "Other",
)
DEFAULT_CATEGORY = "Other"

# ---- 长度限制 ----------------------------------------------------------
MAX_TECH_STACK = 8
MAX_USE_CASES = 5
MAX_EVIDENCE = 5
MAX_SIGNALS = 5
MAX_RISING_CATEGORIES = 5
MAX_WATCH_TOMORROW = 5

LEN_SUMMARY = 120
LEN_PROBLEM = 200
LEN_REASON = 200
LEN_WHY_HOT = 240
LEN_ITEM = 60
LEN_EVIDENCE = 120
LEN_HEADLINE = 120
LEN_SIGNAL = 120

# ---- 幻觉控制 ----------------------------------------------------------
#
# 允许出现在 why_hot.evidence 里的证据键：**只有我们真的喂进 prompt 的字段**。
# 中英两套写法都认，模型用哪种都能对上。
EVIDENCE_KEYS: Dict[str, Sequence[str]] = {
    "stars": ("stars", "star", "星标", "总星"),
    "delta_24h": ("delta_24h", "24h", "24 小时", "日增"),
    "delta_7d": ("delta_7d", "7d", "7 日", "7日", "周增"),
    "trending_rank": ("trending_rank", "trending", "趋势榜", "排名"),
    "created_at": ("created_at", "创建"),
    "updated_at": ("updated_at", "更新时间"),
    "pushed_at": ("pushed_at", "最近提交", "推送"),
    "heat_score": ("heat_score", "heat score", "热度分"),
    "forks": ("forks", "fork"),
    "language": ("language", "语言"),
    "open_issues": ("open_issues", "issue"),
}

# 未经证据支持就不允许出现的外部事件断言
_HALLUCINATION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"融资|投资|估值|收购|并购|上市|轮融",
        r"被[^。；;]{0,12}(公司|大厂|巨头|团队)[^。；;]{0,8}(采用|使用|引入|接入)",
        r"(谷歌|google|微软|microsoft|苹果|apple|meta|openai|英伟达|nvidia|字节|腾讯|阿里|百度|华为)"
        r"[^。；;]{0,10}(采用|背书|官宣|投资|推荐|发布|支持)",
        r"(马斯克|musk|奥特曼|altman|大佬|名人|网红|kol)[^。；;]{0,10}(推荐|转发|点赞|提及)",
        r"(媒体|新闻|报道|头条|hacker\s*news|hn\s*首页|product\s*hunt|reddit|twitter|"
        r"推特|微博|知乎|x\s*平台|youtube|b\s*站)[^。；;]{0,10}(报道|热议|传播|讨论|刷屏|首页|置顶|推荐)",
        r"(发布|上线|推出|开源)了?[^。；;]{0,14}(重大|重磅|里程碑|爆款)",
        r"(病毒式|爆火|刷屏|出圈|一夜之间|席卷)",
        r"(大会|发布会|峰会|conference|keynote)[^。；;]{0,10}(亮相|发布|宣布)",
    )
]

# 没有外部证据时统一使用的中性表述（规格 §14 的示例文案）
NO_EXTERNAL_EVIDENCE_TEXT = (
    "从现有 GitHub 数据看，该项目近期关注度明显上升，"
    "目前没有足够证据确定具体外部驱动事件。"
)


# ----------------------------------------------------------------------
# 基础清洗
# ----------------------------------------------------------------------
def clean_text(value: Any, max_length: int) -> str:
    """
    把模型返回的任意值转成安全的纯文本

    - 非字符串 → str()（dict/list 直接丢弃，返回空串）
    - 压缩所有空白（含换行），避免破坏邮件排版
    - 去掉控制字符
    - 超长截断并加省略号

    Note:
        这里**不做 HTML 转义**——转义是渲染层 ``report.esc()`` 的职责，
        在这里转义会导致数据层出现 ``&amp;`` 之类的脏数据。
    """
    if value is None or isinstance(value, (dict, list, tuple, set, bool)):
        return ""
    text = str(value)
    # 去控制字符（保留可见字符与空格）
    text = "".join(ch for ch in text if ch == " " or ch.isprintable())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if max_length > 0 and len(text) > max_length:
        text = text[: max_length - 1].rstrip() + "…"
    return text


def clean_list(value: Any, *, max_items: int, max_length: int) -> List[str]:
    """把模型返回的列表清洗成去重、限长、限量的字符串列表"""
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        items: Sequence[Any] = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        return []

    cleaned: List[str] = []
    seen = set()
    for item in items:
        text = clean_text(item, max_length)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def clamp_score(value: Any, *, default: int = 0) -> int:
    """把 relevance_score 夹到 0~100（非法值 → default）"""
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        # "85 分" 这类脏数据：抽第一个数字
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        if not match:
            return default
        try:
            number = float(match.group(0))
        except ValueError:  # pragma: no cover
            return default
    if number != number:  # NaN
        return default
    return int(max(0.0, min(100.0, round(number))))


def choose_enum(value: Any, allowed: Sequence[str], default: str) -> str:
    """把模型返回的枚举值归一化（大小写/空白容错，非法 → default）"""
    text = clean_text(value, 40).lower().strip()
    if not text:
        return default
    for option in allowed:
        if text == option:
            return option
    for option in allowed:
        if option in text:
            return option
    return default


def choose_category(value: Any) -> str:
    """归一化 category（不在建议列表里时保留模型原文，但做长度限制）"""
    text = clean_text(value, 40)
    if not text:
        return DEFAULT_CATEGORY
    lowered = text.lower()
    for option in CATEGORIES:
        if lowered == option.lower():
            return option
    return text


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------
@dataclass
class WhyHot:
    """为什么值得关注（evidence-based，禁止编造外部事件）"""

    summary: str = ""
    confidence: str = DEFAULT_CONFIDENCE
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


@dataclass
class RepoAnalysis:
    """单个仓库的 AI 分析结果"""

    full_name: str
    summary_zh: str = ""
    problem: str = ""
    category: str = DEFAULT_CATEGORY
    tech_stack: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    why_hot: WhyHot = field(default_factory=WhyHot)
    maturity: str = DEFAULT_MATURITY
    relevance_score: int = 0
    relevance_reason: str = ""
    recommended_action: str = DEFAULT_ACTION
    recommendation_reason: str = ""
    # 静态字段是否来自缓存（仅用于统计与日志）
    from_cache: bool = False

    @property
    def action_label(self) -> str:
        return ACTION_LABELS.get(self.recommended_action, self.recommended_action)

    @property
    def action_icon(self) -> str:
        return ACTION_ICONS.get(self.recommended_action, "•")

    @property
    def has_content(self) -> bool:
        """是否有任何可展示内容（全空的结果视为失败）"""
        return bool(
            self.summary_zh
            or self.problem
            or self.why_hot.summary
            or self.tech_stack
            or self.use_cases
        )

    def static_fields(self) -> Dict[str, Any]:
        """可缓存的相对静态字段（规格 §18）"""
        return {
            "summary_zh": self.summary_zh,
            "problem": self.problem,
            "category": self.category,
            "tech_stack": list(self.tech_stack),
            "use_cases": list(self.use_cases),
            "maturity": self.maturity,
        }

    def apply_static_fields(self, payload: Dict[str, Any]) -> None:
        """用缓存里的静态字段填充（缓存内容同样要过消毒）"""
        if not isinstance(payload, dict):
            return
        self.summary_zh = clean_text(payload.get("summary_zh"), LEN_SUMMARY)
        self.problem = clean_text(payload.get("problem"), LEN_PROBLEM)
        self.category = choose_category(payload.get("category"))
        self.tech_stack = clean_list(
            payload.get("tech_stack"), max_items=MAX_TECH_STACK, max_length=LEN_ITEM
        )
        self.use_cases = clean_list(
            payload.get("use_cases"), max_items=MAX_USE_CASES, max_length=LEN_ITEM
        )
        self.maturity = choose_enum(payload.get("maturity"), MATURITY_LEVELS, DEFAULT_MATURITY)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "full_name": self.full_name,
            "summary_zh": self.summary_zh,
            "problem": self.problem,
            "category": self.category,
            "tech_stack": list(self.tech_stack),
            "use_cases": list(self.use_cases),
            "why_hot": self.why_hot.to_dict(),
            "maturity": self.maturity,
            "relevance_score": self.relevance_score,
            "relevance_reason": self.relevance_reason,
            "recommended_action": self.recommended_action,
            "recommendation_reason": self.recommendation_reason,
            "from_cache": self.from_cache,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], *, full_name: str = "") -> Optional["RepoAnalysis"]:
        """从（缓存/序列化）字典还原，走同一套消毒"""
        if not isinstance(payload, dict):
            return None
        name = clean_text(payload.get("full_name"), 140) or full_name
        if not name:
            return None
        analysis = parse_repo_analysis(payload, full_name=name)
        if analysis is not None:
            analysis.from_cache = payload.get("from_cache") is True
        return analysis


@dataclass
class DailySynthesis:
    """每日趋势总结"""

    headline: str = ""
    signals: List[str] = field(default_factory=list)
    rising_categories: List[str] = field(default_factory=list)
    watch_tomorrow: List[str] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return bool(self.headline or self.signals or self.rising_categories or self.watch_tomorrow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "headline": self.headline,
            "signals": list(self.signals),
            "rising_categories": list(self.rising_categories),
            "watch_tomorrow": list(self.watch_tomorrow),
        }


@dataclass
class AIUsage:
    """Token / 请求 / 缓存统计（规格 §25）"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    repositories_analyzed: int = 0

    def add_request(
        self,
        *,
        success: bool,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        """累加一次 API 请求的用量"""
        self.requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.prompt_tokens += max(0, int(prompt_tokens or 0))
        self.completion_tokens += max(0, int(completion_tokens or 0))
        counted = int(total_tokens or 0)
        if counted <= 0:
            counted = max(0, int(prompt_tokens or 0)) + max(0, int(completion_tokens or 0))
        self.total_tokens += max(0, counted)

    def merge(self, other: Optional["AIUsage"]) -> None:
        """合并另一份统计"""
        if other is None:
            return
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.requests += other.requests
        self.successful_requests += other.successful_requests
        self.failed_requests += other.failed_requests
        self.cache_hits += other.cache_hits
        self.repositories_analyzed += other.repositories_analyzed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "requests": self.requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "cache_hits": self.cache_hits,
            "repositories_analyzed": self.repositories_analyzed,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "AIUsage":
        usage = cls()
        if not isinstance(payload, dict):
            return usage
        for key in usage.to_dict():
            value = payload.get(key)
            try:
                setattr(usage, key, max(0, int(value)))
            except (TypeError, ValueError):
                continue
        return usage


# ----------------------------------------------------------------------
# 解析 + 校验
# ----------------------------------------------------------------------
def _parse_why_hot(payload: Any) -> WhyHot:
    """解析 why_hot（模型可能直接给字符串）"""
    if isinstance(payload, str):
        return WhyHot(summary=clean_text(payload, LEN_WHY_HOT))
    if not isinstance(payload, dict):
        return WhyHot()
    return WhyHot(
        summary=clean_text(payload.get("summary"), LEN_WHY_HOT),
        confidence=choose_enum(payload.get("confidence"), CONFIDENCE_LEVELS, DEFAULT_CONFIDENCE),
        evidence=clean_list(
            payload.get("evidence"), max_items=MAX_EVIDENCE, max_length=LEN_EVIDENCE
        ),
    )


def parse_repo_analysis(payload: Any, *, full_name: str) -> Optional[RepoAnalysis]:
    """
    把模型返回的一个仓库对象转成 ``RepoAnalysis``

    Args:
        payload: 模型返回的 dict
        full_name: 该仓库的权威 full_name（**以我们的输入为准**，
            不采信模型回写的名字，避免它张冠李戴）

    Returns:
        ``RepoAnalysis``；payload 不是对象时返回 None
    """
    if not isinstance(payload, dict) or not full_name:
        return None

    return RepoAnalysis(
        full_name=full_name,
        summary_zh=clean_text(payload.get("summary_zh"), LEN_SUMMARY),
        problem=clean_text(payload.get("problem"), LEN_PROBLEM),
        category=choose_category(payload.get("category")),
        tech_stack=clean_list(
            payload.get("tech_stack"), max_items=MAX_TECH_STACK, max_length=LEN_ITEM
        ),
        use_cases=clean_list(
            payload.get("use_cases"), max_items=MAX_USE_CASES, max_length=LEN_ITEM
        ),
        why_hot=_parse_why_hot(payload.get("why_hot")),
        maturity=choose_enum(payload.get("maturity"), MATURITY_LEVELS, DEFAULT_MATURITY),
        relevance_score=clamp_score(payload.get("relevance_score")),
        relevance_reason=clean_text(payload.get("relevance_reason"), LEN_REASON),
        recommended_action=choose_enum(
            payload.get("recommended_action"), ACTIONS, DEFAULT_ACTION
        ),
        recommendation_reason=clean_text(payload.get("recommendation_reason"), LEN_REASON),
    )


def parse_daily_synthesis(payload: Any) -> Optional[DailySynthesis]:
    """解析每日趋势总结（内容全空时返回 None，让报告走「不可用」分支）"""
    if not isinstance(payload, dict):
        return None
    synthesis = DailySynthesis(
        headline=clean_text(payload.get("headline"), LEN_HEADLINE),
        signals=clean_list(payload.get("signals"), max_items=MAX_SIGNALS, max_length=LEN_SIGNAL),
        rising_categories=clean_list(
            payload.get("rising_categories"), max_items=MAX_RISING_CATEGORIES, max_length=LEN_ITEM
        ),
        watch_tomorrow=clean_list(
            payload.get("watch_tomorrow"), max_items=MAX_WATCH_TOMORROW, max_length=LEN_ITEM
        ),
    )
    return synthesis if synthesis.has_content else None


# ----------------------------------------------------------------------
# 幻觉控制
# ----------------------------------------------------------------------
def looks_like_external_claim(text: str) -> bool:
    """文本是否声称了「外部驱动事件」（融资 / 大厂采用 / 媒体传播 …）"""
    if not text:
        return False
    return any(pattern.search(text) for pattern in _HALLUCINATION_PATTERNS)


def filter_evidence(evidence: Sequence[str], allowed_keys: Sequence[str]) -> List[str]:
    """
    只保留引用了**实际输入指标**的证据条目

    Args:
        evidence: 模型给出的证据列表
        allowed_keys: 本次真正喂给模型的指标键（如 stars / delta_24h）
    """
    if not evidence:
        return []
    aliases: List[str] = []
    for key in allowed_keys or ():
        aliases.extend(EVIDENCE_KEYS.get(key, (key,)))
    aliases = [alias.lower() for alias in aliases if alias]
    if not aliases:
        return []

    kept: List[str] = []
    for item in evidence:
        lowered = (item or "").lower()
        if any(alias in lowered for alias in aliases):
            kept.append(item)
    return kept


def apply_hallucination_guard(
    analysis: RepoAnalysis, allowed_keys: Sequence[str]
) -> RepoAnalysis:
    """
    「为什么火」幻觉控制（规格 §14）

    1. 丢弃引用不到实际输入的 evidence
    2. evidence 被清空后，confidence 一律降级为 low
    3. summary 里出现无证据支撑的外部事件断言 → 整句替换成中性表述

    这是**就地修改**并返回同一个对象，便于链式使用。
    """
    why_hot = analysis.why_hot
    why_hot.evidence = filter_evidence(why_hot.evidence, allowed_keys)

    if not why_hot.evidence and why_hot.confidence != "low":
        why_hot.confidence = "low"

    if looks_like_external_claim(why_hot.summary):
        # 外部事件永远不可能被 GitHub 指标证明，直接替换，不做「部分保留」
        why_hot.summary = NO_EXTERNAL_EVIDENCE_TEXT
        why_hot.confidence = "low"

    if looks_like_external_claim(analysis.relevance_reason):
        analysis.relevance_reason = ""
    if looks_like_external_claim(analysis.recommendation_reason):
        analysis.recommendation_reason = ""

    return analysis
