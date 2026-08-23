# coding=utf-8
"""
用户兴趣 Profile 与 deterministic keyword score

两件事：
1. 从 ``config/github_radar_profile.yaml`` 读取兴趣画像（容错到极致）
2. 在**完全不调用 AI** 的前提下算出 0~100 的 ``keyword_score``

为什么必须有 deterministic score
--------------------------------
- AI 可能挂掉：挂了也要有一个可用的个性化信号
- AI 可能漂移：同一个项目今天 90 分明天 40 分，deterministic 分数是锚
- 它还决定「哪些 Hot20 之外的项目值得送进 AI」（Priority 1 候选）

评分口径（可解释、可单测）
--------------------------
命中位置的权重::

    repo name    1.00     # 名字里就带关键词，信号最强
    topics       0.90     # 作者自己打的标签
    description  0.70
    readme       0.40     # 只用前若干字符，噪声最大

单个 interest category::

    raw   = Σ(每个命中关键词的最佳位置权重)，同一关键词只算一次
    ratio = min(1.0, raw / SATURATION)      # 命中两处强信号即满分
    cat   = ratio × category.weight

总分::

    primary = max(cat)                      # 主兴趣
    others  = Σ(cat) − primary              # 其它兴趣的广度加成
    score   = 100 × min(1, (primary + 0.25×others) / max_weight)

因此「完全命中权重 1.0 的兴趣」= 100 分；
「只命中 productivity(0.7)」≈ 70 分；跨兴趣命中会有加成但不会超过 100。
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from ..logging_utils import log, warn
from ..models import RepoRecord

DEFAULT_PROFILE_PATH = "config/github_radar_profile.yaml"

# 命中位置权重
FIELD_WEIGHTS: Dict[str, float] = {
    "name": 1.0,
    "topics": 0.9,
    "description": 0.7,
    "readme": 0.4,
}

# 单个兴趣达到满分所需的累计权重
SATURATION = 2.0

# 其它兴趣的广度加成系数
BREADTH_BONUS = 0.25

# 参与匹配的 README 前缀长度（keyword score 只看开头，避免长文档噪声）
README_MATCH_CHARS = 2000

# weight 合法范围
MIN_WEIGHT = 0.01
MAX_WEIGHT = 5.0
DEFAULT_WEIGHT = 1.0

# Priority 1（For You 强候选）的 keyword score 门槛
STRONG_MATCH_THRESHOLD = 60.0

# 内置兜底画像：配置文件缺失 / 损坏 / 全空时使用，保证 Radar 永不崩溃
FALLBACK_PROFILE_NAME = "generic-fallback"
FALLBACK_INTERESTS: Dict[str, Dict[str, Any]] = {
    "ai_agents": {
        "weight": 1.0,
        "keywords": ["AI Agent", "LLM", "agent framework", "MCP", "coding agent"],
    },
    "developer_tools": {
        "weight": 0.8,
        "keywords": ["developer tools", "automation", "workflow", "CLI"],
    },
    "machine_learning": {
        "weight": 0.8,
        "keywords": ["deep learning", "neural network", "machine learning"],
    },
}


@dataclass
class InterestCategory:
    """一个兴趣类别"""

    name: str
    weight: float = DEFAULT_WEIGHT
    keywords: List[str] = field(default_factory=list)


@dataclass
class UserProfile:
    """用户兴趣画像"""

    name: str = FALLBACK_PROFILE_NAME
    interests: List[InterestCategory] = field(default_factory=list)
    # 是否使用了兜底画像（报告/日志里会如实说明）
    is_fallback: bool = False
    source: str = ""

    @property
    def max_weight(self) -> float:
        """最大兴趣权重（归一化用；空画像返回 1.0 避免除零）"""
        weights = [c.weight for c in self.interests if c.weight > 0]
        return max(weights) if weights else DEFAULT_WEIGHT

    def describe(self) -> str:
        keyword_count = sum(len(c.keywords) for c in self.interests)
        suffix = "（兜底画像）" if self.is_fallback else ""
        return (
            f"[AI] profile: {self.name}{suffix} | "
            f"{len(self.interests)} interests / {keyword_count} keywords"
        )


@dataclass
class KeywordMatch:
    """一个仓库的 deterministic 匹配结果"""

    score: float = 0.0
    top_category: str = ""
    matched_keywords: List[str] = field(default_factory=list)
    category_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def is_strong(self) -> bool:
        """是否达到 For You 强候选门槛"""
        return self.score >= STRONG_MATCH_THRESHOLD

    def describe(self) -> str:
        if not self.matched_keywords:
            return "无关键词命中"
        shown = "、".join(self.matched_keywords[:5])
        return f"{self.top_category}: {shown}"


# ----------------------------------------------------------------------
# 加载
# ----------------------------------------------------------------------
def _coerce_weight(value: Any, category: str) -> float:
    """把任意输入转成合法 weight（非法 → 1.0）"""
    if value is None or isinstance(value, bool):
        return DEFAULT_WEIGHT
    try:
        weight = float(value)
    except (TypeError, ValueError):
        warn(f"[AI] profile 兴趣 {category} 的 weight 非法（{value!r}），按 {DEFAULT_WEIGHT} 处理")
        return DEFAULT_WEIGHT
    if weight != weight or weight in (float("inf"), float("-inf")):  # NaN / inf
        warn(f"[AI] profile 兴趣 {category} 的 weight 非法（{value!r}），按 {DEFAULT_WEIGHT} 处理")
        return DEFAULT_WEIGHT
    if weight <= 0:
        warn(f"[AI] profile 兴趣 {category} 的 weight <= 0，按 {DEFAULT_WEIGHT} 处理")
        return DEFAULT_WEIGHT
    return max(MIN_WEIGHT, min(MAX_WEIGHT, weight))


def _coerce_keywords(value: Any) -> List[str]:
    """把任意输入转成关键词列表（去空、去重、保序）"""
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        items: Sequence[Any] = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        return []

    keywords: List[str] = []
    seen = set()
    for item in items:
        if item is None or isinstance(item, (dict, list, tuple, set)):
            continue
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        keywords.append(text)
    return keywords


def _build_interests(raw: Any) -> List[InterestCategory]:
    """把 YAML 的 interests 段转成 InterestCategory 列表（跳过无效项）"""
    if not isinstance(raw, dict):
        return []

    interests: List[InterestCategory] = []
    for name, payload in raw.items():
        category = str(name or "").strip()
        if not category:
            continue
        if not isinstance(payload, dict):
            warn(f"[AI] profile 兴趣 {category} 格式异常，已跳过")
            continue
        keywords = _coerce_keywords(payload.get("keywords"))
        if not keywords:
            # 空关键词的类别没有任何作用，直接跳过（不算错误）
            continue
        interests.append(
            InterestCategory(
                name=category,
                weight=_coerce_weight(payload.get("weight"), category),
                keywords=keywords,
            )
        )
    return interests


def fallback_profile() -> UserProfile:
    """内置兜底画像"""
    return UserProfile(
        name=FALLBACK_PROFILE_NAME,
        interests=_build_interests(FALLBACK_INTERESTS),
        is_fallback=True,
        source="builtin",
    )


def _default_profile_path() -> Path:
    """默认画像路径：<仓库根>/config/github_radar_profile.yaml"""
    return Path(__file__).resolve().parent.parent.parent / DEFAULT_PROFILE_PATH


def load_profile(path: Optional[Union[str, Path]] = None) -> UserProfile:
    """
    加载兴趣画像

    以下任一情况都会安全退回兜底画像（只 warning，绝不抛异常）：
    - 文件不存在 / 读不出来 / 不是合法 YAML
    - PyYAML 不可用
    - 顶层不是对象、interests 段缺失
    - 所有兴趣的关键词都为空

    Args:
        path: 配置文件路径（默认 ``<repo>/config/github_radar_profile.yaml``）

    Returns:
        ``UserProfile``（``is_fallback`` 标明是否用了兜底）
    """
    target = Path(path) if path else _default_profile_path()

    try:
        import yaml  # 项目已有依赖
    except Exception as exc:  # pragma: no cover - 正常环境不会走到
        warn(f"[AI] PyYAML 不可用（{type(exc).__name__}），使用兜底兴趣画像")
        return fallback_profile()

    if not target.is_file():
        warn(f"[AI] 兴趣画像文件不存在（{target}），使用兜底兴趣画像")
        return fallback_profile()

    try:
        with target.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception as exc:
        warn(f"[AI] 兴趣画像读取失败（{type(exc).__name__}: {exc}），使用兜底兴趣画像")
        return fallback_profile()

    if not isinstance(data, dict):
        warn("[AI] 兴趣画像格式异常（顶层不是对象），使用兜底兴趣画像")
        return fallback_profile()

    interests = _build_interests(data.get("interests"))
    if not interests:
        warn("[AI] 兴趣画像没有任何有效关键词，使用兜底兴趣画像")
        return fallback_profile()

    profile_meta = data.get("profile")
    name = ""
    if isinstance(profile_meta, dict):
        name = str(profile_meta.get("name") or "").strip()

    return UserProfile(
        name=name or "personal-tech-radar",
        interests=interests,
        is_fallback=False,
        source=str(target),
    )


# ----------------------------------------------------------------------
# deterministic keyword score
# ----------------------------------------------------------------------
_NON_WORD = re.compile(r"[^0-9a-z一-鿿]+")


def _compile_keyword(keyword: str) -> Optional["re.Pattern"]:
    """
    把关键词编译成匹配用正则

    - 大小写不敏感
    - 多词关键词允许空格 / 下划线 / 连字符 / 点 差异
      （"AI Agent" 能匹配 "ai-agent"、"AI_Agent"）
    - ASCII 关键词加词边界，避免 "MCP" 命中 "mcphersons"
    """
    text = (keyword or "").strip().lower()
    if not text:
        return None

    parts = [re.escape(part) for part in _NON_WORD.split(text) if part]
    if not parts:
        return None

    body = r"[\s_\-\./]*".join(parts)
    ascii_only = all(part.isascii() for part in parts)
    if ascii_only:
        # 前后不能紧邻字母/数字（中文不受影响）
        body = r"(?<![0-9a-z])" + body + r"(?![0-9a-z])"

    try:
        return re.compile(body)
    except re.error:  # pragma: no cover - 上面已 escape，不应发生
        return None


class KeywordIndex:
    """关键词正则缓存（同一个 profile 在一次运行中会用几十次）"""

    def __init__(self, profile: UserProfile):
        self._patterns: Dict[str, List[tuple]] = {}
        for category in profile.interests:
            compiled = []
            for keyword in category.keywords:
                pattern = _compile_keyword(keyword)
                if pattern is not None:
                    compiled.append((keyword, pattern))
            self._patterns[category.name] = compiled

    def patterns(self, category: str) -> List[tuple]:
        return self._patterns.get(category, [])


def build_haystacks(record: RepoRecord, readme: Optional[str] = None) -> Dict[str, str]:
    """
    构造各字段的小写待匹配文本

    Args:
        record: 仓库
        readme: README 文本（可选，只取前 ``README_MATCH_CHARS`` 字符）
    """
    name = f"{record.owner or ''} {record.name or record.full_name or ''}".lower()
    topics = " ".join(record.topics or []).lower()
    description = (record.description or "").lower()
    readme_text = (readme or "")[:README_MATCH_CHARS].lower()
    return {
        "name": name,
        "topics": topics,
        "description": description,
        "readme": readme_text,
    }


def keyword_score(
    record: RepoRecord,
    profile: UserProfile,
    *,
    readme: Optional[str] = None,
    index: Optional[KeywordIndex] = None,
) -> KeywordMatch:
    """
    计算 deterministic keyword score（0~100）

    Args:
        record: 仓库
        profile: 兴趣画像
        readme: README 文本（可选）
        index: 预编译的关键词索引（批量计算时传入可省重复编译）

    Returns:
        ``KeywordMatch``（含总分、主类别、命中关键词）
    """
    if record is None or not profile.interests:
        return KeywordMatch()

    index = index or KeywordIndex(profile)
    haystacks = build_haystacks(record, readme)

    category_scores: Dict[str, float] = {}
    matched_by_category: Dict[str, List[str]] = {}

    for category in profile.interests:
        raw = 0.0
        hits: List[str] = []
        for keyword, pattern in index.patterns(category.name):
            best = 0.0
            for field_name, weight in FIELD_WEIGHTS.items():
                text = haystacks.get(field_name) or ""
                if text and pattern.search(text):
                    best = max(best, weight)
            if best > 0:
                raw += best
                hits.append(keyword)
        if raw <= 0:
            continue
        ratio = min(1.0, raw / SATURATION)
        category_scores[category.name] = round(ratio * category.weight, 4)
        matched_by_category[category.name] = hits

    if not category_scores:
        return KeywordMatch()

    top_category = max(category_scores, key=lambda key: (category_scores[key], key))
    primary = category_scores[top_category]
    others = sum(category_scores.values()) - primary

    raw_total = primary + BREADTH_BONUS * others
    score = 100.0 * min(1.0, raw_total / max(profile.max_weight, MIN_WEIGHT))

    # 命中关键词：主类别优先，其余按类别得分降序补充
    ordered = [top_category] + sorted(
        (key for key in category_scores if key != top_category),
        key=lambda key: (-category_scores[key], key),
    )
    matched: List[str] = []
    for name in ordered:
        for keyword in matched_by_category.get(name, []):
            if keyword not in matched:
                matched.append(keyword)

    return KeywordMatch(
        score=round(max(0.0, min(100.0, score)), 2),
        top_category=top_category,
        matched_keywords=matched,
        category_scores=category_scores,
    )


def score_all(
    records: Sequence[RepoRecord], profile: UserProfile
) -> Dict[str, KeywordMatch]:
    """
    批量计算 keyword score

    Returns:
        ``{full_name_lower: KeywordMatch}``
    """
    index = KeywordIndex(profile)
    scores: Dict[str, KeywordMatch] = {}
    for record in records or []:
        if record is None or not record.full_name:
            continue
        scores[record.full_name.lower()] = keyword_score(record, profile, index=index)
    return scores


def log_profile(profile: UserProfile) -> None:
    """打印画像信息（不含任何敏感数据）"""
    log(profile.describe())
