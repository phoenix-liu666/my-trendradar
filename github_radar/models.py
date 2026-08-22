# coding=utf-8
"""
数据模型

``RepoRecord`` 是候选仓库在整条链路中的唯一载体：
Trending 解析 / API 补全 / 快照写入 / 排名 / 报告 都使用它。

字段缺失时一律使用 ``None`` 或空集合，**绝不伪造数据**。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 候选来源标识
SOURCE_TRENDING = "trending"
SOURCE_SEARCH_NEW = "search:new"
SOURCE_SEARCH_POPULAR = "search:popular"

# 快照中保存的字段
#
# 刻意保持精简：快照每天提交回仓库，字段越多 git 体积增长越快。
# - 不存 description / topics：它们每天都可能变、体积占比最大，
#   而历史快照的唯一职责是提供「当时的 Star 数」用于算差值；
#   报告需要的描述来自当天实时抓取的数据。
# - 不存 html_url：可由 full_name 推导。
SNAPSHOT_FIELDS = (
    "stars",
    "forks",
    "open_issues",
    "language",
    "created_at",
    "pushed_at",
    "trending_rank",
    "trending_stars_today",
)


def _to_int(value: Any) -> Optional[int]:
    """安全转 int，失败返回 None（不伪造 0）"""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_str(value: Any) -> Optional[str]:
    """安全转字符串，空串归一为 None"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass
class RepoRecord:
    """单个候选仓库"""

    full_name: str
    owner: str = ""
    name: str = ""
    html_url: str = ""
    description: Optional[str] = None
    language: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    stars: Optional[int] = None
    forks: Optional[int] = None
    open_issues: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    pushed_at: Optional[str] = None
    trending_rank: Optional[int] = None
    trending_stars_today: Optional[int] = None
    collected_at: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    # 是否已通过 GitHub API 补全（False 表示只有 Trending HTML 的粗略数据）
    api_enriched: bool = False

    def __post_init__(self) -> None:
        self.full_name = (self.full_name or "").strip().strip("/")
        if self.full_name and (not self.owner or not self.name):
            parts = self.full_name.split("/")
            if len(parts) == 2:
                self.owner = self.owner or parts[0]
                self.name = self.name or parts[1]
        if not self.html_url and self.full_name:
            self.html_url = f"https://github.com/{self.full_name}"
        if self.topics is None:
            self.topics = []
        if self.sources is None:
            self.sources = []

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    @classmethod
    def from_api(cls, payload: Dict[str, Any], source: str = "") -> Optional["RepoRecord"]:
        """
        从 GitHub API 的 repository 对象构造

        Args:
            payload: API 返回的仓库 JSON
            source: 来源标识

        Returns:
            RepoRecord；payload 非法（缺 full_name）时返回 None
        """
        if not isinstance(payload, dict):
            return None
        full_name = _clean_str(payload.get("full_name"))
        if not full_name or "/" not in full_name:
            return None

        owner_obj = payload.get("owner") or {}
        owner_login = ""
        if isinstance(owner_obj, dict):
            owner_login = _clean_str(owner_obj.get("login")) or ""

        topics = payload.get("topics")
        if not isinstance(topics, list):
            topics = []
        topics = [str(t) for t in topics if t]

        return cls(
            full_name=full_name,
            owner=owner_login,
            name=_clean_str(payload.get("name")) or "",
            html_url=_clean_str(payload.get("html_url")) or "",
            description=_clean_str(payload.get("description")),
            language=_clean_str(payload.get("language")),
            topics=topics,
            stars=_to_int(payload.get("stargazers_count")),
            forks=_to_int(payload.get("forks_count")),
            open_issues=_to_int(payload.get("open_issues_count")),
            created_at=_clean_str(payload.get("created_at")),
            updated_at=_clean_str(payload.get("updated_at")),
            pushed_at=_clean_str(payload.get("pushed_at")),
            sources=[source] if source else [],
            api_enriched=True,
        )

    # ------------------------------------------------------------------
    # 合并
    # ------------------------------------------------------------------
    def merge(self, other: "RepoRecord") -> None:
        """
        合并另一条同名记录（去重时使用）

        规则：
        - API 数据优先于 Trending HTML 的粗略数据
        - Trending 独有字段（排名、stars today）永远保留
        - 空值不覆盖已有值
        """
        if other is None:
            return

        prefer_other = other.api_enriched and not self.api_enriched

        for attr in (
            "html_url",
            "description",
            "language",
            "created_at",
            "updated_at",
            "pushed_at",
        ):
            other_value = getattr(other, attr, None)
            if other_value and (prefer_other or not getattr(self, attr, None)):
                setattr(self, attr, other_value)

        for attr in ("stars", "forks", "open_issues"):
            other_value = getattr(other, attr, None)
            if other_value is not None and (prefer_other or getattr(self, attr, None) is None):
                setattr(self, attr, other_value)

        if other.topics and (prefer_other or not self.topics):
            self.topics = list(other.topics)

        # Trending 专属信息：谁有用谁的
        if other.trending_rank is not None and self.trending_rank is None:
            self.trending_rank = other.trending_rank
        if other.trending_stars_today is not None and self.trending_stars_today is None:
            self.trending_stars_today = other.trending_stars_today

        if other.owner and not self.owner:
            self.owner = other.owner
        if other.name and not self.name:
            self.name = other.name
        if other.collected_at and not self.collected_at:
            self.collected_at = other.collected_at

        for source in other.sources:
            if source not in self.sources:
                self.sources.append(source)

        self.api_enriched = self.api_enriched or other.api_enriched

    def add_source(self, source: str) -> None:
        """登记来源"""
        if source and source not in self.sources:
            self.sources.append(source)

    # ------------------------------------------------------------------
    # 展示 / 序列化
    # ------------------------------------------------------------------
    @property
    def display_description(self) -> str:
        """描述缺失时的统一占位文案"""
        return self.description or "No description provided."

    @property
    def display_language(self) -> str:
        """语言缺失时的统一占位文案"""
        return self.language or "Unknown"

    def to_snapshot_dict(self) -> Dict[str, Any]:
        """转换为快照中保存的精简字典（None 字段直接省略，不写入假值）"""
        data: Dict[str, Any] = {}
        for key in SNAPSHOT_FIELDS:
            value = getattr(self, key, None)
            if value is None:
                continue
            if isinstance(value, list):
                if value:
                    data[key] = list(value)
                continue
            data[key] = value
        return data

    def to_dict(self) -> Dict[str, Any]:
        """完整字典（调试 / 未来扩展用）"""
        return {
            "full_name": self.full_name,
            "owner": self.owner,
            "name": self.name,
            "html_url": self.html_url,
            "description": self.description,
            "language": self.language,
            "topics": list(self.topics),
            "stars": self.stars,
            "forks": self.forks,
            "open_issues": self.open_issues,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "pushed_at": self.pushed_at,
            "trending_rank": self.trending_rank,
            "trending_stars_today": self.trending_stars_today,
            "collected_at": self.collected_at,
            "sources": list(self.sources),
        }


def dedupe_records(records: List[RepoRecord]) -> List[RepoRecord]:
    """
    按 full_name（忽略大小写）去重并合并

    GitHub 仓库名大小写不敏感，因此用小写做 key；
    输出顺序为首次出现顺序，便于结果稳定可测。

    Args:
        records: 原始记录列表（可能来自多个数据源）

    Returns:
        去重合并后的记录列表
    """
    merged: Dict[str, RepoRecord] = {}
    order: List[str] = []

    for record in records:
        if record is None or not record.full_name or "/" not in record.full_name:
            continue
        key = record.full_name.lower()
        if key in merged:
            merged[key].merge(record)
        else:
            merged[key] = record
            order.append(key)

    return [merged[key] for key in order]
