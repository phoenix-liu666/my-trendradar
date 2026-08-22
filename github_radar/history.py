# coding=utf-8
"""
Star 历史快照与增量计算（本功能的核心）

存储位置：``data/github_radar/YYYY-MM-DD.json``（每天一个文件，直接提交回仓库，
不依赖任何数据库或收费云存储）。另外维护一份 ``latest.json`` 便于快速查看，
但它**不能替代**每日历史文件。

增量口径（必须严格遵守）：
- ``delta_stars_24h`` = 今天 stars − **昨天那份快照**里的 stars
- ``delta_stars_7d``  = 今天 stars − **7 天前那份快照**里的 stars
- ``average_daily_growth_7d`` = delta_stars_7d / 7
- 缺少对应快照、或该仓库当时不在候选池 → 一律 ``None``，绝不用总 Star 冒充增量

为什么用「精确日期」而不是「最近一份历史」：
若某天 workflow 失败，用 2 天前的数据算出来的差值并不是 24h 增长，
标成 24h 就是伪造数据。宁可显示 “—”，也不给错误数字。
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from .logging_utils import log, warn
from .models import RepoRecord
from .timeutils import DEFAULT_TIMEZONE, days_between, shift_date_str

DEFAULT_DATA_DIR = "data/github_radar"
DEFAULT_RETENTION_DAYS = 90
LATEST_FILENAME = "latest.json"
SCHEMA_VERSION = 1

_SNAPSHOT_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")


@dataclass
class StarDelta:
    """单个仓库的 Star 增量（缺历史时为 None）"""

    delta_stars_24h: Optional[int] = None
    delta_stars_7d: Optional[int] = None
    average_daily_growth_7d: Optional[float] = None

    @property
    def has_24h(self) -> bool:
        return self.delta_stars_24h is not None

    @property
    def has_7d(self) -> bool:
        return self.delta_stars_7d is not None


@dataclass
class HistoryStatus:
    """当日可用历史情况（用于日报概览与首日提示）"""

    yesterday_date: Optional[str] = None
    week_ago_date: Optional[str] = None
    has_yesterday: bool = False
    has_week_ago: bool = False
    available_days: int = 0
    matched_24h: int = 0
    matched_7d: int = 0

    @property
    def is_first_run(self) -> bool:
        """是否为「尚无任何可用历史」的首次运行"""
        return not self.has_yesterday and not self.has_week_ago

    def describe(self) -> str:
        """一行日志描述"""
        if self.available_days <= 0:
            return "history: no snapshot available (first run)"
        unit = "day" if self.available_days == 1 else "days"
        return (
            f"history: {self.available_days} {unit} available "
            f"(24h={'yes' if self.has_yesterday else 'no'}, "
            f"7d={'yes' if self.has_week_ago else 'no'})"
        )


class SnapshotStore:
    """每日快照读写"""

    def __init__(self, data_dir: Union[str, Path] = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # 路径 / 列表
    # ------------------------------------------------------------------
    def path_for(self, date_str: str) -> Path:
        return self.data_dir / f"{date_str}.json"

    @property
    def latest_path(self) -> Path:
        return self.data_dir / LATEST_FILENAME

    def exists(self, date_str: str) -> bool:
        return self.path_for(date_str).is_file()

    def available_dates(self) -> List[str]:
        """已存在的快照日期（升序）"""
        if not self.data_dir.is_dir():
            return []
        dates = []
        for item in self.data_dir.iterdir():
            if not item.is_file():
                continue
            match = _SNAPSHOT_NAME_RE.match(item.name)
            if match:
                dates.append(match.group(1))
        return sorted(dates)

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def load(self, date_str: str) -> Optional[Dict[str, Any]]:
        """
        读取某天快照

        Returns:
            快照字典；文件不存在或损坏时返回 None（并记录 warning）
        """
        path = self.path_for(date_str)
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            warn(f"快照读取失败（{path.name}）：{type(exc).__name__}: {exc}")
            return None
        if not isinstance(data, dict):
            warn(f"快照格式异常（{path.name}）：顶层不是对象")
            return None
        return data

    def load_repositories(self, date_str: str) -> Dict[str, Dict[str, Any]]:
        """
        读取某天快照中的仓库映射

        Returns:
            ``{full_name_lower: {字段...}}``；不存在时返回空字典
        """
        data = self.load(date_str)
        if not data:
            return {}
        repositories = data.get("repositories")
        if not isinstance(repositories, dict):
            warn(f"快照 {date_str} 缺少 repositories 字段")
            return {}

        normalized: Dict[str, Dict[str, Any]] = {}
        for full_name, payload in repositories.items():
            if isinstance(payload, dict) and full_name:
                normalized[str(full_name).lower()] = payload
        return normalized

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def save(
        self,
        date_str: str,
        records: Iterable[RepoRecord],
        *,
        generated_at: str,
        timezone: str = DEFAULT_TIMEZONE,
        write_latest: bool = True,
    ) -> Path:
        """
        写入当天快照

        JSON 使用 ``sort_keys=True`` + 两空格缩进，保证每天的 git diff 最小且稳定。

        Args:
            date_str: 日期 "YYYY-MM-DD"
            records: 当天候选仓库
            generated_at: 生成时间（ISO 字符串）
            timezone: 时区名
            write_latest: 是否同时更新 latest.json

        Returns:
            快照文件路径
        """
        repositories: Dict[str, Dict[str, Any]] = {}
        for record in records:
            if not record or not record.full_name:
                continue
            repositories[record.full_name] = record.to_snapshot_dict()

        payload = {
            "schema_version": SCHEMA_VERSION,
            "date": date_str,
            "generated_at": generated_at,
            "timezone": timezone,
            "repository_count": len(repositories),
            "repositories": repositories,
        }

        self.data_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(date_str)
        self._write_json(path, payload)

        if write_latest:
            self._write_json(self.latest_path, payload)

        return path

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        """原子性尽力而为地写 JSON（先写临时文件再替换）"""
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(path)

    # ------------------------------------------------------------------
    # 保留策略
    # ------------------------------------------------------------------
    def prune(self, retention_days: int, today: str) -> List[str]:
        """
        删除超出保留期的每日快照

        Args:
            retention_days: 保留天数（<= 0 表示永久保留，不删除）
            today: 今天日期 "YYYY-MM-DD"

        Returns:
            被删除的日期列表
        """
        if retention_days is None or retention_days <= 0:
            return []
        if not self.data_dir.is_dir():
            return []

        removed: List[str] = []
        for date_str in self.available_dates():
            age = days_between(date_str, today)
            if age is None:
                continue
            if age > retention_days:
                path = self.path_for(date_str)
                try:
                    path.unlink()
                    removed.append(date_str)
                except OSError as exc:
                    warn(f"删除过期快照失败（{path.name}）：{exc}")

        if removed:
            log(f"pruned {len(removed)} snapshot(s) older than {retention_days} days")
        return removed


# ----------------------------------------------------------------------
# 增量计算
# ----------------------------------------------------------------------
def _snapshot_stars(payload: Optional[Dict[str, Any]]) -> Optional[int]:
    """从快照条目中安全读取 stars"""
    if not isinstance(payload, dict):
        return None
    value = payload.get("stars")
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def compute_deltas(
    records: Iterable[RepoRecord],
    yesterday_repositories: Optional[Dict[str, Dict[str, Any]]] = None,
    week_ago_repositories: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, StarDelta]:
    """
    计算每个仓库的 Star 增量

    Args:
        records: 今天的候选仓库
        yesterday_repositories: 昨天快照的仓库映射（key 为小写 full_name）
        week_ago_repositories: 7 天前快照的仓库映射

    Returns:
        ``{full_name_lower: StarDelta}``

    Note:
        - 今天没有 stars（API 失败）→ 全部为 None
        - 昨天/7 天前没有这个仓库 → 对应字段为 None
        - Star 数可能下降（取关），此时返回负值，不做修正（保持真实）
    """
    yesterday_repositories = yesterday_repositories or {}
    week_ago_repositories = week_ago_repositories or {}

    deltas: Dict[str, StarDelta] = {}
    for record in records:
        if not record or not record.full_name:
            continue
        key = record.full_name.lower()
        delta = StarDelta()

        today_stars = record.stars
        if today_stars is not None:
            yesterday_stars = _snapshot_stars(yesterday_repositories.get(key))
            if yesterday_stars is not None:
                delta.delta_stars_24h = today_stars - yesterday_stars

            week_ago_stars = _snapshot_stars(week_ago_repositories.get(key))
            if week_ago_stars is not None:
                delta.delta_stars_7d = today_stars - week_ago_stars
                delta.average_daily_growth_7d = delta.delta_stars_7d / 7.0

        deltas[key] = delta

    return deltas


def load_history(
    store: SnapshotStore,
    today: str,
    records: Optional[Iterable[RepoRecord]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], HistoryStatus]:
    """
    载入昨天与 7 天前的快照，并给出历史可用情况

    Args:
        store: 快照仓库
        today: 今天日期
        records: 今天的候选（用于统计实际能算出增量的仓库数，可选）

    Returns:
        (昨天仓库映射, 7天前仓库映射, HistoryStatus)
    """
    yesterday = shift_date_str(today, -1)
    week_ago = shift_date_str(today, -7)

    yesterday_repos = store.load_repositories(yesterday) if yesterday else {}
    week_ago_repos = store.load_repositories(week_ago) if week_ago else {}

    status = HistoryStatus(
        yesterday_date=yesterday,
        week_ago_date=week_ago,
        has_yesterday=bool(yesterday_repos),
        has_week_ago=bool(week_ago_repos),
        available_days=len([d for d in store.available_dates() if d != today]),
    )

    if records is not None:
        for record in records:
            key = record.full_name.lower()
            if _snapshot_stars(yesterday_repos.get(key)) is not None:
                status.matched_24h += 1
            if _snapshot_stars(week_ago_repos.get(key)) is not None:
                status.matched_7d += 1

    return yesterday_repos, week_ago_repos, status
