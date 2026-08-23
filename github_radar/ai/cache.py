# coding=utf-8
"""
AI 结果缓存

两层缓存，职责完全不同
----------------------
``StaticAnalysisCache``（``data/github_radar/ai_cache/<owner>__<repo>.json``）
    只缓存**相对静态**的字段：summary_zh / problem / category /
    tech_stack / use_cases / maturity。
    这些东西不会因为今天涨了几个 Star 就变化，缓存能省掉绝大部分 Token。
    **绝不缓存** why_hot / relevance_score / PersonalScore —— 那是每日上下文。

``DailyResultStore``（``data/github_radar/ai_cache/daily/<date>.json``）
    缓存**当天已经算出来的 AI 结果**。
    workflow 每天有 4 次兜底 cron，如果第一次邮件失败，后面几次会重跑；
    没有这一层的话，同一天最多会分析 4×30 个仓库，
    直接违反「每天最多 30 个 unique repositories」的硬限制。
    ``--force-run`` 会绕过它（用户明确要求重算）。

失效与容错
----------
- schema_version 变化 → miss
- ``repo pushed_at`` 变化 → miss（项目有更新，静态描述可能过时）
- 超过 TTL（默认 7 天）→ miss
- 文件损坏 / 读不出来 / 字段类型异常 → **fail-open**，当作 miss 重新分析
- 写入失败只 warning，绝不影响日报
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone as dt_timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..logging_utils import log, warn
from ..models import RepoRecord
from ..timeutils import days_between, parse_github_time
from .schemas import AIUsage, DailySynthesis, RepoAnalysis, parse_daily_synthesis

CACHE_DIRNAME = "ai_cache"
DAILY_DIRNAME = "daily"

CACHE_SCHEMA_VERSION = 1
DEFAULT_TTL_DAYS = 7
# 缓存文件保留天数（超过就删掉，避免仓库无限增长）
DEFAULT_CACHE_RETENTION_DAYS = 30

_SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z._-]+")
_DAILY_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")


def cache_key(full_name: str) -> str:
    """
    ``owner/repo`` → 安全的文件名

    Examples:
        >>> cache_key("Owner/Repo.js")
        'owner__repo.js'
    """
    text = (full_name or "").strip().lower().replace("/", "__")
    text = _SAFE_NAME_RE.sub("-", text)
    return text.strip("-") or "unknown"


def _now_utc() -> datetime:
    return datetime.now(dt_timezone.utc)


def _age_days(cached_at: Optional[str]) -> Optional[float]:
    """缓存写入至今的天数（无法解析返回 None）"""
    parsed = parse_github_time(cached_at)
    if parsed is None:
        return None
    return max(0.0, (_now_utc() - parsed).total_seconds() / 86400.0)


@dataclass
class CacheStats:
    """一次运行的缓存统计"""

    hits: int = 0
    misses: int = 0
    stale: int = 0
    corrupted: int = 0
    writes: int = 0


class StaticAnalysisCache:
    """仓库静态分析结果缓存"""

    def __init__(
        self,
        cache_dir: Union[str, Path],
        *,
        ttl_days: int = DEFAULT_TTL_DAYS,
        model: str = "",
    ):
        self.cache_dir = Path(cache_dir)
        self.ttl_days = max(0, int(ttl_days))
        self.model = model or ""
        self.stats = CacheStats()

    def path_for(self, full_name: str) -> Path:
        return self.cache_dir / f"{cache_key(full_name)}.json"

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def get(self, record: RepoRecord) -> Optional[Dict[str, Any]]:
        """
        取出某仓库的静态字段

        Returns:
            静态字段字典；未命中 / 失效 / 损坏时返回 None（fail-open）
        """
        if record is None or not record.full_name:
            return None

        path = self.path_for(record.full_name)
        if not path.is_file():
            self.stats.misses += 1
            return None

        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            # 缓存损坏：当作没有缓存，重新分析（绝不因此失败）
            warn(f"[AI] 缓存损坏（{path.name}）：{type(exc).__name__}，将重新分析")
            self.stats.corrupted += 1
            self.stats.misses += 1
            return None

        if not isinstance(payload, dict):
            warn(f"[AI] 缓存格式异常（{path.name}）：顶层不是对象，将重新分析")
            self.stats.corrupted += 1
            self.stats.misses += 1
            return None

        if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            self.stats.stale += 1
            self.stats.misses += 1
            return None

        cached_name = str(payload.get("full_name") or "").lower()
        if cached_name and cached_name != record.full_name.lower():
            # 文件名归一化理论上可能撞车，撞了就当没命中
            self.stats.misses += 1
            return None

        if self._is_stale(payload, record):
            self.stats.stale += 1
            self.stats.misses += 1
            return None

        static = payload.get("static")
        if not isinstance(static, dict) or not static.get("summary_zh"):
            self.stats.misses += 1
            return None

        self.stats.hits += 1
        return dict(static)

    def _is_stale(self, payload: Dict[str, Any], record: RepoRecord) -> bool:
        """判断缓存是否已经失效"""
        # 1) 项目有新提交 → 静态描述可能过时
        cached_pushed = str(payload.get("repo_pushed_at") or "").strip()
        current_pushed = str(record.pushed_at or "").strip()
        if cached_pushed and current_pushed and cached_pushed != current_pushed:
            return True

        # 2) TTL
        if self.ttl_days > 0:
            age = _age_days(payload.get("cached_at"))
            if age is None or age > self.ttl_days:
                return True

        return False

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def put(self, analysis: RepoAnalysis, record: RepoRecord, *, now_iso: str = "") -> bool:
        """
        写入静态字段

        Returns:
            是否写入成功（失败只 warning，不影响日报）
        """
        if analysis is None or record is None or not record.full_name:
            return False
        if not analysis.summary_zh:
            # 空结果没有缓存价值，也避免把失败固化下来
            return False

        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "full_name": record.full_name,
            "cached_at": now_iso or _now_utc().isoformat(timespec="seconds"),
            "repo_pushed_at": record.pushed_at or "",
            "model": self.model,
            "static": analysis.static_fields(),
        }

        path = self.path_for(record.full_name)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            tmp_path.replace(path)
        except OSError as exc:
            warn(f"[AI] 缓存写入失败（{path.name}）：{exc}")
            return False

        self.stats.writes += 1
        return True

    # ------------------------------------------------------------------
    # 保留策略
    # ------------------------------------------------------------------
    def prune(self, retention_days: int = DEFAULT_CACHE_RETENTION_DAYS) -> int:
        """删除过老的缓存文件（返回删除数量）"""
        if retention_days <= 0 or not self.cache_dir.is_dir():
            return 0

        removed = 0
        for item in self.cache_dir.iterdir():
            if not item.is_file() or item.suffix != ".json":
                continue
            try:
                with item.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                age = _age_days((payload or {}).get("cached_at"))
            except Exception:
                age = None
            if age is None or age > retention_days:
                try:
                    item.unlink()
                    removed += 1
                except OSError:
                    continue

        if removed:
            log(f"[AI] pruned {removed} cache file(s) older than {retention_days} days")
        return removed


class DailyResultStore:
    """当天 AI 结果的复用存储（配合 4 次兜底 cron）"""

    def __init__(self, cache_dir: Union[str, Path]):
        self.daily_dir = Path(cache_dir) / DAILY_DIRNAME

    def path_for(self, date_str: str) -> Path:
        return self.daily_dir / f"{date_str}.json"

    def load(
        self, date_str: str, *, model: str = ""
    ) -> Optional[Tuple[Dict[str, RepoAnalysis], Optional[DailySynthesis], AIUsage]]:
        """
        读取当天已经算好的 AI 结果

        Returns:
            ``(analyses, synthesis, usage)``；不存在 / 损坏 / 模型变了 → None
        """
        path = self.path_for(date_str)
        if not path.is_file():
            return None

        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            warn(f"[AI] 当日 AI 结果损坏（{path.name}）：{type(exc).__name__}，将重新分析")
            return None

        if not isinstance(payload, dict):
            return None
        if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            return None
        if model and str(payload.get("model") or "") != model:
            # 换了模型就重新算，避免混用两种模型的结果
            return None

        analyses: Dict[str, RepoAnalysis] = {}
        raw_analyses = payload.get("analyses")
        if isinstance(raw_analyses, dict):
            for key, item in raw_analyses.items():
                analysis = RepoAnalysis.from_dict(item, full_name=str(key))
                if analysis is not None:
                    analyses[str(key).lower()] = analysis

        if not analyses:
            return None

        synthesis = parse_daily_synthesis(payload.get("synthesis"))
        usage = AIUsage.from_dict(payload.get("usage"))
        return analyses, synthesis, usage

    def save(
        self,
        date_str: str,
        *,
        analyses: Dict[str, RepoAnalysis],
        synthesis: Optional[DailySynthesis],
        usage: AIUsage,
        model: str,
        generated_at: str = "",
    ) -> Optional[Path]:
        """写入当天 AI 结果（失败只 warning）"""
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "date": date_str,
            "model": model,
            "generated_at": generated_at or _now_utc().isoformat(timespec="seconds"),
            "usage": usage.to_dict() if usage else AIUsage().to_dict(),
            "synthesis": synthesis.to_dict() if synthesis else None,
            "analyses": {
                key: analysis.to_dict() for key, analysis in (analyses or {}).items()
            },
        }

        path = self.path_for(date_str)
        try:
            self.daily_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            tmp_path.replace(path)
        except OSError as exc:
            warn(f"[AI] 当日 AI 结果写入失败（{path.name}）：{exc}")
            return None
        return path

    def prune(self, retention_days: int, today: str) -> List[str]:
        """删除过期的当日结果文件"""
        if retention_days <= 0 or not self.daily_dir.is_dir():
            return []

        removed: List[str] = []
        for item in self.daily_dir.iterdir():
            if not item.is_file():
                continue
            match = _DAILY_NAME_RE.match(item.name)
            if not match:
                continue
            age = days_between(match.group(1), today)
            if age is None or age <= retention_days:
                continue
            try:
                item.unlink()
                removed.append(match.group(1))
            except OSError:
                continue
        return removed


def default_cache_dir(data_dir: Union[str, Path]) -> Path:
    """``data/github_radar`` → ``data/github_radar/ai_cache``"""
    return Path(data_dir) / CACHE_DIRNAME
