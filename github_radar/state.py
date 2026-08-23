# coding=utf-8
"""
每日运行状态（幂等控制）

存储位置：``data/github_radar/state/YYYY-MM-DD.json``（每天一个几百字节的小文件，
随 Star 快照一起提交回仓库，不引入数据库、不引入云服务）。

为什么需要它
------------
GitHub Actions 官方明确说明：``schedule`` 触发在平台高负载时**可能延迟、
甚至直接被丢弃**。因此 workflow 每天排了 4 次兜底 cron。
有了兜底就必须有幂等，否则一天会收到 4 封一模一样的日报。

判定规则（顺序不能颠倒）：
- 当天「快照完成 **且** 邮件已发」 → 后续触发直接正常退出（退出码 0）
- 当天「快照完成 **但** 邮件失败」 → 后续触发**必须继续重试邮件**

第二条是关键：判断「今天是否已经发过信」只看状态文件里的 ``email_sent``，
**绝不能**用「快照文件是否已存在」来推断，否则邮件永远补不上。

失败策略：状态文件缺失 / 损坏 / 读不出来，一律按「今天什么都没做」处理
（fail-open）。宁可多发一封，也不要因为一个坏掉的小文件让当天彻底收不到日报。
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .logging_utils import log, warn
from .timeutils import DEFAULT_TIMEZONE, days_between

# 状态目录名（相对于快照目录 data/github_radar/）
DEFAULT_STATE_DIRNAME = "state"

STATE_SCHEMA_VERSION = 1

_STATE_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")


def _as_bool(value: Any) -> bool:
    """
    严格解析布尔位

    只有真正的 JSON ``true`` 才算「已完成」；其它一切（缺失 / null /
    字符串 / 数字）都按未完成处理，保证异常数据只会导致「多跑一次」，
    而不会导致「该发的邮件被跳过」。
    """
    return value is True


def _as_int(value: Any) -> int:
    """安全转 int，失败返回 0"""
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_str(value: Any) -> Optional[str]:
    """安全转字符串，空串归一为 None"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass
class DailyState:
    """某一天的运行状态"""

    date: str
    # ---- 必需字段（workflow / 排查问题时主要看这三个）----
    snapshot_completed: bool = False
    email_sent: bool = False
    completed_at: Optional[str] = None
    # ---- 辅助字段（便于排查「第几次兜底才成功」）----
    snapshot_completed_at: Optional[str] = None
    email_sent_at: Optional[str] = None
    timezone: str = DEFAULT_TIMEZONE
    runs: int = 0
    last_run_at: Optional[str] = None
    last_status: str = ""

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    @property
    def is_complete(self) -> bool:
        """当天是否已经「快照完成 + 邮件已发」"""
        return self.snapshot_completed and self.email_sent

    def should_skip(self, *, needs_email: bool = True) -> bool:
        """
        本次触发是否可以直接跳过

        Args:
            needs_email: 本次运行是否需要发邮件（``--no-email`` 时为 False）

        Note:
            ``needs_email=False`` 时只要求快照完成——因为这种运行本来就不发信，
            再跑一遍也不会产出任何新东西。
        """
        if not self.snapshot_completed:
            return False
        return self.email_sent or not needs_email

    def describe(self) -> str:
        """一行日志描述"""
        return (
            f"daily state {self.date}: "
            f"snapshot={'done' if self.snapshot_completed else 'pending'}, "
            f"email={'sent' if self.email_sent else 'pending'}, "
            f"previous_runs={self.runs}"
        )

    # ------------------------------------------------------------------
    # 更新
    # ------------------------------------------------------------------
    def mark_run_started(self, timestamp: str) -> None:
        """记录一次真正执行（被跳过的触发不计数）"""
        self.runs += 1
        self.last_run_at = timestamp

    def mark_snapshot_completed(self, timestamp: str) -> None:
        """标记当天快照已落盘"""
        self.snapshot_completed = True
        if not self.snapshot_completed_at:
            self.snapshot_completed_at = timestamp
        self._refresh_completed_at(timestamp)

    def mark_email_sent(self, timestamp: str) -> None:
        """标记当天日报已发出"""
        self.email_sent = True
        if not self.email_sent_at:
            self.email_sent_at = timestamp
        self._refresh_completed_at(timestamp)

    def _refresh_completed_at(self, timestamp: str) -> None:
        """两件事都完成后才写 completed_at，且只写第一次"""
        if self.is_complete and not self.completed_at:
            self.completed_at = timestamp

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """转为可写入 JSON 的字典"""
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "date": self.date,
            "snapshot_completed": self.snapshot_completed,
            "email_sent": self.email_sent,
            "completed_at": self.completed_at,
            "snapshot_completed_at": self.snapshot_completed_at,
            "email_sent_at": self.email_sent_at,
            "timezone": self.timezone,
            "runs": self.runs,
            "last_run_at": self.last_run_at,
            "last_status": self.last_status,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], *, date: str) -> "DailyState":
        """
        从字典还原（字段缺失或类型异常时退化为「未完成」）

        Args:
            payload: 状态文件内容
            date: 期望的日期（以文件名为准，避免内容里的 date 写错导致误判）
        """
        return cls(
            date=date,
            snapshot_completed=_as_bool(payload.get("snapshot_completed")),
            email_sent=_as_bool(payload.get("email_sent")),
            completed_at=_as_str(payload.get("completed_at")),
            snapshot_completed_at=_as_str(payload.get("snapshot_completed_at")),
            email_sent_at=_as_str(payload.get("email_sent_at")),
            timezone=_as_str(payload.get("timezone")) or DEFAULT_TIMEZONE,
            runs=_as_int(payload.get("runs")),
            last_run_at=_as_str(payload.get("last_run_at")),
            last_status=_as_str(payload.get("last_status")) or "",
        )


class StateStore:
    """每日状态读写（与 SnapshotStore 风格一致）"""

    def __init__(self, state_dir: Union[str, Path]):
        self.state_dir = Path(state_dir)

    # ------------------------------------------------------------------
    # 路径 / 列表
    # ------------------------------------------------------------------
    def path_for(self, date_str: str) -> Path:
        return self.state_dir / f"{date_str}.json"

    def exists(self, date_str: str) -> bool:
        return self.path_for(date_str).is_file()

    def available_dates(self) -> List[str]:
        """已存在的状态日期（升序）"""
        if not self.state_dir.is_dir():
            return []
        dates: List[str] = []
        for item in self.state_dir.iterdir():
            if not item.is_file():
                continue
            match = _STATE_NAME_RE.match(item.name)
            if match:
                dates.append(match.group(1))
        return sorted(dates)

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def load(self, date_str: str) -> DailyState:
        """
        读取某天状态

        Returns:
            ``DailyState``；文件不存在 / 损坏 / 格式异常时返回全新的空状态
            （fail-open：宁可重跑，也不要漏发）
        """
        path = self.path_for(date_str)
        if not path.is_file():
            return DailyState(date=date_str)

        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            warn(
                f"当日状态读取失败（{path.name}）：{type(exc).__name__}: {exc}"
                "；按「今天尚未运行」处理"
            )
            return DailyState(date=date_str)

        if not isinstance(data, dict):
            warn(f"当日状态格式异常（{path.name}）：顶层不是对象；按「今天尚未运行」处理")
            return DailyState(date=date_str)

        return DailyState.from_dict(data, date=date_str)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def save(self, state: DailyState) -> Optional[Path]:
        """
        写入当天状态

        Returns:
            状态文件路径；写入失败返回 None（只告警，绝不让日报流程失败——
            最坏结果只是下一次兜底触发会重跑一遍）
        """
        path = self.path_for(state.date)
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(state.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            tmp_path.replace(path)
        except OSError as exc:
            warn(f"当日状态写入失败（{path.name}）：{exc}（不影响本次日报，但幂等会失效）")
            return None
        return path

    # ------------------------------------------------------------------
    # 保留策略
    # ------------------------------------------------------------------
    def prune(self, retention_days: int, today: str) -> List[str]:
        """
        删除超出保留期的状态文件（与快照共用同一套保留天数）

        Args:
            retention_days: 保留天数（<= 0 表示永久保留）
            today: 今天日期 "YYYY-MM-DD"

        Returns:
            被删除的日期列表
        """
        if retention_days is None or retention_days <= 0:
            return []
        if not self.state_dir.is_dir():
            return []

        removed: List[str] = []
        for date_str in self.available_dates():
            age = days_between(date_str, today)
            if age is None or age <= retention_days:
                continue
            path = self.path_for(date_str)
            try:
                path.unlink()
                removed.append(date_str)
            except OSError as exc:
                warn(f"删除过期状态失败（{path.name}）：{exc}")

        if removed:
            log(f"pruned {len(removed)} state file(s) older than {retention_days} days")
        return removed
