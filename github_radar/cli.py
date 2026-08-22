# coding=utf-8
"""
GitHub Daily Radar 命令行入口

用法::

    python -m github_radar                 # 完整流程（采集 → 快照 → 报告 → 邮件）
    python -m github_radar --no-email      # 本地测试：不发邮件
    python -m github_radar --date 2026-08-22 --no-email --no-snapshot

退出码：
    0  成功
    1  失败（无任何候选数据 / 邮件发送失败）
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from . import __version__
from .collector import (
    DEFAULT_MAX_DETAIL_REQUESTS,
    DEFAULT_NEW_MIN_STARS,
    DEFAULT_NEW_WINDOW_DAYS,
    collect_candidates,
    summarize_sources,
)
from .github_api import GitHubAPIClient
from .history import (
    DEFAULT_DATA_DIR,
    DEFAULT_RETENTION_DAYS,
    SnapshotStore,
    compute_deltas,
    load_history,
)
from .logging_utils import error, log, warn
from .mailer import EmailConfig, load_email_config, send_report_email
from .ranking import (
    HOT_TODAY_TOP_N,
    NEW_RISING_TOP_N,
    rank_repositories,
    select_hot_today,
    select_new_and_rising,
)
from .report import (
    FIRST_RUN_NOTICE,
    ReportContext,
    ReportSummary,
    build_subject,
    render_html,
    render_text,
)
from .timeutils import DEFAULT_TIMEZONE, now, today_str

DEFAULT_OUTPUT_DIR = "output/github_radar"

EXIT_OK = 0
EXIT_FAILED = 1


def repo_root() -> Path:
    """仓库根目录（本包所在目录的上一级）"""
    return Path(__file__).resolve().parent.parent


def _resolve_path(value: str, default_relative: str) -> Path:
    """
    解析路径：显式传入的相对路径按当前工作目录解析，
    默认值按仓库根目录解析（保证从任何目录运行结果一致）
    """
    if value:
        return Path(value)
    return repo_root() / default_relative


def resolve_timezone(cli_value: Optional[str] = None, env: Optional[Mapping[str, str]] = None) -> str:
    """
    解析时区：命令行 > 环境变量 > config/config.yaml 的 app.timezone > 默认值

    读取 TrendRadar 配置只是「只读复用」，读不到就用默认值，不产生耦合。
    """
    env = env if env is not None else os.environ

    if cli_value:
        return cli_value

    env_value = (env.get("GITHUB_RADAR_TIMEZONE") or "").strip()
    if env_value:
        return env_value

    config_path = repo_root() / "config" / "config.yaml"
    try:
        import yaml  # 项目已有依赖

        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        timezone = ((config.get("app") or {}).get("timezone") or "").strip()
        if timezone:
            return timezone
    except Exception:
        # 配置缺失 / 解析失败 / PyYAML 不可用：静默降级
        pass

    return DEFAULT_TIMEZONE


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog="python -m github_radar",
        description="GitHub Daily Radar - 每日 GitHub 热门/快速增长项目日报",
    )
    parser.add_argument("--version", action="version", version=f"GitHub Daily Radar {__version__}")
    parser.add_argument("--date", default="", help="指定日期 YYYY-MM-DD（默认取配置时区的今天）")
    parser.add_argument("--timezone", default="", help="时区名（默认读取 config/config.yaml）")
    parser.add_argument("--data-dir", default="", help=f"快照目录（默认 {DEFAULT_DATA_DIR}）")
    parser.add_argument("--output-dir", default="", help=f"报告输出目录（默认 {DEFAULT_OUTPUT_DIR}）")
    parser.add_argument(
        "--retention-days",
        type=int,
        default=DEFAULT_RETENTION_DAYS,
        help=f"快照保留天数，0 表示永久保留（默认 {DEFAULT_RETENTION_DAYS}）",
    )
    parser.add_argument("--top", type=int, default=HOT_TODAY_TOP_N, help="Hot Today 数量")
    parser.add_argument("--new-top", type=int, default=NEW_RISING_TOP_N, help="New & Rising 数量")
    parser.add_argument(
        "--new-window-days",
        type=int,
        default=DEFAULT_NEW_WINDOW_DAYS,
        help="新项目时间窗口（天）",
    )
    parser.add_argument(
        "--new-min-stars", type=int, default=DEFAULT_NEW_MIN_STARS, help="新项目最低 Star"
    )
    parser.add_argument(
        "--max-detail-requests",
        type=int,
        default=DEFAULT_MAX_DETAIL_REQUESTS,
        help="单仓库详情 API 请求上限",
    )
    parser.add_argument("--trending-since", default="daily", choices=["daily", "weekly", "monthly"])
    parser.add_argument("--no-email", action="store_true", help="不发送邮件（本地测试用）")
    parser.add_argument("--no-snapshot", action="store_true", help="不写入每日快照")
    parser.add_argument(
        "--no-latest",
        action="store_true",
        help="不写 latest.json（它是当天快照的副本，关闭可减少一半 git 体积增长）",
    )
    parser.add_argument("--no-report-files", action="store_true", help="不写出 HTML/TXT 报告文件")
    parser.add_argument(
        "--token",
        default="",
        help="GitHub token（默认读取环境变量 GITHUB_TOKEN / GH_TOKEN）",
    )
    return parser


def _write_github_output(pairs: Dict[str, str]) -> None:
    """把关键结果写入 GITHUB_OUTPUT，供 workflow 后续步骤使用"""
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    try:
        with open(output_path, "a", encoding="utf-8") as handle:
            for key, value in pairs.items():
                handle.write(f"{key}={value}\n")
    except OSError as exc:
        warn(f"写入 GITHUB_OUTPUT 失败：{exc}")


def _write_report_files(output_dir: Path, date: str, html_body: str, text_body: str) -> None:
    """把报告写到本地文件，便于人工查看与排查（不会被 workflow 提交）"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{date}.html").write_text(html_body, encoding="utf-8")
        (output_dir / f"{date}.txt").write_text(text_body, encoding="utf-8")
        log(f"report files written to {output_dir}")
    except OSError as exc:
        warn(f"写出报告文件失败：{exc}")


def _resolve_token(cli_token: str, env: Mapping[str, str]) -> Optional[str]:
    """解析 GitHub token（绝不打印其内容）"""
    token = (cli_token or env.get("GITHUB_TOKEN") or env.get("GH_TOKEN") or "").strip()
    return token or None


def main(argv: Optional[List[str]] = None) -> int:
    """主流程"""
    args = build_parser().parse_args(argv)
    env = os.environ

    timezone = resolve_timezone(args.timezone, env)
    date = (args.date or "").strip() or today_str(timezone)
    generated_at = now(timezone)

    log(f"GitHub Daily Radar v{__version__} starting | date={date} | timezone={timezone}")

    token = _resolve_token(args.token, env)
    if not token:
        warn("未检测到 GITHUB_TOKEN，将以匿名方式访问 GitHub API（限流更严格）")
    client = GitHubAPIClient(token=token)
    log(f"GitHub API auth: {'token detected' if client.authenticated else 'anonymous'}")

    # ---------- 1. 采集候选 ----------
    collection = collect_candidates(
        client,
        today=date,
        timezone=timezone,
        new_window_days=args.new_window_days,
        new_min_stars=args.new_min_stars,
        max_detail_requests=args.max_detail_requests,
    )
    records = collection.repositories

    if not records:
        error("No GitHub repository data could be collected.")
        error("所有候选来源（Trending / GitHub Search API）均失败，本次不生成也不发送日报。")
        _write_github_output({"status": "failed", "repo_count": "0"})
        return EXIT_FAILED

    log(f"source breakdown: {summarize_sources(records)}")

    # ---------- 2. 写入快照（先落盘，保证即使后续失败也不丢当天数据） ----------
    store = SnapshotStore(_resolve_path(args.data_dir, DEFAULT_DATA_DIR))
    snapshot_path: Optional[Path] = None
    if args.no_snapshot:
        log("snapshot skipped (--no-snapshot)")
    else:
        try:
            snapshot_path = store.save(
                date,
                records,
                generated_at=generated_at.isoformat(timespec="seconds"),
                timezone=timezone,
                write_latest=not args.no_latest,
            )
            log(f"snapshot saved: {snapshot_path} ({len(records)} repositories)")
            store.prune(args.retention_days, date)
        except OSError as exc:
            # 快照写失败不阻断日报，但要显著告警（明天的 24h 增量会缺失）
            warn(f"快照写入失败：{exc}（今日日报仍会生成，但明天的 24h 增量会缺失）")

    # ---------- 3. 历史与增量 ----------
    yesterday_repos, week_ago_repos, status = load_history(store, date, records)
    log(status.describe())
    deltas = compute_deltas(records, yesterday_repos, week_ago_repos)

    # ---------- 4. 排名 ----------
    log("ranking...")
    scored = rank_repositories(records, deltas, reference_time=generated_at)
    hot = select_hot_today(scored, top_n=args.top)
    new_rising = select_new_and_rising(
        scored,
        max_age_days=args.new_window_days,
        min_stars=args.new_min_stars,
        top_n=args.new_top,
    )
    log(f"hot today: {len(hot)} | new & rising: {len(new_rising)}")

    # ---------- 5. 报告 ----------
    notes: List[str] = []
    if status.is_first_run:
        notes.append(FIRST_RUN_NOTICE)
    elif not status.has_week_ago:
        notes.append("7 日 Star 增长需要连续运行 7 天后才会显示。")
    if not collection.trending_ok:
        notes.append("本次未能获取 GitHub Trending 数据，榜单仅基于 GitHub API 候选。")
    if not collection.search_ok:
        notes.append("本次 GitHub Search API 不可用，榜单仅基于 Trending 候选。")
    if client.rate_limited:
        notes.append("本次运行触发了 GitHub API 限流，部分仓库数据可能不完整。")

    summary = ReportSummary(
        date=date,
        generated_at_display=generated_at.strftime("%Y-%m-%d %H:%M"),
        candidate_count=len(records),
        trending_count=collection.trending_count,
        new_repo_count=collection.new_repo_count(max_age_days=args.new_window_days),
        new_window_days=args.new_window_days,
        has_24h_history=status.has_yesterday,
        has_7d_history=status.has_week_ago,
        matched_24h=status.matched_24h,
        matched_7d=status.matched_7d,
        history_days=status.available_days,
        notes=notes,
    )
    context = ReportContext(summary=summary, hot=hot, new_rising=new_rising)

    html_body = render_html(context)
    text_body = render_text(context)
    subject = build_subject(date, len(hot))
    log(f"report generated: {subject}")

    if not args.no_report_files:
        _write_report_files(_resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR), date, html_body, text_body)

    # ---------- 6. 邮件 ----------
    email_status = "skipped"
    if args.no_email:
        log("email skipped (--no-email)")
    else:
        config, missing = load_email_config(env)
        if config is None:
            error(
                "邮件配置缺失：" + ", ".join(missing) +
                "。请在 GitHub Secrets 中配置，或本地测试时加 --no-email。"
            )
            _write_github_output(
                {
                    "status": "email_not_configured",
                    "repo_count": str(len(records)),
                    "snapshot_date": date,
                }
            )
            return EXIT_FAILED

        if not _send(config, subject, html_body, text_body):
            _write_github_output(
                {
                    "status": "email_failed",
                    "repo_count": str(len(records)),
                    "snapshot_date": date,
                }
            )
            return EXIT_FAILED
        email_status = "sent"
        log("email sent.")

    _write_github_output(
        {
            "status": "ok",
            "repo_count": str(len(records)),
            "snapshot_date": date,
            "snapshot_path": str(snapshot_path) if snapshot_path else "",
            "email": email_status,
        }
    )
    log(client.describe())
    log("done.")
    return EXIT_OK


def _send(config: EmailConfig, subject: str, html_body: str, text_body: str) -> bool:
    """发送邮件（独立成函数，便于测试打桩）"""
    return send_report_email(
        config,
        subject=subject,
        html_body=html_body,
        text_body=text_body,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
