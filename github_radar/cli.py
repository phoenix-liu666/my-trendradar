# coding=utf-8
"""
GitHub Daily Radar 命令行入口

用法::

    python -m github_radar                 # 完整流程（采集 → 快照 → 报告 → 邮件）
    python -m github_radar --no-email      # 本地测试：不发邮件
    python -m github_radar --force-run     # 忽略当天状态，强制重跑一遍
    python -m github_radar --skip-ai       # 跳过 DeepSeek 增强，只出基础日报
    python -m github_radar --date 2026-08-22 --no-email --no-snapshot

每日幂等（配合 workflow 的 4 次兜底 cron，见 ``state.py``）：
    - 当天「快照完成 + 邮件已发」 → 直接正常退出，退出码 0，status=skipped
    - 当天「快照完成但邮件失败」 → 继续重试邮件，且**不覆盖**当天已有的正式快照

退出码：
    0  成功（含「当天已完成，本次跳过」）
    1  失败（无任何候选数据 / 邮件发送失败）
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from . import __version__
from .ai import AIReportData, disabled_result, load_ai_config, run_ai_enhancement
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
    ScoredRepo,
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
from .state import DEFAULT_STATE_DIRNAME, DailyState, StateStore
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
        "--state-dir",
        default="",
        help=f"每日状态目录（默认 <data-dir>/{DEFAULT_STATE_DIRNAME}）",
    )
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
        "--force-run",
        action="store_true",
        help="忽略当天状态：即使今天已完成也重跑一遍（会覆盖当天快照并重复发信）",
    )
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="完全不读写每日状态文件（关闭幂等，仅用于本地调试）",
    )
    parser.add_argument(
        "--no-latest",
        action="store_true",
        help="不写 latest.json（它是当天快照的副本，关闭可减少一半 git 体积增长）",
    )
    parser.add_argument("--no-report-files", action="store_true", help="不写出 HTML/TXT 报告文件")
    parser.add_argument(
        "--skip-ai",
        action="store_true",
        help="完全跳过 DeepSeek AI 增强（只出基础日报，便于验证旧功能）",
    )
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

    # ---------- 0. 每日幂等检查（必须在任何网络请求之前） ----------
    #
    # workflow 每天排了 4 次兜底 cron（GitHub 的 schedule 会延迟甚至丢弃），
    # 这里保证「日报每天最多一封、快照每天只有一个正式版本」。
    data_dir = _resolve_path(args.data_dir, DEFAULT_DATA_DIR)
    store = SnapshotStore(data_dir)
    state_store = StateStore(
        Path(args.state_dir) if args.state_dir else data_dir / DEFAULT_STATE_DIRNAME
    )
    use_state = not args.no_state
    state = state_store.load(date) if use_state else DailyState(date=date)
    state.timezone = timezone
    wants_email = not args.no_email

    if use_state:
        log(state.describe())
    else:
        log("daily state disabled (--no-state)")

    if use_state and not args.force_run and state.should_skip(needs_email=wants_email):
        log(f"{date} 今日任务已完成（快照 + 邮件），本次触发正常退出，不重复发送。")
        log("需要重跑：手动运行 workflow 并勾选 force_run，或本地加 --force-run。")
        _write_github_output(
            {
                "status": "skipped",
                "snapshot_date": date,
                "snapshot": "kept",
                "email": "already_sent" if state.email_sent else "skipped",
                "completed_at": state.completed_at or "",
            }
        )
        return EXIT_OK

    timestamp = generated_at.isoformat(timespec="seconds")
    state.mark_run_started(timestamp)

    def finish(run_status: str, exit_code: int, outputs: Dict[str, str]) -> int:
        """统一收尾：落状态 → 写 GITHUB_OUTPUT → 返回退出码

        每条返回路径都要经过这里，尤其是「快照成功但邮件失败」——
        那次运行必须把 snapshot_completed=True 落盘，
        否则下一次兜底触发会重新覆盖当天快照。
        """
        state.last_status = run_status
        if use_state:
            state_store.save(state)
        payload = {"status": run_status, "snapshot_date": date}
        payload.update(outputs)
        _write_github_output(payload)
        return exit_code

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
        return finish("failed", EXIT_FAILED, {"repo_count": "0"})

    log(f"source breakdown: {summarize_sources(records)}")

    # ---------- 2. 写入快照（先落盘，保证即使后续失败也不丢当天数据） ----------
    snapshot_path: Optional[Path] = None
    snapshot_status = "skipped"
    if args.no_snapshot:
        log("snapshot skipped (--no-snapshot)")
    elif state.snapshot_completed and store.exists(date) and not args.force_run:
        # 今天已经有正式快照了：这次是来补发邮件的，不能覆盖它。
        # 一天只保留一个正式版本，既是数据口径要求（明天的 24h 增量必须对着
        # 同一个基准算），也避免 4 次兜底触发把同一个文件改 4 遍。
        snapshot_path = store.path_for(date)
        snapshot_status = "kept"
        log(f"snapshot already completed for {date}: keeping {snapshot_path.name}（不覆盖当天正式版本）")
        store.prune(args.retention_days, date)
    else:
        try:
            snapshot_path = store.save(
                date,
                records,
                generated_at=timestamp,
                timezone=timezone,
                write_latest=not args.no_latest,
            )
            log(f"snapshot saved: {snapshot_path} ({len(records)} repositories)")
            snapshot_status = "written"
            state.mark_snapshot_completed(timestamp)
            store.prune(args.retention_days, date)
        except OSError as exc:
            # 快照写失败不阻断日报，但要显著告警（明天的 24h 增量会缺失）
            warn(f"快照写入失败：{exc}（今日日报仍会生成，但明天的 24h 增量会缺失）")

    if use_state:
        state_store.prune(args.retention_days, date)

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

    # ---------- 4.5 AI 增强（可选，失败必须降级） ----------
    #
    # 位置很关键：快照已经落盘、排名已经算完，所以 AI 这一步无论怎么炸，
    # 都不可能影响 snapshot / Heat Score / Star 数据。
    ai_data = _run_ai(
        scored,
        hot,
        new_rising,
        env=env,
        skip_ai=args.skip_ai,
        data_dir=data_dir,
        date=date,
        github_client=client,
        force=args.force_run,
        now_iso=timestamp,
    )

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
    context = ReportContext(summary=summary, hot=hot, new_rising=new_rising, ai=ai_data)

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
    elif state.email_sent and not args.force_run:
        # 同一天日报最多发送一次：只认状态里的 email_sent，不看快照是否存在
        email_status = "already_sent"
        log(f"email already sent for {date}, skipping（同一天日报最多发送一次）")
    else:
        config, missing = load_email_config(env)
        if config is None:
            error(
                "邮件配置缺失：" + ", ".join(missing) +
                "。请在 GitHub Secrets 中配置，或本地测试时加 --no-email。"
            )
            return finish(
                "email_not_configured",
                EXIT_FAILED,
                {
                    "repo_count": str(len(records)),
                    "snapshot": snapshot_status,
                    "ai": ai_data.status,
                },
            )

        if not _send(config, subject, html_body, text_body):
            return finish(
                "email_failed",
                EXIT_FAILED,
                {
                    "repo_count": str(len(records)),
                    "snapshot": snapshot_status,
                    "ai": ai_data.status,
                },
            )
        email_status = "sent"
        state.mark_email_sent(timestamp)
        log("email sent.")

    log(client.describe())
    log("done.")
    return finish(
        "ok",
        EXIT_OK,
        {
            "repo_count": str(len(records)),
            "snapshot_path": str(snapshot_path) if snapshot_path else "",
            "snapshot": snapshot_status,
            "email": email_status,
            "ai": ai_data.status,
            "ai_tokens": str(ai_data.usage.total_tokens),
            "completed_at": state.completed_at or "",
        },
    )


def _run_ai(
    scored: List[ScoredRepo],
    hot: List[ScoredRepo],
    new_rising: List[ScoredRepo],
    *,
    env: Mapping[str, str],
    skip_ai: bool,
    data_dir: Path,
    date: str,
    github_client: GitHubAPIClient,
    force: bool,
    now_iso: str,
) -> AIReportData:
    """
    执行 AI 增强，并保证它**永远不会让日报失败**

    ``run_ai_enhancement`` 内部已经把每一步都做了降级，这里再包一层
    ``try/except`` 是最后一道防线：哪怕 AI 子系统出现完全预期之外的异常
    （import 失败、配置对象被改坏……），基础日报也照常生成、照常发送。
    """
    try:
        config = load_ai_config(env, skip_ai=skip_ai)
        if not config.enabled:
            log(config.describe())
            return disabled_result(config.disabled_reason, config.model)

        return run_ai_enhancement(
            scored,
            hot,
            new_rising,
            config=config,
            data_dir=data_dir,
            date=date,
            github_client=github_client,
            env=env,
            force=force,
            now_iso=now_iso,
        )
    except Exception as exc:  # 绝不让 AI 影响日报
        warn(f"[AI] AI 增强意外失败（{type(exc).__name__}: {exc}），本次退回基础日报")
        return disabled_result("AI 运行异常")


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
