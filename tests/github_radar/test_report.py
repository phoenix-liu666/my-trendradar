# coding=utf-8
"""HTML / 纯文本日报渲染测试"""

import re
import unittest
from datetime import datetime, timedelta, timezone

from github_radar.history import StarDelta
from github_radar.models import RepoRecord
from github_radar.ranking import rank_repositories, select_hot_today, select_new_and_rising
from github_radar.report import (
    FIRST_RUN_NOTICE,
    ReportContext,
    ReportSummary,
    build_subject,
    fmt_delta,
    fmt_int,
    render_html,
    render_text,
)

NOW = datetime(2026, 8, 22, 8, 10, tzinfo=timezone.utc)


def iso_days_ago(days):
    return (NOW - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_context(*, with_history=True, hot_count=2, notes=None):
    records = [
        RepoRecord(
            full_name="owner/project",
            stars=18342,
            forks=2411,
            language="Python",
            created_at=iso_days_ago(23),
            description="一个用于测试的 <项目> & 描述",
            trending_rank=1,
        ),
        RepoRecord(
            full_name="second/repo",
            stars=900,
            forks=30,
            language=None,
            created_at=iso_days_ago(6),
            description=None,
        ),
    ][:hot_count]

    deltas = {}
    if with_history:
        deltas = {
            "owner/project": StarDelta(
                delta_stars_24h=1826, delta_stars_7d=6214, average_daily_growth_7d=887.7
            ),
            "second/repo": StarDelta(delta_stars_24h=120, delta_stars_7d=400, average_daily_growth_7d=57.1),
        }

    scored = rank_repositories(records, deltas, reference_time=NOW)
    summary = ReportSummary(
        date="2026-08-22",
        generated_at_display="2026-08-22 08:10",
        candidate_count=68,
        trending_count=25,
        new_repo_count=12,
        has_24h_history=with_history,
        has_7d_history=with_history,
        matched_24h=60 if with_history else 0,
        matched_7d=55 if with_history else 0,
        history_days=8 if with_history else 0,
        notes=notes or ([] if with_history else [FIRST_RUN_NOTICE]),
    )
    return ReportContext(
        summary=summary,
        hot=select_hot_today(scored, top_n=20),
        new_rising=select_new_and_rising(scored, top_n=10),
    )


class SubjectTest(unittest.TestCase):
    def test_subject_format(self):
        self.assertEqual(
            build_subject("2026-08-22", 20), "🔥 GitHub Daily Radar | 2026-08-22 | Top20"
        )

    def test_subject_reflects_actual_count(self):
        self.assertIn("Top3", build_subject("2026-08-22", 3))


class FormattingTest(unittest.TestCase):
    def test_fmt_int(self):
        self.assertEqual(fmt_int(18342), "18,342")
        self.assertEqual(fmt_int(None), "—")

    def test_fmt_delta_positive_has_plus_sign(self):
        self.assertEqual(fmt_delta(1826), "+1,826")

    def test_fmt_delta_missing_history_shows_dash(self):
        self.assertEqual(fmt_delta(None), "—")

    def test_fmt_delta_zero_and_negative(self):
        self.assertEqual(fmt_delta(0), "0")
        self.assertEqual(fmt_delta(-12), "-12")


class HTMLReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = render_html(build_context())

    def test_contains_sections_and_repos(self):
        self.assertIn("GitHub Daily Radar", self.html)
        self.assertIn("今日概览", self.html)
        self.assertIn("Hot Today Top", self.html)
        self.assertIn("New &amp; Rising Top", self.html)
        self.assertIn("数据说明", self.html)
        self.assertIn("owner/project", self.html)

    def test_repo_name_links_to_github(self):
        self.assertIn('href="https://github.com/owner/project"', self.html)

    def test_numbers_are_formatted(self):
        self.assertIn("18,342", self.html)
        self.assertIn("+1,826", self.html)
        self.assertIn("+6,214", self.html)

    def test_overview_contains_counts(self):
        self.assertIn("候选仓库数量：68", self.html)
        self.assertIn("Trending 仓库数量：25", self.html)

    def test_description_is_escaped(self):
        self.assertIn("&lt;项目&gt;", self.html)
        self.assertNotIn("<项目>", self.html)
        self.assertIn("&amp;", self.html)

    def test_missing_fields_show_placeholders(self):
        self.assertIn("No description provided.", self.html)
        self.assertIn("Unknown", self.html)

    def test_no_scripts_or_external_resources(self):
        lowered = self.html.lower()
        self.assertNotIn("<script", lowered)
        self.assertNotIn("<img", lowered)
        self.assertNotIn("<link", lowered)
        self.assertNotIn("背景图", lowered)
        # 唯一允许出现的外链是 github.com 仓库地址
        for url in re.findall(r'href="(http[^"]+)"', self.html):
            self.assertTrue(url.startswith("https://github.com/"), url)

    def test_mobile_friendly_and_inline_styles(self):
        self.assertIn('name="viewport"', self.html)
        self.assertIn("max-width:640px", self.html)
        self.assertIn("style=", self.html)
        self.assertNotIn("display:flex", self.html)

    def test_heat_score_displayed(self):
        self.assertIn("Heat Score", self.html)


class FirstRunReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = build_context(with_history=False)
        cls.html = render_html(cls.context)
        cls.text = render_text(cls.context)

    def test_first_run_notice_present(self):
        self.assertIn("首次运行", self.html)
        self.assertIn(FIRST_RUN_NOTICE, self.text)

    def test_deltas_show_dash_instead_of_total_stars(self):
        self.assertIn("—", self.html)
        self.assertNotIn("+18,342", self.html)  # 绝不能把总 Star 当增量
        self.assertIn("18,342", self.html)      # 总 Star 本身仍然显示

    def test_history_flags_reported(self):
        self.assertIn("是否已有 24h 历史：否", self.html)
        self.assertIn("是否已有 7d 历史：否", self.html)


class EmptyReportTest(unittest.TestCase):
    def test_empty_sections_render_placeholder(self):
        context = ReportContext(summary=ReportSummary(date="2026-08-22"))
        html = render_html(context)
        text = render_text(context)
        self.assertIn("今日没有可用的候选仓库。", html)
        self.assertIn("今日没有符合条件的新项目", html)
        self.assertIn("今日没有可用的候选仓库。", text)


class TextReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = render_text(build_context())

    def test_plain_text_has_no_html_tags(self):
        self.assertNotIn("<div", self.text)
        self.assertNotIn("<table", self.text)

    def test_contains_key_information(self):
        self.assertIn("GitHub Daily Radar", self.text)
        self.assertIn("owner/project", self.text)
        self.assertIn("https://github.com/owner/project", self.text)
        self.assertIn("+1,826", self.text)
        self.assertIn("Heat Score", self.text)

    def test_new_rising_section_shows_growth_speed(self):
        self.assertIn("成长速度", self.text)


if __name__ == "__main__":
    unittest.main()
