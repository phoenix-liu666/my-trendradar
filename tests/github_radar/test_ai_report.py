# coding=utf-8
"""AI 增强日报渲染测试（区块 / HTML 转义 / 降级）"""

import unittest

from github_radar.ai.pricing import ModelPricing, estimate_cost
from github_radar.ai.profile import KeywordMatch
from github_radar.ai.result import (
    STATUS_FAILED,
    STATUS_OK,
    SYNTHESIS_UNAVAILABLE_TEXT,
    AIReportData,
    disabled_result,
)
from github_radar.ai.schemas import AIUsage, DailySynthesis, parse_repo_analysis
from github_radar.ai.scoring import build_for_you
from github_radar.report import ReportContext, ReportSummary, render_html, render_text

from .ai_helpers import analysis_payload, make_scored

PRICING = ModelPricing(input_price_per_1m=1.0, output_price_per_1m=2.0)


def build_ai(
    *,
    analyses=None,
    synthesis=None,
    scored=None,
    keywords=None,
    status=STATUS_OK,
    usage=None,
    notes=None,
    reused=False,
):
    analyses = analyses or {}
    scored = scored or []
    usage = usage or AIUsage(
        prompt_tokens=82_431,
        completion_tokens=11_283,
        total_tokens=93_714,
        requests=5,
        successful_requests=5,
        cache_hits=9,
        repositories_analyzed=len(analyses),
    )
    return AIReportData(
        enabled=True,
        status=status,
        model="deepseek-v4-flash",
        analyses=analyses,
        for_you=build_for_you(scored, analyses, keywords or {}),
        synthesis=synthesis,
        usage=usage,
        cost=estimate_cost(usage.prompt_tokens, usage.completion_tokens, PRICING),
        notes=list(notes or []),
        reused=reused,
    )


def build_context(ai=None, hot=None):
    hot = hot if hot is not None else [make_scored("owner/project", score=62.1)]
    summary = ReportSummary(date="2026-08-23", candidate_count=90, trending_count=25)
    return ReportContext(summary=summary, hot=hot, new_rising=[], ai=ai)


def analysis(full_name="owner/project", **overrides):
    return parse_repo_analysis(analysis_payload(full_name, **overrides), full_name=full_name)


class AiDisabledTest(unittest.TestCase):
    """AI 未启用 → 邮件里完全看不到 AI 区块（原始基础日报）"""

    def test_no_ai_object_renders_base_report(self):
        html = render_html(build_context(None))
        for marker in ("今日 GitHub 技术信号", "For You", "AI 使用情况"):
            with self.subTest(marker=marker):
                self.assertNotIn(marker, html)
        self.assertIn("Hot Today Top", html)

    def test_disabled_result_renders_base_report(self):
        html = render_html(build_context(disabled_result("GITHUB_RADAR_AI_ENABLED 未开启")))
        self.assertNotIn("AI 使用情况", html)
        self.assertIn("Hot Today Top", html)

    def test_text_report_has_no_ai_sections(self):
        text = render_text(build_context(disabled_result("skip_ai")))
        self.assertNotIn("For You", text)
        self.assertNotIn("AI 使用情况", text)


class SignalsSectionTest(unittest.TestCase):
    """23. 日报顶部的 📡 今日 GitHub 技术信号"""

    def setUp(self):
        self.scored = [make_scored("owner/project", score=62.1)]
        self.analyses = {"owner/project": analysis()}

    def test_renders_headline_and_signals(self):
        synthesis = DailySynthesis(
            headline="Coding Agent 与 MCP 工具链继续保持较高热度。",
            signals=["Agent 开发工具占 Hot20 中较高比例", "两个 Scientific AI 新项目增长较快"],
            rising_categories=["Coding Agent"],
            watch_tomorrow=["本地部署类 AI 工具"],
        )
        html = render_html(
            build_context(build_ai(analyses=self.analyses, synthesis=synthesis, scored=self.scored))
        )
        self.assertIn("📡 今日 GitHub 技术信号", html)
        self.assertIn("Coding Agent 与 MCP 工具链继续保持较高热度。", html)
        self.assertIn("Agent 开发工具占 Hot20 中较高比例", html)
        self.assertIn("升温方向：", html)
        self.assertIn("明日关注：", html)

    def test_missing_synthesis_shows_unavailable_but_keeps_the_report(self):
        html = render_html(
            build_context(build_ai(analyses=self.analyses, synthesis=None, scored=self.scored))
        )
        self.assertIn("📡 今日 GitHub 技术信号", html)
        self.assertIn(SYNTHESIS_UNAVAILABLE_TEXT, html)
        self.assertIn("Hot Today Top", html)     # 邮件本体照常

    def test_text_version(self):
        synthesis = DailySynthesis(headline="今天很热闹", signals=["a", "b"])
        text = render_text(
            build_context(build_ai(analyses=self.analyses, synthesis=synthesis, scored=self.scored))
        )
        self.assertIn("今日主线：今天很热闹", text)
        self.assertIn("• a", text)


class ForYouSectionTest(unittest.TestCase):
    """20. For You 展示"""

    def setUp(self):
        self.scored = [make_scored("owner/project", score=62.1)]
        self.analyses = {
            "owner/project": analysis(
                relevance_score=98,
                recommended_action="study",
                recommendation_reason="架构值得研究。",
                relevance_reason="正是你关注的 Coding Agent 方向。",
            )
        }
        self.keywords = {"owner/project": KeywordMatch(score=100.0, top_category="ai_agents")}

    def html(self):
        return render_html(
            build_context(
                build_ai(analyses=self.analyses, scored=self.scored, keywords=self.keywords)
            )
        )

    def test_all_required_fields_are_rendered(self):
        html = self.html()
        self.assertIn("🎯 For You Top1", html)
        self.assertIn("owner/project", html)
        self.assertIn("Personal Score", html)
        self.assertIn("Heat Score", html)
        self.assertIn("相关度", html)
        self.assertIn("98/100", html)
        self.assertIn("一句话", html)
        self.assertIn("它解决什么问题", html)
        self.assertIn("为什么与你相关", html)
        self.assertIn("推荐原因", html)

    def test_action_is_translated(self):
        self.assertIn("阅读源码/架构", self.html())

    def test_personal_score_value_is_shown(self):
        # 0.55*98 + 0.25*62.1 + 0.20*100 = 89.4
        self.assertIn("89.4", self.html())

    def test_empty_for_you_shows_placeholder(self):
        html = render_html(build_context(build_ai(analyses={}, scored=self.scored, status=STATUS_FAILED)))
        self.assertIn("今日暂无 AI 个性化推荐", html)

    def test_text_version(self):
        text = render_text(
            build_context(
                build_ai(analyses=self.analyses, scored=self.scored, keywords=self.keywords)
            )
        )
        self.assertIn("🎯 #1 owner/project", text)
        self.assertIn("Personal Score", text)
        self.assertIn("推荐: 📖 阅读源码/架构", text)


class ListEnhancementTest(unittest.TestCase):
    """21. Hot20 / Rising10 的 AI 增强与降级"""

    def test_ai_fields_are_appended_to_hot_cards(self):
        scored = [make_scored("owner/project", score=62.1)]
        html = render_html(
            build_context(build_ai(analyses={"owner/project": analysis()}, scored=scored), hot=scored)
        )
        self.assertIn("为什么值得关注", html)
        self.assertIn("Developer Tool", html)

    def test_base_fields_are_preserved(self):
        scored = [make_scored("owner/project", score=62.1, stars=18342)]
        html = render_html(
            build_context(build_ai(analyses={"owner/project": analysis()}, scored=scored), hot=scored)
        )
        self.assertIn("18,342", html)
        self.assertIn("Heat Score", html)
        self.assertIn("+100", html)          # 24h 增量

    def test_repo_without_analysis_falls_back_to_base_card(self):
        scored = [make_scored("owner/project"), make_scored("other/repo")]
        html = render_html(
            build_context(build_ai(analyses={"owner/project": analysis()}, scored=scored), hot=scored)
        )
        self.assertIn("other/repo", html)
        # AI 增强块用专门的底色渲染：两个仓库里只有一个有 AI 数据
        from github_radar.report import COLOR_AI_BG

        self.assertEqual(html.count(COLOR_AI_BG), 1)


class UsageSectionTest(unittest.TestCase):
    """27. 邮件底部的 📊 AI 使用情况"""

    def setUp(self):
        self.scored = [make_scored("owner/project")]
        self.ai = build_ai(analyses={"owner/project": analysis()}, scored=self.scored)

    def test_usage_numbers_are_rendered(self):
        html = render_html(build_context(self.ai))
        self.assertIn("📊 AI 使用情况", html)
        self.assertIn("deepseek-v4-flash", html)
        self.assertIn("82,431", html)
        self.assertIn("11,283", html)
        self.assertIn("93,714", html)
        self.assertIn("缓存命中：9", html)

    def test_input_tokens_are_split_into_cache_hit_and_miss(self):
        usage = AIUsage(
            prompt_tokens=82_431,
            prompt_cache_hit_tokens=60_000,
            prompt_cache_miss_tokens=22_431,
            completion_tokens=11_283,
            total_tokens=93_714,
            requests=5,
            cache_hits=9,
            repositories_analyzed=28,
        )
        ai = build_ai(
            analyses={"owner/project": analysis()}, scored=self.scored, usage=usage
        )
        html = render_html(build_context(ai))

        self.assertIn("输入 Tokens：82,431", html)
        self.assertIn("├─ 缓存命中：60,000", html)
        self.assertIn("└─ 缓存未命中：22,431", html)
        self.assertIn("输出 Tokens：11,283", html)
        self.assertIn("总 Tokens：93,714", html)

    def test_repository_cache_row_is_distinct_from_token_cache(self):
        html = render_html(build_context(self.ai))
        self.assertIn("仓库缓存命中：9", html)

    def test_missing_cache_detail_shows_everything_as_miss(self):
        usage = AIUsage(prompt_tokens=50_000, completion_tokens=1_000, total_tokens=51_000)
        ai = build_ai(analyses={"owner/project": analysis()}, scored=self.scored, usage=usage)
        html = render_html(build_context(ai))

        self.assertIn("├─ 缓存命中：0", html)
        self.assertIn("└─ 缓存未命中：50,000", html)

    def test_cache_rows_appear_in_the_text_version(self):
        usage = AIUsage(
            prompt_tokens=1000, prompt_cache_hit_tokens=600, prompt_cache_miss_tokens=400
        )
        ai = build_ai(analyses={"owner/project": analysis()}, scored=self.scored, usage=usage)
        text = render_text(build_context(ai))

        self.assertIn("├─ 缓存命中：600", text)
        self.assertIn("└─ 缓存未命中：400", text)

    def test_cost_is_marked_as_estimated(self):
        html = render_html(build_context(self.ai))
        self.assertIn("预估今日费用", html)
        self.assertIn("约 ¥", html)
        self.assertIn("不代表最终账单", html)
        self.assertIn("DeepSeek 当前官方定价", html)

    def test_reused_result_is_disclosed(self):
        ai = build_ai(analyses={"owner/project": analysis()}, scored=self.scored, reused=True)
        self.assertIn("复用当天已生成的 AI 结果", render_html(build_context(ai)))

    def test_notes_appear_in_the_explanation(self):
        ai = build_ai(
            analyses={"owner/project": analysis()},
            scored=self.scored,
            notes=["本次有部分 AI 批次失败，对应项目显示的是基础数据。"],
        )
        self.assertIn("部分 AI 批次失败", render_html(build_context(ai)))

    def test_text_version(self):
        text = render_text(build_context(self.ai))
        self.assertIn("📊 AI 使用情况", text)
        self.assertIn("总 Tokens：93,714", text)


class HtmlEscapingTest(unittest.TestCase):
    """29. HTML escaping：模型输出绝不能变成可执行 HTML"""

    def malicious(self):
        payload = "<script>alert(1)</script>"
        return {
            "owner/project": analysis(
                summary_zh=payload,
                problem=f"<img src=x onerror={payload}>",
                relevance_reason=payload,
                recommendation_reason=payload,
                tech_stack=[payload],
                use_cases=[payload],
                why_hot={"summary": payload, "confidence": "low", "evidence": [payload]},
            )
        }

    def test_script_tags_are_escaped_everywhere(self):
        scored = [make_scored("owner/project")]
        html = render_html(
            build_context(build_ai(analyses=self.malicious(), scored=scored), hot=scored)
        )
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        # onerror 这类属性注入同样只能作为文本出现：尖括号已经被转义，标签不成立
        self.assertNotIn("<img", html)
        self.assertIn("&lt;img src=x onerror=", html)

    def test_synthesis_is_escaped(self):
        synthesis = DailySynthesis(
            headline="<script>alert('headline')</script>",
            signals=["<b>bold</b>"],
            rising_categories=["<i>cat</i>"],
        )
        html = render_html(
            build_context(
                build_ai(
                    analyses={"owner/project": analysis()},
                    scored=[make_scored("owner/project")],
                    synthesis=synthesis,
                )
            )
        )
        self.assertNotIn("<script>alert(", html)
        self.assertNotIn("<b>bold</b>", html)
        self.assertIn("&lt;b&gt;bold&lt;/b&gt;", html)

    def test_model_name_is_escaped(self):
        ai = build_ai(analyses={"owner/project": analysis()}, scored=[make_scored("owner/project")])
        ai.model = "<script>x</script>"
        html = render_html(build_context(ai))
        self.assertNotIn("<script>x</script>", html)

    def test_quotes_and_ampersands_are_escaped(self):
        analyses = {"owner/project": analysis(summary_zh='A & B "quoted"')}
        html = render_html(
            build_context(build_ai(analyses=analyses, scored=[make_scored("owner/project")]))
        )
        self.assertIn("&amp;", html)
        self.assertIn("&quot;", html)


if __name__ == "__main__":
    unittest.main()
