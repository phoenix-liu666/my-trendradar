# coding=utf-8
"""workflow 触发配置的契约测试

github-radar.yml 里有三条属性正是「定时可靠性」这次修复的核心，
被改回去只会悄无声息地退化（当天收不到日报 / 收到重复邮件），
所以在这里钉住：

1. 每天 4 次北京时间兜底 cron —— GitHub 的 schedule 会被延迟甚至丢弃
2. 每条 cron 都带 timezone: Asia/Shanghai —— 少写一个就偏移 8 小时
3. force_run 默认关闭 —— 手动运行同样遵守每日幂等，避免误发重复邮件
"""

import re
import unittest
from pathlib import Path

import yaml

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2] / ".github" / "workflows" / "github-radar.yml"
)

# 北京时间 08:17 / 08:37 / 08:57 / 09:17
EXPECTED_CRONS = ["17 8 * * *", "37 8 * * *", "57 8 * * *", "17 9 * * *"]
EXPECTED_TIMEZONE = "Asia/Shanghai"


def load_triggers():
    """读取 workflow 的 on: 段

    PyYAML 按 YAML 1.1 解析，裸 ``on:`` 会变成布尔键 True，这里两种都兜住。
    """
    with WORKFLOW_PATH.open("r", encoding="utf-8") as handle:
        workflow = yaml.safe_load(handle)
    triggers = workflow.get(True)
    if triggers is None:
        triggers = workflow.get("on")
    return triggers


class ScheduleTest(unittest.TestCase):
    """4 次兜底触发"""

    def setUp(self):
        self.schedule = load_triggers()["schedule"]

    def test_has_four_fallback_triggers(self):
        self.assertEqual(len(self.schedule), 4)

    def test_crons_are_beijing_morning_slots(self):
        self.assertEqual([entry["cron"] for entry in self.schedule], EXPECTED_CRONS)

    def test_every_cron_declares_the_timezone(self):
        """漏掉 timezone 的那条会按 UTC 解释，实际晚 8 小时触发"""
        for entry in self.schedule:
            with self.subTest(cron=entry["cron"]):
                self.assertEqual(entry.get("timezone"), EXPECTED_TIMEZONE)


class WorkflowDispatchTest(unittest.TestCase):
    """手动运行的开关"""

    def setUp(self):
        self.inputs = load_triggers()["workflow_dispatch"]["inputs"]

    def test_force_run_defaults_to_off(self):
        """手动运行默认也要幂等：忘记取消勾选不应该导致当天重复发信"""
        self.assertIs(self.inputs["force_run"]["default"], False)

    def test_debug_switches_default_to_off(self):
        for name in ("skip_email", "skip_commit", "skip_ai"):
            with self.subTest(input=name):
                self.assertIs(self.inputs[name]["default"], False)

    def test_all_switches_are_boolean(self):
        for name in ("force_run", "skip_email", "skip_commit", "skip_ai"):
            with self.subTest(input=name):
                self.assertEqual(self.inputs[name]["type"], "boolean")

    def test_skip_ai_exists(self):
        """30. skip_ai：手动运行时可以完全跳过 DeepSeek，用来验证旧功能"""
        self.assertIn("skip_ai", self.inputs)


class AiEnvironmentTest(unittest.TestCase):
    """AI 相关环境变量必须接进 workflow，且都能不配置

    Repository Variable 名称不能以 GITHUB_ 开头（GitHub 保留前缀，UI 里
    创建不出来），所以仓库侧叫 RADAR_AI_*，workflow 里映射成 Python
    内部使用的 GITHUB_RADAR_AI_*：env 的**键**是内部名，**值**取仓库变量。
    """

    def setUp(self):
        with WORKFLOW_PATH.open("r", encoding="utf-8") as handle:
            workflow = yaml.safe_load(handle)
        steps = workflow["jobs"]["radar"]["steps"]
        self.step = next(step for step in steps if step.get("id") == "radar")
        self.env = self.step["env"]
        self.script = self.step["run"]

    def test_ai_switch_comes_from_repository_variables(self):
        self.assertEqual(
            self.env["GITHUB_RADAR_AI_ENABLED"], "${{ vars.RADAR_AI_ENABLED }}"
        )

    def test_api_key_comes_from_secrets(self):
        """API Key 只能来自 Secrets，绝不能写成 vars 或硬编码"""
        self.assertEqual(self.env["DEEPSEEK_API_KEY"], "${{ secrets.DEEPSEEK_API_KEY }}")

    def test_model_and_limit_come_from_variables(self):
        self.assertEqual(self.env["DEEPSEEK_MODEL"], "${{ vars.DEEPSEEK_MODEL }}")
        self.assertEqual(
            self.env["GITHUB_RADAR_AI_REPO_LIMIT"], "${{ vars.RADAR_AI_REPO_LIMIT }}"
        )

    def test_no_repository_variable_uses_the_reserved_github_prefix(self):
        """GITHUB_ 是 GitHub 的保留前缀：这种仓库变量根本创建不出来

        写成 ``vars.GITHUB_*`` 不会报错，只会永远取到空字符串 ——
        AI 静默关闭、日报静默退回基础版，所以在这里钉住。
        """
        text = WORKFLOW_PATH.read_text(encoding="utf-8")
        offenders = re.findall(r"vars\.(GITHUB_\w+)", text)
        self.assertEqual(offenders, [])

    def test_skip_ai_input_is_wired_to_the_cli_flag(self):
        self.assertIn("inputs.skip_ai", self.script)
        self.assertIn("--skip-ai", self.script)

    def test_no_secret_is_hardcoded(self):
        text = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertNotIn("sk-", text)
        self.assertNotIn("Bearer ", text)

    def test_ai_status_is_reported_in_the_summary(self):
        text = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertIn("steps.radar.outputs.ai", text)


if __name__ == "__main__":
    unittest.main()
