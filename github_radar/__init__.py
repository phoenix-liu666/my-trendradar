# coding=utf-8
"""
GitHub Daily Radar
==================

每天自动发现 GitHub 热门 / 快速增长项目，维护 Star 历史快照，
生成 Top20 日报并通过邮件推送。

设计原则
--------
1. 与 TrendRadar 原有新闻/RSS 链路完全解耦：本包不在 import 时依赖
   ``trendradar.*``（``trendradar/__init__.py`` 会拖入 litellm / boto3
   等重量级依赖），仅在发邮件时**惰性**复用其 SMTP 预设，且失败可降级。
2. 只依赖项目已有依赖：``requests``（必需）、``pytz`` / ``PyYAML``（可选）。
3. 所有网络请求都有超时、有限重试与优雅降级；单点失败不影响整体日报。
4. Star 增量只来自「每日 snapshot 差值」，绝不把总 Star 当成 24h 增长。

模块划分
--------
- ``logging_utils``  统一日志前缀 ``[GitHubRadar]`` 与敏感信息脱敏
- ``timeutils``      时区 / 日期 / 项目年龄计算
- ``models``         ``RepoRecord`` 数据模型与快照序列化
- ``github_api``     GitHub REST API 客户端（重试 / 限流 / 容错）
- ``trending``       GitHub Trending 页面抓取与 HTML 解析
- ``collector``      候选仓库池构建与去重
- ``history``        每日快照读写、保留策略、Star 增量计算
- ``ranking``        GitHub Heat Score 与榜单筛选
- ``report``         HTML 邮件日报 + 纯文本 fallback
- ``mailer``         轻量 SMTP 适配器
- ``cli``            命令行入口（``python -m github_radar``）
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
