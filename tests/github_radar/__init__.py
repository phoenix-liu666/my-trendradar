# coding=utf-8
"""GitHub Daily Radar 测试

全部使用标准库 ``unittest``（不新增 pytest 依赖），且**不访问真实网络**：
所有 HTTP 交互都通过注入的假会话 / 假客户端完成。

运行方式::

    python -m unittest discover -s tests -t .
    # 或（本机已装 pytest 时）
    pytest tests
"""
