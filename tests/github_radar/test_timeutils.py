# coding=utf-8
"""时间/日期工具测试"""

import unittest
from datetime import datetime, timezone

from github_radar import timeutils


class DateMathTest(unittest.TestCase):
    def test_shift_date_str(self):
        self.assertEqual(timeutils.shift_date_str("2026-08-22", -1), "2026-08-21")
        self.assertEqual(timeutils.shift_date_str("2026-08-22", -7), "2026-08-15")
        self.assertEqual(timeutils.shift_date_str("2026-03-01", -1), "2026-02-28")
        self.assertEqual(timeutils.shift_date_str("2026-01-01", -1), "2025-12-31")

    def test_shift_invalid_date(self):
        self.assertIsNone(timeutils.shift_date_str("not-a-date", -1))
        self.assertIsNone(timeutils.shift_date_str("", -1))

    def test_days_between(self):
        self.assertEqual(timeutils.days_between("2026-05-24", "2026-08-22"), 90)
        self.assertEqual(timeutils.days_between("2026-08-22", "2026-08-22"), 0)
        self.assertIsNone(timeutils.days_between("bad", "2026-08-22"))


class ParseTest(unittest.TestCase):
    def test_parse_github_time_with_z(self):
        parsed = timeutils.parse_github_time("2026-08-01T12:00:00Z")
        self.assertEqual(parsed.year, 2026)
        self.assertIsNotNone(parsed.tzinfo)

    def test_parse_github_time_with_offset(self):
        parsed = timeutils.parse_github_time("2026-08-01T12:00:00+08:00")
        self.assertIsNotNone(parsed.tzinfo)

    def test_parse_invalid(self):
        self.assertIsNone(timeutils.parse_github_time(None))
        self.assertIsNone(timeutils.parse_github_time(""))
        self.assertIsNone(timeutils.parse_github_time("garbage"))

    def test_format_created_display(self):
        self.assertEqual(
            timeutils.format_created_display("2026-07-30T10:00:00Z"), "2026-07-30"
        )
        self.assertEqual(timeutils.format_created_display(None), "—")


class AgeTest(unittest.TestCase):
    def test_age_in_days(self):
        reference = datetime(2026, 8, 22, tzinfo=timezone.utc)
        age = timeutils.age_in_days("2026-08-12T00:00:00Z", reference)
        self.assertAlmostEqual(age, 10.0, places=3)

    def test_future_date_is_clamped_to_zero(self):
        reference = datetime(2026, 8, 22, tzinfo=timezone.utc)
        self.assertEqual(timeutils.age_in_days("2026-09-01T00:00:00Z", reference), 0.0)

    def test_unknown_created_at(self):
        self.assertIsNone(timeutils.age_in_days(None))


class TimezoneTest(unittest.TestCase):
    def test_now_is_timezone_aware(self):
        current = timeutils.now("Asia/Shanghai")
        self.assertIsNotNone(current.tzinfo)

    def test_today_str_format(self):
        self.assertRegex(timeutils.today_str("Asia/Shanghai"), r"^\d{4}-\d{2}-\d{2}$")

    def test_unknown_timezone_falls_back(self):
        tz = timeutils.get_timezone("Not/AZone")
        self.assertIsNotNone(tz)
        self.assertIsNotNone(datetime.now(tz).tzinfo)


if __name__ == "__main__":
    unittest.main()
