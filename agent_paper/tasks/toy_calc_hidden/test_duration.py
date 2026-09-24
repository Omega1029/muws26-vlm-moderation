import unittest

from calc.ops import parse_duration


class TestDuration(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_duration("1h30m"), 5400)
        self.assertEqual(parse_duration("45m"), 2700)
        self.assertEqual(parse_duration("90s"), 90)
