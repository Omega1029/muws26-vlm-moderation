import unittest

from calc.ops import mean


class TestMean(unittest.TestCase):
    def test_values(self):
        self.assertEqual(mean([1, 2, 3]), 2)
        self.assertEqual(mean([5]), 5)

    def test_empty(self):
        with self.assertRaises(ValueError):
            mean([])
