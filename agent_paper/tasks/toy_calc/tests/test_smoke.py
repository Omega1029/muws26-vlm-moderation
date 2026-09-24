import unittest

from calc import ops


class Smoke(unittest.TestCase):
    def test_import(self):
        self.assertTrue(callable(ops.mean))
