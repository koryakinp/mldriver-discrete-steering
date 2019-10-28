import unittest
from classes.config import Config
import os


class TestConfig(unittest.TestCase):

    def setUp(self):
        self.cfg = Config('tests/config.json')

    def test_get_config_value(self):
        actual = self.cfg.get('GAMMA')
        expected = 0.90
        self.assertEqual(actual, expected)

    def test_set_config_value(self):
        self.cfg.set_config_value('GAMMA', 0.999)
        expected = 0.999
        actual = self.cfg.get('GAMMA')
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
