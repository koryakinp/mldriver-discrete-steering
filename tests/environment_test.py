from classes.environment import Environment
import unittest
from consts import *


class Test_Environemnt(unittest.TestCase):

    def test_start_episode(self):
        environment = Environment(OBS_DEPTH)
        x, done = environment.start_episode()
        actual = x.shape
        expected = (OBS_DEPTH, OBS_SIZE, OBS_SIZE)
        self.assertEqual(actual, expected)

    def test_step(self):
        environment = Environment(OBS_DEPTH)
        environment.start_episode()
        r, s, done = environment.step(1)
        actual = s.shape
        expected = (OBS_DEPTH, OBS_SIZE, OBS_SIZE)
        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()
