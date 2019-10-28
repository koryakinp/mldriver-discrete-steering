import unittest
from classes.config import Config
from classes.memory import Memory
import numpy as np
import os


class TestMemory(unittest.TestCase):

    def setUp(self):
        self.cfg = Config('tests/config.json')
        self.cfg.set_config_value('GAMMA', 0.9)

    def test_save(self):
        memory = Memory(self.cfg)

        for i in range(3):
            memory.save_action(1)
            memory.save_reward(1)
            memory.save_value(1)
            memory.save_frame(np.random.rand(128, 128, 1))
            memory.save_state(np.random.rand(128, 128, 5))

        self.assertEqual(len(memory.actions), 3)
        self.assertEqual(len(memory.rewards), 3)
        self.assertEqual(len(memory.values), 3)
        self.assertEqual(len(memory.frames), 3)
        self.assertEqual(len(memory.states), 3)

    def test_get_advantages(self):
        memory = Memory(self.cfg)

        for value in [7, 13, 14, 10, -11]:
            memory.save_value(value)

        for reward in [10, 11, 10, 9, -12]:
            memory.save_reward(reward)

        actual = memory.get_advantages()
        expected = [14.7, 10.6, 5, -10.9, -1]

        self.assertEqual(len(expected), len(actual))

        for i in range(len(expected)):
            self.assertAlmostEqual(actual[i], expected[i], 5)


if __name__ == '__main__':
    unittest.main()
