import unittest
from classes.memory import Memory


class Test_Memory(unittest.TestCase):
    def test_baseline1(self):
        memory = Memory(3, 0.9)
        memory.save(None, 1.0, None)
        memory.save(None, 2.0, None)
        memory.save(None, 3.0, None)
        self.assertEqual(memory.baseline(), 2.0)

    def test_baseline2(self):
        memory = Memory(3, 0.9)
        memory.save(None, 1.0, None)
        memory.save(None, 2.0, None)
        memory.save(None, 3.0, None)
        memory.save(None, 4.0, None)
        memory.save(None, 5.0, None)
        self.assertEqual(memory.baseline(), 4.0)

    def test_baseline3(self):
        memory = Memory(3, 0.9)
        memory.save(None, 1.0, None)
        memory.save(None, 2.0, None)
        self.assertEqual(memory.baseline(), 1.5)

    def test_episode_reward(self):
        memory = create_memory()

        actual1 = memory.episode_reward(0, 0)
        expected1 = 74.35370334
        self.assertAlmostEqual(expected1, actual1, 5)

        actual2 = memory.episode_reward(5, 0)
        expected2 = 55.04192
        self.assertAlmostEqual(expected2, actual2, 5)

    def test_episode_reward_with_baseline(self):
        memory = create_memory()

        actual1 = memory.episode_reward(0, 5)
        expected1 = 40.04423315
        self.assertAlmostEqual(expected1, actual1, 5)

        actual2 = memory.episode_reward(5, 5)
        expected2 = 31.61397
        self.assertAlmostEqual(expected2, actual2, 5)

    def test_episode_advantages(self):
        memory = create_memory()
        actual = memory.episode_advantage()
        expected = [
            0.120486,
            -0.068147,
            3.055594,
            0.970862,
            3.098938,
            4.352355,
            -2.032736,
            -3.571727,
            -0.837273,
            -3.354545,
            -2.818182]

        self.assertEqual(len(expected), len(actual))

        for i in range(len(expected)):
            self.assertAlmostEqual(actual[i], expected[i], 5)


def create_memory():
    memory = Memory(100, 0.9)
    memory.save(None, 11.0, None)
    memory.save(None, 8.0, None)
    memory.save(None, 13.0, None)
    memory.save(None, 9.0, None)
    memory.save(None, 10.0, None)
    memory.save(None, 17.0, None)
    memory.save(None, 12.0, None)
    memory.save(None, 8.0, None)
    memory.save(None, 13.0, None)
    memory.save(None, 10.0, None)
    memory.save(None, 8.0, None)
    return memory

if __name__ == '__main__':
    unittest.main()
