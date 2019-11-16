import sys
from classes.environment import Environment
from classes.config import Config
from classes.unity_env_provider import UnityEnvironmentProvider
import unittest
from unittest.mock import Mock
import numpy as np
from mlagents.envs import UnityEnvironment


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def frames_helper(state, targets, use_diff=False):

    if use_diff:
        targets = np.array(targets)
        targets = ((targets/255) + 1)/2

    for i in range(len(targets)):
        if not np.all(state[:, :, :, i] == targets[i]):
            return False

    return True


def fill_step_mock(brain, arr):
    return {
        brain: Struct(**{
            'local_done': [False],
            'rewards': [1],
            'visual_observations': [[arr]]
        })
    }


class TestEnvironment(unittest.TestCase):

    def setUp(self):
        self.env_provider_mock = Mock(spec=UnityEnvironmentProvider)
        self.env_mock = Mock(spec=UnityEnvironment)
        self.brain_name = 'default_brain'
        self.env_mock.brain_names = [self.brain_name]

    def test_step_1(self):
        self.env_mock.step.return_value = fill_step_mock(
            self.brain_name, np.random.rand(10, 10, 1))
        self.env_provider_mock.provide.return_value = self.env_mock
        cfg = Config('tests/config.json')
        cfg.set_config_value('OBS_SIZE', 10)
        cfg.set_config_value('FRAMES_SKIP', 4)
        cfg.set_config_value('FRAMES_LOOKBACK', 5)
        cfg.set_config_value('USE_DIFF', False)
        env = Environment(self.env_provider_mock, cfg)
        res = env.step(1)
        self.assertEqual(res["stacked_observation"].shape, (1, 10, 10, 5))

    def test_step_2(self):
        self.env_mock.step.return_value = fill_step_mock(
            self.brain_name, np.random.rand(10, 10, 1))
        self.env_provider_mock.provide.return_value = self.env_mock

        cfg = Config('tests/config.json')
        cfg.set_config_value('OBS_SIZE', 10)
        cfg.set_config_value('FRAMES_SKIP', 4)
        cfg.set_config_value('FRAMES_LOOKBACK', 5)
        cfg.set_config_value('USE_DIFF', True)

        env = Environment(self.env_provider_mock, cfg)
        res = env.step(1)

        self.assertEqual(res["stacked_observation"].shape, (1, 10, 10, 5))

    def test_step_3(self):

        vo_mock = Mock()
        vo_mock.side_effect = [
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 0.17)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 0.23)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 0.02)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 0.11)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 0.90)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 0.14)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 0.10)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 0.19))
        ]

        self.env_mock.step = vo_mock
        self.env_provider_mock.provide.return_value = self.env_mock

        cfg = Config('tests/config.json')
        cfg.set_config_value('OBS_SIZE', 10)
        cfg.set_config_value('FRAMES_SKIP', 0)
        cfg.set_config_value('FRAMES_LOOKBACK', 5)
        cfg.set_config_value('USE_DIFF', True)

        env = Environment(self.env_provider_mock, cfg)

        res = env.step(1)
        self.assertTrue(
            frames_helper(
                res["stacked_observation"], [0, 0, 0, 0, 0], True))
        res = env.step(1)
        self.assertTrue(
            frames_helper(
                res["stacked_observation"], [0, 0, 0, 0, 0.06], True))
        res = env.step(1)
        self.assertTrue(
            frames_helper(
                res["stacked_observation"], [0, 0, 0, 6, -21], True))
        res = env.step(1)
        self.assertTrue(
            frames_helper(
                res["stacked_observation"], [0, 0, 6, -21, 9], True))
        res = env.step(1)
        self.assertTrue(
            frames_helper(
                res["stacked_observation"], [0, 6, -21, 9, -2], True))
        res = env.step(1)
        self.assertTrue(
            frames_helper(
                res["stacked_observation"], [6, -21, 9, -2, 5], True))
        res = env.step(1)
        self.assertTrue(
            frames_helper(
                res["stacked_observation"], [-21, 9, -2, 5, -4], True))
        res = env.step(1)
        self.assertTrue(
            frames_helper(
                res["stacked_observation"], [9, -2, 5, -4, 9], True))

    def test_step_4(self):

        vo_mock = Mock()
        vo_mock.side_effect = [
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 0.17)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 0.23)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 0.20)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 0.11)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 0.90)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 0.14)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 0.10)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 0.19))
        ]

        self.env_mock.step = vo_mock

        self.env_provider_mock.provide.return_value = self.env_mock

        cfg = Config('tests/config.json')
        cfg.set_config_value('OBS_SIZE', 10)
        cfg.set_config_value('FRAMES_SKIP', 2)
        cfg.set_config_value('FRAMES_LOOKBACK', 2)
        cfg.set_config_value('USE_DIFF', True)

        env = Environment(self.env_provider_mock, cfg)

        res = env.step(1)
        self.assertTrue(frames_helper(res["stacked_observation"], [0, 0]))
        res = env.step(1)
        self.assertTrue(frames_helper(res["stacked_observation"], [0, 6]))
        res = env.step(1)
        self.assertTrue(frames_helper(res["stacked_observation"], [0, -15]))
        res = env.step(1)
        self.assertTrue(frames_helper(res["stacked_observation"], [0, -6]))
        res = env.step(1)
        self.assertTrue(frames_helper(res["stacked_observation"], [6, -14]))
        res = env.step(1)
        self.assertTrue(frames_helper(res["stacked_observation"], [-15, 12]))
        res = env.step(1)
        self.assertTrue(frames_helper(res["stacked_observation"], [-6, -1]))
        res = env.step(1)
        self.assertTrue(frames_helper(res["stacked_observation"], [-14, 10]))


if __name__ == '__main__':
    unittest.main()
