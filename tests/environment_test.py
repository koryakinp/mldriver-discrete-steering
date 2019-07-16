import sys
from classes.environment import Environment
from classes.unity_env_provider import UnityEnvironmentProvider
import unittest
from unittest.mock import Mock
from consts import *
import numpy as np
sys.path.append('../mlagents')
from mlagents.envs import UnityEnvironment


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def frames_helper(state, targets):
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


class Test_Environemnt(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Test_Environemnt, self).__init__(*args, **kwargs)

    def setUp(self):
        self.env_provider_mock = Mock(spec=UnityEnvironmentProvider)
        self.env_mock = Mock(spec=UnityEnvironment)
        self.brain_name = 'default_brain'
        self.env_mock.brain_names = [self.brain_name]

    def test_step_1(self):
        self.env_mock.step.return_value = fill_step_mock(
            self.brain_name, np.random.rand(10, 10, 1))
        self.env_provider_mock.provide.return_value = self.env_mock
        env = Environment(self.env_provider_mock, 5, 4, False)
        reward, state, done = env.step(1)

        self.assertEquals(state.shape, (1, 10, 10, 5))

    def test_step_2(self):
        self.env_mock.step.return_value = fill_step_mock(
            self.brain_name, np.random.rand(10, 10, 1))
        self.env_provider_mock.provide.return_value = self.env_mock
        env = Environment(self.env_provider_mock, 5, 4, True)
        reward, state, done = env.step(1)

        self.assertEquals(state.shape, (1, 10, 10, 5))

    def test_step_3(self):

        vo_mock = Mock()
        vo_mock.side_effect = [
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 17)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 23)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 2)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 11)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 9)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 14)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 10)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 19))
        ]

        self.env_mock.step = vo_mock

        self.env_provider_mock.provide.return_value = self.env_mock
        env = Environment(self.env_provider_mock, 5, 0, True)

        reward, state, done = env.step(1)
        self.assertTrue(frames_helper(state, [0, 0, 0, 0, 0]))
        reward, state, done = env.step(1)
        self.assertTrue(frames_helper(state, [0, 0, 0, 0, 6]))
        reward, state, done = env.step(1)
        self.assertTrue(frames_helper(state, [0, 0, 0, 6, -21]))
        reward, state, done = env.step(1)
        self.assertTrue(frames_helper(state, [0, 0, 6, -21, 9]))
        reward, state, done = env.step(1)
        self.assertTrue(frames_helper(state, [0, 6, -21, 9, -2]))
        reward, state, done = env.step(1)
        self.assertTrue(frames_helper(state, [6, -21, 9, -2, 5]))
        reward, state, done = env.step(1)
        self.assertTrue(frames_helper(state, [-21, 9, -2, 5, -4]))
        reward, state, done = env.step(1)
        self.assertTrue(frames_helper(state, [9, -2, 5, -4, 9]))

    def test_step_4(self):

        vo_mock = Mock()
        vo_mock.side_effect = [
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 17)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 23)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 2)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 11)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 9)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 14)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 10)),
            fill_step_mock(self.brain_name, np.full((10, 10, 1), 19))
        ]

        self.env_mock.step = vo_mock

        self.env_provider_mock.provide.return_value = self.env_mock
        env = Environment(self.env_provider_mock, 2, 2, True)

        reward, state, done = env.step(1)
        self.assertTrue(frames_helper(state, [0, 0]))
        reward, state, done = env.step(1)
        self.assertTrue(frames_helper(state, [0, 6]))
        reward, state, done = env.step(1)
        self.assertTrue(frames_helper(state, [0, -15]))
        reward, state, done = env.step(1)
        self.assertTrue(frames_helper(state, [0, -6]))
        reward, state, done = env.step(1)
        self.assertTrue(frames_helper(state, [6, -14]))
        reward, state, done = env.step(1)
        self.assertTrue(frames_helper(state, [-15, 12]))
        reward, state, done = env.step(1)
        self.assertTrue(frames_helper(state, [-6, -1]))
        reward, state, done = env.step(1)
        self.assertTrue(frames_helper(state, [-14, 10]))

if __name__ == '__main__':
    unittest.main()
