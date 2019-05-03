from mlagents.envs import UnityEnvironment
from collections import deque
import numpy as np
from consts import *


class Environment:
    def __init__(self):
        self.env = UnityEnvironment(file_name=ENV_NAME, worker_id=0)
        self.default_brain = self.env.brain_names[0]

    def start_episode(self):
        info = self.env.reset(train_mode=True)[self.default_brain]
        visual_observation = info.visual_observations[0][0]
        done = info.local_done[0]
        visual_observation = np.expand_dims(visual_observation, 0)
        return visual_observation, done

    def step(self, action):
        info = self.env.step([action])[self.default_brain]
        reward = info.rewards[0]
        visual_observation = info.visual_observations[0][0]
        done = info.local_done[0]
        visual_observation = np.expand_dims(visual_observation, 0)
        return reward, visual_observation, done
