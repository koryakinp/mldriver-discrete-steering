from mlagents.envs import UnityEnvironment
from collections import deque
from consts import *
import numpy as np


class Environment:
    def __init__(self, frames_lookback, frames_skip, experiment_id):
        self.experiment_id = experiment_id
        self.env = UnityEnvironment(file_name=ENV_NAME, worker_id=5)
        self.frames_skip = frames_skip
        self.frames_lookback = frames_lookback
        self.default_brain = self.env.brain_names[0]
        queue_size = frames_lookback + (frames_lookback - 1) * frames_skip
        self.state = deque(maxlen=queue_size)
        self.frame_counter = 1

    def start_episode(self):
        info = self.env.reset(train_mode=True)[self.default_brain]
        visual_observation = info.visual_observations[0][0]
        done = info.local_done[0]
        self.fill_state(visual_observation)
        return self.get_state()

    def step(self, action):
        info = self.env.step([[action]])[self.default_brain]
        reward = info.rewards[0]
        visual_observation = info.visual_observations[0][0]
        done = info.local_done[0]

        if(len(self.state) == 0):
            self.fill_state(visual_observation)
        else:
            self.state.append(visual_observation)

        visual_observation = self.get_state()

        if(done):
            self.state.clear()

        return reward, visual_observation, done

    def get_state(self):
        arr = np.array(self.state)
        arr = arr[::self.frames_skip + 1]
        return np.transpose(arr, (3, 1, 2, 0))

    def fill_state(self, visual_observation):
        for i in range(0, self.state.maxlen):
            self.state.append(visual_observation)
