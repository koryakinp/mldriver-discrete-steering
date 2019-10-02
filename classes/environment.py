from collections import deque
from consts import *
import numpy as np
import logging


class Environment:
    def __init__(
            self,
            env_provider,
            obs_size,
            frames_lookback,
            frames_skip,
            use_diff):
        self.env = env_provider.provide()
        self.obs_size = obs_size
        self.frames_skip = frames_skip
        self.frames_lookback = frames_lookback
        self.default_brain = self.env.brain_names[0]
        queue_size = frames_lookback + (frames_lookback - 1) * frames_skip
        if use_diff:
            queue_size += frames_skip + 1

        self.state = deque(maxlen=queue_size)
        self.frame_counter = 1
        self.use_diff = use_diff

    def start_episode(self):

        logging.info('Starting environment..')

        info = self.env.reset(train_mode=True)[self.default_brain]
        visual_observation = info.visual_observations[0][0]
        self.__fill_state(visual_observation)
        done = info.local_done[0]

        stacked_observation = self.__get_state()

        if self.use_diff:
            stacked_observation = self.__apply_diff(stacked_observation)

        res = {
            "stacked_observation": stacked_observation,
            "visual_observation": visual_observation,
            "done": done
        }

        return res

    def step(self, action):
        step_info = self.env.step([[action]])
        info = step_info[self.default_brain]
        reward = info.rewards[0]
        visual_observation = info.visual_observations[0][0]
        done = info.local_done[0]

        if len(self.state) == 0:
            self.__fill_state(visual_observation)
        else:
            self.state.append(visual_observation)

        stacked_observation = self.__get_state()

        if done:
            self.state.clear()

        if self.use_diff:
            stacked_observation = self.__apply_diff(stacked_observation)

        res = {
            "stacked_observation": stacked_observation,
            "visual_observation": visual_observation,
            "reward": reward,
            "done": done
        }

        return res

    def __get_state(self):
        arr = np.array(self.state)
        arr = arr[::self.frames_skip + 1]
        return np.transpose(arr, (3, 1, 2, 0))

    def __fill_state(self, visual_observation):
        for i in range(0, self.state.maxlen):
            self.state.append(visual_observation)

    def __apply_diff(self, frames):
        if self.frames_lookback == 1:
            raise Exception('Can not apply diff for a single frame')

        output = []

        for i in range(1, self.frames_lookback + 1):
            output.append(frames[0, :, :, i] - frames[0, :, :, i - 1])

        output = np.stack(output, axis=2)
        output = np.expand_dims(output, axis=0)
        return output
