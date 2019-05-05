from mlagents.envs import UnityEnvironment
from collections import deque
import numpy as np
from consts import *
from PIL import Image


class Environment:
    def __init__(self):
        self.env = UnityEnvironment(file_name=ENV_NAME, worker_id=2)
        self.default_brain = self.env.brain_names[0]
        self.frame_counter = 0
        self.episode_counter = 0

    def start_episode(self):
        info = self.env.reset(train_mode=True)[self.default_brain]
        visual_observation = info.visual_observations[0][0]
        done = info.local_done[0]
        visual_observation = np.expand_dims(visual_observation, 0)
        visual_observation = np.expand_dims(visual_observation, 0)
        # save_frame(visual_observation, True)
        return visual_observation, done

    def step(self, action):
        info = self.env.step([action])[self.default_brain]
        reward = info.rewards[0]
        visual_observation = info.visual_observations[0][0]
        done = info.local_done[0]
        visual_observation = np.expand_dims(visual_observation, 0)
        visual_observation = np.expand_dims(visual_observation, 0)
        # save_frame(visual_observation)
        return reward, visual_observation, done

    def save_frame(self, visual_observation, new_episode=False):
        frame = np.squeeze(visual_observation)
        frame = (frame * 255).astype(np.uint8)
        im = Image.fromarray(test, 'L')

        self.frame_counter = self.frame_counter + 1
        if new_episode:
            self.episode_counter = self.episode_counter + 1

        im.save('sample-episodes/foo{0}-{1}.jpeg'.format(
            self.episode_counter, self.frame_counter))
