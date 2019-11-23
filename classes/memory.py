import numpy as np


class Memory:
    def __init__(self, cfg):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.frames = []
        self.GAMMA = cfg.get('GAMMA')

    def save_state(self, s):
        self.states.append(s)

    def save_action(self, a):
        self.actions.append(a)

    def save_value(self, v):
        self.values.append(v)

    def save_reward(self, r):
        self.rewards.append(r)

    def save_frame(self, f):
        self.frames.append(np.transpose(f, [2, 1, 0]))

    def get_actions(self):
        return np.array(self.actions)

    def get_states(self):
        arr = np.array(self.states)
        arr = np.squeeze(arr, axis=1)
        return arr

    def get_advantages(self):

        total_actions = len(self.actions)
        adv = np.zeros(total_actions)
        discounted_rewards = self.get_discounted_rewards()

        for i in range(total_actions):
            dr = discounted_rewards[i:]
            adv[i] = sum(dr) - self.values[i]

        return adv

    def get_discounted_rewards(self):
        temp = 0
        res = np.zeros(len(self.rewards))
        for (idx, reward) in enumerate(self.rewards[::-1]):
            res[idx] = reward + temp * self.GAMMA
            temp = res[idx]

        return np.flip(res)

    def get_frames(self):
        frames = np.swapaxes(self.frames, 1, 3)
        return np.rint(frames * 255)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.frames = []
