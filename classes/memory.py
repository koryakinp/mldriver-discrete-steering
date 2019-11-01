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
        return np.squeeze(self.states, axis=1)[:-1]

    def get_advantages(self):
        return np.array(self.rewards) + \
            np.full(len(self.rewards), self.GAMMA) * \
            np.append(self.values[1:], 0) - \
            np.array(self.values)

    def get_true_values(self):
        temp = 0
        res = np.zeros(len(self.rewards))
        for (idx, reward) in enumerate(self.rewards[::-1]):
            res[idx] = reward + temp * self.GAMMA
            temp = res[idx]

        return res

    def get_frames(self):
        frames = np.swapaxes(self.frames, 1, 3)
        return np.rint(frames * 255)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.frames = []
