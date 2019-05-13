from collections import deque
from consts import *
import numpy as np


class Memory:
    def __init__(self):
        self.actions = []
        self.rewards = []
        self.states = []
        self.bl = deque(maxlen=BASELINE_PERIOD)
        queue_size = FRAMES_LOOKBACK + (FRAMES_LOOKBACK - 1) * FRAMES_SKIP
        self.transition = deque(maxlen=queue_size)

    def save(self, a, r, s):
        self.actions.append(a)
        self.rewards.append(r)
        self.bl.append(r)
        self.transition.append(s)
        self.states.append(np.array(self.transition)[::FRAMES_SKIP + 1])

    def fill_transition(self, s):
        for i in range(0, self.transition.maxlen):
            self.transition.append(s)

    def clear(self):
        self.actions = []
        self.rewards = []
        self.states = []

    def baseline(self):
        summ = float(sum(self.bl))
        lenn = float(len(self.bl))

        return summ/lenn

    def episode_reward(self, t, baseline):
        total = 0
        steps = len(self.rewards) - t
        gammas = np.full(steps, GAMMA) ** np.arange(0, steps)

        for idx, reward in enumerate(self.rewards[t:]):
            total += (reward - baseline) * gammas[idx]

        return total

    def episode_advantage(self):
        bl = self.baseline()
        advantages = np.zeros_like(self.rewards)
        for i in range(len(self.rewards)):
            advantages[i] = self.episode_reward(i, bl)

        return advantages

    def episode_states(self):
        arr = np.array(self.states)
        arr = np.transpose(arr, (0, 4, 2, 3, 1))
        return np.squeeze(arr)

    def episode_actions(self):
        return self.actions

    def get_state(self):
        arr = np.array(self.transition)
        arr = arr[::FRAMES_SKIP + 1]
        return np.transpose(arr, (3, 1, 2, 0))
