from collections import deque
import numpy as np


class Memory:
    def __init__(self, baseline_period, gamma):
        self.GAMMA = gamma
        self.actions = []
        self.rewards = []
        self.states = []
        self.bl = deque(maxlen=baseline_period)

    def save(self, a, r, s):
        self.actions.append(a)
        self.rewards.append(r)
        self.states.append(s)
        self.bl.append(r)

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
        gammas = np.full(steps, self.GAMMA) ** np.arange(0, steps)

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
        return self.states

    def episode_actions(self):
        return self.actions
