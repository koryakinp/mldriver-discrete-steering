from collections import deque
import numpy as np


class Memory:
    def __init__(self, frames_lookback, frames_skip, gamma):
        self.actions = []
        self.rewards = []
        self.states = []
        queue_size = frames_lookback + (frames_lookback - 1) * frames_skip
        self.transition = deque(maxlen=queue_size)
        self.frames_skip = frames_skip
        self.frames_lookback = frames_lookback
        self.gamma = gamma

    def save(self, a, r, s):
        self.actions.append(a)
        self.rewards.append(r)
        self.transition.append(s)
        self.states.append(np.array(self.transition)[::self.frames_skip + 1])

    def fill_transition(self, s):
        for i in range(0, self.transition.maxlen):
            self.transition.append(s)

    def clear(self):
        self.actions = []
        self.rewards = []
        self.states = []

    def episode_rewards(self):
        episode_rewards = []
        for t in range(len(self.rewards)):
            total = sum(
                self.gamma**i * r for i, r in enumerate(self.rewards[t:]))
            episode_rewards.append(total)
        return episode_rewards

    def episode_states(self):
        arr = np.array(self.states)
        arr = np.transpose(arr, (0, 4, 2, 3, 1))
        return np.squeeze(arr)

    def episode_actions(self):
        return self.actions

    def get_state(self):
        arr = np.array(self.transition)
        arr = arr[::self.frames_skip + 1]
        return np.transpose(arr, (3, 1, 2, 0))
