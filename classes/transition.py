import numpy as np
from consts import *


class Transition:
    def __init__(self, state, action, reward, estimated_value, done, frame):
        self.state = state
        self.action = action
        self.reward = reward
        self.estimated_value = estimated_value
        self.done = done
        self.true_value = None
        self.frame = frame

    def Q(self):
        return self.reward + GAMMA * self.estimated_value

    def A(self):
        return self.Q() - self.estimated_value
