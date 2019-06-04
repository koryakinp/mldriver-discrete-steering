from collections import deque
from classes.transition import Transition
from consts import *
import numpy as np
import tensorflow as tf


class Memory:
    def __init__(self):
        self.transitions = np.array([], dtype=object)
        self.best_score = 0
        self.cur_score = 0
        self.best_run = np.empty([1, OBS_SIZE, OBS_SIZE])
        self.cur_run = np.empty([1, OBS_SIZE, OBS_SIZE])

    def save(self, s, a, r, done, v):
        transition = Transition(s, a, r, v, done)
        self.transitions = np.append(self.transitions, transition)

        if done:
            if self.cur_score > self.best_score:
                self.best_score = self.cur_score
                self.best_run = self.cur_run

            self.cur_score = 0
            self.cur_run = np.empty([1, OBS_SIZE, OBS_SIZE])

        self.cur_score += r

        frame = s[:, :, :, FRAMES_LOOKBACK - 1]
        self.cur_run = np.append(self.cur_run, frame, axis=0)

    def compute_true_value(self):
        temp = 0
        for transition in self.transitions[::-1]:
            is_done = 0 if transition.done else 1
            transition.true_value = transition.reward + temp * GAMMA * is_done
            temp = transition.true_value

    def get_rollout(self):
        advantages = [transition.A() for transition in self.transitions]
        values = [transition.true_value for transition in self.transitions]
        states = [transition.state for transition in self.transitions]
        actions = [transition.action for transition in self.transitions]
        rewards = [transition.reward for transition in self.transitions]

        states = np.array(states).squeeze()

        return advantages, values, states, actions, rewards

    def get_best(self):
        bs = self.best_score
        br = self.best_run
        br = np.expand_dims(br, axis=3)
        br = br * 255
        br = br.astype(np.uint8)
        br = tf.convert_to_tensor(br)

        return bs, br

    def clear(self):
        self.transitions = np.array([], dtype=object)
        self.best_score = 0
        self.cur_score = 0
        self.best_run = np.empty([1, OBS_SIZE, OBS_SIZE])
        self.cur_run = np.empty([1, OBS_SIZE, OBS_SIZE])
