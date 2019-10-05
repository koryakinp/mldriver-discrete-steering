from collections import deque
from classes.transition import Transition
from consts import *
import numpy as np
import tensorflow as tf


class Memory:
    def __init__(self, best_score):
        self.transitions = np.array([], dtype=object)
        self.best_score = best_score
        self.cur_score = 0
        self.record_beaten = False

    def save(self, save_result):

        s = save_result["state"]
        a = save_result["action"]
        r = save_result["reward"]
        done = save_result["done"]
        v = save_result["value"]

        transition = Transition(s, a, r, v, done)
        self.transitions = np.append(self.transitions, transition)

        self.cur_score += r

        if done:
            if self.cur_score > self.best_score:
                self.best_score = self.cur_score
                self.record_beaten = True

            self.cur_score = 0

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

        states = np.squeeze(states)

        rollout_res = {
            "values": values,
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "advantages": advantages,
            "record_beaten": self.record_beaten
        }

        return rollout_res

    def get_best(self):
        return self.best_score

    def clear(self):
        self.transitions = np.array([], dtype=object)
        self.cur_score = 0
        self.record_beaten = False
