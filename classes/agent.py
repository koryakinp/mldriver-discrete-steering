from consts import *
from classes.memory import Memory
from classes.network import Network
import numpy as np
import tensorflow as tf
import os
import os.path as path


class PolicyGradientAgent:

    def __init__(self, env):
        self.env = env
        self.network = Network(LR)
        self.memory = Memory(BASELINE_PERIOD, GAMMA)
        self.saver = tf.train.Saver()

    def get_action(self, session, state):
        action_prob = session.run(
          self.network.softmax, feed_dict={self.network.X: state})
        action_prob = np.squeeze(action_prob)
        return np.random.choice([0, 1, 2], p=action_prob)

    def train(self, session, summ_writer):
        episode = 1
        while True:

            s, done = self.env.start_episode()
            while not done:

                a = self.get_action(session, s)
                s_prev = s
                r, s, done = self.env.step([a])
                self.memory.save(a, r, s_prev)

            self.learn(session)
            episode_rewards = sum(self.memory.rewards)
            self.log_scalar(summ_writer, 'rewards', episode_rewards, episode)
            print(episode_rewards)
            self.memory.clear()
            episode = episode + 1

            self.save_model(session)

    def learn(self, session):
        advantages = self.memory.episode_advantage()
        actions = self.memory.episode_actions()
        states = np.array(self.memory.episode_states())
        states = np.squeeze(states, axis=1)

        session.run(self.network.optimizer, feed_dict={
            self.network.X: states,
            self.network.A: advantages,
            self.network.Y: actions})

    def log_scalar(self, writer, tag, value, step):
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)])
        writer.add_summary(summary, step)

    def save_model(self, session):
        self.saver.save(
            session, "./checkpoints/mldriver.ckpt".format(episode))
