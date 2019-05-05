from consts import *
from classes.memory import Memory
from classes.network import Network
import numpy as np
import tensorflow as tf
import os
import os.path as path
from PIL import Image


class PolicyGradientAgent:

    def __init__(self, env, session, network):
        self.env = env
        self.network = network
        self.memory = Memory(BASELINE_PERIOD, GAMMA)
        self.saver = tf.train.Saver()
        self.frame_counter = 0
        self.episode_counter = 0
        self.global_step = 0
        self.session = session
        self.summ_writer = tf.summary.FileWriter(
            os.path.join('summaries', 'rewards'), self.session.graph)

    def get_action(self, state):
        action_prob = self.session.run(
          self.network.softmax, feed_dict={self.network.X: state})
        action_prob = np.squeeze(action_prob)
        return np.random.choice([0, 1, 2], p=action_prob)

    def train(self, save_frames=False):

        s = self.env.start_episode()

        while True:
            a = self.get_action(s)
            s_prev = s
            r, s, done = self.env.step([a])
            self.memory.save(a, r, s_prev)
            self.frame_counter += 1
            self.global_step += 1
            if save_frames:
                self.save_frame(s)

            if done:
                self.learn()
                self.log_progress()
                self.memory.clear()
                self.frame_counter = 1
                self.episode_counter += 1
                self.save_model()

    def learn(self):
        advantages = self.memory.episode_advantage()
        actions = self.memory.episode_actions()
        states = np.array(self.memory.episode_states())
        states = np.squeeze(states, axis=1)
        self.session.run(self.network.optimizer, feed_dict={
            self.network.X: states,
            self.network.A: advantages,
            self.network.Y: actions})

    def save_model(self):
        self.saver.save(self.session, CHECKPOINT_FILE)

    def save_frame(self, s):
        frame = np.squeeze(s)
        frame = (frame * 255).astype(np.uint8)
        im = Image.fromarray(frame, 'L')
        im.save('sample-episodes/foo{0}-{1}.jpeg'.format(
            self.episode_counter, self.frame_counter))

    def log_progress(self):
        episode_rewards = sum(self.memory.rewards)
        summary = tf.Summary(
            value=[tf.Summary.Value(
                tag='rewards', simple_value=episode_rewards)])
        self.summ_writer.add_summary(summary, self.global_step)
        message = 'episode: {0} | frames: {1} | reward: {2}'
        print(message.format(
            self.episode_counter,
            self.frame_counter,
            episode_rewards))
