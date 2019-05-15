from consts import *
from classes.memory import Memory
from classes.network import Network
import numpy as np
import tensorflow as tf
import os
import os.path as path
from PIL import Image


class PolicyGradientAgent:

    def __init__(self, env, session, network, experiment_id):
        self.env = env
        self.network = network
        self.memory = Memory()
        self.saver = tf.train.Saver()
        self.frame_counter = 0
        self.episode_counter = 0
        self.global_step = 0
        self.session = session
        self.experiment_id = experiment_id
        self.summ_writer = tf.summary.FileWriter(
            os.path.join('summaries', experiment_id, 'rewards'),
            self.session.graph)

    def get_action(self, state):
        action_prob = self.session.run(
          self.network.softmax, feed_dict={self.network.X: state})
        action_prob = np.squeeze(action_prob)
        return np.random.choice([0, 1, 2], p=action_prob)

    def train(self, save_frames=False):

        frame = self.env.start_episode()
        self.memory.fill_transition(frame)

        while True:
            frame_prev = frame
            s = self.memory.get_state()
            a = self.get_action(s)
            r, frame, done = self.env.step([a])
            self.memory.save(a, r, frame_prev)
            self.frame_counter += 1
            self.global_step += 1

            if save_frames:
                self.save_block(s)

            if done:
                self.learn()
                self.log_progress()
                self.memory.clear()
                self.frame_counter = 1
                self.episode_counter += 1
                if self.global_step % SAVE_MODEL_STEPS == 0:
                    self.save_model()

    def learn(self):
        advantages = self.memory.episode_advantage()
        actions = self.memory.episode_actions()
        states = self.memory.episode_states()
        self.session.run(self.network.optimizer, feed_dict={
            self.network.X: states,
            self.network.A: advantages,
            self.network.Y: actions})

    def save_model(self):
        self.saver.save(self.session, CHECKPOINT_FILE)

    def save_block(self, s):
        for i in range(0, FRAMES_LOOKBACK):
            self.save_frame(np.squeeze(s)[:, :, i], i)

    def save_frame(self, frame, slice):
        frame = (frame * 255).astype(np.uint8)
        im = Image.fromarray(frame, 'L')
        filename = 'episode{0}-frame{1}-slice{2}.jpeg'.format(
            self.episode_counter, self.frame_counter, slice)
        fullpath = os.path.join(
            'summaries', self.experiment_id, 'sample-episodes', filename)
        im.save(fullpath)

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
