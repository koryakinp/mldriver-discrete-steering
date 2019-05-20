from consts import *
from classes.memory import Memory
import numpy as np
import tensorflow as tf
import os
import os.path as path
from PIL import Image


class PolicyGradientAgent:

    def __init__(
            self, env, session, value_network, policy_network, experiment_id):
        self.env = env
        self.value_network = value_network
        self.policy_network = policy_network
        self.memory = Memory(FRAMES_LOOKBACK, FRAMES_SKIP, GAMMA, MEMORY_SIZE)
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
        action_prob = self.policy_network.get_action_prob(self.session, state)
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
                rewards, policy_loss, value_loss = self.learn()
                self.log_progress(
                    rewards, np.mean(policy_loss), np.mean(value_loss))
                self.memory.clear()
                self.frame_counter = 1
                self.episode_counter += 1
                if self.global_step % SAVE_MODEL_STEPS == 0:
                    self.save_model()

    def learn(self):
        episode_rewards = self.memory.episode_rewards()
        actions = self.memory.episode_actions()
        states = self.memory.episode_states()

        value_estimation = self.value_network.get_value(self.session, states)
        value_estimation = np.squeeze(np.array(value_estimation))

        advantages = episode_rewards - value_estimation

        policy_loss = self.policy_network.update(
            self.session, states, actions, advantages)

        s, r = self.memory.sample_from_experiences(len(states))

        value_loss = self.value_network.update(self.session, s, r)

        return sum(self.memory.episode_rewards()), policy_loss, value_loss

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

    def log_progress(self, rewards, policy_loss, value_loss):
        summary_rewards = tf.Summary(
            value=[tf.Summary.Value(
                tag='rewards', simple_value=rewards)])

        summary_policy_loss = tf.Summary(
            value=[tf.Summary.Value(
                tag='policy_loss', simple_value=policy_loss)])

        summary_value_loss = tf.Summary(
            value=[tf.Summary.Value(
                tag='value_loss', simple_value=value_loss)])

        self.summ_writer.add_summary(summary_rewards, self.episode_counter)
        self.summ_writer.add_summary(summary_policy_loss, self.episode_counter)
        self.summ_writer.add_summary(summary_value_loss, self.episode_counter)

        message = 'episode: {0} | frames: {1} | reward: {2}'
        print(
            message.format(self.episode_counter, self.frame_counter, rewards))
