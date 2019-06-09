from consts import *
from classes.memory import Memory
from classes.policy import Policy
from utils import tensor_to_gif_summ
from pympler import muppy
from pympler import summary
import numpy as np
import tensorflow as tf
import os
import os.path as path


class PolicyGradientAgent:

    def __init__(
            self, env, sess, experiment_id):
        self.env = env
        self.memory = Memory()
        self.sess = sess
        self.policy = Policy(
            sess, [OBS_SIZE, OBS_SIZE, FRAMES_LOOKBACK], NUMBER_OF_ACTIONS)
        self.global_step = 1
        self.saver = tf.train.Saver()
        self.experiment_id = experiment_id
        self.summ_writer = tf.summary.FileWriter(
            os.path.join('summaries', experiment_id), sess.graph)
        self.record_score = 0
        self.batch_episode_counter = 0

    def play(self, state):
        while self.batch_episode_counter < BUFFER_SIZE:
            a, v = self.policy.step(state)
            r, next_state, done = self.env.step(a)
            self.memory.save(state, a, r, done, v)
            state = next_state
            if done:
                self.batch_episode_counter += 1

        self.batch_episode_counter = 0
        self.memory.compute_true_value()
        advs, values, states, acts, rewards = self.memory.get_rollout()
        return advs, values, states, acts, rewards, state

    def learn(self):

        state = self.env.start_episode()

        while True:
            advs, values, states, actions, rewards, state = self.play(state)
            best_score, best_run = self.memory.get_best()
            self.memory.clear()

            vl, pl = self.policy.optimize(states, actions, values, advs)

            episode_len = len(actions)/BUFFER_SIZE
            episode_reward = sum(rewards)/BUFFER_SIZE

            self.log_scalar('value_loss', vl, self.global_step)
            self.log_scalar('policy_loss', pl, self.global_step)
            self.log_scalar('episode_length', episode_len, self.global_step)
            self.log_scalar('episode_reward', episode_reward, self.global_step)

            if best_score > self.record_score:
                self.log_gif('best_run', best_run, self.global_step)
                self.record_score = best_score

            print('=======')

            if self.global_step % SAVE_MODEL_STEPS == 0:
                self.save_model()

                all_objects = muppy.get_objects()
                sum1 = summary.summarize(all_objects)
                summary.print_(sum1)

            del advs
            del values
            del states
            del actions
            del rewards

            all_objects = None
            sum1 = None

            self.global_step += 1

    def save_model(self):
        path = os.path.join(
            'summaries', self.experiment_id, 'checkpoints', CHECKPOINT_FILE)
        self.saver.save(
            self.sess, path, self.global_step, write_meta_graph=False)

    def log_scalar(self, tag, value, step):
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.summ_writer.add_summary(summary, step)
        print('episode: {0} | {1}: {2}'.format(step, tag, value))

    def log_gif(self, tag, images, step):
        tensor_summ = tf.summary.tensor_summary(tag, images)
        tensor_value = self.sess.run(tensor_summ)
        self.summ_writer.add_summary(tensor_to_gif_summ(tensor_value), step)
