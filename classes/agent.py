from classes.memory import Memory
from classes.policy import Policy
from utils import *
from pympler import muppy
from pympler import summary
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
import os
import os.path as path
import logging
import moviepy.editor as mpy
import tempfile
from PIL import Image, ImageDraw, ImageFont


class PolicyGradientAgent:

    def __init__(self, env, cfg, experiment_id):
        self.env = env
        self.cfg = cfg
        self.GS = tf.Variable(
            0, name='global_step', trainable=False, dtype=tf.int32)
        self.BC = tf.Variable(
            0, name='batch_count', trainable=False, dtype=tf.int32)
        self.CN = tf.Variable(
            0, name='checkpoint_count', trainable=False, dtype=tf.int32)
        self.RECORD = tf.Variable(
            0, name='record', trainable=False, dtype=tf.float32)
        self.policy = Policy(cfg)
        self.sess = get_session(experiment_id)
        self.batch_count = self.sess.run(tf.assign(self.BC, self.BC+1))
        self.global_step = self.sess.run(tf.assign(self.GS, self.GS+1))
        self.checkpoint_number = self.sess.run(tf.assign(self.CN, self.CN+1))
        self.record_run = self.sess.run(self.RECORD)
        self.memory = Memory(cfg)
        self.saver = tf.train.Saver(max_to_keep=0)
        self.experiment_id = experiment_id
        self.summ_writer = tf.summary.FileWriter(
            os.path.join('output', experiment_id, 'summaries'),
            self.sess.graph)
        self.REPLAY_BUFFER_SIZE = self.cfg.get('REPLAY_BUFFER_SIZE')
        self.BATCH_SIZE = self.cfg.get('BATCH_SIZE')
        self.SAVE_MODEL_STEPS = self.cfg.get('SAVE_MODEL_STEPS')
        self.CHECKPOINT_FILE = self.cfg.get('CHECKPOINT_FILE')

    def learn(self):

        while True:

            experiences = []

            for i in range(0, self.REPLAY_BUFFER_SIZE):
                episode_experience = self.play_episode()
                experiences.append(episode_experience)
                step = self.global_step + i
                reward = sum(episode_experience.rewards)
                self.log('reward', reward, step, False)

                if reward > self.record_run:
                    frames = episode_experience.get_frames()
                    self.log_gif('best_run', frames, step)
                    self.record_run = reward
                    self.sess.run(tf.assign(self.RECORD, self.record_run))
                    logging.info('Record beaten: {0}'.format(self.record_run))

                logging.info('Episode: {0} | reward: {1}'.format(step, reward))

            states, actions, values, advantages = self.get_experience_batches(
                experiences)

            opt_results = []

            for i in range(len(actions)):
                opt_res = self.policy.optimize(
                    states[i], actions[i], values[i], advantages[i], self.sess)

                opt_results.append(opt_res)

            value_loss = np.array(
               [q["value_loss"] for q in opt_results]).mean()
            policy_loss = np.array(
               [q["policy_loss"] for q in opt_results]).mean()
            entropy = np.array(
                [q["entropy"] for q in opt_results]).mean()
            total_loss = np.array(
                [q["total_loss"] for q in opt_results]).mean()

            self.log('value_loss', value_loss, self.global_step)
            self.log('policy_loss', policy_loss, self.global_step)
            self.log('entropy', entropy, self.global_step)
            self.log('total_loss', total_loss, self.global_step)

            print('=======')

            if self.batch_count % self.SAVE_MODEL_STEPS == 0:
                self.check_model()
                self.save_model()

                all_objects = muppy.get_objects()
                sum1 = summary.summarize(all_objects)
                summary.print_(sum1)

                all_objects.clear()
                sum1.clear()

                del all_objects
                del sum1

            opt_results.clear()
            experiences.clear()

            self.global_step = self.sess.run(
                tf.assign(self.GS, self.GS+self.REPLAY_BUFFER_SIZE))
            self.batch_count = self.sess.run(
                tf.assign(self.BC, self.BC+1))

    def get_experience_batches(self, experiences):
        states = np.concatenate(
            [state.get_states() for state in experiences], axis=0)

        actions = np.concatenate(
            [action.get_actions() for action in experiences], axis=0)

        values = np.concatenate(
            [value.get_true_values() for value in experiences], axis=0)

        advantages = np.concatenate(
            [adv.get_advantages() for adv in experiences], axis=0)

        states, actions, values, advantages = shuffle(
            states, actions, values, advantages)

        states = np.array_split(states, self.BATCH_SIZE)
        actions = np.array_split(actions, self.BATCH_SIZE)
        values = np.array_split(values, self.BATCH_SIZE)
        advantages = np.array_split(advantages, self.BATCH_SIZE)

        return states, actions, values, advantages

    def play_episode(self):
        memory = Memory(self.cfg)
        step_result = self.env.start_episode()

        memory.save_state(step_result['stacked_observation'])
        memory.save_frame(step_result['visual_observation'])

        while not step_result["done"]:
            a, v = self.policy.play(
                step_result['stacked_observation'], self.sess)

            memory.save_value(v)
            memory.save_action(a)

            step_result = self.env.step(a)

            memory.save_state(step_result['stacked_observation'])
            memory.save_frame(step_result['visual_observation'])
            memory.save_reward(step_result['reward'])

        del step_result

        return memory

    def check_model(self):
        for trainable_variable in tf.trainable_variables():
            self.sess.run(
                tf.check_numerics(trainable_variable, 'invalid tensor'))

    def save_model(self):
        path = os.path.join(
            'output', self.experiment_id, 'checkpoints', self.CHECKPOINT_FILE)
        self.saver.save(self.sess, path, self.CN, write_meta_graph=False)
        self.sess.run(tf.assign(self.CN, self.CN + 1))

    def log(self, tag, value, step, write_line=True):
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.summ_writer.add_summary(summary, step)
        if write_line:
            msg = 'step: {0} | {1}: {2}'.format(step, tag, value)
            logging.info(msg)

    def log_gif(self, tag, images, step):
        images = np.array(images)
        images = tf.convert_to_tensor(images)
        tensor_summ = tf.summary.tensor_summary(tag, images)
        tensor_value = self.sess.run(tensor_summ)
        self.summ_writer.add_summary(tensor_to_gif_summ(tensor_value), step)

    def save_gif(self, frames, values, rewards):

        frames = np.array(frames)
        frames = np.transpose(frames, (0, 2, 3, 1))
        frames = frames * 255
        frames = frames.astype('uint8')
        frames = list(frames)

        for index, frame in enumerate(frames):
            value = 'Value: ' + str(round(values[index], 2))
            reward = 'Reward: ' + str(round(rewards[index], 2))
            frames[index] = apply_text(frame, [value, reward])

        clip = mpy.ImageSequenceClip(frames, fps=1)
        folder_path = os.path.join('output', self.experiment_id, 'summaries')
        filename = str(self.global_step) + '_sample_batch.gif'
        full_path = os.path.join(folder_path, filename)
        clip.write_gif(full_path, verbose=True)


def apply_text(frame, data):
    frame = np.squeeze(frame)
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    for i in range(0, len(data)):
        draw.text((2, i * 12 + 2), data[i], fill='white')
    frame = np.array(img)
    return np.expand_dims(frame, 2)
