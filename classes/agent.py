from consts import *
from classes.memory import Memory
from classes.policy import Policy
from utils import *
from pympler import muppy
from pympler import summary
import numpy as np
import tensorflow as tf
import os
import os.path as path
import logging
import moviepy.editor as mpy
import tempfile
from PIL import Image, ImageDraw, ImageFont


class PolicyGradientAgent:

    def __init__(self, env, experiment_id):
        self.env = env
        self.GS = tf.Variable(
            0, name='global_step', trainable=False, dtype=tf.int32)
        self.RECORD = tf.Variable(
            0, name='record', trainable=False, dtype=tf.float32)
        self.policy = Policy([
            OBS_SIZE, OBS_SIZE, FRAMES_LOOKBACK], NUMBER_OF_ACTIONS)
        self.sess = get_session(experiment_id)
        self.global_step = self.sess.run(tf.assign(self.GS, self.GS+1))
        self.record_run = self.sess.run(self.RECORD)
        self.memory = Memory(self.record_run)
        self.saver = tf.train.Saver(max_to_keep=0)
        self.experiment_id = experiment_id
        self.summ_writer = tf.summary.FileWriter(
            os.path.join('output', experiment_id, 'summaries'),
            self.sess.graph)
        self.record_score = 0
        self.batch_episode_counter = 0

    def learn(self):

        env_step_result = self.env.start_episode()
        save_result = {
            "state": None,
            "action": None,
            "reward": None,
            "value": None,
            "frame": None
        }

        logging.info('Starting trainig loop..')

        while True:

            while self.batch_episode_counter < BUFFER_SIZE:

                pol_step_result = self.policy.play(
                    env_step_result['stacked_observation'], self.sess)

                save_result.update(pol_step_result)
                save_result["state"] = env_step_result['stacked_observation']
                save_result["frame"] = env_step_result['visual_observation']
                save_result["done"] = env_step_result['done']

                env_step_result = self.env.step(pol_step_result['action'])
                save_result["reward"] = env_step_result['reward']

                self.memory.save(save_result)

                if env_step_result["done"]:
                    self.batch_episode_counter += 1
                    msg = 'Episode: {0} | Batch: {1}'.format(
                        self.batch_episode_counter, self.global_step)
                    logging.info(msg)

            self.batch_episode_counter = 0
            self.memory.compute_true_value()

            rollout_res = self.memory.get_rollout()

            opt_result = self.policy.optimize(
                rollout_res["states"],
                rollout_res["actions"],
                rollout_res["values"],
                rollout_res["advantages"], self.sess)

            episode_len = len(rollout_res["actions"])/BUFFER_SIZE
            episode_reward = sum(rollout_res["rewards"])/BUFFER_SIZE

            self.log_scalar(
                'value_loss', opt_result["value_loss"], self.global_step)
            self.log_scalar(
                'policy_loss', opt_result["policy_loss"], self.global_step)
            self.log_scalar(
                'entropy', opt_result["entropy"], self.global_step)
            self.log_scalar(
                'total_loss', opt_result["total_loss"], self.global_step)
            self.log_scalar(
                'episode_length', episode_len, self.global_step)
            self.log_scalar(
                'episode_reward', episode_reward, self.global_step)

            if rollout_res["record_beaten"]:
                best_score, best_run = self.memory.get_best()
                self.log_gif('best_run', best_run, self.global_step)
                self.record_score = best_score
                self.sess.run(tf.assign(self.RECORD, self.record_score))
                logging.info('Record beaten: {0}'.format(best_score))

            self.memory.clear()

            print('=======')

            if self.global_step % SAVE_MODEL_STEPS == 0:
                self.check_model()
                self.save_model()

                all_objects = muppy.get_objects()
                sum1 = summary.summarize(all_objects)

                total = [x[2] for x in sum1]
                print(total)
                total = total/1024
                print(total)
                total = sum(total)
                print(total)

                for item in sum1[:15]:
                    msg = 'Name: {0} | Count: {1} | Memory {2} MB'.format(item[0], item[1], item[2]/1024)
                    logging.info(msg)

                summary.print_(sum1)

            del rollout_res
            del opt_result

            all_objects = None
            sum1 = None

            self.global_step = self.sess.run(tf.assign(self.GS, self.GS+1))

    def check_model(self):
        for trainable_variable in tf.trainable_variables():
            self.sess.run(
                tf.check_numerics(trainable_variable, 'invalid tensor'))

    def save_model(self):
        assign_gs = tf.assign(self.GS, self.global_step)
        self.sess.run(assign_gs)
        path = os.path.join(
            'output', self.experiment_id, 'checkpoints', CHECKPOINT_FILE)
        self.saver.save(
            self.sess, path, self.global_step, write_meta_graph=False)

    def log_scalar(self, tag, value, step):
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.summ_writer.add_summary(summary, step)
        msg = 'step: {0} | {1}: {2}'.format(step, tag, value)
        logging.info(msg)

    def log_gif(self, tag, images, step):
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
