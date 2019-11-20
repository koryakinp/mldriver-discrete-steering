import numpy as np
import tensorflow as tf


def conv(inputs, nf, ks, strides):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=nf,
        kernel_size=ks,
        padding='same',
        strides=(strides, strides),
        activation=tf.nn.relu)


def maxpool(inputs, pool_size, strides):
    return tf.layers.max_pooling2d(
        inputs,
        pool_size,
        strides)


def fc(inputs, n, act=tf.nn.relu):
    return tf.layers.dense(inputs=inputs, units=n, activation=act)


class Policy():
    def __init__(self, cfg):

        obs_size = cfg.get('OBS_SIZE')
        obs_depth = cfg.get('FRAMES_LOOKBACK')
        num_of_actions = cfg.get('NUMBER_OF_ACTIONS')
        value_loss_coefficient = cfg.get('VALUE_LOSS_K')
        entropy_coefficient = cfg.get('ENTROPY_K')
        learning_rate = cfg.get('LEARNING_RATE')

        self.X = tf.placeholder(
            tf.float32, [None, obs_size, obs_size, obs_depth])

        self.A = tf.placeholder(tf.int32, [None])
        self.ADV = tf.placeholder(tf.float32, [None])
        self.R = tf.placeholder(tf.float32, [None])

        cv1 = conv(self.X, 32, 8, 4)
        cv2 = conv(cv1, 64, 4, 2)
        cv3 = conv(cv2, 64, 3, 1)
        flat = tf.layers.flatten(cv3)
        fc1 = fc(flat, 1024)
        fc2 = fc(fc1, 512)
        fc3 = fc(fc2, 256)
        actor = fc(fc3, num_of_actions, act=None)
        critic = fc(fc3, 1, act=None)

        self.v0 = tf.squeeze(critic)

        prob = tf.nn.softmax(actor)
        dist = tf.distributions.Categorical(logits=prob)
        self.a0 = tf.squeeze(dist.sample())

        self.value_loss = tf.reduce_mean(
            tf.square(tf.squeeze(critic) - self.R))
        action_one_hot = tf.one_hot(
            self.A, num_of_actions, dtype=tf.float32)
        neg_log_prob = -tf.log(tf.clip_by_value(prob, 1e-10, 1.0))
        self.policy_loss = tf.reduce_mean(
            tf.reduce_sum(neg_log_prob * action_one_hot, axis=1) * self.ADV)

        self.entropy = tf.reduce_mean(
            tf.reduce_sum(prob * neg_log_prob, axis=1))

        self.loss = \
            self.policy_loss + \
            self.value_loss * value_loss_coefficient - \
            self.entropy * entropy_coefficient

        self.adam = tf.train.RMSPropOptimizer(
            learning_rate, decay=0.99).minimize(self.loss)

    def play(self, ob, sess):
        a, v = sess.run([self.a0, self.v0], {self.X: ob})
        return a, v

    def optimize(self, s, a, r, adv, sess):
        pl, vl, ent, total, _ = sess.run([
            self.policy_loss,
            self.value_loss,
            self.entropy,
            self.loss,
            self.adam], {
                self.X: s,
                self.A: a,
                self.ADV: adv,
                self.R: r})

        res = {
            "policy_loss": pl,
            "value_loss": vl,
            "entropy": ent,
            "total_loss": total
        }

        return res
