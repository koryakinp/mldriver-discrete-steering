import numpy as np
import tensorflow as tf
from consts import *


def conv(inputs, nf, ks, strides):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=nf,
        kernel_size=ks,
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
    def __init__(self, ob_space, ac_space):
        nh, nw, nc = ob_space
        self.X = tf.placeholder(tf.float32, [None, nh, nw, nc])

        self.A = tf.placeholder(tf.int32, [None])
        self.ADV = tf.placeholder(tf.float32, [None])
        self.R = tf.placeholder(tf.float32, [None])

        h1 = conv(self.X, 32, 4, 2)
        pool1 = maxpool(h1, 2, 2)
        h2 = conv(pool1, 48, 4, 2)
        pool2 = maxpool(h2, 2, 2)
        h3 = conv(pool2, 64, 4, 1)
        flat = tf.layers.flatten(h3)
        h4 = fc(flat, 1024, act=tf.nn.sigmoid)
        h5 = fc(h4, 1024, act=tf.nn.sigmoid)
        actor = fc(h5, ac_space, act=None)
        critic = fc(h5, 1, act=None)

        self.v0 = tf.squeeze(critic)

        prob = tf.nn.softmax(actor)
        dist = tf.distributions.Categorical(logits=prob)
        self.a0 = tf.squeeze(dist.sample())

        self.value_loss = tf.reduce_mean(
            tf.square(tf.squeeze(critic) - self.R))
        action_one_hot = tf.one_hot(
            self.A, NUMBER_OF_ACTIONS, dtype=tf.float32)
        neg_log_prob = -tf.log(tf.clip_by_value(prob, 1e-10, 1.0))
        self.policy_loss = tf.reduce_mean(
            tf.reduce_sum(neg_log_prob * action_one_hot, axis=1) * self.ADV)

        self.entropy = tf.reduce_mean(
            tf.reduce_sum(prob * neg_log_prob, axis=1))

        self.loss = \
            self.policy_loss + \
            self.value_loss * VALUE_LOSS_K - \
            self.entropy * ENTROPY_K

        self.adam = tf.train.RMSPropOptimizer(LR).minimize(self.loss)

    def play(self, ob, sess):
        a, v = sess.run([self.a0, self.v0], {self.X: ob})

        res = {
            "action": a,
            "value": v
        }

        return res

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
