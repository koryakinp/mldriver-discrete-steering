import numpy as np
import tensorflow as tf
from consts import *


def sample(logits):
    dist = tf.distributions.Categorical(logits=tf.nn.softmax(logits))
    return dist.sample()


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
    def __init__(self, sess, ob_space, ac_space):
        nh, nw, nc = ob_space
        X = tf.placeholder(tf.float32, [None, nh, nw, nc])

        A = tf.placeholder(tf.int32, [None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])

        h1 = conv(X, 32, 8, 4)
        pool1 = maxpool(h1, 2, 2)
        h2 = conv(pool1, 64, 4, 2)
        pool2 = maxpool(h2, 2, 2)
        h3 = conv(pool2, 64, 3, 1)
        h3 = tf.layers.flatten(h3)
        h4 = fc(h3, 512)
        pi = fc(h4, ac_space, act=None)
        vf = fc(h4, 1, act=None)

        v0 = tf.squeeze(vf)
        a0 = tf.squeeze(sample(pi))

        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pi, labels=A)
        policy_loss = tf.reduce_mean(ADV * neg_log_prob)
        value_loss = tf.reduce_mean(tf.squared_difference(tf.squeeze(vf), R))
        loss = policy_loss + value_loss * VALUE_LOSS_K

        adam = tf.train.AdamOptimizer(LR).minimize(loss)

        def step(ob, *_args, **_kwargs):
            return sess.run([a0, v0], {X: ob})

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X: ob})

        def optimize(s, a, r, adv):
            pl, vl, _ = sess.run(
                [policy_loss, value_loss, adam], {X: s, A: a, ADV: adv, R: r})

            return pl, vl

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.optimize = optimize
