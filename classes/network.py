from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers import LSTM, Reshape, TimeDistributed
import tensorflow as tf
from consts import *


class Network:
    def __init__(self):
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(
                tf.float32, [None, 1, OBS_SIZE, OBS_SIZE, 1], name="X")
            self.Y = tf.placeholder(tf.int32, [None], name="Y")
            self.A = tf.placeholder(tf.float32, [None], name="G")

        with tf.name_scope('layers'):
            conv1 = TimeDistributed(
                Convolution2D(
                    32,
                    (8, 8),
                    activation='relu',
                    strides=2,
                    input_shape=(1, OBS_SIZE, OBS_SIZE, 1)))(self.X)
            conv2 = TimeDistributed(
                Convolution2D(
                    32,
                    (6, 6),
                    strides=2,
                    activation='relu'))(conv1)
            conv3 = TimeDistributed(
                Convolution2D(
                    32,
                    (4, 4),
                    activation='relu'))(conv2)
            flat = TimeDistributed(Flatten())(conv3)
            lstm = LSTM(256)(flat)
            self.logits = Dense(3, name="pred")(lstm)
            self.softmax = tf.nn.softmax(self.logits)
            self.output = tf.argmax(self.logits, name="output")

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.Y)
            loss = tf.reduce_mean(neg_log_prob * self.A)

        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(LR).minimize(loss)
