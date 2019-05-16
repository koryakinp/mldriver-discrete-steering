from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers import LSTM, Reshape, TimeDistributed, MaxPooling2D
import tensorflow as tf
from consts import *


class Network:
    def __init__(self):
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(
                tf.float32,
                [None, OBS_SIZE, OBS_SIZE, FRAMES_LOOKBACK], name="X")
            self.Y = tf.placeholder(tf.int32, [None], name="Y")
            self.A = tf.placeholder(tf.float32, [None], name="G")

        with tf.name_scope('layers'):
            conv1 = Convolution2D(
                32, (8, 8), strides=(2, 2), activation='relu')(self.X)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            conv2 = Convolution2D(
                32, (6, 6), strides=(2, 2), activation='relu')(conv1)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
            conv3 = Convolution2D(
                32, (4, 4), activation='relu')(conv2)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
            flat = Flatten()(pool3)

            dense = Dense(256, activation='relu')(flat)

            self.logits = Dense(3, name="pred")(dense)
            self.softmax = tf.nn.softmax(self.logits)
            self.output = tf.argmax(self.logits, name="output")

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.Y)
            loss = neg_log_prob * self.A

        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(LR).minimize(loss)
