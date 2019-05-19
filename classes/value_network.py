from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D
from consts import *
import tensorflow as tf


class ValueNetwork:
    def __init__(self):
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(
                tf.float32,
                [None, OBS_SIZE, OBS_SIZE, FRAMES_LOOKBACK], name="X")
            self.Y = tf.placeholder(tf.float32, [None], name="Y")

        with tf.name_scope('layers'):
            conv1 = Convolution2D(32, (4, 4), activation='relu')(self.X)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            conv2 = Convolution2D(32, (4, 4), activation='relu')(conv1)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
            conv3 = Convolution2D(32, (4, 4), activation='relu')(conv2)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
            flat = Flatten()(pool3)
            dense = Dense(256, activation='relu')(flat)
            self.value = Dense(1, name="value")(dense)

        with tf.name_scope('value_loss'):
            self.loss = tf.squared_difference(self.value, self.Y)

        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(
                VALUE_NETWORK_LR).minimize(self.loss)

    def get_value(self, session, x):
        return session.run(self.value, feed_dict={self.X: x})

    def update(self, session, x, y):
        loss, _ = session.run(
            [self.loss, self.optimizer], feed_dict={self.X: x, self.Y: y})
        return loss
