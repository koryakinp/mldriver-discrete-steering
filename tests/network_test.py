from classes.network import Network
import unittest
import numpy as np
import tensorflow as tf
from consts import *


class Test_Netowrk(unittest.TestCase):

    def test_forward(self):
        network = Network()
        x = np.random.rand(1, OBS_SIZE, OBS_SIZE, FRAMES_LOOKBACK)
        y = np.array([0, 1, 0])
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        actions = session.run(network.softmax, feed_dict={
            network.X: x,
            network.Y: y})

        self.assertTrue(True)

    def test_backward(self):
        network = Network()
        x = np.random.rand(3, OBS_SIZE, OBS_SIZE, FRAMES_LOOKBACK)
        y = np.array([0, 1, 0])
        a = np.random.rand(3)
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        actions = session.run(network.optimizer, feed_dict={
            network.X: x,
            network.Y: y,
            network.A: a})

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
