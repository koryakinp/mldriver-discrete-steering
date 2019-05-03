from classes.environment import Environment
from classes.agent import PolicyGradientAgent
from collections import deque
from consts import *
import os
import tensorflow as tf
import sys

if not os.path.exists('summaries'):
    os.mkdir('summaries')
if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')
if not os.path.exists(os.path.join('summaries', 'rewards')):
    os.mkdir(os.path.join('summaries', 'rewards'))

checkpoint = ''

if(len(sys.argv) > 1):
    checkpoint = sys.argv[1]

env = Environment()
agent = PolicyGradientAgent(env)

session = tf.Session()

if(checkpoint == ''):
    session.run(tf.global_variables_initializer())
else:
    saver = tf.train.Saver()
    saver.restore(session, "checkpoints/{0}".format(checkpoint))

summ_writer = tf.summary.FileWriter(
    os.path.join('summaries', 'rewards'), session.graph)

agent.train(session, summ_writer)
