import sys
import os
import tensorflow as tf


def create_folders():
    if not os.path.exists('summaries'):
        os.mkdir('summaries')
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists(os.path.join('summaries', 'rewards')):
        os.mkdir(os.path.join('summaries', 'rewards'))
    if not os.path.exists('sample-episodes'):
        os.mkdir('sample-episodes')


def get_session():
    checkpoint = ''

    if(len(sys.argv) > 1):
        checkpoint = sys.argv[1]

    session = tf.Session()
    if(checkpoint == ''):
        session.run(tf.global_variables_initializer())
    else:
        saver = tf.train.Saver()
        saver.restore(session, "checkpoints/{0}".format(checkpoint))
    return session
