import sys
import os
import tensorflow as tf
import uuid


def create_folders():
    if not os.path.exists('summaries'):
        os.mkdir('summaries')

    experiment_id = str(uuid.uuid4())

    os.mkdir(os.path.join('summaries', experiment_id))
    os.mkdir(os.path.join('summaries', experiment_id, 'rewards'))
    os.mkdir(os.path.join('summaries', experiment_id, 'checkpoints'))
    os.mkdir(os.path.join('summaries', experiment_id, 'sample-episodes'))

    return experiment_id


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
