import sys
import os
import tensorflow as tf
import tempfile
import moviepy.editor as mpy
import numpy as np
import uuid
from PIL import Image
import os.path as path
import sys


def get_experiment_id():

    if len(sys.argv) != 2:
        raise Exception('Must provide experiment id or "new" keyword')

    if sys.argv[1] == 'new':
        return str(uuid.uuid4())

    if os.path.exists(os.path.join('output', sys.argv[1])):
        return sys.argv[1]

    raise Exception('Can not find experiment {0}'.format(sys.argv[1]))


def get_session(experiment_id):
    session = tf.Session()
    if os.path.exists(os.path.join('output', experiment_id)):
        saver = tf.train.Saver()
        saver.restore(
            session, tf.train.latest_checkpoint(
                os.path.join('output', experiment_id, 'checkpoints')))
    else:
        session.run(tf.global_variables_initializer())
    return session


def create_folders(experiment_id):
    if not os.path.exists('output'):
        os.mkdir(os.path.join('output'))

    if not os.path.exists(os.path.join('output', experiment_id)):
        os.mkdir(os.path.join('output', experiment_id))
        os.mkdir(os.path.join('output', experiment_id, 'checkpoints'))
        os.mkdir(os.path.join('output', experiment_id, 'summaries'))


def tensor_to_gif_summ(summ):
    if isinstance(summ, bytes):
        summary_proto = tf.Summary()
        summary_proto.ParseFromString(summ)
        summ = summary_proto

    summary = tf.Summary()
    for value in summ.value:
        tag = value.tag
        images_arr = tf.make_ndarray(value.tensor)

        if len(images_arr.shape) == 5:
            # concatenate batch dimension horizontally
            images_arr = np.concatenate(list(images_arr), axis=-2)
        if len(images_arr.shape) != 4:
            raise ValueError('Tensors must be 4-D or 5-D for gif summary.')
        if images_arr.shape[-1] != 1:
            raise ValueError('Tensors must have 3 channels.')

        # encode sequence of images into gif string
        clip = mpy.ImageSequenceClip(list(images_arr), fps=30)
        with tempfile.NamedTemporaryFile() as f:
            filename = f.name + '.gif'
        clip.write_gif(filename, verbose=False)
        with open(filename, 'rb') as f:
            encoded_image_string = f.read()

        image = tf.Summary.Image()
        image.height = images_arr.shape[-3]
        image.width = images_arr.shape[-2]
        image.colorspace = 1  # code for 'RGB'
        image.encoded_image_string = encoded_image_string
        summary.value.add(tag=tag, image=image)
    return summary
