import sys
import os
import tensorflow as tf
import tempfile
import moviepy.editor as mpy
import numpy as np
import uuid


def create_folders():
    if not os.path.exists('summaries'):
        os.mkdir('summaries')

    experiment_id = str(uuid.uuid4())

    os.mkdir(os.path.join('summaries', experiment_id))

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
        clip = mpy.ImageSequenceClip(list(images_arr), fps=10)
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
