import numpy as np
import tensorflow as tf
import sys


checkpoint = sys.argv[1] if len(sys.argv) == 2 else None
print(checkpoint)


def print_vars(variables, session):
    for n in variables:
        if n.name == 'global_step:0':
            print(n.name)
            val = session.run(n)
            print(val)
            print('=====')

global_step = 1

X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.int32, [None, 2])
GS = tf.Variable(1, name='global_step', trainable=False)

fc1 = tf.layers.dense(inputs=X, units=3, activation=tf.nn.relu)
fc2 = tf.layers.dense(inputs=fc1, units=3, activation=tf.nn.relu)
out = tf.layers.dense(inputs=fc2, units=2, activation=tf.nn.softmax)

loss = tf.losses.mean_squared_error(Y, out)
adam = tf.train.AdamOptimizer(0.1).minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())


for i in range(10):
    global_step += 1
    x = np.random.rand(10, 3)
    y = np.eye(2)[np.random.choice(2, 10)]
    assign = tf.assign(GS, global_step)
    session.run([adam, assign], {X: x, Y: y})

saver = tf.train.Saver()
print_vars(tf.global_variables(), session)
saver.save(session, './checkpoints/', global_step, write_meta_graph=False)


session2 = tf.Session()
saver.restore(session2, tf.train.latest_checkpoint('./checkpoints/'))
print_vars(tf.global_variables(), session2)
