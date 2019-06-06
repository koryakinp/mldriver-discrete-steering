from classes.environment import Environment
from classes.agent import PolicyGradientAgent
from utils import *
from consts import *
import numpy as np

clear()
experiment_id = create_folders()
env = Environment(FRAMES_LOOKBACK, FRAMES_SKIP, experiment_id)
session = tf.Session()

agent = PolicyGradientAgent(env, session, experiment_id)
session.run(tf.global_variables_initializer())

agent.learn()
