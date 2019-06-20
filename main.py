from classes.environment import Environment
from classes.agent import PolicyGradientAgent
from utils import *
from consts import *
import numpy as np

experiment_id = get_experiment_id()
env = Environment(FRAMES_LOOKBACK, FRAMES_SKIP, experiment_id)
agent = PolicyGradientAgent(env, experiment_id)
create_folders(experiment_id)
agent.learn()
