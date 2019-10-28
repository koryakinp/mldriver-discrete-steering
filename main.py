from classes.environment import Environment
from classes.agent import PolicyGradientAgent
from classes.unity_env_provider import UnityEnvironmentProvider
from utils import *
from consts import *
import numpy as np
import logging


try:
    experiment_id = get_experiment_id()
    env_provider = UnityEnvironmentProvider()
    env = Environment(
      env_provider, OBS_SIZE, FRAMES_LOOKBACK, FRAMES_SKIP, USE_DIFF)
    agent = PolicyGradientAgent(env, experiment_id)
    create_folders(experiment_id)
    agent.learn()
except Exception as e:
    logging.exception("message")
