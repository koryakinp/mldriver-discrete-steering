from classes.environment import Environment
from classes.agent import PolicyGradientAgent
from classes.config import Config
from classes.unity_env_provider import UnityEnvironmentProvider
from utils import *
from consts import *
import numpy as np
import logging


try:
    experiment_id = get_experiment_id()
    cfg = Config('config.json')
    env_provider = UnityEnvironmentProvider(cfg)
    env = Environment(env_provider, cfg)
    agent = PolicyGradientAgent(env, cfg, experiment_id)
    create_folders(experiment_id)
    agent.learn()
except Exception as e:
    logging.exception("message")
