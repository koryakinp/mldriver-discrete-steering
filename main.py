from classes.environment import Environment
from classes.policy_network import PolicyNetwork
from classes.value_network import ValueNetwork
from classes.agent import PolicyGradientAgent
from classes.memory import Memory
from utils import *
import uuid
import numpy as np


experiment_id = create_folders()
env = Environment()
policy_network = PolicyNetwork()
value_network = ValueNetwork()
session = get_session()
agent = PolicyGradientAgent(
  env, session, value_network, policy_network, experiment_id)
agent.train()
