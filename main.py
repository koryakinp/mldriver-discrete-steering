from classes.environment import Environment
from classes.network import Network
from classes.agent import PolicyGradientAgent
from utils import *
import uuid


experiment_id = create_folders()
env = Environment()
network = Network()
session = get_session()
agent = PolicyGradientAgent(env, session, network, experiment_id)
agent.train()
