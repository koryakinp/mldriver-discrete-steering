from mlagents.envs import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name="environments/mldriver-discrete-steering", worker_id=1)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]