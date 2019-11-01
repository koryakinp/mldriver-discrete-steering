import json
import os


class Config:

    def __init__(self, path):

        self.consts = {
            "OBS_SIZE": 128,
            "SAVE_MODEL_STEPS": 3,
            "BATCH_SIZE": 64,
            "ENV_NAME": "environments/mldriver-discrete-steering",
            "CHECKPOINT_FILE": "mldriver.ckpt",
            "NUMBER_OF_ACTIONS": 3,
            "GAMMA": None,
            "LEARNING_RATE": None,
            "FRAMES_LOOKBACK": None,
            "FRAMES_SKIP": None,
            "VALUE_LOSS_K": None,
            "ENTROPY_K": None,
            "USE_DIFF": None,
            "REPLAY_BUFFER_SIZE": None,
        }

        if path is not None:
            with open(path) as f:
                values = json.load(f)
                self.set_value(values, "GAMMA")
                self.set_value(values, "LEARNING_RATE")
                self.set_value(values, "FRAMES_LOOKBACK")
                self.set_value(values, "FRAMES_SKIP")
                self.set_value(values, "VALUE_LOSS_K")
                self.set_value(values, "ENTROPY_K")
                self.set_value(values, "USE_DIFF")
                self.set_value(values, "REPLAY_BUFFER_SIZE")

    def get(self, key):
        if key not in self.consts:
            raise Exception('{0} not found.'.format(key))
        else:
            return self.consts[key]

    def set_value(self, values, key):
        if key not in self.consts or key not in values:
            raise Exception('{0} not found.'.format(key))
        else:
            self.consts[key] = values[key]

    def set_config_value(self, key, value):
        if key in self.consts:
            self.consts[key] = value
        else:
            raise Exception('{0} not found.'.format(key))
