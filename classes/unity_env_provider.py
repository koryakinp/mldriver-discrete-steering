from mlagents.envs import UnityEnvironment
from consts import *


class UnityEnvironmentProvider:
    def provide(self):
        connected = False
        worker_id = 0
        while not connected and worker_id < 100:
            try:
                env = UnityEnvironment(file_name=ENV_NAME, worker_id=worker_id)
                connected = True
            except:
                msg = 'Could not create UnityEnvironment with worker_id {0}'
                print(msg.format(worker_id))
                worker_id += 1
                msg = 'Trying re-create UnityEnvironment with worker_id {0}'
                print(msg.format(worker_id))
                pass

        return env
