from mlagents.envs import UnityEnvironment


class UnityEnvironmentProvider:

    def __init__(self, cfg):
        self.env_name = cfg.get('ENV_NAME')

    def provide(self):
        connected = False
        worker_id = 0
        while not connected and worker_id < 100:
            try:
                env = UnityEnvironment(
                    file_name=self.env_name, worker_id=worker_id)
                connected = True
            except Exception:
                msg = 'Could not create UnityEnvironment with worker_id {0}'
                print(msg.format(worker_id))
                worker_id += 1
                msg = 'Trying re-create UnityEnvironment with worker_id {0}'
                print(msg.format(worker_id))

        if worker_id >= 100:
            raise Exception("Can not create UnityEnvironment")

        return env
