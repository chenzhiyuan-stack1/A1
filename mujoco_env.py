import os
from os import path

import mujoco_py
import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import spaces

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')

class MujocoEnv(mujoco_env.MujocoEnv):
    """
    My own wrapper around MujocoEnv.

    The caller needs to declare
    """
    def __init__(
            self,
            model_path,
            frame_skip=1,
            model_path_is_local=True,
            automatically_set_obs_and_action_space=False,
    ):
        if model_path_is_local:
            model_path = get_asset_xml(model_path)
        if automatically_set_obs_and_action_space:
            self.first_try = True # just find the dimension
            mujoco_env.MujocoEnv.__init__(self, model_path, frame_skip)
            self.first_try = False
            self.ref_sim = mujoco_py.MjSim(self.model)
            self.ref_data = self.ref_sim.data
            # use a symmetric and normalized Box action space
            bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
            low, high = bounds.T
            low = low/33.5
            high = high/33.5
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)


def get_asset_xml(xml_name):
    return os.path.join(ENV_ASSET_DIR, xml_name)
