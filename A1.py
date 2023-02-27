import numpy as np
import os

from mujoco_env import MujocoEnv

from utilities import imitation_task

REF_DIR = os.path.join(os.path.dirname(__file__), 'ref')

class A1Env(MujocoEnv):
    def __init__(self, use_motor_mode=True):
        # self.init_serialization(locals())
        self.step_counter = 0
        self.env_step_counter = 0
        
        if use_motor_mode:
            xml_path = 'a1_motor.xml'
        else:
            xml_path = 'a1_pos.xml'
        
        super().__init__(
            xml_path,
            frame_skip=1,
            automatically_set_obs_and_action_space=True,
        )

        self.time_step = (self.dt / self.frame_skip) # in self.dt we do self.frame_skip 个 mjstep()
        self.imitation_task = imitation_task.ImitationTask(ref_motion_filenames=[os.path.join(REF_DIR, 'pace.txt')],
                                      enable_cycle_sync=True,
                                      tar_frame_steps=[1, 2, 10, 30],
                                      ref_state_init_prob=0.9,
                                      warmup_time=0.25)
        self.hard_reset = True
        self.reset()
        self.hard_reset = False

    def step(self, a):
        # used only in init
        # just know about obervation dimension
        if(self.first_try):
            ob = self._get_obs()
            reward = 0
            done = False
            return ob, reward, done, {}

        self.do_simulation(a, self.frame_skip)
        
        reward = self._reward()
        
        s = self.state_vector()

        if self.imitation_task and hasattr(self.imitation_task, 'done'):
            done = self.imitation_task.done(self)
        else:
            done = False
        
        ob = self._get_obs()
        
        self.step_counter += self.frame_skip
        
        if self.imitation_task and hasattr(self.imitation_task, 'update'):
            self.imitation_task.update(self)      
        
        self.env_step_counter += 1
        return ob, reward, done, {}

    def _get_obs(self):
        # this is gym ant obs, should use rllab?
        # if position is needed, override this in subclasses
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        self.env_step_counter = 0
        self.step_counter = 0
        NUM_LEGS = 4
        INIT_MOTOR_ANGLES = np.array([0, 0.9, -1.8] * NUM_LEGS)
        HIP_JOINT_OFFSET = 0.0
        UPPER_LEG_JOINT_OFFSET = 0.0
        KNEE_JOINT_OFFSET = 0.0
        JOINT_OFFSETS = np.array(
        [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
        JOINT_DIRECTIONS = np.ones(12) 
        root_pos = np.array([0., 0., 0.32])
        root_rot = np.array([1., 0., 0., 0.])
        joint_pose = (INIT_MOTOR_ANGLES + JOINT_OFFSETS) * JOINT_DIRECTIONS
        qpos = np.zeros(19)
        qpos[0:3] = root_pos
        qpos[3:7] = root_rot
        qpos[7:19] = joint_pose
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)        
        self.set_state(qpos, qvel)
        if self.imitation_task and hasattr(self.imitation_task, 'reset'):
            self.imitation_task.reset(self)

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_time_since_reset(self):
        return self.time_step * self.step_counter
    
    def _reward(self):
        if self.imitation_task:
            return self.imitation_task(self)
        return 0

if __name__ == "__main__":
    env = A1Env()
    while True:
        env.reset()
        for i in range(1000):
            ob, reward, done, _ = env.step(env.action_space.sample())  # take a random action
            env.render()
