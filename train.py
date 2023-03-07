import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env
from A1 import A1Env

checkpoint_callback = CheckpointCallback(
  save_freq=10000000,
  save_path="./logs/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[512, 256], vf=[512, 256]))

new_logger = configure("./debug_log", ["stdout", "csv", "tensorboard"])

env = A1Env()
check_env(env)
model = PPO("MlpPolicy", env, gamma=0.95, policy_kwargs=policy_kwargs, verbose=2)
model.set_logger(new_logger)
model.learn(total_timesteps=200000000, tb_log_name="first_run", callback=checkpoint_callback)
model.save("A1")
del model # remove to demonstrate saving and loading

