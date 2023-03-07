import gym

from stable_baselines3 import PPO
from A1 import A1Env

from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env

new_logger = configure("./debug_log", ["stdout", "csv", "tensorboard"])
env = A1Env()
check_env(env)
model = PPO("MlpPolicy", env, verbose=2)
model.set_logger(new_logger)
model.learn(total_timesteps=25000000, tb_log_name="first_run")
model.save("A1")
del model # remove to demonstrate saving and loading

