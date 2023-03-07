import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from A1 import A1Env

env = A1Env()

model = PPO.load("A1")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, dones, info = env.step(action)
    # env.render()