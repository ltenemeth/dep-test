import ray
from ray.tune.registry import register_env
from ray.rllib.agents import dqn
import gym_gazebo

def env_creator(env_config):
    import gym
    return gym.make("GazeboCartPole-v0")  # or return your own custom env

register_env("my_env", env_creator)
ray.init()
trainer = dqn.DQNAgent(env="my_env", config={
    "env_config": {},  # config to pass to env creator
})

while True:
    print(trainer.train())

