import gym
import gym_feature_gridworld  # needed to register
# from stable_baselines.common.env_checker import check_env  # to check provided feature_gridworld env

env = gym.make('FeatureGridworld-v0')
# env = gym.make('CartPole-v1')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])
#
# # (From stable-baselines:) It will check your custom environment and output additional warnings if needed
# check_env(env)