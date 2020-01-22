from gym.envs.registration import register

register(
    id='FeatureGridworld-v0',
    entry_point='gym_feature_gridworld.envs:FeatureGridworldEnv',
    max_episode_steps=500,
)