from gym.envs.registration import register

register(
    id='feature-gridworld-v0',
    entry_point='gym_feature_gridworld.envs:FeatureGridworldEnv',
)