import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

class FeatureGridworldEnv(gym.Env):
    """
    FeatureGridworldEnv

    \phi(s, a) agent_position-action features corresponding to
    "What do I see if I am standing in agent_position s and look in direction a?"
    They are currently
        0. is first cell a fire?
        1. is first cell a gold bar?
        2. # of fires seen
        3. # of gold bars seen
        4. cumulative distance to all fires seen
        5. cumulative distance to all gold bars seen
        6. # of fires before first gold bar
        7. # of gold bars before first fire
        8. distance to wall or blocked cell

    """
    metadata = {'render.modes': ['human']}
    action_names = {0: "n", 1: "e", 2: "s", 3: "w"}
    compass = {0: [1, 0], 1: [0, 1], 2: [-1, 0], 3: [0, -1]}
    feature_names = {0: "is_first_fire", 1: "is_first_gold", 2: "num_fires", 3: "num_golds",
                     4: "cumu_dist_fires", 5: "cumu_dist_golds", 6: "num_fire_before_gold",
                     7: "num_gold_before_fire", 8: "distance_to_wall"}
    num_actions = len(action_names)
    num_features_per_action = len(feature_names)
    num_features = int(num_features_per_action * num_actions)  # Note: feature vector for every action!

    def __init__(self, num_rows=7, num_cols=7, predefined_layout=True):
        self.num_rows = num_rows
        self.num_cols = num_cols
        feature_lower_bounds = np.zeros(FeatureGridworldEnv.num_features)
        feature_upper_bounds = np.ones(FeatureGridworldEnv.num_features)
        self.observation_space = spaces.Box(low=feature_lower_bounds, high=feature_upper_bounds)
        self.action_space = spaces.Discrete(FeatureGridworldEnv.num_actions)
        # self.stochasticity_eps = 0
        self.compass = FeatureGridworldEnv.compass

        ''' -- -- Create grid -- -- '''
        self.grid = np.full(shape=(num_rows, num_cols), fill_value=" ")
        self.num_remaining_gold = np.sum(self.grid == "g")
        self.agent_position = np.array([0, 0])
        self.rewards = {" ": -1,
                        "g": 10,
                        "f": -5}
        self.done = False
        self.predefined_layout = True
        self.reset()

    def step(self, action):
        """
        Parameters
        ----------
        action : integer 0 <= action <= 3 corresponding to the 4 cardinal directions.

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last agent_position change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        assert action in self.action_names.keys()
        if np.random.uniform(0, 1) < self.stochasticity_eps:
            action = np.random.choice(FeatureGridworldEnv.action_names.keys())

        old_position = self.agent_position.copy()
        new_position = self.agent_position.copy()
        candidate_position = self.agent_position.copy() + self.compass[action]
        if (0 <= candidate_position[0] < self.num_rows and
                0 <= candidate_position[1] < self.num_cols and
                self.grid[candidate_position[0], candidate_position[1]] != "w"):
            new_position = candidate_position

        new_cell = self.grid[new_position[0], new_position[1]]
        reward = self.rewards[new_cell]
        if new_cell == "g":
            self.num_remaining_gold -= 1
        self.grid[old_position[0], old_position[1]] = " "
        self.grid[new_position[0], new_position[1]] = "A"
        self.agent_position = new_position

        self.done = self.check_terminal()
        ob = self._get_feature_values()

        return ob, reward, self.done, None

    def _get_feature_values(self):
        features = np.zeros(self.num_features)
        for action in range(self.num_actions):
            if action == 0:
                features[:self.num_features_per_action] = self._feature_values_of_vector(
                    self.grid[(self.agent_position[0] + 1):, self.agent_position[1]])
            elif action == 1:
                features[self.num_features_per_action: (2 * self.num_features_per_action)] \
                    = self._feature_values_of_vector(self.grid[self.agent_position[0],
                                                     (self.agent_position[1] + 1):])
            elif action == 2:
                features[(2 * self.num_features_per_action): (3 * self.num_features_per_action)] \
                    = self._feature_values_of_vector(self.grid[:self.agent_position[0],
                                                     self.agent_position[1]][::-1])
            elif action == 3:
                features[(3 * self.num_features_per_action): (4 * self.num_features_per_action)] \
                    = self._feature_values_of_vector(self.grid[self.agent_position[0],
                                                     :self.agent_position[1]][::-1])
        return features

    def _feature_values_of_vector(self, gaze):
        """
        Fill up features array with array indices given the
        0. is first cell a fire?
        1. is first cell a gold bar?
        2. # of fires seen
        3. # of gold bars seen
        4. cumulative distance to all fires seen
        5. cumulative distance to all gold bars seen
        6. # of fires before first gold bar
        7. # of gold bars before first fire
        8. distance to wall or blocked cell
        """
        features = np.zeros(self.num_features_per_action)
        has_found_fire = False
        has_found_gold = False
        features[8] = len(gaze)
        for ix in range(len(gaze)):
            cell = gaze[ix]
            if cell == "f":
                if ix == 0:
                    features[0] = 1
                features[2] += 1
                features[4] += ix
                has_found_fire = True
                if not has_found_gold:
                    features[6] += 1
            elif cell == "g":
                if ix == 0:
                    features[0] = 1
                features[3] += 1
                features[5] += ix
                has_found_gold = True
                if not has_found_fire:
                    features[7] += 1
            elif cell == "w":
                features[8] = ix
                break
        return features

    def check_terminal(self):
        return self.num_remaining_gold == 0

    def reset(self):
        if self.predefined_layout:
            self.grid = np.flipud(np.array([["g", "f", " ", " ", " ", " ", "g"],    # top
                                            ["w", " ", "f", "w", "w", "w", "f"],
                                            ["g", " ", " ", "f", " ", " ", "g"],
                                            [" ", "w", " ", " ", " ", " ", "f"],
                                            [" ", "f", "w", "w", "f", " ", " "],
                                            [" ", " ", "g", "f", "g", "g", "g"],
                                            ["f", "A", "f", "g", " ", "f", "f"]]))  # bottom
            self.agent_position = np.array([0, 1])
        else:
            raise NotImplementedError
        self.num_remaining_gold = np.sum(self.grid == "g")
        assert self.num_remaining_gold > 0
        assert (self.num_rows, self.num_rows) == self.grid.shape
        self.done = False

    def render(self, mode='human'):
        self._print_world()

    def _print_world(self):
        # self.grid[self.agent_position[0], self.agent_position[1]] = "A"
        print(np.flipud(self.grid))

    def close(self):
        pass


env = FeatureGridworldEnv()
env.step(1)