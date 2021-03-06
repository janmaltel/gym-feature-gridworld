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
    feature_names = ["is_first_fire",
                     "is_first_gold",
                     # "num_fires",
                     # "num_golds",
                     # 4: "cumu_dist_fires", 5: "cumu_dist_golds",
                     "num_fire_before_gold",
                     "num_gold_before_fire",
                     "distance_to_wall",
                     "is_first_wall"]

    # feature_names = {0: "is_first_fire", 1: "is_first_gold",
    #                  2: "num_fires", 3: "num_golds",
    #                  # 4: "cumu_dist_fires", 5: "cumu_dist_golds",
    #                  6: "num_fire_before_gold", 7: "num_gold_before_fire",
    #                  8: "distance_to_wall"}
    num_actions = len(action_names)
    num_features_per_action = len(feature_names)
    num_features = int(num_features_per_action * num_actions)  # Note: feature vector for every action!

    def __init__(self, num_rows=7, num_cols=7, predefined_layout="simple"):
        super(FeatureGridworldEnv, self).__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        assert self.num_cols == self.num_rows, "Only quadratic worlds implemented atm."
        self.max_dist = self.num_cols - 1  # -1 because agent has to stand somewhere (--> full num_rows/cols not possible)
        self.max_cumu = self.max_dist * (self.max_dist + 1) / 2  # n * (n+1) / 2
        self.feature_max_values = np.array([1, 1,                                    # is __ first
                                            # self.max_dist, self.max_dist,            # num of __
                                            # self.max_cumu, self.max_cumu,            # cumulative distances
                                            self.max_dist - 1, self.max_dist - 1,    # num of __ before __
                                            self.max_dist,                           # distance to wall
                                            1                                        # is wall first
                                            ])
        feature_lower_bounds = np.zeros(FeatureGridworldEnv.num_features)
        feature_upper_bounds = np.ones(FeatureGridworldEnv.num_features)
        self.observation_space = spaces.Box(low=feature_lower_bounds, high=feature_upper_bounds)
        self.action_space = spaces.Discrete(FeatureGridworldEnv.num_actions)
        # self.stochasticity_eps = 0
        self.compass = FeatureGridworldEnv.compass
        self.observation_space_per_action = spaces.Box(low=feature_lower_bounds[:self.num_features_per_action],
                                                       high=feature_upper_bounds[:self.num_features_per_action])

        '''  -- -- Create grids -- -- 
        self.plot_grid is mainly for env.render(). It plots an "A" for the agent.
        self.true_grid does not care about the agent (which is also available from self.agent_position),
                       instead, it is for reward computation and game progress (fires stay fires and are
                       not overridden by the agent "A")  
        '''
        self.plot_grid = np.full(shape=(num_rows, num_cols), fill_value=" ")
        self.true_grid = np.full(shape=(num_rows, num_cols), fill_value=" ")
        self.num_remaining_gold = int(np.sum(self.plot_grid == "g"))
        self.agent_position = np.array([0, 0])
        self.last_pos_was_fire = False
        self.reward_scale_factor = 10
        self.rewards = {" ": -1 / self.reward_scale_factor,  # Transition into empty cell
                        "A": -1 / self.reward_scale_factor,  # Hitting the wall (staying on the same cell)
                        "g": 12 / self.reward_scale_factor,  # Gold
                        "f": -7 / self.reward_scale_factor}  # Fires
        self.done = False
        self.predefined_layout = predefined_layout
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
        # if np.random.uniform(0, 1) < self.stochasticity_eps:
        #     action = np.random.choice(FeatureGridworldEnv.action_names.keys())

        old_position = self.agent_position.copy()
        new_position = self.agent_position.copy()
        candidate_position = self.agent_position.copy() + self.compass[action]
        if (0 <= candidate_position[0] < self.num_rows and
                0 <= candidate_position[1] < self.num_cols and
                self.plot_grid[candidate_position[0], candidate_position[1]] != "w"):
            new_position = candidate_position

        agent_has_moved = np.any(old_position != new_position)
        new_cell = self.true_grid[new_position[0], new_position[1]]
        reward = self.rewards[new_cell]
        if agent_has_moved:
            # new_cell = self.true_grid[new_position[0], new_position[1]]
            # reward = self.rewards[new_cell]
            if self.last_pos_was_fire:
                # Fire is still burning even if leaving the field.
                self.plot_grid[old_position[0], old_position[1]] = "f"
                self.last_pos_was_fire = False
            else:
                self.plot_grid[old_position[0], old_position[1]] = " "
            if new_cell == "g":
                self.true_grid[new_position[0], new_position[1]] = " "
                self.num_remaining_gold -= 1
            elif new_cell == "f":
                self.last_pos_was_fire = True
            self.plot_grid[new_position[0], new_position[1]] = "A"
            self.agent_position = new_position
            self.done = self.check_terminal()

        ob = self._get_feature_values()
        return ob, reward, self.done, {}

    def _get_feature_values(self):
        features = np.zeros(self.num_features)
        for action in range(self.num_actions):
            if action == 0:
                features[:self.num_features_per_action] = self._feature_values_of_vector(
                    self.plot_grid[(self.agent_position[0] + 1):, self.agent_position[1]])
            elif action == 1:
                features[self.num_features_per_action: (2 * self.num_features_per_action)] \
                    = self._feature_values_of_vector(self.plot_grid[self.agent_position[0],
                                                     (self.agent_position[1] + 1):])
            elif action == 2:
                features[(2 * self.num_features_per_action): (3 * self.num_features_per_action)] \
                    = self._feature_values_of_vector(self.plot_grid[:self.agent_position[0],
                                                     self.agent_position[1]][::-1])
            elif action == 3:
                features[(3 * self.num_features_per_action): (4 * self.num_features_per_action)] \
                    = self._feature_values_of_vector(self.plot_grid[self.agent_position[0],
                                                     :self.agent_position[1]][::-1])
        return features

    def _feature_values_of_vector(self, gaze):
        """
        Fill up features array with array indices given the
        0. is first cell a fire?
        1. is first cell a gold bar?
        2. # of fires seen
        3. # of gold bars seen

        4. distance to wall or blocked cell
        """
        features = np.zeros(self.num_features_per_action)
        has_found_fire = False
        has_found_gold = False
        len_gaze = len(gaze)
        # features[4] = len_gaze
        if len_gaze == 0:
            features[5] = 1
        for ix in range(len_gaze):
            cell = gaze[ix]
            if cell == "f":
                if ix == 0:
                    features[0] = 1
                # features[2] += 1
                # features[4] += ix
                has_found_fire = True
                if not has_found_gold:
                    features[2] += 1
            elif cell == "g":
                if ix == 0:
                    features[1] = 1
                # features[3] += 1
                # features[5] += ix
                has_found_gold = True
                if not has_found_fire:
                    features[3] += 1
            elif cell == "w":
                if ix == 0:
                    features[5] = 1
                features[4] = ix
                break
        features = features / self.feature_max_values
        return features

    # def _feature_values_of_vector_old(self, gaze):
    #     """
    #     Fill up features array with array indices given the
    #     0. is first cell a fire?
    #     1. is first cell a gold bar?
    #     2. # of fires seen
    #     3. # of gold bars seen
    #     4. cumulative distance to all fires seen
    #     5. cumulative distance to all gold bars seen
    #     6. # of fires before first gold bar
    #     7. # of gold bars before first fire
    #     8. distance to wall or blocked cell
    #     """
    #     features = np.zeros(self.num_features_per_action)
    #     has_found_fire = False
    #     has_found_gold = False
    #     features[8] = len(gaze)
    #     for ix in range(len(gaze)):
    #         cell = gaze[ix]
    #         if cell == "f":
    #             if ix == 0:
    #                 features[0] = 1
    #             features[2] += 1
    #             features[4] += ix
    #             has_found_fire = True
    #             if not has_found_gold:
    #                 features[6] += 1
    #         elif cell == "g":
    #             if ix == 0:
    #                 features[1] = 1
    #             features[3] += 1
    #             features[5] += ix
    #             has_found_gold = True
    #             if not has_found_fire:
    #                 features[7] += 1
    #         elif cell == "w":
    #             features[8] = ix
    #             break
    #     features = features / self.feature_max_values
    #     return features

    def check_terminal(self):
        return self.num_remaining_gold == 0

    def reset(self):
        if self.predefined_layout == "simple":
            self.plot_grid = np.flipud(np.array([["g", "f", " ", " ", " ", "g", "g"],  # top
                                                 [" ", " ", " ", " ", "w", " ", "f"],
                                                 ["g", " ", " ", "f", " ", " ", "g"],
                                                 [" ", " ", " ", " ", " ", " ", " "],
                                                 [" ", "f", "w", " ", "f", " ", " "],
                                                 [" ", " ", " ", " ", "g", "g", "g"],
                                                 ["f", "A", " ", "g", " ", "f", "f"]]))  # bottom
            self.agent_position = np.array([0, 1])
        elif self.predefined_layout == "hidden_gold":
            self.plot_grid = np.flipud(np.array([["g", "f", " ", " ", " ", " ", "g"],  # top
                                                 ["w", " ", " ", "w", "w", "w", "f"],
                                                 ["g", " ", " ", "f", " ", " ", "g"],
                                                 [" ", "w", " ", " ", " ", " ", "f"],
                                                 [" ", "f", "w", "w", "f", " ", " "],
                                                 [" ", " ", "g", "f", "g", "g", "g"],
                                                 ["f", "A", "f", "g", " ", "f", "f"]]))  # bottom
            self.agent_position = np.array([0, 1])
        elif self.predefined_layout == "no_hidden_gold":
            self.plot_grid = np.flipud(np.array([[" ", "f", " ", " ", " ", " ", "g"],  # top
                                                 ["w", " ", " ", " ", "w", "w", "f"],
                                                 ["g", " ", " ", " ", " ", " ", "g"],
                                                 [" ", "w", " ", " ", " ", " ", "f"],
                                                 [" ", " ", "w", "w", "f", " ", " "],
                                                 [" ", " ", "g", "f", "g", "g", "g"],
                                                 ["f", "A", "f", "g", " ", "f", "f"]]))  # bottom
            self.agent_position = np.array([0, 1])
        else:
            raise NotImplementedError
        self.true_grid = self.plot_grid.copy()
        self.num_remaining_gold = int(np.sum(self.plot_grid == "g"))
        assert self.num_remaining_gold > 0
        assert (self.num_rows, self.num_rows) == self.plot_grid.shape
        self.done = False
        self.last_pos_was_fire = False
        ob = self._get_feature_values()
        return ob

    def render(self, mode='human'):
        self._print_plot_grid()

    def _print_plot_grid(self):
        # self.plot_grid[self.agent_position[0], self.agent_position[1]] = "A"
        print(np.flipud(self.plot_grid))

    def _print_true_grid(self):
        # self.plot_grid[self.agent_position[0], self.agent_position[1]] = "A"
        print(np.flipud(self.true_grid))

    def close(self):
        pass

