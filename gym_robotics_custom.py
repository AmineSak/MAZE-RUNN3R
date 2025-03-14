import gymnasium as gym
import numpy as np
from gymnasium import Env, ObservationWrapper, RewardWrapper


class MazeObservationWrapper(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(6,), dtype="float64"
        )

    def observation(self, observation):
        green_ball_observation = observation["observation"]
        goal_observation = observation["desired_goal"]

        transformed_observation = np.concatenate(
            [green_ball_observation, goal_observation]
        )

        return transformed_observation


class MazeRewardWrapper(RewardWrapper):
    def __init__(self, env: Env):
        super().__init__(env)

    def reward(self, base_reward):
        return -1 if base_reward == 0 else 1


def MazeObservationTF(observation):
    assert (observation["observation"], True)
    green_ball_observation = observation["observation"]

    assert (observation["desired_goal"], True)
    goal_observation = observation["desired_goal"]

    transformed_observation = np.concatenate([green_ball_observation, goal_observation])

    return transformed_observation
