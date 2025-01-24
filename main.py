import gymnasium as gym
import gymnasium_robotics
from gym_robotics_custom import MazeObservationWrapper

gym.register_envs(gymnasium_robotics)

example_map = [[1, 1, 1, 1, 1],
               [1, 0, 0, 0, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 0, 0, 1],
               [1, 1, 1, 1, 1]]

env = gym.make('PointMaze_UMaze-v3',render_mode = "human",maze_map=example_map)

wrapped_env = MazeObservationWrapper(env)
# Reset the environment to generate the first observation
observation, info = wrapped_env.reset(seed=42)

for _ in range(1000):
    action = wrapped_env.action_space.sample()
    wrapped_env.step(action)

print(observation)