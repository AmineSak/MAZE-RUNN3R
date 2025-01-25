import gymnasium as gym
import gymnasium_robotics
from gym_robotics_custom import MazeObservationWrapper, MazeRewardWrapper

gym.register_envs(gymnasium_robotics)

maze = [[1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]]

env = gym.make('PointMaze_Large_Diverse_GR-v3',render_mode = "RGB", maze_map = maze)

wrapped_env = MazeObservationWrapper(env)
wrapped_env = MazeRewardWrapper(wrapped_env)
# Reset the environment to generate the first observation
observation, info = wrapped_env.reset(seed=42)

episode_over = False
episode_len = 0
while not episode_over:
    episode_len += 1 
    action = wrapped_env.action_space.sample()
    observation, reward, terminated, truncated, info  = wrapped_env.step(action)
    print("reward", reward)
    episode_over = terminated or truncated

env.close()
print(episode_len)