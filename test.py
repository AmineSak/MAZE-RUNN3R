
import gymnasium as gym
import gymnasium_robotics
from gym_robotics_custom import MazeObservationWrapper
from agent import Agent
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

gym.register_envs(gymnasium_robotics)

MEDIUM_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]]
env = gym.make('PointMaze_UMaze-v3',maze_map=MEDIUM_MAZE,max_episode_steps=1000, render_mode="rgb_array")
env = MazeObservationWrapper(env)

env = RecordVideo(env, video_folder="recordings")
env = RecordEpisodeStatistics(env)

# Initialize agent and environment
agent = Agent(observation_space_size=6,
                action_space_size=2,
                load_existing=True)
observation, _ = env.reset()
done = False
score = 0
n_steps=0
while not done:
    action, _, _ = agent.choose_action(observation)
    observation_, reward, _, truncated, info = env.step(action)
    done = info["success"] or truncated
    score += reward
    n_steps += 1

env.close()

print(f"Steps {n_steps}", f"score {score}")
        


