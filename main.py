import torch
import gymnasium as gym
import gymnasium_robotics
from gym_robotics_custom import MazeObservationWrapper, MazeRewardWrapper
from agent import Agent
import numpy as np
import matplotlib.pyplot as plt
import time

gym.register_envs(gymnasium_robotics)


maze = [[1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]]
env = gym.make('PointMaze_Large_Diverse_GR-v3',render_mode='human', maze_map = maze)

wrapped_env = MazeObservationWrapper(env)
wrapped_env = MazeRewardWrapper(wrapped_env)



# Agent Initialization
horizon = 20
batch_size = 10
n_epochs = 10
lr = 0.0003
agent = Agent(observation_space_size=6, action_space_size=2,lr=lr,n_epochs=n_epochs,load_existing=True)

n_games = 300
train_iters = 0
best_score = -float("inf")
score_history = []
avg_score = 0
n_steps = 0

tic = time.time()
for i in range(n_games):
    # Reset the environment to generate the first observation
    observation, _ = wrapped_env.reset()
    done = False
    score = 0
    while not done:
        action, prob, val = agent.choose_action(observation)

        observation_, reward, _, _, info = wrapped_env.step(action)
    
        done =  info["success"]
        score += reward
        n_steps += 1
        agent.memorize(observation, action, prob, reward, done, val)
        
        if n_steps % horizon == 0:
            agent.train()
            train_iters += 1
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-5:])
    
    if avg_score > best_score:
        best_score = avg_score
        agent.save_model()
    
    print(f"episode {i}", "score %.1f" %score, "avg score %.1f" %avg_score)
toc = time.time()
print("Time gpu", toc-tic)
    
def plot_results(score_history):
    plt.figure(figsize=(12, 6))
    plt.plot(score_history, label='Score per Episode', color='blue')
    plt.title('PPO Training Performance')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.axhline(y=np.mean(score_history[-5:]), color='red', linestyle='--', label='Average Score (last 5 episodes)')
    plt.legend()
    plt.grid()
    plt.show()

# Call the visualization function after training
plot_results(score_history)

