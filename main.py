import gymnasium as gym
import gymnasium_robotics
from gym_robotics_custom import MazeObservationWrapper, MazeRewardWrapper
from agent import Agent
import numpy as np
import time
from utils import plot_results

gym.register_envs(gymnasium_robotics)


maze = [[1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]]
MAX_STEPS = 1000
env = gym.make('PointMaze_UMaze-v3',maze_map = maze,max_episode_steps=MAX_STEPS)

wrapped_env = MazeObservationWrapper(env)
wrapped_env = MazeRewardWrapper(wrapped_env)

# Agent Initialization
horizon = 30
n_epochs = 10
lr = 0.0001
agent = Agent(observation_space_size=6, action_space_size=2,lr=lr,n_epochs=n_epochs,load_existing=False)

n_games = 300
best_score = -float("inf")
score_history = []
avg_score = 0
n_steps = 0

tic = time.time()
for i in range(n_games):
    # Reset the environment to generate the first observation
    observation, _ = wrapped_env.reset(options={"goal_cell": (3,1), "reset_cell": (1,1)})
    done = False
    score = 0
    while not done:
        action, prob, val = agent.choose_action(observation)

        observation_, reward, _, truncated, info = wrapped_env.step(action)
    
        done =  info["success"] or truncated
        score += reward
        n_steps += 1
        agent.memorize(observation, action, prob, reward, done, val)
        
        if n_steps % horizon == 0:
            agent.train()
        observation = observation_
    score_history.append(score + MAX_STEPS)
    avg_score = np.mean(score_history[-5:])
    
    if avg_score > best_score:
        best_score = avg_score
        agent.save_model()
    
    print(f"episode {i}", f"score {score + MAX_STEPS}" , "avg score %.1f" %avg_score)
toc = time.time()
print("Time gpu", toc-tic)
    

# Call the visualization function after training with hyperparameters
hyperparams = {
    'horizon': horizon,
    'n_epochs': agent.n_epochs,
    'learning_rate': agent.lr,
    'n_games': n_games,
    'max_steps': MAX_STEPS,
    'c1': agent.value_loss_coeff,
    'c2': agent.entropy_loss_coeff
}
plot_results(score_history, hyperparams)

