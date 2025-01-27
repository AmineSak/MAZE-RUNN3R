from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import gymnasium_robotics
from gym_robotics_custom import MazeObservationWrapper
from agent import Agent
import numpy as np
from datetime import datetime
import os

gym.register_envs(gymnasium_robotics)

def train_and_log(horizon, n_epochs, lr, n_games, max_steps,c1,c2,dense):
    maze = [[1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],]
    dense_reward = 'PointMaze_UMazeDense-v3'
    if dense:
        env = gym.make(dense_reward,maze_map = maze,max_episode_steps=max_steps)
    else:
        env = gym.make('PointMaze_UMaze-v3',maze_map = maze,max_episode_steps=max_steps)
    wrapped_env = MazeObservationWrapper(env)
    
    # Generate a unique directory for this run
    run_name = f"run_De{dense}_h{horizon}_ep{n_epochs}_lr{lr}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(log_dir)

    # Initialize agent and environment
    agent = Agent(observation_space_size=6, action_space_size=2,
                  lr=lr,
                  n_epochs=n_epochs,
                  load_existing=True,
                  value_loss_coeff=c1,
                  entropy_loss_coeff=c2)
    score_history = []
    best_score = -float("inf")
    avg_score = 0
    n_steps = 0
    window_size = 50

    for i in range(n_games):
        observation, _ = wrapped_env.reset(options={"goal_cell": (1, 3), "reset_cell": (1, 1)})
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, _, truncated, info = wrapped_env.step(action)

            done = info["success"] or truncated
            score += reward
            n_steps += 1
            agent.memorize(observation, action, prob, reward, done, val)

            if n_steps % horizon == 0:
                total_loss, policy_loss, value_loss, entropy_loss = agent.train()
                writer.add_scalar("Loss/Total Loss", total_loss, n_steps)
                writer.add_scalar("Loss/Actor Loss", policy_loss, n_steps)
                writer.add_scalar("Loss/Critic Loss", value_loss, n_steps)
                writer.add_scalar("Loss/Entropy Loss", entropy_loss, n_steps)
                
            observation = observation_

        score_history.append(score + max_steps)
        avg_score = np.mean(score_history[-window_size:])

        # Log metrics for each episode
        writer.add_scalar("Performance/Score", score + max_steps, i)
        writer.add_scalar("Performance/Average Score", avg_score, i)
        writer.add_scalar("Performance/Steps", n_steps, i)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_model()

        print(f"episode {i}", f"score {score + max_steps}", "avg score %.1f" % avg_score)

    # Log hyperparameters
    writer.add_hparams(
        {
            'horizon': horizon,
            'n_epochs': n_epochs,
            'learning_rate': lr,
            'n_games': n_games,
            'max_steps': max_steps,
            'c1': c1,
            'c2': c2,
            'dense_reward': dense
        },
        {
            'best_avg_score': best_score
        }
    )

    writer.close()

train_and_log(c1=0.5,c2=0.01,horizon=64,n_epochs=10,lr=0.0003,n_games=1000,max_steps=200, dense=True)
