from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import gymnasium_robotics
from gym_robotics_custom import MazeObservationWrapper
from agent import Agent
import numpy as np
from datetime import datetime
import os
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from tqdm import tqdm

gym.register_envs(gymnasium_robotics)

def train_and_log(horizon, n_epochs, lr, n_games, max_steps,c1,c2,h_size):
    # Generate a unique directory for this run
    run_name = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M')}_DeTrue_hs{h_size}_lr{lr}"
    log_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(log_dir)
    
    
    # Env setup
    maze = [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]]
    env = gym.make('PointMaze_UMazeDense-v3',maze_map=maze,max_episode_steps=max_steps)
    wrapped_env = MazeObservationWrapper(env)
    
    # training_period = n_games // 10 

    # wrapped_env = RecordVideo(wrapped_env, video_folder=log_dir,
    #               episode_trigger=lambda x: x % training_period == 0)
    # wrapped_env = RecordEpisodeStatistics(wrapped_env)
    
    # Initialize agent
    agent = Agent(observation_space_size=6,
                  action_space_size=2,
                  h_size=h_size,
                  lr=lr,
                  n_epochs=n_epochs,
                  load_existing=True,
                  value_loss_coeff=c1,
                  entropy_loss_coeff=c2,
                  n_games = n_games,
                  policy_clip=0.1)
    score_history = []
    best_score = -float("inf")
    avg_score = 0
    n_steps = 0
    window_size = 100

    for i in tqdm(range(n_games)):
        agent.policy_clip *= (1- i//n_games)
        observation, _ = wrapped_env.reset(options={"goal_cell": (6,6), "reset_cell": (1,1)})
        done = False
        score = 0
        ep_steps = 0
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
            ep_steps += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-window_size:])

        # Log metrics for each episode
        writer.add_scalar("Performance/Score", score, i)
        writer.add_scalar("Performance/Average Score", avg_score, i)
        writer.add_scalar("Performance/Steps", n_steps, i)

        if avg_score > best_score:
            best_score = avg_score
            
        if i % 100 == 0:
            agent.save_model()

        print(f"episode {i}", "score %.1f" % score, "avg score %.1f" % avg_score,"episode steps",ep_steps)
    

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
            'hidden_size': h_size,
            "policy_clip": agent.policy_clip
        },
        {
            "best_avg_score": best_score
        }
    )

    writer.close()

# Learning rate tuning
train_and_log(c1=0.5,c2=0.01,horizon=2048,n_epochs=10,lr=0.00001,n_games=10000,max_steps=1500,h_size=128)


