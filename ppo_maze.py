import re
import time

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch as T
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import wandb
from gym_robotics_custom import MazeObservationWrapper
from utils import *

gym.register_envs(gymnasium_robotics)


def get_envs(env_id: str, num_actors: int, max_steps_per_episode):
    def make_env():
        env = gym.make(env_id, max_episode_steps=max_steps_per_episode)
        if re.match(r"PointMaze", env_id):
            env = MazeObservationWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env

    envs = gym.vector.SyncVectorEnv([make_env for _ in range(num_actors)])
    return envs


def save_model_with_artifact(model, exp_name):
    filename = f"checkpoints/model_{exp_name}.pth"
    T.save(model.state_dict(), filename)

    # Create a W&B artifact
    artifact = wandb.Artifact("model-checkpoint", type="model")
    artifact.add_file(filename)

    # Log the artifact
    wandb.log_artifact(artifact)


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), gain=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.prod(envs.single_observation_space.shape), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(
                nn.Linear(64, np.prod(envs.single_action_space.shape)), gain=1.0
            ),
        )
        self.actor_logstd = nn.Parameter(
            T.zeros(np.prod(envs.single_action_space.shape))
        )

    def forward(self, x):
        actor_mean = self.actor_mean(x)
        actor_std = T.exp(self.actor_logstd)
        dist = Normal(actor_mean, actor_std)
        action = dist.sample()
        value = self.critic(x)
        return action, dist.log_prob(action).sum(1), value, dist.entropy()


def train(
    exp_name,
    time_steps,
    anneal_lr=True,
    env_id="PointMaze_UMaze-v3",
    max_steps_per_episode=1000,
    seed=1,
    num_actors=3,
    epochs=10,
    batch_size=64,
    clip_coeff=0.1,
    gamma=0.99,
    gae_lambda=0.95,
    lr=2.5e-4,
    entropy_coeff=0.01,
    value_coeff=0.5,
    horizon=2048,
    mini_batch_size=32,
):

    args = {
        "env_id": env_id,
        "exp_name": exp_name,
        "num_actors": int(num_actors),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "clip_coeff": float(clip_coeff),
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "lr": float(lr),
        "entropy_coeff": float(entropy_coeff),
        "value_coeff": float(value_coeff),
        "horizon": int(horizon),
        "seed": int(seed),
    }
    run_name = (
        f"{args['env_id']}__{args['exp_name']}__{args['seed']}__{int(time.time())}"
    )
    envs = get_envs(env_id, num_actors, max_steps_per_episode)
    wandb.init(
        project="PPO implementation",
        sync_tensorboard=True,
        config=args,
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in args.items()])),
    )
    device = T.device("mps" if T.backends.mps.is_available() else "cpu")

    agent = Agent(envs).to(device)
    optimizer = T.optim.Adam(agent.parameters(), lr=args["lr"])

    # Memory setup
    obs = T.zeros((horizon, num_actors) + envs.single_observation_space.shape).to(
        device
    )
    actions = T.zeros((horizon, num_actors) + envs.single_action_space.shape).to(device)
    values = T.zeros((horizon, num_actors)).to(device)
    logprobs = T.zeros((horizon, num_actors)).to(device)
    rewards = T.zeros((horizon, num_actors)).to(device)
    dones = T.zeros((horizon, num_actors)).to(device)

    next_obs = T.tensor(envs.reset()[0], dtype=T.float32).to(device)
    num_updates = time_steps // batch_size
    global_steps = 0
    start_time = time.time()

    for i in range(num_updates):
        if anneal_lr:
            frac = 1.0 - (i - 1.0) / num_updates
            lrnow = frac * lr
            optimizer.param_groups[0]["lr"] = lrnow
        for step in range(horizon):
            global_steps += num_actors
            obs[step] = next_obs
            with T.no_grad():
                action, logprob, value, _ = agent(next_obs)
                values[step] = value.flatten()

            observation, reward, _, truncated, info = envs.step(action.cpu().numpy())

            done = [t or te for t, te in zip(info["success"], truncated)]
            done, observation = T.tensor(done).to(device), T.tensor(
                observation, dtype=T.float32
            ).to(device)
            reward = T.tensor(reward, dtype=T.float32).to(device)
            actions[step] = action
            logprobs[step] = logprob
            dones[step] = done
            rewards[step] = reward
        next_obs = observation
        # Calculating advantages
        advantages = T.zeros((horizon, num_actors)).to(device)
        next_values = T.zeros_like(values)
        next_values[:-1] = values[1:]
        for t in reversed(range(horizon - 1)):
            advantages[t] = (
                rewards[t]
                + gamma * next_values[t + 1]
                - values[t]
                + gamma * gae_lambda * advantages[t + 1]
            )
        returns = advantages + values

        # Training
        f_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        f_logprobs = logprobs.reshape(-1)
        f_values = values.reshape(-1)
        f_advantages = advantages.reshape(-1)
        f_returns = returns.reshape(-1)

        b_inds = np.arange(batch_size)
        for epoch in range(epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                batch = b_inds[start:end]
                _, newlogprobs, newvalues, entropies = agent(f_obs[batch])

                prob_ratios = T.exp(newlogprobs - f_logprobs[batch])
                mini_advantages = f_advantages[batch]

                clipped_ratios = T.clamp(prob_ratios, 1 - clip_coeff, 1 + clip_coeff)
                policy_loss = T.min(
                    prob_ratios * mini_advantages, clipped_ratios * mini_advantages
                ).mean()

                value_loss = ((newvalues - f_returns[batch]) ** 2).mean()

                entropy_loss = entropies.mean()

                total_loss = (
                    -policy_loss
                    - entropy_coeff * entropy_loss
                    + value_coeff * value_loss
                )

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            save_model_with_artifact(agent, exp_name)
        y_pred, y_true = f_values.cpu().numpy(), f_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_steps
        )
        writer.add_scalar("losses/value_loss", value_loss.item(), global_steps)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_steps)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_steps)
        writer.add_scalar("losses/explained_variance", explained_var, global_steps)
        print("SPS:", int(global_steps / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_steps / (time.time() - start_time)), global_steps
        )
    envs.close()
    writer.close()


if __name__ == "__main__":
    train(
        env_id="PointMaze_MediumDense-v3",
        num_actors=8,
        time_steps=2000000,
        max_steps_per_episode=1500,
        exp_name="Maze_dense_reward",
    )
