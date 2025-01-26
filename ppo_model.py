import torch
import torch.nn as nn
import numpy as np
import os


class ActorCriticNetwork(nn.Module):
    def __init__(self, observation_space_size, action_space_size,chkpt_dir='tmp/ppo'):
        """Initialize the PPO neural network model with shared layers for policy and value functions.
        
        Args:
            observation_space_size (int): Dimension of the observation/state space
            action_space_size (int): Dimension of the action space
            
        The network architecture consists of:
        - Shared layers: Two fully connected layers (obs_size → 32 → 32)
        - Policy head: Two fully connected layers with tanh activation (32 → 32 → action_size)
        - Value head: Two fully connected layers (32 → 32 → 1)
        """
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir,"actor_critic_nn")
        
        self.shared_layers = nn.Sequential(
            nn.Linear(observation_space_size,264),
            nn.ReLU(),
            nn.Linear(264,264),
            nn.ReLU()
        )
        
        # Policy layers for mean and standard deviation
        self.policy_layers = nn.Sequential(
            nn.Linear(264, 264),
            nn.ReLU(),
            nn.Linear(264, action_space_size),  # Mean for each action dimension
        )
        
        self.std_layers = nn.Sequential(
            nn.Linear(264, 264),
            nn.ReLU(),
            nn.Linear(264, action_space_size),  # Std dev for each action dimension
            nn.Softplus()  # Ensure positive standard deviation
        )
        
        # Value function layers
        self.value_layers = nn.Sequential(
            nn.Linear(264, 264),
            nn.ReLU(),
            nn.Linear(264, 1)  # Output the value estimate
        )
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.to(self.device)

    def value(self, observation):
        z = self.shared_layers(observation)
        value = self.value_layers(z) 
        return value
    
    def policy(self, observation):
        z = self.shared_layers(observation)
        mean = self.policy_layers(z)  # Mean output
        std = self.std_layers(z)      # Std dev output
        return mean, std  # Return mean and std for the normal distribution
    
    def forward(self,observation):
        mean, std = self.policy(observation)
        value = self.value(observation)
        return mean, std, value
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
    
class PPOMemory:
    """A memory buffer for storing and processing PPO training data.
    
    This class implements a memory buffer that stores observations, actions, rewards,
    and other necessary data for training a PPO (Proximal Policy Optimization) agent.
    
    Args:
        batch_size (int): The size of batches to use when generating training data
    """
    def __init__(self, batch_size):
        self.observations = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.actions = []
        self.log_probs = []
        
        self.batch_size = batch_size
    
    def generate_batches(self):
        """Generate randomized batches of stored memories for training.
        
        Returns:
            tuple: Contains:
                - observations: List of stored observations
                - rewards: List of stored rewards
                - values: List of stored value estimates
                - actions: List of stored actions
                - dones: List of stored terminal flags
                - log_probs: List of stored action log probabilities
                - batches: List of indices for each batch
        """
        n_obs = len(self.observations)
        batch_starts = np.arange(0, n_obs, self.batch_size)
        indices = np.arange(n_obs, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_starts]
        
        # Convert tensors to CPU before converting to numpy arrays
        values = [v.cpu().detach().numpy() if torch.is_tensor(v) else v for v in self.values]
        log_probs = [lp.cpu().detach().numpy() if torch.is_tensor(lp) else lp for lp in self.log_probs]
        
        return  np.array(self.observations),\
                np.array(self.rewards),\
                np.array(values),\
                np.array(self.actions),\
                np.array(self.dones),\
                np.array(log_probs),\
                batches
    
    def store_memory(self, obs, act, log_prob, rew, done, val):
        """Store a single step of interaction data in memory.
        
        Args:
            obs: The observation/state
            act: The action taken
            log_prob: Log probability of the action
            rew: The reward received
            done: Boolean indicating if episode terminated
            val: The value estimate
        """
        self.observations.append(obs)
        self.rewards.append(rew)
        self.actions.append(act)
        self.dones.append(done)
        self.values.append(val)
        self.log_probs.append(log_prob)
    
    def clear_memory(self):
        """Clear all stored memories, resetting the buffer."""
        self.observations = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.actions = []
        self.log_probs = []