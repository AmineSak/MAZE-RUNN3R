import torch.nn as nn
import numpy as np

class ActorCriticNetwork(nn.Module):
    def __init__(self, observation_space_size, action_space_size):
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
        
        self.shared_layers = nn.Sequential(
            nn.Linear(observation_space_size,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU()
        )
        
        # Policy layers for mean and standard deviation
        self.policy_layers = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_space_size),  # Mean for each action dimension
        )
        
        self.std_layers = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_space_size),  # Std dev for each action dimension
            nn.Softplus()  # Ensure positive standard deviation
        )
        
        # Value function layers
        self.value_layers = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output the value estimate
        )

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
    
    def generate_bataches(self):
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
        indices = np.arange(n_obs)
        np.random.shuffle(indices)
        batches = np.array([indices[i:i+self.batch_size] for i in batch_starts])
        
        return  self.observations,\
                self.rewards,\
                self.values,\
                self.actions,\
                self.dones,\
                self.log_probs,batches
    
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
        
        