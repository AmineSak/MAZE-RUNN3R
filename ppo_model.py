import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import os


class ActorNN(nn.Module):
    def __init__(self,observation_space_size, action_space_size,h_size=264,chkpt_dir="checkpoints/ppo"):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir,"actor")
        
        self.fc1 = nn.Linear(observation_space_size,h_size)
        self.fc2 = nn.Linear(h_size,h_size)
        self.fc3 = nn.Linear(h_size,h_size)
        self.fc_mean = nn.Linear(h_size,action_space_size)
        
        self.log_std = nn.Parameter(torch.zeros(action_space_size))
        
        self._initialize_weights()
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self,observation):
        x = F.tanh(self.fc1(observation))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        mean = self.fc_mean(x)
        std = torch.exp(self.log_std)
        return mean,std
    
    def _initialize_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc_mean]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
    

class CriticNN(nn.Module):
    def __init__(self,observation_space_size,h_size=264,chkpt_dir="checkpoints/ppo"):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir,"critic")
        
        self.fc1 = nn.Linear(observation_space_size,h_size)
        self.fc2 = nn.Linear(h_size,h_size)
        self.fc3 = nn.Linear(h_size,h_size)
        self.fc_value = nn.Linear(h_size,1)
        
        self._initialize_weights()
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self,observation):
        x = F.tanh(self.fc1(observation))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        value = self.fc_value(x)
        return value
    
    def _initialize_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc_value]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            
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
