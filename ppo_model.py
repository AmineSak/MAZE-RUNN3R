import torch.nn as nn
import numpy as np

class ActorCriticNetwork(nn.Module):
    def __init__(self, observation_space_size, action_space_size):
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
        policy = self.policy(observation)
        value = self.value(observation)
        return policy, value
    
class PPOMemory:
    def __init__(self, batch_size):
        self.observations = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.actions = []
        self.log_probs = []
        
        self.batch_size = batch_size
    
    def _get_dict(self):
        memory = {
            "observations": self.observations,
            "rewards": self.rewards,
            "values": self.values,
            "dones": self.dones,
            "log_probs": self.log_probs
        }
        return memory
    
    def generate_bataches(self):
        n_obs = len(self.observations)
        batch_starts = np.arange(0, n_obs, self.batch_size)
        indices = np.arange(n_obs)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_starts]
        
        return  self.observations,\
                self.rewards,\
                self.values,\
                self.actions,\
                self.dones,\
                self.log_probs,batches
    
    def store_memory(self,obs, act, log_prob, rew, done, val):
        self.observations.append(obs)
        self.rewards.append(rew)
        self.actions.append(act)
        self.dones.append(done)
        self.values.append(val)
        self.log_probs.append(log_prob)
    
    def clear_memory(self):
        self.observations = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.actions = []
        self.log_probs = []
        
        