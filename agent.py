import torch
import torch.optim as optim
from ppo_model import ActorCriticNetwork, PPOMemory
import numpy as np


class Agent:
    def __init__(self,
                 observation_space_size, 
                 action_space_size,
                 actor_critic_model,
                 alpha=0.0003,
                 lr=0.003,
                 gamma=0.99,
                 gae_lambda=0.95,
                 batch_size=64,
                 n_epochs = 10,
                 horizon=2048, # Number of timesteps between each policy update
                 policy_clip=0.2
                 ):
        self.obs_size = observation_space_size
        self.act_size = action_space_size
        self.actor_critic_model = actor_critic_model(self.obs_size,self.act_size)
        self.alpha = alpha
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.policy_clip = policy_clip
        self.horizon = horizon
        self.n_epochs = n_epochs
        
        self.policy_params = list(self.actor_critic_model.shared_layers) +\
                             list(self.actor_critic_model.policy_layers)
        self.value_params = list(self.actor_critic_model.shared_layers) +\
                            list(self.actor_critic_model.value_layers)
        self.policy_optimizer = optim.Adam( self.policy_params, lr=self.lr)
        self.value_optimizer = optim.Adam( self.value_params, lr=self.lr)
        
        self.memory = PPOMemory(self.batch_size)
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    def memorize(self,obs, act, log_prob, rew, done, val):
        self.memory.store_memory(obs, act, log_prob, rew, done, val)
    
    def choose_action(self, obs):
        obs = torch.tensor([obs], dtype=torch.float32).to(self.device)
        mean, std, value = self.actor_critic_model(obs)
        
        # Create a normal distribution with the output mean and standard deviation
        normal_dist = torch.distributions.Normal(mean, std)

        # Sample an action from the distribution
        action = normal_dist.sample()  # Stochastic action
        
        log_prob = normal_dist.log_prob(action)  # Log probability of the sampled action
        
        return action, log_prob, value
    
    def train(self):
        for _ in range(self.n_epochs):
            obss,rews,vals,acts,dones,old_log_probs,batches = self.memory.generate_bataches()
            
            advs = np.zeros(len(rews), dtype=np.float32)
            for t in reversed(range(len(rews))):
                next_val = 0 if dones[t] else vals[t+1] if t+1 < len(rews) else 0
                delta = rews[t] + self.gamma * next_val - vals[t] 
                gae = delta 
                
            
            
    

