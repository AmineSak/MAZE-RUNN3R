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
                 policy_clip=0.2,
                 value_loss_coeff = 0.5,
                 entropy_loss_coeff = 0.5,
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
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        
        self.policy_params = list(self.actor_critic_model.shared_layers) +\
                             list(self.actor_critic_model.policy_layers)
        self.value_params = list(self.actor_critic_model.shared_layers) +\
                            list(self.actor_critic_model.value_layers)

        self.optimizer = optim.Adam(self.actor_critic_model.parameters(),lr = self.lr)
        
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
        
        log_prob = torch.squeeze(normal_dist.log_prob(action)).item()  # Log probability of the sampled action
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        
        return action, log_prob, value
    
    def train(self):
        for _ in range(self.n_epochs):
            obss,rews,vals,acts,dones,old_log_probs,batches = self.memory.generate_bataches()
            
            next_vals = np.concatenate([vals[1:],[0]])
            deltas = [r + self.gamma* next_val - val for r,next_val,val in zip(rews,vals,next_vals)]
            gaes = [deltas[-1]]
            for t in reversed(range(len(deltas) - 1)):
                gaes.append(deltas[t] + self.gamma*self.gae_lambda*gaes[-1])
            gaes = torch.tensor(np.array(gaes[::-1])).to(self.device)
            
            vals = torch.tensor(vals).to(self.device)
            for batch in batches:
                obss_ = torch.tensor(obss[batch], dtype=torch.float32).to(self.device)
                old_log_probs_ = torch.tensor(old_log_probs[batch], dtype=torch.float32).to(self.device)
                acts_ = torch.tensor(acts[batch], dtype=torch.float32).to(self.device)
                
                mean, std, new_vals = self.actor_critic_model(obss_)
                
                new_vals = torch.squeeze(new_vals)
                dist = torch.distributions.Normal(mean,std)
                
                new_log_probs = dist.log_prob(acts_).sum(dim=-1)
                probs_ratio = torch.exp(new_log_probs - old_log_probs_)
                
                weighted_log_probs = gaes[batch] * probs_ratio
                clamped_log_probs = torch.clamp(probs_ratio, 1 - self.policy_clip,\
                    1 + self.policy_clip) * gaes[batch]
                
                policy_loss = - torch.min(weighted_log_probs,clamped_log_probs).mean()
                
                returns = gaes[batch] + vals
                value_loss = (returns - new_vals)**2
                value_loss = value_loss.mean()
                
                entropy_loss = dist.entropy().sum(dim=-1)
                entropy_loss = entropy_loss.mean()
                
                total_loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_loss_coeff * entropy_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        self.memory.clear_memory()
                
                
                
                
                
            
    

