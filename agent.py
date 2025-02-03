import torch
import torch.optim as optim
from ppo_model import ActorNN, CriticNN, PPOMemory
import numpy as np
import os 


class Agent:
    def __init__(self,
                 observation_space_size, 
                 action_space_size,
                 h_size=128,
                 lr=0.0003,
                 gamma=0.99,
                 gae_lambda=0.95,
                 batch_size=64,
                 n_epochs = 10,
                 policy_clip=0.2,
                 value_loss_coeff = 0.5,
                 entropy_loss_coeff = 0.01,
                 load_existing=False,
                 n_games=1000):
        self.obs_size = observation_space_size
        self.act_size = action_space_size
        self.actor_nn = ActorNN(self.obs_size,self.act_size,h_size=h_size)
        self.critic_nn = CriticNN(self.obs_size,h_size=h_size)
        
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff

        self.actor_optimizer = optim.Adam(self.actor_nn.parameters(),lr=self.lr)
        self.actor_scheduler = optim.lr_scheduler.LinearLR(self.actor_optimizer,start_factor=1.,end_factor=0.,total_iters=n_games)
        self.critic_optimizer = optim.Adam(self.critic_nn.parameters(),lr=self.lr)
        self.critic_scheduler = optim.lr_scheduler.LinearLR(self.critic_optimizer,start_factor=1.,end_factor=0.,total_iters=n_games)
        
        self.memory = PPOMemory(batch_size)
        
        self.policy_loss = float('inf')
        self.value_loss = float('inf')
        self.entropy_loss = float('inf')
        self.total_loss = float('inf')
        
        # Load existing model if specified
        if load_existing and os.path.exists(self.actor_nn.checkpoint_file) and os.path.exists(self.critic_nn.checkpoint_file):
            self.load_model()
    
    def memorize(self,obs, act, log_prob, rew, done, val):
        self.memory.store_memory(obs, act, log_prob, rew, done, val)
        
    def save_model(self):
        print('... saving model ...')
        self.actor_nn.save_checkpoint()
        self.critic_nn.save_checkpoint()

    def load_model(self):
        print('... loading model ...')
        self.actor_nn.load_checkpoint()
        self.critic_nn.load_checkpoint()
    
    def choose_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.actor_nn.device)
        mean, std= self.actor_nn(obs)
        value = self.critic_nn(obs)
        
        # Create a normal distribution with the output mean and standard deviation
        normal_dist = torch.distributions.Normal(mean, std)

        # Sample an action from the distribution
        action = normal_dist.sample()  # Stochastic action
        
        log_prob = torch.squeeze(normal_dist.log_prob(action)).sum(dim=-1)  # Log probability of the sampled action
        
        # Move tensors to CPU and detach
        action = action.cpu().detach()
        log_prob = log_prob.cpu().detach()
        value = value.cpu().detach()

        return action, log_prob, value
    
    def train(self):
        for _ in range(self.n_epochs):
            obss,rews,vals,acts,dones,old_log_probs,batches = self.memory.generate_batches()
            values = vals
            next_vals = values[1:] + [0]
            deltas = [r + self.gamma* next_val - val for r,next_val,val in zip(rews,values,next_vals)]
            gaes = [deltas[-1]]
            for t in reversed(range(len(deltas) - 1)):
                gaes.append(deltas[t] + self.gamma*self.gae_lambda*gaes[-1])
            gaes = np.array(gaes[::-1], dtype=np.float32)
            gaes = torch.from_numpy(gaes).to(self.actor_nn.device)
            
            # Convert values list to numpy array first
            vals = np.array(values, dtype=np.float32)
            vals = torch.from_numpy(vals).to(self.actor_nn.device)
            for batch in batches:
                obss_ = torch.tensor(obss[batch], dtype=torch.float32).to(self.actor_nn.device)
                old_log_probs_ = torch.tensor(old_log_probs[batch]).to(self.actor_nn.device)
                acts_ = torch.tensor(acts[batch]).to(self.actor_nn.device)
                
                mean, std = self.actor_nn(obss_)
                new_vals = self.critic_nn(obss_)
                
                new_vals = torch.squeeze(new_vals)
                dist = torch.distributions.Normal(mean,std)
                
                new_log_probs = dist.log_prob(acts_).sum(dim=-1)
                probs_ratio = torch.exp(new_log_probs - old_log_probs_)
                
                weighted_log_probs = gaes[batch] * probs_ratio
                clamped_log_probs = torch.clamp(probs_ratio, 1 - self.policy_clip,\
                    1 + self.policy_clip) * gaes[batch]
                
                policy_loss = - torch.min(weighted_log_probs,clamped_log_probs).mean()
                self.policy_loss = policy_loss
                
                returns = gaes[batch] + vals[batch]
                value_loss = (returns - new_vals)**2
                value_loss = value_loss.mean()
                self.value_loss = value_loss
                
                entropy_loss = dist.entropy().sum(dim=-1)
                entropy_loss = entropy_loss.mean()
                self.entropy_loss = entropy_loss
                
                total_loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_loss_coeff * entropy_loss
                self.total_loss = total_loss
                
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step() 
        self.memory.clear_memory()
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        return self.total_loss,\
                self.policy_loss,\
                self.value_loss,\
                self.entropy_loss
        
                
                
                
                
                
            
    

