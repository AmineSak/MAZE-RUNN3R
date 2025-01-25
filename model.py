import torch.nn as nn

class ActorCriticNetwork(nn.Module):
    def __init__(self, observation_space_size, action_space_size):
        super().__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(observation_space_size,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU()
        )
        
        self.policy_layers = nn.Sequential(
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,action_space_size)
        )
        
        self.value_layers = nn.Sequential(
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def value(self, observation):
        z = self.shared_layers(observation)
        value = self.value_layers(z) 
        return value
    
    def policy(self, observation):
        z = self.shared_layers(observation)
        policy = self.policy_layers(z) 
        return policy
    
    def forward(self,observation):
        policy = self.policy(observation)
        value = self.value(observation)
        return policy, value