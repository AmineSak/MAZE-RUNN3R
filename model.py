import torch.nn as nn

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.LL1 = nn.Linear()