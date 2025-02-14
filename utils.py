import torch.nn as nn

def layer_init(layer, gain=5/3, bias=0):
    nn.init.orthogonal_(layer.weight,gain)
    nn.init.constant_(layer.bias, bias)
    return layer