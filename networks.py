import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np


def identity(x):
    """Return input without any change."""
    return x


"""
DQN, DDQN, A2C critic, VPG critic, TRPO critic, PPO critic, DDPG actor, TD3 actor
"""
class MLP(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 output_limit=1.0,
                 hidden_sizes=(64,64), 
                 activation=F.relu, 
                 output_activation=identity,
                 use_output_layer=True,
                 use_actor=False,):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.output_limit = output_limit
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.use_output_layer = use_output_layer
        self.use_actor = use_actor

        # Set hidden layers
        self.hidden_layers = nn.ModuleList()
        in_size = self.input_size
        for next_size in self.hidden_sizes:
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc)

        # Set output layers
        if self.use_output_layer:
            self.output_layer = nn.Linear(in_size, self.output_size)
        else:
            self.output_layer = identity

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_activation(self.output_layer(x))
        # If the network is used as actor network, make sure output is in correct range
        x = x * self.output_limit if self.use_actor else x   
        return x



"""
VPG actor, TRPO actor, PPO actor
"""
class GaussianPolicy(MLP):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 output_limit=1.0,
                 hidden_sizes=(64,64),
                 activation=torch.tanh,
    ):
        super(GaussianPolicy, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

        self.output_limit = output_limit
        self.log_std = np.ones(output_size, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.Tensor(self.log_std))

    def forward(self, x, pi=None, use_pi=True):
        mu = super(GaussianPolicy, self).forward(x)
        std = torch.exp(self.log_std)

        dist = torch.distributions.Normal(mu, std)
        if use_pi:
            pi = dist.sample()
        log_pi = dist.log_prob(pi).sum(dim=-1)

        # Make sure outputs are in correct range
        mu = mu * self.output_limit
        pi = pi * self.output_limit
        return mu, std, pi, log_pi