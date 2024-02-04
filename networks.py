import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# https://github.com/dongminlee94/deep_rl/tree/main

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
                 use_actor=False,
                 initialization='uniform'):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.output_limit = output_limit
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.use_output_layer = use_output_layer
        self.use_actor = use_actor

        if initialization == 'uniform':
            self.initialization = torch.nn.init.uniform_
        elif initialization == 'normal':
            self.initialization = torch.nn.init.normal_

        # Set hidden layers
        self.hidden_layers = nn.ModuleList()
        in_size = self.input_size
        for next_size in self.hidden_sizes:
            fc = nn.Linear(in_size, next_size)

            if initialization == 'dirichlet':
                alpha = torch.tensor([0.5 for x in range (fc.weight.shape[1])])  # concentration parameters
                dirichlet = torch.distributions.Dirichlet(alpha)
                samples = dirichlet.sample([fc.weight.shape[0],])
                fc.weight.data = samples
            else:
                self.initialization(fc.weight)

            in_size = next_size
            self.hidden_layers.append(fc)

        # Set output layers
        if self.use_output_layer:
            self.output_layer = nn.Linear(in_size, self.output_size)

            if initialization == 'dirichlet':
                alpha = torch.tensor([0.5 for x in range (self.output_layer.weight.shape[1])])  # concentration parameters
                dirichlet = torch.distributions.Dirichlet(alpha)
                samples = dirichlet.sample([self.output_layer.weight.shape[0],])
                self.output_layer.weight.data = samples
            else:
                self.initialization(self.output_layer.weight)
        else:
            self.output_layer = identity


    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_activation(self.output_layer(x))
        # If the network is used as actor network, make sure output is in correct range
        x = x * self.output_limit if self.use_actor else x   
        return x

class CategoricalPolicy(MLP):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes=(64,),
                 activation=torch.relu,
                 initialization='uniform',
                 temperature=0.1
    ):
        super(CategoricalPolicy, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            initialization=initialization)
        self.temperature = temperature

    def forward(self, x, pi=None, use_pi=True):
        x = super(CategoricalPolicy, self).forward(x)
        pi = F.softmax(x/self.temperature, dim=-1)

        dist = torch.distributions.categorical.Categorical(pi)
        action = dist.sample()

        log_pi = dist.log_prob(action)
        return action, None, pi, log_pi


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


class UniformPolicy(MLP):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 output_limit=1.0,
                 hidden_sizes=(64,64),
                 activation=torch.tanh,
    ):
        super(UniformPolicy, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

        self.output_limit = output_limit

    def forward(self, x, pi=None, use_pi=True):
        high = super(UniformPolicy, self).forward(x)
        
        # ensure high >= low
        high = torch.max(torch.tensor([0.001]), high)

        low = torch.zeros(self.output_size)

        dist = torch.distributions.uniform.Uniform(low, high)

        if use_pi:
            pi = dist.sample()
        log_pi = dist.log_prob(pi).sum(dim=-1)

        # Make sure outputs are in correct range
        high = high * self.output_limit

        # maintain consistent format as GuassianPolicy
        return high, _, pi, log_pi


class BetaPolicy(MLP):
    def __init__(self, 
                 input_size, 
                 output_size,                 
                 output_limit=1.0,
                 hidden_sizes=(64,64),
                 activation=torch.tanh,
    ):
        super(BetaPolicy, self).__init__(
            input_size=input_size,
            output_size=output_size*2,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

        self.output_limit = output_limit

    def forward(self, x, pi=None, use_pi=True):
        mlp_preds = super(BetaPolicy, self).forward(x)

        # first (output_size) params predict concentration 1 of beta
        # second (output size) params predict concentration 2 of beta
        concentration1 = mlp_preds[0:output_size]
        concentration0 = mlp_preds[output_size:]

        dist = torch.distributions.beta.Beta(concentration1, concentration0)

        if use_pi:
            pi = dist.sample()
        log_pi = dist.log_prob(pi).sum(dim=-1)

        # Make sure outputs are in correct range
        concentration1 = concentration1 * self.output_limit
        concentration0 = concentration0 * self.output_limit

        # maintain consistent format as GuassianPolicy
        return concentration1, concentration0, pi, log_pi


class DirichletPolicy(MLP):
    def __init__(self, 
                 input_size, 
                 output_size,                 
                 output_limit=1.0,
                 hidden_sizes=(64,64),
                 activation=torch.tanh,
    ):
        super(DirichletPolicy, self).__init__(
            input_size=input_size,
            output_size=output_size*2,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

        self.output_limit = output_limit

    def forward(self, x, pi=None, use_pi=True):
        concentration = super(DirichletPolicy, self).forward(x)

        dist = torch.distributions.dirichlet.Dirichlet(concentration)

        if use_pi:
            pi = dist.sample()
        log_pi = dist.log_prob(pi).sum(dim=-1)

        # Make sure outputs are in correct range
        concentration = concentration * self.output_limit

        # maintain consistent format as GuassianPolicy
        return concentration, _, pi, log_pi
