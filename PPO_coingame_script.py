import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import unittest
import gc

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys

sys.path.append('/Users/scottmerrill/Documents/UNC/Research/coingame/evoenv')
import evoenv
from evoenv.envs.coin_game import CoinGame
from players import PPOPlayer, DQNPlayer
from population import Population
from networks import MLP, GaussianPolicy, CategoricalPolicy
from memoryBuffers import ReplayBuffer, Buffer
import coinGameExperiment

from datetime import datetime

"""DQN Test"""
# environment settings
n = 3

env = CoinGame
env_options = {'grid_shape':(n,n),
               'n_coins':1,
               'coin_payoffs':np.array([[1, 0], [1, -2]], dtype=np.float_)}

env_description = {'obs_dim':4*n*n,
                   'act_dim':4,
                   'act_limit':1}

env_dict = {'env':env,
            'env_options':env_options,
            'env_description':env_description}


# player settings
base_player_options = {'save_path': r'/Users/scottmerrill/Documents/UNC/Research/coingame/data/'}
ppo_models = [{'actor_model':CategoricalPolicy,
               'critic_model':MLP},

              {'actor_model':CategoricalPolicy,
              'critic_model':MLP}]

ppo_model_params = [#actor network
                    {'input_size':4*n**2, \
                    'output_size':4,
                    'output_limit':1.0,
                    'hidden_sizes':(2*4*n**2,4*n**2),
                    'activation':torch.relu},

                    # critic network
                    {'input_size':4*n**2,
                    'output_size':1,
                    'hidden_sizes':(2*4*n**2,4*n**2),
                    'activation':torch.relu}]

ppo_player_options = {'steps':0,
                    'gamma':0.99,
                    'lam':0.97,
                    'sample_size':100,
                    'train_policy_iters':80,
                    'train_vf_iters':80,
                    'clip_param':0.1,
                    'target_kl':0.01,
                    'policy_lr':1e-3,
                    'vf_lr':1e-3}

player_dict = {'player_class':PPOPlayer,
               'base_player_options':base_player_options,
               'additional_player_options':ppo_player_options,
               'player_models':ppo_models,
               'player_model_params':ppo_model_params}

rounds = 100
timesteps = 500
count = 0

# N, d tuples
population_search = [
                    (2, 1),
                    #(20, 2),
                    (24, 4),
                    #(24, 6),
                    #(60, 2),
#                    (100, 5),
#                    (100, 10),
                    #(1000, 2),
                    #(1000, 5),
                    #(1000, 10)
]

# b large relative to c
# b small relative to c
# c 0, b negative
coin_payoffs = [
                        np.array([[1, 0], [0, -10]], dtype=np.float_),
                        np.array([[1, 0], [1, -10]], dtype=np.float_),
                        np.array([[10, 0], [1, -1]], dtype=np.float_),
                        np.array([[5, 0], [1, -1]], dtype=np.float_),
                        np.array([[0.2, 0], [1, -1]], dtype=np.float_),
                        np.array([[0, 1], [-1, -1]], dtype=np.float_),
                        np.array([[1, 0], [-1, -1]], dtype=np.float_),
]


for coin_payoff in coin_payoffs:
    env_options['coin_payoffs'] = coin_payoff
    print(env_dict)
    for N, d in population_search:
        print(N, d)
        ## Population Settings
        population_dict = {'N':N,
                            'd':d}

        experiment = coinGameExperiment.CoinGameExperiment(env_dict=env_dict,
                                                           population_dict=population_dict,
                                                           player_dict=player_dict,
                                                           device=device,
                                                           save_name='ppo')

        total_games = rounds*N/2
        total_timesteps = total_games * timesteps
        print(f'Starting timesteps:{timesteps}, rounds:{rounds}, N:{N}, d:{d}, PPO')
        print(f'Total Games: {total_games}, Total Timesteps {total_timesteps}')
        start = datetime.now()
        print(start)
        ppo_df, dqn_players, dqn_players_df = experiment.play_multi_rounds(rounds, timesteps, count)
        print(datetime.now()-start)
        del experiment
        gc.collect()
