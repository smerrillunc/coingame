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
from networks import MLP, GaussianPolicy
from memoryBuffers import ReplayBuffer, Buffer
import coinGameExperiment

from datetime import datetime

"""DQN Test"""

print("DQN Test")
# Population count and subpopulatinos
N = 4*2*10
d = 4
population_options = {'N':N,
                      'd':d}

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

## Population Settings
population_dict = {'N':N,
                    'd':d}


# player settings
base_player_options = {'save_path': r'/Users/scottmerrill/Documents/UNC/Research/coingame/data/'}
dqn_player_options = {'steps':0,
                      'gamma':0.99,
                      'epsilon':1.0,
                      'epsilon_decay':0.995,
                      'buffer_size':int(1e4),
                      'batch_size':64,
                      'target_update_step':100}

dqn_models = [{'model':MLP}]
dqn_models_params = [{'input_size': 4 * n ** 2, \
                      'output_size': 4,
                      'output_limit': 1.0,
                      'hidden_sizes': (64, 64),
                      'activation': torch.tanh}]



player_dict = {'player_class':DQNPlayer,
               'base_player_options':base_player_options,
               'additional_player_options':dqn_player_options,
               'player_models':dqn_models,
               'player_model_params':dqn_models_params}


experiment = coinGameExperiment.CoinGameExperiment(env_dict=env_dict,
                                                  population_dict=population_dict,
                                                  player_dict=player_dict,
                                                 device=device,
                                                 save_name='dqn.csv')






rounds = 1
timesteps= 500
count = 10
#dqn_df, dqn_players, dqn_players_df = experiment.play_multi_rounds(rounds, timesteps, count)
print("DONE DQN")

"""PPO Test"""

ppo_models = [{'actor_model':GaussianPolicy,
               'critic_model':MLP},

              {'actor_model':GaussianPolicy,
              'critic_model':MLP}]


ppo_model_params = [
                #actor network
                {'input_size':4*n**2, \
                'output_size':4,
                'output_limit':1.0,
                'hidden_sizes':(64,64),
                'activation':torch.tanh},

                # critic network
                {'input_size':4*n**2,
                'output_size':1,
                'hidden_sizes':(128,64),
              'activation':torch.tanh}]

ppo_player_options = {'steps':0,
                    'gamma':0.99,
                    'lam':0.97,
                    'hidden_sizes':(64,64),
                    'sample_size':2048,
                    'train_policy_iters':80,
                    'train_vf_iters':80,
                    'clip_param':0.2,
                    'target_kl':0.01,
                    'policy_lr':3e-4,
                    'vf_lr':1e-3}

player_dict = {'player_class':PPOPlayer,
               'base_player_options':base_player_options,
               'additional_player_options':ppo_player_options,
               'player_models':ppo_models,
               'player_model_params':ppo_model_params}


experiment = coinGameExperiment.CoinGameExperiment(env_dict=env_dict,
                                                   population_dict=population_dict,
                                                   player_dict=player_dict,
                                                   device=device,
                                                   save_name='ppo.csv')

rounds = 1
timesteps = 1000
count = 10
total_games = rounds*N/2
total_timesteps = total_games * timesteps
print(f'Starting timesteps:{timesteps}, rounds:{rounds}, N:{N}, d:{d}, PPO')
print(f'Total Games: {total_games}, Total Timesteps {total_timesteps}')
start = datetime.now()
print(start)
ppo_df, dqn_players, dqn_players_df = experiment.play_multi_rounds(rounds, timesteps, count)
print(datetime.now()-start)
