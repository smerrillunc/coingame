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
from evoenv.envs.enumerated_stochastic_game import EnumeratedStochasticGame, MatrixGame

from players import PPOPlayer, DQNPlayer
from population import Population
from networks import MLP, GaussianPolicy, CategoricalPolicy
from memoryBuffers import ReplayBuffer, Buffer
import coinGameExperiment

from datetime import datetime

"""DQN Test"""

def generate_two_state_game(c, b):
    # defining rewards
    blue_payoff = (0.5*(b-c), 0.5*(b))
    red_payoff = (0.5*(b), 0.5*(b-c))

    ################################################################
    #blue_coin = np.array([ [(b, 0), blue_payoff],
    #                        [(b, 0), blue_payoff]], dtype='object')
    ################################################################

    # had to force array into tuples for some reason?
    blue_coin = np.empty((2, 2), dtype='object')
    blue_coin[0, 0] = (b, 0)
    blue_coin[0, 1] = blue_payoff
    blue_coin[1, 0] = (b, 0)
    blue_coin[1, 1] = blue_payoff


    ################################################################
    #red_coin = np.array([[(0, b), (0, b)],
    #                    [red_payoff, red_payoff]], dtype='object')
    ################################################################
    red_coin = np.empty((2, 2), dtype='object')
    red_coin[0, 0] = (0, b)
    red_coin[0, 1] = (0, b)
    red_coin[1, 0] = red_payoff
    red_coin[1, 1] = red_payoff

    rewards = {
        0:blue_coin,
        1:red_coin
    }
    return rewards

# environment settings
n = 3

# defining this stochastic games
env = EnumeratedStochasticGame
env_options = {'rewards':generate_two_state_game(1, 1),}


# two states, but returned as a single scalar (0 or 1), two actions
env_description = {'obs_dim':1,
                   'act_dim':2,
                   'act_limit':1}

env_dict = {'env':env,
            'env_options':env_options,
            'env_description':env_description}


# player settings
base_player_options = {}
dqn_model = [{'model':MLP}]

dqn_model_params = [
                    {'input_size':1, \
                    'output_size':2,
                    'output_limit':1.0,
                    'hidden_sizes':(64,),
                    'activation':torch.relu}]

dqn_player_options = {}

player_dict = {'player_class':DQNPlayer,
               'base_player_options':base_player_options,
               'additional_player_options':dqn_player_options,
               'player_models':dqn_model,
               'player_model_params':dqn_model_params}


# setting total timesteps to 200k or 100k/state
rounds = 20
timesteps = 200
count = 0

# N, d tuples
population_search = []
for N in [20, 40, 60, 80, 100]:
    population_search.extend([(N, 2), (N, int(N/2))])


cb_vals = [(1, 1),(5, 1),(1, 5),(0, 1),(1,0)]

for c, b in cb_vals:
    env_options['rewards'] = generate_two_state_game(c, b)
    for N, d in population_search:
        print(N, d)
        print(env_dict)
        ## Population Settings
        population_dict = {'N':N,
                            'd':d}

        experiment = coinGameExperiment.CoinGameExperiment(env_dict=env_dict,
                                                           population_dict=population_dict,
                                                           player_dict=player_dict,
                                                           device=device,
                                                           save_path=r'/Users/scottmerrill/Documents/UNC/Research/coingame/data/DQN/')

        total_games = rounds*N/2
        total_timesteps = total_games * timesteps
        print(f'Starting timesteps:{timesteps}, rounds:{rounds}, N:{N}, d:{d}, DQN')
        print(f'Total Games: {total_games}, Total Timesteps {total_timesteps}')
        start = datetime.now()
        print(start)
        ppo_df, dqn_players, dqn_players_df = experiment.play_multi_rounds(rounds, timesteps, count)
        print(datetime.now()-start)
        del experiment
        gc.collect()