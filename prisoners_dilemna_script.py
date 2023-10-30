import matplotlib
import matplotlib.pyplot as plt

import torch
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
from evoenv.envs.enumerated_stochastic_game import EnumeratedStochasticGame, MatrixGame
from players import PPOPlayer, DQNPlayer,  VPGPlayer
import coinGameExperiment

from datetime import datetime
from cg_utils import section_to_dict, build_reward_matrix

import configparser
import argparse

parser = argparse.ArgumentParser(description='Read file content.')
parser.add_argument('filename', metavar='FILE', type=str, help='Path to the input file')
args = parser.parse_args()

# Access the file name using args.filename
file_name = args.filename

config_file_path = f'configs/{file_name}'
config = configparser.ConfigParser()
config.read(config_file_path)

save_path = str(config.get('general', 'save_path'))

# experiment settings
rounds = int(config.get('experiment', 'rounds'))
timesteps = int(config.get('experiment', 'timesteps'))
count = int(config.get('experiment', 'count'))

# algo settings
memory = int(config.get('model_type', 'memory'))
hidden_size_multiple = int(config.get('model_type', 'hidden_size_multiple'))
algo = str(config.get('model_type', 'algo'))
save_path = save_path + f'PD/{algo}/'

# environment settings
state_space = int(config.get('env', 'state_space'))
actions_space = int(config.get('env', 'actions_space'))
players_per_game = int(config.get('env', 'players_per_game'))
a = int(config.get('env', 'a'))
b = int(config.get('env', 'b'))
c = int(config.get('env', 'c'))
d2 = int(config.get('env', 'd2'))

input_size = state_space + (state_space + players_per_game*actions_space) * (memory)

base_player_options = {'memory':memory}

if algo == 'PPO' or algo == 'VPG':
    if 'PPO' in algo:
        player_class = PPOPlayer

    else:
        player_class = VPGPlayer

    models = [section_to_dict(config, 'models'), section_to_dict(config, 'models')]

    actor_config = section_to_dict(config, 'actor_config')
    critic_config = section_to_dict(config, 'critic_config')

    actor_config['input_size'] = input_size
    actor_config['hidden_sizes'] = (hidden_size_multiple*input_size,)
    actor_config['output_size'] = int(actions_space)

    critic_config['input_size'] = input_size
    critic_config['output_size'] = int(1)
    critic_config['hidden_sizes'] = (hidden_size_multiple*input_size,)
    model_params = [actor_config, critic_config]

    player_options = section_to_dict(config, 'learning_params')
    player_options['sample_size'] = timesteps # train after each round

else:
    player_class = DQNPlayer
    models = [section_to_dict(config, 'models')]

    model_params = section_to_dict(config, 'model_params')

    model_params['input_size'] = input_size
    model_params['hidden_sizes'] = (hidden_size_multiple*input_size,)
    model_params['output_size'] = int(actions_space)
    model_params = [model_params]

    player_options = section_to_dict(config, 'learning_params')
    player_options['target_update_step'] = timesteps # update after each round

player_dict = {'player_class':player_class,
               'base_player_options':base_player_options,
               'additional_player_options':player_options,
               'player_models':models,
               'player_model_params':model_params}


# defining this stochastic games
env = MatrixGame

rewards = build_reward_matrix(a, b, c, d2)
env_options = {'rewards':rewards}

# two states, but returned as a single scalar (0 or 1), two actions
env_description = {'obs_dim':input_size,
                   'act_dim':actions_space,
                   'act_limit':1}

env_dict = {'env':env,
            'env_options':env_options,
            'env_description':env_description}


# N, d tuples
population_search = []
for N in [20]:
    population_search.extend([(N, 1), (N, int(N/2))])

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
                                                       save_path=save_path)

    total_games = rounds*N/2
    total_timesteps = total_games * timesteps
    print(f'Starting timesteps:{timesteps}, rounds:{rounds}, N:{N}, d:{d}, {algo}')
    print(f'Total Games: {total_games}, Total Timesteps {total_timesteps}')
    start = datetime.now()
    print(start)
    ppo_df, dqn_players, dqn_players_df = experiment.play_multi_rounds(rounds, timesteps, count)
    print(datetime.now()-start)
    del experiment
    gc.collect()