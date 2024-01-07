import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from evoenv.envs.enumerated_stochastic_game import EnumeratedStochasticGame, MatrixGame
from players import PPOPlayer, DQNPlayer,  VPGPlayer
import coinGameExperiment

from cg_utils import section_to_dict, prisoner_dilemna_payoff

import optuna
import configparser
import sqlite3
import argparse


# Define the objective function to optimize
def objective(trial):

    # make parameters
    rounds = 1000
    count=0
    timesteps = 10

    N = 6
    d = 3

    population_dict = {'N': N,
                       'd': d}

    # environment settings
    state_space = 1
    actions_space = 2
    memory = 1

    # players memory
    players_per_game = 2

    # input size, we need an additional state for every memory lookback
    # we also are appending to the state each players action
    input_size = state_space + (state_space + players_per_game * actions_space) * (memory)

    b = 5
    c = 1

    rewards = prisoner_dilemna_payoff(b, c)

    env = MatrixGame
    env_options = {'rewards': rewards}

    # two states, but returned as a single scalar (0 or 1), two actions
    env_description = {'obs_dim': input_size,
                       'act_dim': actions_space,
                       'act_limit': 1}

    env_dict = {'env': env,
                'env_options': env_options,
                'env_description': env_description}

    config_file_path = 'configs/pd_vpg.ini'
    config = configparser.ConfigParser()
    config.read(config_file_path)
    base_player_options = {'memory': 1}

    vpg_models = [section_to_dict(config, 'models'), section_to_dict(config, 'models')]

    actor_config = section_to_dict(config, 'actor_config')
    critic_config = section_to_dict(config, 'critic_config')

    hidden_size_multiple = trial.suggest_int('hidden_size_multiple', 1, 10)

    actor_config['input_size'] = input_size
    actor_config['hidden_sizes'] = (hidden_size_multiple * input_size,)
    actor_config['output_size'] = int(actions_space)

    critic_config['input_size'] = input_size
    critic_config['output_size'] = int(1)
    critic_config['hidden_sizes'] = (hidden_size_multiple * input_size,)

    vpg_model_params = [actor_config, critic_config]


    # Define your custom experiment logic here
    lam = trial.suggest_float('lam', 0.25, 0.99)
    gamma = trial.suggest_float('gamma', 0.25, 0.99)
    policy_lr = trial.suggest_float('policy_lr', 0.001, 0.1)
    vf_lr = trial.suggest_float('vf_lr', 0.001, 0.05)
    buffer_multiple = trial.suggest_int('buffer_multiple', 1, 10)

    train_vf_iters = trial.suggest_int('train_vf_iters', 1, 20, 2)
    activation_function = trial.suggest_int('activation_function', 0, 4)
    initialization = trial.suggest_int('initialization', 0, 2)

    activation_funcs = [F.relu, F.leaky_relu, F.elu, F.tanh, F.sigmoid]
    initializations = ['uniform', 'normal', 'dirichlet']

    #vpg_model_params
    vpg_model_params[0]['activation'] = activation_funcs[activation_function]
    vpg_model_params[1]['activation'] = activation_funcs[activation_function]
    vpg_model_params[0]['initialization'] = initializations[initialization]
    vpg_model_params[1]['initialization'] = initializations[initialization]

    vpg_player_options = {'gamma':gamma,
                          'lam':lam,
                          'policy_lr':policy_lr,
                          'vf_lr':vf_lr,
                          'train_vf_iters':train_vf_iters,
                          'sample_size':timesteps,
                          'buffer_multiple':buffer_multiple}

    player_dict = {'player_class': VPGPlayer,
                       'base_player_options': base_player_options,
                       'additional_player_options': vpg_player_options,
                       'player_models': vpg_models,
                       'player_model_params': vpg_model_params}

    experiment = coinGameExperiment.CoinGameExperiment(env_dict=env_dict,
                                                       population_dict=population_dict,
                                                       player_dict=player_dict,
                                                       device=device,
                                                       save_path=save_path)

    df, _, _ = experiment.play_multi_rounds(rounds, timesteps, count)

    mutual_cooperation = df[df['round']==df['round'].max()]['mutual_cooperation_flag'].sum()
    mutual_defection = df[df['round']==df['round'].max()]['mutual_defection_flag'].sum()
    p1_exploit = df[df['round']==df['round'].max()]['p1_exploit_flag'].sum()
    p2_exploit = df[df['round']==df['round'].max()]['p2_exploit_flag'].sum()
    exploit = df[df['round']==df['round'].max()]['exploit_flag'].sum()

    #optuna.upload_artifact(trial, f'{experiment.save_path}/{experiment.save_name}.png')
    trial.set_user_attr('image_path', f'{experiment.save_path}/{experiment.save_name}.png')

    # mutual_cooperation_flag
    trial.set_user_attr('mutual_cooperation_flag', str(mutual_cooperation))
    trial.set_user_attr('mutual_defection_flag', str(mutual_defection))
    trial.set_user_attr('p1_exploit_flag', str(p1_exploit))
    trial.set_user_attr('p2_exploit_flag', str(p2_exploit))
    trial.set_user_attr('exploit_flag', str(exploit))
    trial.set_user_attr('optimize_flag', str(optimize_flag))

    if optimize_flag == 1:
        score = mutual_cooperation
    elif optimize_flag == 2:
        score = mutual_defection
    elif optimize_flag == 3:
        score = exploit
    else:
        score = 0

    return score

parser = argparse.ArgumentParser(description='Read file content.')

parser.add_argument("-o", "--optimize", default=1, type=int, help='optimize flag; 1 = mutual cooperation, 2 = mutual defection, 3 = exploitations')
args = parser.parse_args()
optimize_flag = int(args.optimize)

if optimize_flag == 1:
    study_name = 'vpg_mutual_cooperation'
elif optimize_flag == 2:
    study_name = 'vpg_mutual_defection'
else:
    study_name = 'vpg_exploitations'

#save_path = '/proj/mcavoy_lab/projects/coingame/'
save_path='/Users/scottmerrill/Documents/UNC/Research/coingame/data/PPO/'
#artifact_store = FileSystemArtifactStore(base_path=save_path + 'artifacts')

db_path = save_path + r'optimize.db'
conn = sqlite3.connect(db_path)

# Create an Optuna study with SQLite storage
storage_name = f'sqlite:///{db_path}?study_name={study_name}'

# Create a study object and specify the direction ('minimize' or 'maximize')
study = optuna.create_study(direction='maximize',
                            study_name=study_name,
                            storage=storage_name,
                            load_if_exists=True)

# Optimize the study, passing the objective function and the number of trials
study.optimize(objective, n_trials=10000)

# Get the best parameters and the corresponding score
best_params = study.best_params
best_score = study.best_value

df = study.trials_dataframe()
df.to_csv(save_path + 'all_trials.csv')