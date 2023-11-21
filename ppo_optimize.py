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
import os

sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}/evoenv')

from evoenv.envs.enumerated_stochastic_game import EnumeratedStochasticGame, MatrixGame
from players import PPOPlayer, DQNPlayer,  VPGPlayer
import coinGameExperiment

from datetime import datetime
from cg_utils import section_to_dict, prisoner_dilemna_payoff

import optuna
import configparser
import mlflow

# Define the objective function to optimize
def objective(trial):

    mlflow.set_experiment(experiment_name="PPO")

    with mlflow.start_run():
        save_path = '/Users/scottmerrill/Documents/UNC/Research/coingame/data/PPO/'

        # make parameters
        rounds = 200
        count=0
        timesteps = 100
        N = 20
        d = 1
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

        config_file_path = 'configs/pd_ppo.ini'
        config = configparser.ConfigParser()
        config.read(config_file_path)
        base_player_options = {'memory': 1}
        ppo_models = [section_to_dict(config, 'models'), section_to_dict(config, 'models')]

        actor_config = section_to_dict(config, 'actor_config')
        critic_config = section_to_dict(config, 'critic_config')

        hidden_size_multiple = 4

        actor_config['input_size'] = input_size
        actor_config['hidden_sizes'] = (hidden_size_multiple * input_size,)
        actor_config['output_size'] = 2

        critic_config['input_size'] = input_size
        critic_config['output_size'] = int(1)
        critic_config['hidden_sizes'] = (hidden_size_multiple * input_size,)
        ppo_model_params = [actor_config, critic_config]

        # Define your custom experiment logic here
        gamma = trial.suggest_float('gamma', 0.5, 0.99)  # Example parameter ranges
        lam = trial.suggest_float('lam', 0.5, 0.99)  # Example parameter ranges
        policy_lr = trial.suggest_float('policy_lr', 0.001, 0.1)  # Example parameter ranges
        vf_lr = policy_lr
        clip_param = trial.suggest_float('clip_param', 0.01, 0.5)  # Example parameter ranges
        target_kl = trial.suggest_float('target_kl', 0.01, 0.1)  # Example parameter ranges

        train_vf_iters = trial.suggest_int('train_vf_iters', 10, 20)  # Example parameter ranges
        train_policy_iters = train_vf_iters

        ppo_player_options = {'gamma':gamma,
                              'lam':lam,
                              'policy_lr':policy_lr,
                              'vf_lr':vf_lr,
                              'clip_param':clip_param,
                              'target_kl':target_kl,
                              'train_vf_iters':train_vf_iters,
                              'train_policy_iters':train_policy_iters,
                              'sample_size':timesteps}

        player_dict = {'player_class': PPOPlayer,
                           'base_player_options': base_player_options,
                           'additional_player_options': ppo_player_options,
                           'player_models': ppo_models,
                           'player_model_params': ppo_model_params}

        experiment = coinGameExperiment.CoinGameExperiment(env_dict=env_dict,
                                                           population_dict=population_dict,
                                                           player_dict=player_dict,
                                                           device=device,
                                                           save_path=save_path)

        df, _, _ = experiment.play_multi_rounds(rounds, timesteps, count)

        # output and compute scores
        df = df.groupby('round').aggregate({'total_reward': 'mean'})
        score = abs(df['total_reward'].max() - df['total_reward'].min())

        mlflow.set_tag("dataset", "Prisoner's Dilemna")
        mlflow.log_param("gamma", gamma)
        mlflow.log_param("lam", lam)
        mlflow.log_param("policy_lr", policy_lr)
        mlflow.log_param("vf_lr", vf_lr)
        mlflow.log_param("clip_param", clip_param)
        mlflow.log_param("target_kl", target_kl)
        mlflow.log_param("train_vf_iters", train_vf_iters)
        mlflow.log_param("train_policy_iters", train_policy_iters)
        mlflow.log_param("hidden_size_multiple", hidden_size_multiple)
        mlflow.log_param("b", b)
        mlflow.log_param("c", c)
        mlflow.log_param("N", N)
        mlflow.log_param("d", d)
        mlflow.log_param("rounds", rounds)
        mlflow.log_param("timesteps", timesteps)
        mlflow.log_metric("score", score)
        save_artifact_loc = f'{experiment.save_path}/{experiment.save_name}.png'
        mlflow.log_artifact(save_artifact_loc)

        del experiment

    return score

# Create a study object and specify the direction ('minimize' or 'maximize')
study = optuna.create_study(direction='maximize')

# Optimize the study, passing the objective function and the number of trials
study.optimize(objective, n_trials=100)

# Get the best parameters and the corresponding score
best_params = study.best_params
best_score = study.best_value

