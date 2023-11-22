from collections import deque
from itertools import chain

import torch
import pandas as pd
import numpy as np

from population import Population
from players import PPOPlayer, DQNPlayer

import itertools

import os
from datetime import datetime
import logging
import sys
import matplotlib.pyplot as plt

from evoenv.envs.enumerated_stochastic_game import EnumeratedStochasticGame, MatrixGame

class CoinGameExperiment():
  """
  Description:  A class to perform the experiments of interest on the CoinGame.  This class
  generates populations of players, performs population migrations, runs games, and records
  results
  

  Inputs:
    env: The environment class
    env_options: options to pass to the environment class
    N: The population size to run experiments on
    d: The numbe of subpopulations
    model: a nn.module to use for value function approximation
    policy_init_options: Initial policy distribution for players
  """

  def __init__(self,
               env_dict, 
               population_dict, 
               player_dict, 
               device,
               save_policy=False,
               save_path=r'/Users/scottmerrill/Documents/UNC/Research/coingame/data/PPO/',
               save_name='results'):

    # setup the environment according to options passed    
    self.env = env_dict['env'](**env_dict['env_options'])
    self.act_dim = env_dict['env_description']['act_dim']
    self.obs_dim = env_dict['env_description']['obs_dim']

    if not hasattr(self.env, "GRID_ACTIONS"):
      self.env.GRID_ACTIONS = {0:"C",1: "D"}

    self.n = env_dict['env_options'].get('grid_shape', (2,2))[0]
    self.n_coins = env_dict['env_options'].get('n_coins', 1)
    self.coin_payoffs = env_dict['env_options'].get('coin_payoffs', [0, 0])

    # setup population params    
    self.N = population_dict["N"]
    self.d = population_dict["d"]
    
    # add the save path to the players
    self.device = device
    self.save_name = save_name
    self.states_df = self.generate_all_states()
    self.save_policy = save_policy

    self.player_dict = player_dict

    # Define the directory path with the current date as the name
    if save_path:
      self.save_path = CoinGameExperiment.configure_save_directory(save_path)
      self.player_dict['base_player_options']['save_path'] = self.save_path

      # make summary file
      self.make_summary_file()
      # add logging
      self.add_logger()

    else:
      player_dict['base_player_options']['configure_save_path'] = False
      self.logger=None
      self.save_path = None

    # also updare base player options with environment descriptions (state, action, etc.)
    self.player_dict['base_player_options'].update(env_dict['env_description'])

    # create population object
    self.population = Population(player_dict=player_dict,
                                 N=self.N,
                                 d=self.d)

  @staticmethod
  def convert_state(state):
    """
    Description: Checks the type of state and converts it to a flattened numpy
    numpy array
    """
    if isinstance(state, tuple):
      state = state[0].flatten()
    else:
      state = np.array([state])
    return state

  @staticmethod
  def make_one_hot_dict(num_actions = 4):
    """
    Description: This method returns a dictionary of one hot encodings
    """
    one_hot_dict = {}
    for x in range(0, num_actions):
      one_hot_dict[x] = [0 for y in range(0, num_actions)]
      one_hot_dict[x][x] = 1
    return one_hot_dict


  ###############################
  ###### GAME PLAY METHODS ######
  ###############################
  @staticmethod
  def play_game(env, players, timesteps, device='cpu'):
    """
    Description:  This function will play a single game between players in array
    some starting state.  The resulting networks of the players will be updated in place.

    Inputs:
      env: The environment in which p1 and p2 compete
      players: a list of players
      timesteps: the amount of timesteps to run the game for
    Output:
      env: updated environment
      players: with modified policeis
      df: a dataframe with statistics on the game
    """

    # by design blue player should already be first, but just in case
    # does this even matter?
    if players[0].color == 'r':
      players[0], players[1] = players[1], players[0]

    df = pd.DataFrame()

    # Environment should be confiugred to the starting state for which we want to
    # add this later...
    state, actions = env.reset()

    # convert state to numpy array
    state = CoinGameExperiment.convert_state(state)


    # if it's not assume state is already an integer value returned from enumerated stochastic env
    # possible actions
    #action_space = [x for x in range(0, self.act_dim)]

    # map these actions to numeric moves
    # for grid world
    action_space = list(env.GRID_ACTIONS.keys())
    action_map = {x:list(action_space)[x] for x in range(len(action_space))}
    one_hot_dict = CoinGameExperiment.make_one_hot_dict(len(action_space))

    # state queue.  The first list in the queue corresponds to
    # the current state.  The remaining lists refer to the previous states
    # note that the previous states are of size len(actions) longer than the current active state.
    memory = players[0].memory
    prev_states = deque(maxlen=memory)
    # initialize state queue for previous states
    for i in range(memory):
      prev_states.append([0 for x in range(len(state) + len(players)*len(action_space))])

    # The full state is a flattened vector of the current state and (previous state, a0, a1) tuples
    # depending on the setting for memory this can be multiple
    state_full = np.concatenate([np.array(state).flatten(), np.array(list(chain.from_iterable(prev_states)))])

    # convert this to tensor
    state_full = torch.tensor(state_full, dtype=torch.float32, device=device).unsqueeze(0)

    blue_distance, red_distance, coin_color = CoinGameExperiment.get_player_distance(env)

    start_timestep = 0
    # do this loop until terminal state
    for i in range(timesteps):
          actions = []
          one_hot_actions = []

          # select an action for each player
          for player in players:
            action = player.select_action(state_full)
            one_hot_actions.append(one_hot_dict[action])
            actions.append(action)

          # take a step in the environment (for this env we need to map numeric
          # actions to "U, D, L or R")
          # for grid world
          observation, _, rewards, _ = env.step(tuple(map(lambda x: action_map[x], actions)))

          # dequeou oldest prev_state in memory and add newest one now that we have action pairs
          prev_states.pop()
          prev_states.insert(0, np.append(state, one_hot_actions))

          # Coin was collected
          if sum([abs(x) for x in rewards]) > 0:
            # here we compute whether red or blue player was defectors and get distances
            blue_label, red_label = CoinGameExperiment.is_collaborator_defector(blue_distance, red_distance, coin_color, rewards)

            # store metrics related to the episode
            tmp = {
              'start_timestep':start_timestep,
              'end_timestep':i,
              'episode_length':i - start_timestep,
              'red_distance':red_distance,
              'blue_distance':blue_distance,
              'red_reward':rewards[1],
              'blue_reward':rewards[0],
              'total_reward': rewards[0]+rewards[1],
              'coin_color':coin_color,
              'red_label':red_label,
              'blue_label':blue_label,
              'blue_population': players[0].population,
              'red_population': players[1].population,
              'blue_player_id':players[0].player_id,
              'red_player_id':players[1].player_id}

            df = pd.concat([df, pd.DataFrame(tmp, index=[0])])

            # start timestep of next episode
            start_timestep = i

            # get red and blue distances again for new boardstate
            blue_distance, red_distance, coin_color = CoinGameExperiment.get_player_distance(env)

          # convert observation to flattened numpy array
          observation = CoinGameExperiment.convert_state(observation)
          observation_full = np.concatenate([np.array(observation).flatten(), np.array(list(chain.from_iterable(prev_states)))])
          observation_full = torch.torch.tensor(observation_full, dtype=torch.float32, device=device).flatten().unsqueeze(0)

          # update experience replay memory and update policies
          # question do we need to store actions/rewards of all players if we're selfish
          # optimizers?
          for idx, player in enumerate(players):
            player.steps += 1

            # can be different depending on player type
            player.add_experience(state_full.cpu(),
                                 actions[idx],
                                 rewards[idx],
                                observation_full.cpu(),
                                  done=False)
            # Will only acutally train when the number of experience is equal to sample size for PPO player
            # and when the number of experience is greater than batch_size for dqn player
            player.train_model()

          # set state to next state
          state = observation
          state_full = observation_full

    return env, players, df


  @staticmethod
  def play_paired_games(env, player_pairs, players, timesteps, device='cpu'):
    """
    Description:  This function will play all of the games in the player_pairs
    matrix and record episodic rewards

    Inputs:
      env: environment in which players will play
      player_pair: a matrix of player ids pairings who will play agains each other
      players: a list of player objects
      timesteps: the number of timesteps taken in each individual game

    Outputs:
      df: a dataframe with statistics on the game
    """

    df = pd.DataFrame()

    # x is a population indexer
    # y is a match pair indexer
    for x in range(player_pairs.shape[0]):
      for y in range(player_pairs.shape[1]):
        # reset environment
        state, actions = env.reset()

        # player_pairs[x, y] correspond to all the players in this particular game
        # this generalizes so if theres an array of 10 player ids stored at this
        # location, then 10 players will play in this match.
        player_pair = players[player_pairs[x, y].astype(int)]
        # play the game between these players append rewards
        env, player_pair, tmp = CoinGameExperiment.play_game(env, player_pair, timesteps, device)
        df = pd.concat([df, tmp])
    return df


  def play_multi_rounds(self, rounds, timesteps, count):
    """
    Description: This function plays many roudns of CoinGame.  Reach round is a pairing of all
    N players in the population
    

    Inputs:
      rounds: number of rounds to run
      timesteps: the number of timesteps taken in each individual game
      count: the maximum number of players in each round to play with a different population
    Outputs:
      df: a df describing the games, rewards, etc.
      players: a list of player objects that have been trained and played in rounds # of games
      players_df: a df consisting of information related to the players in the players array.  Includes info on
      how these players were initialized, the population they belong to, the color they belong to, etc.
    """
    # initialize output df
    df = pd.DataFrame()

    # initialize game, payoff, rewards, rounds, etc.
    state, actions = self.env.reset()

    for round_idx in range(rounds):
      start = datetime.now()
      print(f'Round {round_idx}, Start Time {start}')

      if self.logger:
        self.logger.info(f'Round {round_idx}, Start Time {start}')

      # pair players for this particular round
      #player_pairs = self.population.pair_players(self.population.d,
      #                                            self.population.blue_players,
      #                                            self.population.red_players)


      # single population pairing, ignore color and population
      player_pairs = self.population.random_pairing(np.append(self.population.blue_players, self.population.red_players))

      # note we have to divide count by two b
      player_pairs = self.population.population_migration(player_pairs, count//2)
      player_pairs = self.population.population_migration(player_pairs, count//2)

      # play all the games in player_pairings and record reward
      tmp = self.play_paired_games(self.env, player_pairs, self.population.players, timesteps, self.device)

      end = datetime.now()
      total_time = end-start
      time_per_game = total_time / (self.N/2)
      time_per_timestep = time_per_game / timesteps

      tmp['round'] = round_idx
      tmp['round_time'] = total_time
      tmp['time_per_game'] = time_per_game
      tmp['time_per_timestep'] = time_per_timestep

      df = pd.concat([df, tmp])

      if self.save_path:
        self.logger.info(f'Round {round_idx}, Total Time {total_time}, Time/Game: {time_per_game}, Time/timestep: {time_per_timestep}')
        print(f'Round {round_idx}, Total Time {total_time}, Time/Game: {time_per_game}, Time/timestep: {time_per_timestep}')

        if (round_idx+1) % 100 == 0:
            self.make_plots(df, timesteps, count)
            df.to_csv(f'{self.save_path}/{self.save_name}.csv')

    if self.save_path:
      self.make_plots(df, timesteps, count)
      df.to_csv(f'{self.save_path}/{self.save_name}.csv')

      # save the policy of each player for each state
      for player in self.population.players:
        _ = player.save_network()
        if self.save_policy:
          _ = player.save_policy(self.states_df,
                                   cols=['A1', 'A2', 'A3', 'A4'])

    return df, self.population.players, self.population.players_df


  ###############################
  #### GAME METRICS METHODS #####
  ###############################
  @staticmethod
  def get_player_distance(env):
    """
    Description: This function applies to only single coin, single player environments.  Given
    an environment it will return the distance each player is to the coin

    Inputs:
      env: the environment variable to get player distances from
    outputs:
      blue_distance: the distance the red player is from the coin
      red_distance:: the distance the blue player is from the coin
      coin_color: the color of the coin
    """
    state = env._state

    # no metric of distance in matrix 1 step games
    if isinstance(env, EnumeratedStochasticGame):
      if state == 0:
        coin_color = 'b'
      else:
        coin_color = 'r'
      return 0, 0, coin_color

    # if state[2] is not all zeros it's a blue coin
    if state[2].sum() != 0:
      coin_loc = np.unravel_index(state[2].argmax(), state[2].shape)
      coin_color = 'b'
    else:
      coin_loc = np.unravel_index(state[3].argmax(), state[3].shape)
      coin_color = 'r'

    # distances is just manhattan distance b/w player and coin locs
    red_distance = np.abs(env._agent_pos['r'] - coin_loc).sum()
    blue_distance = np.abs(env._agent_pos['b'] - coin_loc).sum()
    return blue_distance, red_distance, coin_color


  @staticmethod
  def is_collaborator_defector(blue_distance, red_distance, coin_color, rewards):
    """
    Description: This function applies to only single coin, single player environments.  It will
    return for each player if they are a collaborator or a defector.  A collaborator is defined
    as a player who is initially closer to the coin of the opposing player, but choses not to
    pick it up.  A defector is similarly a player who's initially at least as close to the coin
    as the other player and choses to pick up the coin.  This function requires that a negatirve
    reward is recieved by picking up a coin of the wrong color

    Inputs:
      blue_distance: The initial distance of the blue player to the coin
      red_distance:  The initial distance of the red player to the coin
      coin_color::  The color of the coin.
      rewards: The reward vector.
    Outputs:
      blue_label: 1 = collaborator, -1 = defector, 0 = neither
      red_label: 1 = collaborator, -1 = defector, 0 = neither
    """

    if coin_color == 'b':
      # blue player can't collaborate or defect if it's their color
      blue_label = 0

      # if red player is closer to the blue coin
      if (red_distance <= blue_distance):
        # If red collects the coin (the blue reward isn't negative)
        # then red player is a collaborator
        if (rewards[0] < 0):
          red_label = -1
        else:
          red_label = 1

      else:
        # If we're further from the coin we can still defect if we collect the coin
        # assuming blue takes wrong move
        if (rewards[0] < 0):
          red_label = -1
        else:
          red_label = 0

    else:
      # red player can't collaborate or defect if it's their color
      red_label = 0

      # if red player is closer to the blue coin
      if (blue_distance <= red_distance):
        # If blue collects the coin (the red reward isn't negative)
        # then red player is a collaborator
        if (rewards[1] < 0):
          blue_label = -1
        else:
          blue_label = 1

      else:
        # If we're further from the coin we can still defect if we collect the coin
        # assuming blue takes wrong move
        if (rewards[1] < 0):
          blue_label = -1
        else:
          blue_label = 0

    return blue_label, red_label


  ###############################
  ##### MISC GAMEPLAY UTILS #####
  ###############################
  @staticmethod
  def set_state(env, state_config):
      """
      Description: Function to set the environment to a particular state.
      Inputs:
        env: environment to set the state for
        state_config: 1X3 list with [state, blue_pos, red_pos]
      Outputs:
        env with state set
      """

      env._state = state_config[0]
      env._agent_pos['b'] = state_config[1]
      env._agent_pos['r'] = state_config[2]
      return env

  def add_logger(self):
    # Create a logger
    self.logger = logging.getLogger('CoinGame')
    self.logger.setLevel(logging.INFO)

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(f'{self.save_path}/log.log')
    # Create a stream handler to capture stdout
    stream_handler = logging.StreamHandler(sys.stdout)

    # Create a formatter to format log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    self.logger.addHandler(file_handler)
    self.logger.addHandler(stream_handler)

  def make_plots(self, df, timesteps, count):
    df['total_reward'] = df['red_reward'] + df['blue_reward']

    means = df.groupby('round').aggregate({'total_reward': 'mean',
                                           'red_reward': 'mean',
                                           'blue_reward': 'mean'}).reset_index()

    plt.figure(figsize=(12, 12))
    plt.plot(means['round'], means['total_reward'], label='total_reward', color='green')
    plt.plot(means['round'], means['red_reward'], label='red_reward', color='red')
    plt.plot(means['round'], means['blue_reward'], label='blue_reward', color='blue')
    plt.grid(True)
    plt.legend(loc='best')
    plt.xlabel('Round')
    plt.ylabel('Mean Reward')
    plt.title(f'Population Size: {self.N}, Subpopulations: {self.d}, Migration Count: {count}, timesteps/round:{timesteps}')
    plt.savefig(f'{self.save_path}/{self.save_name}.png')
    return 1

  def make_summary_file(self):
    # create file in director
    with open(f'{self.save_path}/experiment_summary.txt', 'w') as file:
      file.write(f'N:{self.N}\n')
      file.write(f'd:{self.d}\n')

      file.write(f'grid:({self.n}, {self.n})\n')
      file.write(f'n_coins:{self.n_coins}\n')
      file.write(f'payoffs:{self.coin_payoffs}\n')

      if self.player_dict['player_class'] == PPOPlayer:
        file.write(f'player_class: PPOPlayer\n')
        for idx, model_params in enumerate(self.player_dict['player_model_params']):
          if idx == 0:
            file.write(f'\nACTOR NETWORK\n')
          elif idx == 1:
            file.write(f'\nCritic NETWORK\n')
          for key, value in model_params.items():
            try:
              file.write(f'{key}:{value}\n')
            except:
              continue

      else:
        file.write(f'player_class: DQNPlayer\n')
        file.write(f'\nVALUE NETWORK\n')
        for model_params in self.player_dict['player_model_params']:
          for key, value in model_params.items():
            try:
              file.write(f'{key}:{value}\n')
            except:
              continue

      for key, value in self.player_dict['additional_player_options'].items():
        try:
          file.write(f'{key}:{value}\n')
        except:
          continue

  @staticmethod
  def configure_save_directory(save_path):
    # Define the directory path with the current date as the name
    today = datetime.now().strftime("%Y%m%d")
    save_path = save_path + f'{today}'

    # Make a directory in save path for current date
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory '{today}' created.")
    else:
        print(f"Directory '{today}' already exists.")
        pass

    # this makes a directory for specific trial on a date
    i = 1
    while True:
      if not os.path.exists(save_path + f'/{i}'):
          save_path = save_path + f'/{i}'
          os.makedirs(save_path)
          print(f"Save Path created for Experiment number {i} on '{today}'")
          break
      else:
        i = i + 1
    return save_path


  def generate_all_states(self):
    length = self.n*self.n*4
    ones = self.n_coins
    which = np.array(list(itertools.combinations(range(length), ones)))
    grid = np.zeros((len(which), length), dtype="int8")
    grid[np.arange(len(which))[None].T, which] = 1
    return pd.DataFrame(grid)

