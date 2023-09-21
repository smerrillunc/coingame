import math
import random
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import torch.optim as optim

from coingame.population import Population

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

  def __init__(self, env, env_options, population_options, player, player_options, player_models, player_model_params, device):
    # setup the environment according to options passed    
    self.env = env(**env_options)

    # setup population params    
    self.N = population_options["N"]
    self.d = population_options["d"]
    
    # create population object
    self.population = Population(playerClass=player,
                                 player_options=player_options,
                                 models=player_models,
                                 model_options=player_model_params,
                                 N=self.N,
                                 d=self.d)

    self.device = device
    pass

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

    # take the first np array from state and convert to tensor
    state = torch.tensor(state[0], dtype=torch.float32, device=device).flatten().unsqueeze(0)

    # possible actions
    action_space = list(env.GRID_ACTIONS.keys())

    # map these actions to numeric moves
    action_map = {x:list(action_space)[x] for x in range(len(action_space))}

    blue_distance, red_distance, coin_color = CoinGameExperiment.get_player_distance(env)

    start_timestep = 0
    # do this loop until terminal state
    for i in range(timesteps):
          actions = []

          # select an action for each player
          for player in players:
            action = player.select_action(state)
            actions.append(action)

          # take a step in the environment (for this env we need to map numeric
          # actions to "U, D, L or R")
          observation, _, rewards, _ = env.step(list(map(lambda x: action_map[x], actions)))

          # Coin was collected
          if np.abs(rewards).sum() > 0:
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

          next_state = torch.tensor(observation[0], dtype=torch.float32, device=device).flatten().unsqueeze(0)

          # update experience replay memory and update policies
          # question do we need to store actions/rewards of all players if we're selfish
          # optimizers?
          for idx, player in enumerate(players):
            player.steps += 1

            # can be different depending on player type
            player.add_experience(state,
                                 actions[idx],
                                 rewards[idx],
                                  next_state,
                                  done=False)
            # Will only acutally train when the number of experience is equal to sample size for PPO player
            # and when the number of experience is greater than batch_size for dqn player
            player.train_model()
          # set state to next state
          state = next_state

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

    for round in range(rounds):
      # pair players for this particular round
      player_pairs = self.population.pair_players(self.population.d,
                                                  self.population.blue_players,
                                                  self.population.red_players)

      # note we have to divide count by two b
      player_pairs = self.population.population_migration(player_pairs, count//2)
      player_pairs = self.population.population_migration(player_pairs, count//2)

      # play all the games in player_pairings and record reward
      tmp = self.play_paired_games(self.env, player_pairs, self.population.players, timesteps, self.device)
      df = pd.concat([df, tmp])


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

    # if state[2] is not all zeros it's a blue coin
    if state[2].sum() != 0:
      coin_loc = np.unravel_index(state[2].argmax(), state[2].shape)
      coin_color = 'b'
    else:
      coin_loc = np.unravel_index(state[3].argmax(), state[3].shape)
      coin_color = 'r'

    # distances is just manhattan distance b/w player and coin locs
    red_distance = np.abs(env._agent_pos['r'] - coin_loc).sum()
    blue_distance =  np.abs(env._agent_pos['b'] - coin_loc).sum()
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