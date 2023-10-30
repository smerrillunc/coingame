import math
import random
from collections import namedtuple, deque
from itertools import count
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import torch.optim as optim

class Population():
  """
  Description: Thic class stores all functions related to creating population of players.

  Inputs:
    playerClass: A class for player objects we will be creating a population of
    N: The number of players to create a population of
    d: The number of subpopulations to create
    model: A DQN model the players will use to update their value functions or policies
    policy_init_options: the initial policy player objects will follow
  """

  def __init__(self, player_dict, N, d):
    self.N = N
    self.d = d

    self.playerClass = player_dict['player_class']
    self.base_player_options = player_dict['base_player_options']
    self.additional_player_options = player_dict['additional_player_options']
    self.models = player_dict['player_models']
    self.model_options = player_dict['player_model_params']

    # generate players
    self.players, self.players_df = self.generate_population()

    # create nddarrys of red and blue players
    self.blue_players, self.red_players = self.get_red_blue_players()

    pass


  ###############################
  ### PLAYER CREATION METHODS ####
  ###############################
  def generate_population(self):
    """
    Description: This method will generate a population of players, assign them colors,
    population group, and initialize their networks.

    Inputs:
      self: to access the following attributes
        N: The population size.  We will create N/2 red and N/2 blue players
        d: The number of subpopulation groups
        model: The nn.Module network the players will optimize
        policy_init_options: a list of dictionaries specifying the initial distributions to
        use for actions.  Ex.
            policy_init_options = [{'function':np.random.uniform, 'params':{'size':size}}, # uniform
                                  {'function':np.random.beta,'params':{'a':a, 'b':b, 'size':size}}, # beta
                                  {'function':np.random.dirichlet,'params':{'alpha':[alpha for x in range(size)], 'size':size}}] # dirichlet

    Outputs:
      players = a list of player objects
      players_df = a dataframe describing players objects
    """

    # red players
    red_params = np.array([['r' for x in range(self.N//2)], # player color
            [i % self.d for i in range(self.N//2)], # player population
            ])

    # blue players
    blue_params = np.array([['b' for x in range(self.N//2)], # player color
                          [i % self.d for i in range(self.N//2)], # player population
                           ] )

    # parameters to initialize player objects with
    player_inits = np.hstack([red_params, blue_params]).T

    # create a df from player_inits for future use
    players_df = pd.DataFrame(player_inits, columns=['color', 'population'])
    players_df = players_df.reset_index().rename(columns={'index':'player_id'})

    # array of player objects
    #players = np.array([self.playerClass(**x) for (idx, x) in players_df.iterrows()])


    # PPO player setting actor/critic nets
    if len(self.model_options) == 2:
      players = np.array([self.playerClass(**self.models[0],
                                           **self.additional_player_options,
                                           actor_model_params=self.model_options[0],
                                           critic_model_params = self.model_options[1],
                                           base_player_params={**{'player_id': x['player_id'],
                                                                  'color': x['color'],
                                                                  'population': x['population']},
                                                               **self.base_player_options}) for (idx, x) in players_df.iterrows()])

    # DQN setting only model
    else:
      players = np.array([self.playerClass(**self.models[0],
                                           **self.additional_player_options,
                                           model_params=self.model_options[0],
                                           base_player_params={**{'player_id': x['player_id'],
                                                                  'color': x['color'],
                                                                  'population': x['population']},
                                                               **self.base_player_options}) for (idx, x) in players_df.iterrows()])
    return players, players_df


  def get_red_blue_players(self):
    """
    Description: This method splits the players df into two seprate arrays of red and blue players
    where red_players(i, j) is the j'th red player in the i'th population

    Inputs:
      self
    Outputs:
      blue_players: ndarray of blue player ids
      red_players: ndarray of red player ids
    """

    red_players = np.zeros((self.d, self.N//(2*self.d)))
    blue_players = np.zeros((self.d, self.N//(2*self.d)))

    for population_id in self.players_df.population.unique():
        red_players[int(population_id)] = self.players_df[(self.players_df.population == population_id) &\
                                           (self.players_df.color == 'r')]['player_id'].values
        
        blue_players[int(population_id)] = self.players_df[(self.players_df.population == population_id) &\
                                                     (self.players_df.color == 'b')]['player_id'].values

    return blue_players, red_players


  @staticmethod
  def pair_players(d, blue_players, red_players):
      """
      Description: This function will pair players within a subpopulation randomly
      given some probability of being paired to another subpopulation

      Inputs:
        d: the number of subpopulations
        blue_players: blue player nddarray (d, # blue players in population d)
        red_players: red player nddarray (d, # blue players in population d)
      Output:
        population_pairs: a list of tuples with players paired
      """

      # initialize population pairs
      population_pairs = []

      # for each population group match players within that group
      # can we do this in a more efficient vectorized way?
      for population_id in range(d):
        population_pairs.append(np.array(list(zip( shuffle(blue_players[population_id]),
                                                           red_players[population_id]))))

      return np.array(population_pairs)


  @staticmethod
  def population_migration(player_pairs, count, color='r'):
      """
      Description: This function will perform migration.  We specifically randomly select
      count of red or blue players to move.  Note there is a 1 / N chance that
      a player could be randomly paired with itself (not move subpopulations)
      Inputs:
        player_pairs: A numpy array of player matchups
        count: The number of players to migrate populations
        color: The color of the player to different population
      Outputs:
        player_pairs: updated such that count of players in population a move
        to poulation b
      """

      # select population indexes
      idx_population1 = np.random.choice(player_pairs.shape[0], count, replace=True)
      idx_population2 = np.random.choice(player_pairs.shape[0], count, replace=True)

      # randomly generate indexes to swap
      idx_pair1 = np.random.choice(player_pairs.shape[1], count, replace=True)
      idx_pair2 = np.random.choice(player_pairs.shape[1], count, replace=True)

      # color indexes
      if color == 'r':
        color_idx = np.ones((1, count), dtype=int)
      else:
        color_idx = np.zeros((1, count), dtype=int)

      # swap red players
      player_pairs[idx_population1, idx_pair1, color_idx], player_pairs[idx_population2, idx_pair2, color_idx] = \
      player_pairs[idx_population2, idx_pair2, color_idx], player_pairs[idx_population1, idx_pair1, color_idx]

      # note copy is pass by value so done in place, but here we return the
      # array anyway.
      return player_pairs
