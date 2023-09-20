
import math
import random
from collections import namedtuple, deque
from itertools import count
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import unittest


class TestCoinGame(unittest.TestCase):
    """
    Description: This is a unit test class to test various functionalities.
    """

    @staticmethod
    def _map_opponents(players_df, player_pairs):
      """
      Description: This is a helper function that takes in a players df and opponent mathcup
      pairs and adds 3 columns to the dataframe corresponding to the opponent they will be
      playing, the opponent's color and their opponent's id

      Inputs:
        players_df: a dataframe containing all players and information about each
        player_pairs: a numpy array corresponding to the pairings of opponents
      """
                
      # matchup dict is a dictionary that maps p1 -> p2
      # matchup dict to is a dictionary that maps p2 -> p1
      matchup_dict = dict(zip(player_pairs[:,:,0].flatten(), player_pairs[:,:,1].flatten()))
      matchup_dict2 = dict([(value, key) for key, value in matchup_dict.items()])

      # the combination of these is a dictioanry that will map player_id -> opponent
      all_matchups = {**matchup_dict, **matchup_dict2}

      # create dictionaries of colors and dictionary mappings
      player_colors_dict = dict(zip(players_df['player_id'], players_df['color']))
      player_populations_dict = dict(zip(players_df['player_id'], players_df['population']))

      # map opponents ids, colors and poulations
      players_df['opponent'] = players_df['player_id'].map(all_matchups)
      players_df['opponent_color'] = players_df['opponent'].map(player_colors_dict)
      players_df['opponent_populations'] = players_df['opponent'].map(player_populations_dict)

      return players_df


    @staticmethod
    def _initialize_population(N=1000, d=4):
      """
      Description: This is a helper function that generates a population with basic
      initializations

      Inputs:
        N: a dataframe containing all players and information about each
        d: a numpy array corresponding to the pairings of opponents
      Outputs:
        players: A list of player objects
        players_df: A dataframe describing the players created
      """
      # gnerate population with these default params
      policy_init_options = [None]
      n_observations, n_actions = 3**2, 4
      model = DQN(n_observations, n_actions)
      population = Population(playerClass=Player,
                            N=N,
                            d=d,
                            model=model,
                            policy_init_options=policy_init_options)
      
      return population.players, population.players_df, population.blue_players, population.red_players


    def test_population_generation(self, N=1000, d=4):
        """
        Description: This function checks to ensure players within a population are
        all valid players.  Specifically
          1.  No two players share the same player_id
          2.  Each population group is the same size
          3.  There are an equal number of players with each color
          4.  All initial policies valid (the cummulative distribution integrates to 1)

        Inputs:
          N: a dataframe containing all players and information about each
          d: a numpy array corresponding to the pairings of opponents
        Outputs:
          None
        """
        # gnerate population
        players, players_df, blue_players, red_players = TestCoinGame._initialize_population(N, d)

        # check each player has a different ID
        assert len(players_df) == len(players_df.player_id.unique()), "Non-unique player IDs"
        
        # check each population has the same size
        assert len(np.unique(players_df.groupby('population').count().iloc[:,1].values)) == 1, "Population Sizes are different"

        # check each color has the same number of players
        assert len(np.unique(players_df.groupby('color').count().iloc[:,1].values)) == 1, "Color Sizes are different"

        # check to make sure if all initial polices are valid (cummulative probablities some to one for each player)
        init_policies = np.array([population.players[x].init_policy.sum() for x in range(len(population.players))])
        assert np.allclose(init_policies, np.ones(init_policies.shape), rtol=1e-05, atol=1e-08, equal_nan=False), "Initial Polices not well defined"
        
        return 1


    def test_population_pairing(self, N=1000, d=4):
        """
        Description: This function checks the initial pairing of players in a population.
        It checks:
          1. To make sure all players are assigned one and only one opponenet
          2. To make sure players are matched with someone of the opposite color
          3. To make sure players are matched with someone in their population
        Inputs:
          N: a dataframe containing all players and information about each
          d: a numpy array corresponding to the pairings of opponents
        Outputs:
          None
        """
        # gnerate population
        players, players_df, blue_players, red_players = TestCoinGame._initialize_population(N, d)

        # pair players for this particular round
        player_pairs = Population.pair_players(d, blue_players, red_players)

        # map opponents
        players_df = TestCoinGame._map_opponents(players_df, player_pairs)

        # check to make sure each player has an opponent
        assert players_df.opponent.isna().sum() == 0, "Not all players have an opponent"

        # check to make sure each player's opponent is of the other color
        assert (players_df['color'] == players_df['opponent_color']).sum() == 0, "Players are matched up with players of the same color"

        # check to make sure each player's opponent is of the same population
        assert (players_df['population'] == players_df['opponent_populations']).sum() == len(players_df), "Players are matched up with players of a different population"

        return 1
        

    def test_population_migration(self, N=1000, d=4, count=10):
        """
        Description: This function checks the population migration functionality.  
        It checks to see if the number of players matched up with an opponent in 
        a different population matches what is expected after running the population
        migration function.

        Inputs:
          N: a dataframe containing all players and information about each
          d: a numpy array corresponding to the pairings of opponents
          count: the number of red and blue players to move to a different population
        Outputs:
          None
        """

        # gnerate population
        players, players_df, blue_players, red_players = TestCoinGame._initialize_population(N, d)

        # pair players for this particular round
        player_pairs = Population.pair_players(d, blue_players, red_players)

        # population migration
        player_pairs = Population.population_migration(player_pairs, count)

        # map opponents
        players_df = TestCoinGame._map_opponents(players_df, player_pairs)

        # make sure players are still matched up against someone of a different color
        assert (players_df['color'] == players_df['opponent_color']).sum() == 0, "Players are matched up with players of the same color"

        # note because the players df has an index for each player in the population,
        # a single pair will show up twice if two players are in different populations
        diff_pops = (players_df['population'] != players_df['opponent_populations']).sum() // 2

        # count * 2 here since a single will effect two different matchups
        assert diff_pops < count * 2, "Population migrations yielded more migrations than expected"

        return 1


    def test_player_distances(self):
      """
      Description: This function tests the player_distance function that determins how
      far away each player is from the given coin.
      """
  
      # test case 1 #
      env = CoinGame(grid_shape=(3,3))
      state, actions = env.reset()

      env._state = np.array([[[0, 0, 0],
                              [0, 0, 1],
                              [0, 0, 0]],

                            [[0, 0, 1],
                              [0, 0, 0],
                              [0, 0, 0]],

                            [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]],

                            [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]])
      env._agent_pos = {'b': np.array([1, 2]), 'r': np.array([0, 2])}

      blue_distance, red_distance, coin_color = CoinGameExperiment.get_player_distance(env)

      assert blue_distance == 1, "Blue distance is wrong"
      assert red_distance == 2, "Red distance is wrong"
      assert coin_color == 'b', "Coin color is wrong"


      # test case 2 #
      env._state = np.array([[[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 1]],

                          [[0, 1, 0],
                            [0, 0, 0],
                            [0, 0, 0]],

                          [[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]],

                          [[0, 0, 1],
                            [0, 0, 0],
                            [0, 0, 0]]]
                            )
      env._agent_pos = {'b': np.array([2, 2]), 'r': np.array([0, 1])}


      blue_distance, red_distance, coin_color = CoinGameExperiment.get_player_distance(env)

      assert blue_distance == 2, "Blue distance is wrong"
      assert red_distance == 1, "Red distance is wrong"
      assert coin_color == 'r', "Coin color is wrong"

      return 1


    def test_is_collaborator(self):
      """
      Description: This function tests the is_collaborator evaluation metric, which
      determins if a player is a collaborator for a given round
      """

      
      # Test Case 1 #
      blue_distance = 2
      red_distance = 1
      coin_color = 'b'
      rewards = [1, 0]
      blue_label, red_label = CoinGameExperiment.is_collaborator_defector(blue_distance, red_distance, coin_color, rewards)
      assert blue_label == 0, "Blue Label is Wrong"
      assert red_label == 1, "Red Label is Wrong"

      # Test Case 2 #
      rewards = [-1, 1]
      blue_label, red_label = CoinGameExperiment.is_collaborator_defector(blue_distance, red_distance, coin_color, rewards)
      assert blue_label == 0, "Blue Label is Wrong"
      assert red_label == -1, "Red Label is Wrong"

      # Test Case 3 #
      blue_distance = 2
      red_distance = 2
      coin_color = 'r'
      rewards = [1, -1]
      blue_label, red_label = CoinGameExperiment.is_collaborator_defector(blue_distance, red_distance, coin_color, rewards)
      assert blue_label == -1, "Blue Label is Wrong"
      assert red_label == 0, "Red Label is Wrong"

      # Test Case 4 #
      rewards = [0, 1]
      blue_label, red_label = CoinGameExperiment.is_collaborator_defector(blue_distance, red_distance, coin_color, rewards)
      assert blue_label == 1, "Blue Label is Wrong"
      assert red_label == 0, "Red Label is Wrong"

      return 1
