import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('/Users/scottmerrill/Documents/UNC/Research/coingame/evoenv')
from evoenv.matrixtools.matrixtools import FloatTupleDtype
from networks import MLP, CategoricalPolicy, GaussianPolicy

def section_to_dict(config, section):
  section_dict = {}
  options = config.options(section)
  for option in options:
    val = config.get(section, option)
    if val == 'CategoricalPolicy':
      section_dict[option] = CategoricalPolicy
    elif val == 'MLP':
      section_dict[option] = MLP
    elif val == 'GaussianPolicy':
      section_dict[option] = GaussianPolicy

    else:
      try:
        val = float(val)
        if val == round(val, 0):
          section_dict[option] = int(val)
        else:
          section_dict[option] = val
      except:
        section_dict[option] = config.get(section, option)
  return section_dict

def build_reward_matrix(a, b, c, d):

    rewards = np.array(
      [
        [(a, a), (c, b)],
        [(b, c), (d, d)]
      ], dtype=FloatTupleDtype(2)
    )
    return rewards


def prisoner_dilemna_payoff(b, c):
    R = b - c
    S = -c
    T = b
    P = 0

    rewards = np.array(
      [
        [(R, R), (S, T)],
        [(T, S), (P, P)]
      ], dtype=FloatTupleDtype(2)
    )

    return rewards


def plot_reward_hist(dfs, titles=[]):
  fig, axes = plt.subplots(nrows=1, ncols=len(dfs), figsize=(10, 5), sharey=True, )

  for i in range(0, len(dfs)):
    df = dfs[i].copy()
    df['total_reward'] = df['red_reward'] + df['blue_reward']

    means = df.groupby('round').aggregate({'total_reward': 'mean',
                                           'red_reward': 'mean',
                                           'blue_reward': 'mean'}).reset_index()

    axes[i].plot(means['round'], means['total_reward'], label='total_reward', color='green')
    axes[i].plot(means['round'], means['red_reward'], label='red_reward', color='red')
    axes[i].plot(means['round'], means['blue_reward'], label='blue_reward', color='blue')
    axes[i].grid(True)

    axes[i].set_xlabel('Round')

    if i == 0:
      axes[i].set_ylabel('Mean Reward')

    if titles:
      axes[i].set_title(titles[i])

  plt.legend(loc='best')

def show_payoff_mat(payoff_matrix):
    vals = np.array([sum(x) for x in payoff_matrix.flatten()]).reshape(payoff_matrix.shape)
    annots = np.array([str(x) for x in payoff_matrix.flatten()], dtype='str').reshape(payoff_matrix.shape)
    print(annots)
    print(vals)

    sns.heatmap(vals, annot=annots, fmt='', linewidth=.5)

    plt.title('Payoff Matrix')

    plt.xticks([x + 0.5 for x in range(2)], ['C', 'D'])
    plt.yticks([x + 0.5 for x in range(2)], ['C', 'D'])

    plt.show()

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