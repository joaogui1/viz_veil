import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

experiments_mapping = { 
                        "Adam's ε": "epsilon",
                        "Batch Size": "batch_sizes",
                        "Conv. Activation Function": "layer_funct_conv",
                        "Convolutional Normalization": "normalizations_convs", 
                        "Convolutional Width": "CNN_widths",
                        "Dense Activation Function": "layer_funct_dense",
                        "Dense Normalization": "normalizations",
                        "Dense Width": "widths",
                        "Discount Factor": "gammas",
                        "Exploration ε": "eps_train",
                        "Learning Rate": "learning_rate",
                        "Minimum Replay History": "min_replay_history",
                        "Number of Atoms": "num_atoms", 
                        "Number of Convolutional Layers": "convs", 
                        "Number of Dense Layers": "depths",
                        "Replay Capacity": "replay_capacity",
                        "Reward Clipping": "clip_rewards",
                        "Target Update Period": "target_update_periods",
                        "Update Horizon": "update_horizon",
                        "Update Period": "update_periods",
                        "Weight Decay": "weightdecay",
                    }

ATARI_100K_GAMES = [
            'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
            'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
            'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo',
            'Krull', 'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert',
            'RoadRunner', 'Seaquest', 'UpNDown'
            ]


def decorate_axis(ax, wrect=10, hrect=10, labelsize='large'):
  # Hide the right and top spines
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.spines['left'].set_linewidth(2)
  ax.spines['bottom'].set_linewidth(2)
  # Deal with ticks and the blank space at the origin
  ax.tick_params(length=0.1, width=0.1, labelsize=labelsize)
  # Pablos' comment
  ax.spines['left'].set_position(('outward', hrect))
  ax.spines['bottom'].set_position(('outward', wrect))


def plot_iqm(all_experiments, colors=None, hp_values=None):
  IQM = lambda x: metrics.aggregate_iqm(x) # Interquartile Mean
  OG = lambda x: metrics.aggregate_optimality_gap(x, 1.0) # Optimality Gap

  aggregate_func = lambda x: np.array([IQM(x), OG(x)])
  aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
      all_experiments, aggregate_func, reps=50000)

  fig, _ = plot_utils.plot_interval_estimates(
      aggregate_scores, 
      aggregate_interval_estimates,
      metric_names = ['IQM', 'Optimality Gap'],
      algorithms=hp_values,
      colors=colors,
      xlabel_y_coordinate=-0.3,
      xlabel='Human Normalized Score')
  return fig

def split_plot_iqm(dict_100k, dict_40M, colors=None, hp_values=None):
  IQM = lambda x: metrics.aggregate_iqm(x) # Interquartile Mean

  algorithms = list(dict_100k.keys() | dict_40M.keys())
  colors = dict(zip(algorithms, sns.color_palette("pastel")))
  
  dict_100k = {k + "@100k":v for (k, v) in dict_100k.items()}
  dict_40M = {k + "@40M":v for (k, v) in dict_40M.items()}
  if "normalization" in dict_100k.keys():
    dict_100k["No Normalization (default)"] = dict_100k.pop("normalization (default)")
    dict_40M["No Normalization (default)"] = dict_40M.pop("normalization (default)")

  all_experiments = {**dict_100k, **dict_40M}
  hp_values = list(all_experiments.keys())
  print("Interval plots:", hp_values)
  colors = {**{k:colors[k.split("@100k")[0]] for k in dict_100k}, 
            **{k:colors[k.split("@40M")[0]] for k in dict_40M}}

  aggregate_func = lambda x: np.array([IQM(x)])
  aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
      all_experiments, aggregate_func, reps=50000)

  y_gap = -0.4 if len(hp_values) <= 4 else -0.2 if len(hp_values) < 8 else -0.1
  fig, _ = plot_utils.plot_interval_estimates(
      aggregate_scores, 
      aggregate_interval_estimates,
      metric_names = ['IQM'],
      algorithms=hp_values,
      colors=colors,
      subfigure_width=5.0,
      xlabel_y_coordinate=y_gap,
      xlabel='Human Normalized Score')
  return fig


def plot_human_normalized(all_experiments, scale='100k', ax=None, colors=None):
  all_experiments = {k.split("_")[-1]:v for (k, v) in all_experiments.items()}
  if "normalization" in all_experiments.keys():
    all_experiments["No Normalization (default)"] = all_experiments.pop("normalization")
  algorithms = list(all_experiments.keys())

  print('algorithms:', algorithms)
  
  if scale == '100k':
    frames = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) - 1
  if scale == '40M':
    frames = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40]) - 1
  all_experiments = {algorithm: score[:, :, frames] for algorithm, score
                            in all_experiments.items()}
  iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame])
                                for frame in range(scores.shape[-1])])
  iqm_scores, iqm_cis = rly.get_interval_estimates(all_experiments, iqm, reps=2000)

  if ax is None:
    fig, ax = plt.subplots(figsize=(10, 8))
  else:
    fig = None
  ax = plot_utils.plot_sample_efficiency_curve(
      0.01*(frames+1) if scale == "100k" else (frames + 1), 
      iqm_scores, iqm_cis, 
      algorithms=algorithms,
      xlabel=f'',
      ylabel='IQM Human Normalized Score',
      colors=colors,
      legend=True,
      ax=ax)
  return fig, ax


def plot_all_games(df):
  ylabels = {'returns': 'Return'}
  envs = set(df['returns'].env)

  num_cols = 4
  num_rows = 7 
  col, row = 0, 0
  fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * 8 * num_cols, 4 * 8 * num_rows))
  for env in envs:
    for key in ylabels.keys():
      env_df = df[key]
      env_df = env_df[env_df.env == env]
      env_df = env_df[env_df['step'] % 4 == 0]
      ax = axes[row, col]
      sns.lineplot(x='step', y='val', hue='sweep', data=env_df, ax=ax)
      title = f'{env}'
      ax.set_title(title, fontsize=22)
      ax.set_ylabel(ylabels[key], fontsize=18)
      xlabel = 'Step'
      ax.set_xlabel(xlabel, fontsize=18)
      if key == 'losses':
        ax.set_yscale('symlog')
      col += 1
      if col == num_cols:
        col = 0
        row += 1
  return fig

def plot_game(agent, env, scale):
  num_cols = 7
  num_rows = 3
  fig, axes = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 8 * num_rows))
  data_path = f'data/{scale}_experiments/curves_all_games/*.pickle'
  col, row = 0, 0
  for filename in glob.glob(data_path):
    if "atoms" in filename and agent != "DER":
        continue
    with open(filename, mode='rb') as f:
        data = pickle.load(f)
    
    keys = sorted(list(data.keys()))
    hp_key = keys[0] if agent == "DER" else keys[1]
    print(hp_key, scale)
    aux = data[hp_key]['returns']
    env_data = aux[aux['env'] == env]
    hparam = hp_key[len("DER_"):] if agent == "DER" else hp_key[len("DrQ_eps_"):]
    param_name = list(experiments_mapping.keys())[list(experiments_mapping.values()).index(hparam)]

    env_data = env_data.rename(columns={"sweep": param_name})
    if scale == "100k":
      env_data = env_data[env_data["step"] <= 10]
    else:
      env_data = env_data[env_data["step"] <= 40]
    
    ax = axes[row, col]
    sns.lineplot(x='step', y='val', hue=param_name, data=env_data, ax=ax)
    title = hp_key[len(agent) + 1:]
    ax.set_title(title, fontsize=22)
    ylabel = 'Returns' if col == 0 else ""
    ax.set_ylabel(ylabel, fontsize=18)
    xlabel = 'Step' if row == num_rows - 1 else ""
    ax.set_xlabel(xlabel, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(prop=dict(size=18))
    col += 1
    if col == num_cols:
        col = 0
        row += 1
  return fig


def plot_hparam(agent, param, scale):
  num_cols = 7
  num_rows = 4
  fig, axes = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 8 * num_rows))
  data_path = f'data/{scale}_experiments/curves_all_games/{param}.pickle'
  col, row = 0, 0
  if "num_atoms" == param and agent != "DER":
      return fig
  with open(data_path, mode='rb') as f:
      data = pickle.load(f)
  
  keys = sorted(list(data.keys()))
  hp_key = keys[0] if agent == "DER" else keys[1]

  param_name = list(experiments_mapping.keys())[list(experiments_mapping.values()).index(param)]
  
  for env in np.unique(data[hp_key]['returns']['env']):
    env_data = data[hp_key]['returns'][data[hp_key]['returns']['env'] == env]
    env_data = env_data.rename(columns={"sweep": param_name})
    ax = axes[row, col]
    if scale == "100k":
      env_data = env_data[env_data["step"] <= 10]
    else:
      env_data = env_data[env_data["step"] <= 40]
    sns.lineplot(x='step', y='val', hue=param_name, data=env_data, ax=ax)
    
    title = env
    ax.set_title(title, fontsize=22)
    ylabel = 'Returns' if col == 0 else ""
    ax.set_ylabel(ylabel, fontsize=18)
    xlabel = 'Step' if row == num_rows - 1 else ""
    ax.set_xlabel(xlabel, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    if col==2 and row==0:
      h, l = ax.get_legend_handles_labels()
      ph = [plt.plot([], marker="", ls="")[0]]
      handles = ph + h
      labels = [param_name] + l
      print(l, len(h))
      ax.legend(handles, labels, prop={'size': 24}, bbox_to_anchor=(2.8, 1.4), ncol=7)
      for legobj in ax.get_legend().legendHandles:
        legobj.set_linewidth(6.0)
    else:
      ax.get_legend().remove()
    
    col += 1
    if col == num_cols:
        col = 0
        row += 1
  return fig


def split_plot(dict_100k, dict_40M):
  fig, all_axes = plt.subplots(ncols=2, nrows=1, sharey=True,
                              gridspec_kw={'width_ratios':[1, 2]},
                              figsize=(14, 6))

  if "normalization" in dict_100k.keys():
    dict_100k["No Normalization (default)"] = dict_100k.pop("normalization")
    dict_40M["No Normalization (default)"] = dict_40M.pop("normalization")
  algorithms = list(set(dict_100k.keys()) | set(dict_40M.keys()))
  colors = zip(algorithms, sns.color_palette("pastel"))
  colors = {k:v for (k, v) in colors}
  
  _, all_axes[0] = plot_human_normalized(dict_100k, scale="100k", ax=all_axes[0], colors=colors) 
  _, all_axes[1] = plot_human_normalized(dict_40M, scale="40M", ax=all_axes[1], colors=colors)
  
  all_axes[0].legend(loc='upper left', prop={'size': 'xx-large'})

  all_axes[1].set_ylabel('')
  plot_utils._decorate_axis(all_axes[1], hrect=-4, ticklabelsize='xx-large')
  all_axes[1].legend(prop={'size': 'xx-large'})
  all_axes[1].spines['left'].set_linewidth(3)
  all_axes[1].spines['left'].set_linestyle('-.')

  fig.subplots_adjust(wspace=0.0)
  return fig


def plot_rank_correlation(all_experiments, colors=None, hp_values=None):
  ...