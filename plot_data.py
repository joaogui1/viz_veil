import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
import seaborn as sns

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

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

def plot(all_experiments, colors=None, hp_values=None):
  IQM = lambda x: metrics.aggregate_iqm(x) # Interquartile Mean
  OG = lambda x: metrics.aggregate_optimality_gap(x, 1.0) # Optimality Gap
  MEAN = lambda x: metrics.aggregate_mean(x)
  MEDIAN = lambda x: metrics.aggregate_median(x)

  aggregate_func = lambda x: np.array([MEDIAN(x), IQM(x), MEAN(x), OG(x)])
  aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
      all_experiments, aggregate_func, reps=50000)

  fig, _ = plot_utils.plot_interval_estimates(
      aggregate_scores, 
      aggregate_interval_estimates,
      metric_names = ['Median', 'IQM', 'Mean', 'Optimality Gap'],
      algorithms=hp_values,
      colors=colors,
      xlabel_y_coordinate=-0.8,
      xlabel='Human Normalized Score')
  return fig


my_str="""alien 7127.70 227.80 297638.17 ± 37054.55 464232.43 ± 7988.66 741812.63
amidar 1719.50 5.80 29660.08 ± 880.39 31331.37 ± 817.79 28634.39
assault 742.00 222.40 67212.67 ± 6150.59 110100.04 ± 346.06 143972.03
asterix 8503.30 210.00 991384.42 ± 9493.32 999354.03 ± 12.94 998425.00
asteroids 47388.70 719.10 150854.61 ± 16116.72 431072.45 ± 1799.13 6785558.64
atlantis 29028.10 12850.00 1528841.76 ± 28282.53 1660721.85 ± 14643.83 1674767.20
bank_heist 753.10 14.20 23071.50 ± 15834.73 27117.85 ± 963.12 1278.98
battle_zone 37187.50 2360.00 934134.88 ± 38916.03 992600.31 ± 1096.19 848623.00
beam_rider 16926.50 363.90 300509.80 ± 13075.35 390603.06 ± 23304.09 4549993.53
berzerk 2630.40 123.70 61507.83 ± 26539.54 77725.62 ± 4556.93 85932.60
bowling 160.70 23.10 251.18 ± 13.22 161.77 ± 99.84 260.13
boxing 12.10 0.10 100.00 ± 0.00 100.00 ± 0.00 100.00
breakout 30.50 1.70 790.40 ± 60.05 863.92 ± 0.08 864.00
centipede 12017.00 2090.90 412847.86 ± 26087.14 908137.24 ± 7330.99 1159049.27
chopper_command 7387.80 811.00 999900.00 ± 0.00 999900.00 ± 0.00 991039.70
crazy_climber 35829.40 10780.50 565909.85 ± 89183.85 729482.83 ± 87975.74 458315.40
defender 18688.90 2874.50 677642.78 ± 16858.59 730714.53 ± 715.54 839642.95
demon_attack 1971.00 152.10 143161.44 ± 220.32 143913.32 ± 92.93 143964.26
double_dunk -16.40 -18.60 23.93 ± 0.06 24.00 ± 0.00 23.94
enduro 860.50 0.00 2367.71 ± 8.69 2378.66 ± 3.66 2382.44
fishing_derby -38.70 -91.70 86.97 ± 3.25 90.34 ± 2.66 91.16
freeway 29.60 0.00 32.59 ± 0.71 34.00 ± 0.00 33.03
frostbite 4334.70 65.20 541280.88 ± 17485.76 309077.30 ± 274879.03 631378.53
gopher 2412.50 257.60 117777.08 ± 3108.06 129736.13 ± 653.03 130345.58
gravitar 3351.40 173.00 19213.96 ± 348.25 21068.03 ± 497.25 6682.70
hero 30826.40 1027.00 114736.26 ± 49116.60 49339.62 ± 4617.76 49244.11
ice_hockey 0.90 -11.20 63.64 ± 6.48 86.59 ± 0.59 67.04
jamesbond 302.80 29.00 135784.96 ± 9132.28 158142.36 ± 904.45 41063.25
kangaroo 3035.00 52.00 24034.16 ± 12565.88 18284.99 ± 817.25 16763.60
krull 2665.50 1598.00 251997.31 ± 20274.39 245315.44 ± 48249.07 269358.27
kung_fu_master 22736.30 258.50 206845.82 ± 11112.10 267766.63 ± 2895.73 204824.00
montezuma_revenge 4753.30 0.00 9352.01 ± 2939.78 3000.00 ± 0.00 0.00
ms_pacman 6951.60 307.30 63994.44 ± 6652.16 62595.90 ± 1755.82 243401.10
name_this_game 8049.00 2292.30 54386.77 ± 6148.50 138030.67 ± 5279.91 157177.85
phoenix 7242.60 761.40 908264.15 ± 28978.92 990638.12 ± 6278.77 955137.84
pitfall 6463.70 -229.40 18756.01 ± 9783.91 0.00 ± 0.00 0.00
pong 14.60 -20.70 20.67 ± 0.47 21.00 ± 0.00 21.00
private_eye 69571.30 24.90 79716.46 ± 29515.48 40700.00 ± 0.00 15299.98
qbert 13455.00 163.90 580328.14 ± 151251.66 777071.30 ± 190653.94 72276.00
riverraid 17118.00 1338.50 63318.67 ± 5659.55 93569.66 ± 13308.08 323417.18
road_runner 7845.00 11.50 243025.80 ± 79555.98 593186.78 ± 88650.69 613411.80
robotank 11.90 2.20 127.32 ± 12.50 144.00 ± 0.00 131.13
seaquest 42054.70 68.40 999997.63 ± 1.42 999999.00 ± 0.00 999976.52
skiing -4336.90 -17098.10 -4202.60 ± 607.85 -3851.44 ± 517.52 -29968.36
solaris 12326.70 1236.30 44199.93 ± 8055.50 67306.29 ± 10378.22 56.62
space_invaders 1668.70 148.00 48680.86 ± 5894.01 67898.71 ± 1744.74 74335.30
star_gunner 10250.00 664.00 839573.53 ± 67132.17 998600.28 ± 218.66 549271.70
surround 6.50 -10.00 9.50 ± 0.19 10.00 ± 0.00 9.99
tennis -8.30 -23.80 23.84 ± 0.10 24.00 ± 0.00 0.00
time_pilot 5229.20 3568.00 405425.31 ± 17044.45 460596.49 ± 3139.33 476763.90
tutankham 167.60 11.40 2354.91 ± 3421.43 483.78 ± 37.90 491.48
up_n_down 11693.20 533.40 623805.73 ± 23493.75 702700.36 ± 8937.59 715545.61
venture 1187.50 0.00 2623.71 ± 442.13 2258.93 ± 29.90 0.40
video_pinball 17667.90 0.00 992340.74 ± 12867.87 999645.92 ± 57.93 981791.88
wizard_of_wor 4756.50 563.50 157306.41 ± 16000.00 183090.81 ± 6070.10 197126.00
yars_revenge 54576.90 3092.90 998532.37 ± 375.82 999807.02 ± 54.85 553311.46
zaxxon 9173.30 32.50 249808.90 ± 58261.59 370649.03 ± 19761.32 725853.90"""


human_scores = my_str.split('\n')


def plot_human_normalized(all_experiments, scale='100k', ax=None, colors=None):
  all_experiments = {k.split("_")[-1]:v for (k, v) in all_experiments.items()}
  if "normalization" in all_experiments.keys():
    all_experiments["No Normalization"] = all_experiments.pop("normalization")
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
      frames+1, iqm_scores, iqm_cis, 
      algorithms=algorithms,
      xlabel=f'Number of Frames (in {"thousands" if scale == "100k" else "millions"})',
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

def plot_game(agent, env):
  num_cols = 6
  num_rows = 3
  fig, axes = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 8 * num_rows))
  data_path = 'data/100k_experiments/curves_all_games/*.pickle'
  col, row = 0, 0
  for filename in glob.glob(data_path):
    if "optimizer" in filename:
        continue
    if "atoms" in filename and agent != "DER":
        continue
    with open(filename, mode='rb') as f:
        data = pickle.load(f)
    
    keys = sorted(list(data.keys()))
    hp_key = keys[0] if agent == "DER" else keys[1]
    env_data = data[hp_key]['returns'][data[hp_key]['returns']['env'] == env]
    
    ax = axes[row, col]
    sns.lineplot(x='step', y='val', hue='sweep', data=env_data, ax=ax)
    title = hp_key[len(agent) + 1:]#hp_key.removeprefix(agent + '_')
    ax.set_title(title, fontsize=22)
    ax.set_ylabel('Returns', fontsize=18)
    xlabel = 'Step'
    ax.set_xlabel(xlabel, fontsize=18)
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
    dict_100k["No Normalization"] = dict_100k.pop("normalization")
    dict_40M["No Normalization"] = dict_40M.pop("normalization")
  algorithms = list(dict_100k.keys())
  colors = zip(algorithms, sns.color_palette("pastel"))
  colors = {k:v for (k, v) in colors}
  
  _, all_axes[0] = plot_human_normalized(dict_100k, scale="100k", ax=all_axes[0], colors=colors)

  axes = all_axes[1:]
  
  _, all_axes[1] = plot_human_normalized(dict_40M, scale="40M", ax=all_axes[1], colors=colors)
  
  all_axes[0].legend(loc='upper left', prop={'size': 'large'})

  for ax in axes:
    ax.set_ylabel('')
    plot_utils._decorate_axis(ax, hrect=-4, ticklabelsize='xx-large')
    ax.spines['left'].set_linewidth(3)
    ax.spines['left'].set_linestyle('-.')

  fig.subplots_adjust(wspace=0.0)
  return fig


def inset_plot(dict_100k, dict_40M):
  fig, ax = plt.subplots(figsize=(14, 6))

  ax2 = ax.inset_axes([0.1, 0.7, 0.55/2, 0.5/2])

  # ax2 = plt.axes([0, 0, 1, 1])
  # ip = InsetPosition(ax, [0.1, 0.7, 0.55/2, 0.5/2])  # Second parameter is [xpos, ypos, width, height]
  # ax2.set_axes_locator(ip)
  mark_inset(ax, ax2, loc1=2, loc2=4, fc='none', ec='0.5')

  algorithms = [k.split("_")[-1] for k in dict_100k.keys()]
  colors = zip(algorithms, sns.color_palette("pastel"))
  colors = {k:v for (k, v) in colors}
  
  _, ax2 = plot_human_normalized(dict_100k, scale="100k", ax=ax2, colors=colors)

  # ax2.set_xlim(0, 10)
  # ax2.set_ylim(0.0, 1.0)

  _, ax = plot_human_normalized(dict_40M, scale="40M", ax=ax, colors=colors)
  
  ax2.set_ylabel('')
  ax.get_legend().remove()
  plot_utils._decorate_axis(ax, hrect=-4, ticklabelsize='xx-large')

  fig.subplots_adjust(wspace=0.0)
  return fig