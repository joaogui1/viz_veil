import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_score_hist(data, bins=20, figsize=(28, 14), 
                    fontsize='xx-large', N=6, extra_row=1,
                    names=None):
  keys = data.keys()
  data_r = dict()
  for key in keys:
    data_r[f'{key}'] = data[key]

  num_tasks = data[list(keys)[0]].shape[1]
  if names is None:
    names = ATARI_100K_GAMES
  N1 = (num_tasks// N) + extra_row
  #print('N1:', N1)
  fig, ax = plt.subplots(nrows=N1, ncols=N, figsize=figsize)
  for i in range(N):
    for j in range(N1):
      idx = j * N + i
      if idx < num_tasks:
        ax[j, i].set_title(names[idx], fontsize=fontsize)
        for key in keys:
          data_r[f'{key}'] = data[f'{key}'][:, idx]

        g = sns.histplot(data_r, bins=bins, ax=ax[j,i], kde=False, legend= False)
        if i ==1 and j==4:
          g= sns.histplot(data_r, bins=bins, ax=ax[j,i], kde=False, legend= True)
          sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

      else:
        ax[j, i].axis('off')
      
      decorate_axis(ax[j, i], wrect=5, hrect=5, labelsize='xx-large')
      ax[j, i].xaxis.set_major_locator(plt.MaxNLocator(4))
      if idx % N == 0:
        ax[j, i].set_ylabel('Count', size=fontsize)
      else:
        ax[j, i].yaxis.label.set_visible(False)
      ax[j, i].grid(axis='y', alpha=0.1)
  return fig