import os
import pickle

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from rliable import library as rly
from rliable import metrics

from plot_data import plot_game, plot_hparam, ATARI_100K_GAMES, experiments_mapping
from utils import get_agent_metric, get_this_metric


# for scale in ["100k", "40M"]:
#     for ag in ["DrQ_eps", "DER"]:
#         for game in ATARI_100K_GAMES:
#             fig = plot_game(ag, game, scale)
#             save_dir = f"figures/{scale}_experiments/game_comparison/{game}"
#             if not os.path.isdir(save_dir):
#                 os.makedirs(save_dir)
#             fig.savefig(f"{save_dir}/{ag}.png", bbox_inches='tight')


# for scale in ["100k", "40M"]:
#     for ag in ["DrQ_eps", "DER"]:
#         for k,hparam in experiments_mapping.items():
#             fig = plot_hparam(ag, hparam, scale)
#             save_dir = f"figures/{scale}_experiments/hparam_comparison/{hparam}"
#             if not os.path.isdir(save_dir):
#                 os.makedirs(save_dir)
#             fig.savefig(f"{save_dir}/{ag}.png", bbox_inches='tight')

"""
Get THIS metric between the 26 environments
"""


# pd_dict = {'Algorithm': [], 'HParam': [], 'Value': []}

# for k, hparam in experiments_mapping.items():
#     with open(f'data/100k_experiments/final_perf/{hparam}.pickle', mode='rb') as f:
#         data = pickle.load(f)
#     keys = list(data.keys())
#     for ag, hp_key in zip(["DER", "DrQ_eps"], keys):
#         if ag == "DrQ_eps" and hparam == "num_atoms":
#             continue
#         if hparam == "normalizations":
#             continue
#         hparam_val =  get_this_metric(data[hp_key])
#         pd_dict['Algorithm'].append(ag)
#         pd_dict['HParam'].append(k)
#         pd_dict['Value'].append(hparam_val[0])
# df = pd.DataFrame(pd_dict)
# print(df.columns)
# sns.barplot(data=df, x='HParam', y='Value', hue='Algorithm')
# plt.xticks(rotation = 90)
# plt.savefig("figures/100k_experiments/importance_score/THIS_environments.pdf", bbox_inches='tight')
# plt.savefig("figures/100k_experiments/importance_score/THIS_environments.png", bbox_inches='tight')

"""
Get THIS metric between the 2 agents
"""

# pd_dict = {'Data Regime': [], 'HParam': [], 'Value': []}

# for k, hparam in experiments_mapping.items():
#     data = dict()
#     if hparam == "num_atoms":
#             continue
#     if hparam == "normalizations":
#             continue
#     with open(f'data/100k_experiments/final_perf/{hparam}.pickle', mode='rb') as f:
#         data["100k"] = pickle.load(f)
#     with open(f'data/40M_experiments/final_perf/{hparam}.pickle', mode='rb') as f:
#         data["40M"] = pickle.load(f)
#     keys = list(data["100k"].keys())
#     for data_regime, hp_key in zip(["100k", "40M"], keys):
#         hparam_val =  get_agent_metric(data[data_regime][f"DrQ_eps_{hparam}"], data[data_regime][f"DER_{hparam}"])
#         pd_dict['Data Regime'].append(data_regime)
#         pd_dict['HParam'].append(k)
#         pd_dict['Value'].append(hparam_val[0])
# df = pd.DataFrame(pd_dict)
# sns.barplot(data=df, x='HParam', y='Value', hue='Data Regime')
# plt.xticks(rotation = 90)
# plt.savefig("figures/split/importance_score/THIS_agents.pdf", bbox_inches='tight')
# plt.savefig("figures/split/importance_score/THIS_agents.png", bbox_inches='tight')

# for data_regime in ["100k", "40M"]:
#     sns.barplot(data=df[df['Data Regime'] == data_regime], x='HParam', y='Value')
#     plt.xticks(rotation = 90)
#     plt.savefig(f"figures/{data_regime}_experiments/importance_score/THIS_agents.pdf", bbox_inches='tight')
#     plt.savefig(f"figures/{data_regime}_experiments/importance_score/THIS_agents.pdf", bbox_inches='tight')

"""THIS metric between data regimes"""
pd_dict = {'Agent': [], 'HParam': [], 'Value': []}

for k, hparam in experiments_mapping.items():
    if hparam == "num_atoms":
        continue
    if hparam == "normalizations":
        continue
    data = dict()
    with open(f'data/100k_experiments/final_perf/{hparam}.pickle', mode='rb') as f:
        data["100k"] = pickle.load(f)
    with open(f'data/40M_experiments/final_perf/{hparam}.pickle', mode='rb') as f:
        data["40M"] = pickle.load(f)
    keys = list(data["100k"].keys())
    for agent, hp_key in zip(["DrQ_eps", "DER"], keys):
        hparam_val =  get_agent_metric(data["100k"][f"{agent}_{hparam}"],
                                       data["40M"][f"{agent}_{hparam}"])
        pd_dict['Agent'].append(agent)
        pd_dict['HParam'].append(k)
        pd_dict['Value'].append(hparam_val[0])
df = pd.DataFrame(pd_dict)
sns.barplot(data=df, x='HParam', y='Value', hue='Agent')
plt.xticks(rotation = 90)
plt.savefig("figures/split/importance_score/THIS_data_regimes.pdf", bbox_inches='tight')
plt.savefig("figures/split/importance_score/THIS_data_regimes.png", bbox_inches='tight')