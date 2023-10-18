import os
import pickle

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import kendalltau

from rliable import library as rly
from rliable import metrics

from plot_data import plot_game, plot_hparam, ATARI_100K_GAMES, experiments_mapping
from utils import get_metric


for scale in ["100k", "40M"]:
    for ag in ["DrQ_eps", "DER"]:
        for game in ATARI_100K_GAMES:
            fig = plot_game(ag, game, scale)
            save_dir = f"figures/{scale}_experiments/game_comparison/{game}"
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            fig.savefig(f"{save_dir}/{ag}.png", bbox_inches='tight')


# for scale in ["100k", "40M"]:
#     for ag in ["DrQ_eps", "DER"]:
#         for k,hparam in experiments_mapping.items():
#             fig = plot_hparam(ag, hparam, scale)
#             save_dir = f"figures/{scale}_experiments/hparam_comparison/{hparam}"
#             if not os.path.isdir(save_dir):
#                 os.makedirs(save_dir)
#             fig.savefig(f"{save_dir}/{ag}.png", bbox_inches='tight')

# drq_w = dict()
# der_w = dict()
# W_dict = {"DrQ_eps" : dict(),
#           "DER": dict()}

# for k, hparam in experiments_mapping.items():
#     with open(f'data/40M_experiments/final_perf/{hparam}.pickle', mode='rb') as f:
#         data = pickle.load(f)
#     keys = list(data.keys())
#     for ag, hp_key in zip(["DER", "DrQ_eps"], keys):
#         if ag == "DrQ_eps" and hparam == "num_atoms":
#             continue
#         if hparam == "normalizations":
#             continue
#         print(ag, hp_key)
#         W_dict[ag][k] = get_metric(data[hp_key])
# for ag in ["DER", "DrQ_eps"]:
#     fig, ax = plt.subplots()
#     ax.bar(list(W_dict[ag].keys()), [v[0] for v in W_dict[ag].values()] 
#         #    ,yerr=[v[1] for v in W_dict[ag].values()]
#            )
#     plt.xticks(list(W_dict[ag].keys()), rotation='vertical')
#     save_dir = f"figures/40M_experiments/importance_score/"
#     if not os.path.isdir(save_dir):
#         os.makedirs(save_dir)
#     ax.figure.savefig(f"{save_dir}/{ag}.pdf", bbox_inches='tight')

"""
Get Kendall's Tau between the 2 agents
"""
# get_iqm = lambda data: rly.get_interval_estimates(data, lambda x: np.array([metrics.aggregate_iqm(x)]),reps=500)[0]

# kendall_taus = dict()
# for k, hparam in experiments_mapping.items():
#     if hparam == "num_atoms":
#         continue
#     with open(f"data/40M_experiments/final_perf/{hparam}.pickle", mode='rb') as f:
#         data = pickle.load(f)
#     scores_DER, scores_DrQ = data[f"DER_{hparam}"], data[f"DrQ_eps_{hparam}"]
#     iqm_DER = get_iqm(scores_DER)
#     iqm_DrQ = get_iqm(scores_DrQ)
#     tau = kendalltau(list(iqm_DER.values()), list(iqm_DrQ.values())).correlation
#     print(tau, hparam)
#     kendall_taus[k] = tau
# fig, ax = plt.subplots()
# ax.bar(list(kendall_taus.keys()), list(kendall_taus.values()))
# plt.xticks(list(kendall_taus.keys()), rotation='vertical')
# ax.figure.savefig("figures/kendall_tau_agents.pdf", bbox_inches='tight')