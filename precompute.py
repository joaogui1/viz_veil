import os
import pickle

from matplotlib import pyplot as plt

from plot_data import plot_game, plot_hparam, ATARI_100K_GAMES, experiments_mapping
from utils import kendall_w


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

drq_w = dict()
der_w = dict()
W_dict = {"DrQ_eps" : dict(),
          "DER": dict()}

for k, hparam in experiments_mapping.items():
    with open(f'data/40M_experiments/final_perf/{hparam}.pickle', mode='rb') as f:
        data = pickle.load(f)
    keys = list(data.keys())
    for ag, hp_key in zip(["DER", "DrQ_eps"], keys):
        if ag == "DrQ_eps" and hparam == "num_atoms":
            continue
        if hparam == "normalizations":
            continue
        print(ag, hp_key)
        W_dict[ag][k] = kendall_w(data[hp_key])

fig = plt.bar(list(W_dict["DER"].keys()), list(W_dict["DER"].values()))
plt.xticks(list(W_dict["DER"].keys()), rotation='vertical')
plt.show()
# fig = plot_hparam(ag, hparam, scale)
# save_dir = f"figures/40M_experiments/kendall_w/{hparam}"
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# fig.savefig(f"{save_dir}/{ag}.png", bbox_inches='tight')
