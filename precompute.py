import os

from plot_data import plot_game, plot_hparam, ATARI_100K_GAMES, experiments_mapping


# for scale in ["100k", "40M"]:
#     for ag in ["DrQ_eps", "DER"]:
#         for game in ATARI_100K_GAMES:
#             fig = plot_game(ag, game, scale)
#             save_dir = f"figures/{scale}_experiments/game_comparison/{game}"
#             if not os.path.isdir(save_dir):
#                 os.makedirs(save_dir)
#             fig.savefig(f"{save_dir}/{ag}.png", bbox_inches='tight')


for scale in ["100k", "40M"]:
    for ag in ["DrQ_eps", "DER"]:
        for k,hparam in experiments_mapping.items():
            fig = plot_hparam(ag, hparam, scale)
            save_dir = f"figures/{scale}_experiments/hparam_comparison/{hparam}"
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            fig.savefig(f"{save_dir}/{ag}.png", bbox_inches='tight')
