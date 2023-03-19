import os

from plot_data import plot_game, ATARI_100K_GAMES

for scale in ["100k", "40M"]:
    for ag in ["DrQ_eps", "DER"]:
        for game in ATARI_100K_GAMES:
            fig = plot_game(ag, game, scale)
            save_dir = f"figures/{scale}_experiments/game_comparison/{game}"
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            fig.savefig(f"{save_dir}/{ag}.png", bbox_inches='tight')