import os
import pickle

import matplotlib.pyplot as plt
from plot_data import plot, plot_human_normalized #, plot_all_games
import seaborn as sns

agents = ["DrQ_eps", "DER"]

experiments_mapping = { "Activation Function": "layer_funct",
                        "Adam's epsilon": "epsilon",
                        "Batch Size": "batch_sizes",
                        "Convolutional Normalization": "convs_normalization", 
                        "Dense Normalization": "normalizations",
                        "Discount Factor": "gammas",
                        "Learning Rate": "learning_rate",
                        "Minimum Replay History": "min_replay_history",
                        "Number of Atoms": "num_atoms", 
                        "Number of Convolutional Layers": "convs", 
                        # "Number of Dense Layers": "depths",
                        "Replay Capacity": "replay_capacity",
                        "Reward Clipping": "clip_rewards",
                        "Target Update Period": "target_update_periods",
                        "Update Horizon": "update_horizon",
                        "Update Period": "update_periods",
                        "Weight Decay": "weightdecay",
                        "Width": "widths",
                    }

for hyperparameter, hyp in experiments_mapping.items():
    # if hyp in ["gammas", "layer_funct", "convs_normalization", "min_replay_history", "num_atoms", "update_horizon"]:
    #     shim = "40M_experiments"
    #     shim2 = "split"
    # else:
    #     shim = "100k_experiments"
    #     shim2 = "100k_experiments"

    shims = ["100k_experiments", "40M_experiments"]
    
    for shim in shims:
        colors = None
        try:
            with open(f'data/{shim}/final_perf/{hyp}.pickle', mode='rb') as f:
                data = pickle.load(f)
        except:
            data = None

        try:
            with open(f'data/{shim}/human_normalized_curve/{hyp}.pickle', mode='rb') as f:
                data2 = pickle.load(f)

        except:
            data2 = None

        # with open(f'data/{shim}/curves_all_games/{hyp}.pickle', mode='rb') as f:
        #     data3 = pickle.load(f)


        for ag in agents:
            if ag == "DrQ_eps" and hyp == "num_atoms":
                continue
            if data is not None:
                print(f"working!\n{ag} and {hyp}\n")
                data[f'{ag}_{hyp}'] = {k.split("_")[-1]:v for (k, v) in data[f'{ag}_{hyp}'].items()}
                hp_values = list(data[f'{ag}_{hyp}'].keys())

                colors = zip(hp_values, sns.color_palette("pastel"))
                colors = {k:v for (k, v) in colors}

                fig = plot(data[f'{ag}_{hyp}'], colors, hp_values)
                
                
                save_dir = f"figures/{shim}/IQM/{hyperparameter}"
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                fig.savefig(f"{save_dir}/{ag}.png", bbox_inches='tight')

            if data2 is not None:
                fig2, _ = plot_human_normalized(
                                                data2[f'{ag}_{hyp}'],
                                                scale=shim.split('_')[0],
                                                colors=colors
                                                )
            
                save_dir = f"figures/{shim}/HNS/{hyperparameter}"
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                fig2.savefig(f"{save_dir}/{ag}.png", bbox_inches='tight')

            # fig3 = plot_all_games(data3[f'{ag}_{hyp}'])

            # save_dir = f"figures/{shim}/all_games/{hyperparameter}"
            # if not os.path.isdir(save_dir):
            #     os.makedirs(save_dir)
            # fig3.savefig(f"{save_dir}/{ag}.png")
            plt.close()

