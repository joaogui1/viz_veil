import os
import pickle

import matplotlib.pyplot as plt
from plot_data import split_plot

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
                        "Number of Dense Layers": "depths",
                        "Replay Capacity": "replay_capacity",
                        "Reward Clipping": "clip_rewards",
                        "Target Update Period": "target_update_periods",
                        "Update Horizon": "update_horizon",
                        "Update Period": "update_periods",
                        "Weight Decay": "weightdecay",
                        "Width": "widths",
                    }

for hyperparameter, hyp in experiments_mapping.items():
    shims = ["100k_experiments", "40M_experiments"]

    with open(f'data/{shims[0]}/human_normalized_curve/{hyp}.pickle', mode='rb') as f:
        data100k = pickle.load(f)
    
    with open(f'data/{shims[1]}/human_normalized_curve/{hyp}.pickle', mode='rb') as f:
        data40M = pickle.load(f)


    for ag in agents:
        if ag == "DrQ_eps" and hyp == "num_atoms":
            continue
        data100k[f'{ag}_{hyp}'] = {k.split("_")[-1]:v for (k, v) in data100k[f'{ag}_{hyp}'].items()}
        data40M[f'{ag}_{hyp}'] = {k.split("_")[-1]:v for (k, v) in data40M[f'{ag}_{hyp}'].items()}
        fig = split_plot(data100k[f'{ag}_{hyp}'], data40M[f'{ag}_{hyp}'])
        plt.xlabel("Number of Frames (in Millions)", x=0.2)
        
        save_dir = f"figures/split/HNS/{hyperparameter}"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        fig.savefig(f"{save_dir}/{ag}.png", bbox_inches='tight')
        fig.savefig(f"{save_dir}/{ag}.pdf", bbox_inches='tight')

        plt.close()

