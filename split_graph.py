import os
import pickle

import matplotlib.pyplot as plt
from plot_data import plot_iqm, split_plot

agents = ["DrQ_eps", "DER"]

experiments_mapping = { 
                        "Adam's epsilon": "epsilon",
                        "Batch Size": "batch_sizes",
                        "Conv. Activation Function": "layer_funct_conv",
                        "Convolutional Normalization": "normalizations_convs", 
                        "Convolutional Width": "CNN_widths",
                        "Dense Activation Function": "layer_funct_dense",
                        "Dense Normalization": "normalizations",
                        "Dense Width": "widths",
                        "Discount Factor": "gammas",
                        "Learning Rate": "learning_rate",
                        "Minimum Replay History": "min_replay_history",
                        "Number of Atoms": "num_atoms", 
                        "Number of Convolutional Layers": "convs", 
                        "Number of Dense Layers": "depths",
                        # "Replay Capacity": "replay_capacity",
                        "Reward Clipping": "clip_rewards",
                        "Target Update Period": "target_update_periods",
                        "Update Horizon": "update_horizon",
                        "Update Period": "update_periods",
                        "Weight Decay": "weightdecay",
                    }

for hyperparameter, hyp in experiments_mapping.items():
    shims = ["100k_experiments", "40M_experiments"]

    with open(f'data/{shims[0]}/human_normalized_curve/{hyp}.pickle', mode='rb') as f:
        data100k = pickle.load(f)
    
    with open(f'data/{shims[1]}/human_normalized_curve/{hyp}.pickle', mode='rb') as f:
        data40M = pickle.load(f)
    
    # with open(f'data/{shims[0]}/final_perf/{hyp}.pickle', mode='rb') as f:
    #     final_perf_100k = pickle.load(f)
    
    # with open(f'data/{shims[1]}/final_perf/{hyp}.pickle', mode='rb') as f:
    #     final_perf_40M = pickle.load(f)


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

        # final_perf_100k[f'{ag}_{hyp}'] = {k.split("_")[-1]:v for (k, v) in final_perf_100k[f'{ag}_{hyp}'].items()}
        # final_perf_40M[f'{ag}_{hyp}'] = {k.split("_")[-1]:v for (k, v) in final_perf_40M[f'{ag}_{hyp}'].items()}
        # fig_iqm = plot_iqm(final_perf_100k[f'{ag}_{hyp}'], final_perf_40M[f'{ag}_{hyp}'])
        
        # save_dir = f"figures/split/IQM/{hyperparameter}"
        # if not os.path.isdir(save_dir):
        #     os.makedirs(save_dir)
        # fig_iqm.savefig(f"{save_dir}/{ag}.png", bbox_inches='tight')
        # fig_iqm.savefig(f"{save_dir}/{ag}.pdf", bbox_inches='tight')

        plt.close()

