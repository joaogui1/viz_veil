import os
import pickle

import matplotlib.pyplot as plt
from plot_data import split_plot, experiments_mapping, split_plot_iqm

agents = ["DrQ_eps", "DER"]


for hyperparameter, hyp in experiments_mapping.items():
    shims = ["100k_experiments", "40M_experiments"]

    with open(f'data/{shims[0]}/human_normalized_curve/{hyp}.pickle', mode='rb') as f:
        data100k = pickle.load(f)
    
    with open(f'data/{shims[1]}/human_normalized_curve/{hyp}.pickle', mode='rb') as f:
        data40M = pickle.load(f)
    
    try:
        with open(f'data/{shims[0]}/final_perf/{hyp}.pickle', mode='rb') as f:
            final_perf_100k = pickle.load(f)
    except:
        print(f"data/{shims[0]}/final_perf/{hyp}.pickle not found")
        continue
    
    try:
        with open(f'data/{shims[1]}/final_perf/{hyp}.pickle', mode='rb') as f:
            final_perf_40M = pickle.load(f)
    except:
        print(f"data/{shims[1]}/final_perf/{hyp}.pickle not found")
        continue


    for ag in agents:
        if ag == "DrQ_eps" and hyp == "num_atoms":
            continue
        data100k[f'{ag}_{hyp}'] = {k.split("_")[-1]:v for (k, v) in data100k[f'{ag}_{hyp}'].items()}
        data40M[f'{ag}_{hyp}'] = {k.split("_")[-1]:v for (k, v) in data40M[f'{ag}_{hyp}'].items()}
        fig = split_plot(data100k[f'{ag}_{hyp}'], data40M[f'{ag}_{hyp}'])
        plt.xlabel("Number of Frames (in Millions)", x=0.2)

        save_dir = f"figures/test/split/HNS/{hyperparameter}"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        fig.savefig(f"{save_dir}/{ag}.png", bbox_inches='tight')
        fig.savefig(f"{save_dir}/{ag}.pdf", bbox_inches='tight')

        print(f"{ag}_{hyp}")
        # final_perf_100k[f'{ag}_{hyp}'] = {k.split("_")[-1]:v for (k, v) in final_perf_100k[f'{ag}_{hyp}'].items()}
        # final_perf_40M[f'{ag}_{hyp}'] = {k.split("_")[-1]:v for (k, v) in final_perf_40M[f'{ag}_{hyp}'].items()}
        # fig_iqm = split_plot_iqm(final_perf_100k[f'{ag}_{hyp}'],
        #                          final_perf_40M[f'{ag}_{hyp}'])

        # save_dir = f"figures/split/IQM/{hyperparameter}"
        # if not os.path.isdir(save_dir):
        #     os.makedirs(save_dir)
        # fig_iqm.savefig(f"{save_dir}/{ag}.png", bbox_inches='tight')
        # fig_iqm.savefig(f"{save_dir}/{ag}.pdf", bbox_inches='tight')

        plt.close()

