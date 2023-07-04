import os
import pickle

import matplotlib.pyplot as plt
from plot_data import plot_iqm, plot_human_normalized
import seaborn as sns
from plot_data import experiments_mapping

agents = ["DrQ_eps", "DER"]

for hyperparameter, hyp in experiments_mapping.items():
    shims = ["100k_experiments", "40M_experiments"]
    colors = None
    for ag in agents:

        for shim in shims:
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

            if ag == "DrQ_eps" and hyp == "num_atoms":
                continue

            if data is not None:
                data[f'{ag}_{hyp}'] = {k.split("_")[-1]:v for (k, v) in data[f'{ag}_{hyp}'].items()}
                if "normalization" in data[f'{ag}_{hyp}'].keys():
                    data[f'{ag}_{hyp}']["No Normalization"] = data[f'{ag}_{hyp}'].pop("normalization")
                hp_values = list(data[f'{ag}_{hyp}'].keys())
                colors = zip(hp_values, sns.color_palette("pastel"))
                colors = {k:v for (k, v) in colors}

                fig = plot_iqm(data[f'{ag}_{hyp}'], colors, hp_values)
                
                
                save_dir = f"figures/{shim}/IQM/{hyperparameter}"
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                fig.savefig(f"{save_dir}/{ag}.pdf", bbox_inches='tight')

            if data2 is not None:
                data2[f'{ag}_{hyp}'] = {k.split("_")[-1]:v for (k, v) in data2[f'{ag}_{hyp}'].items()}
                if "normalization" in data2[f'{ag}_{hyp}'].keys():
                    data2[f'{ag}_{hyp}']["No Normalization"] = data2[f'{ag}_{hyp}'].pop("normalization")
                hp_values = list(data2[f'{ag}_{hyp}'].keys())
                colors = zip(hp_values, sns.color_palette("pastel"))
                colors = {k:v for (k, v) in colors}
                fig2, _ = plot_human_normalized(
                                                data2[f'{ag}_{hyp}'],
                                                scale=shim.split('_')[0],
                                                colors=colors
                                                ) 
                save_dir = f"figures/{shim}/HNS/{hyperparameter}"
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                fig2.savefig(f"{save_dir}/{ag}.pdf", bbox_inches='tight')

            plt.close()

