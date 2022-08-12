import streamlit as st

from functools import partial
import pickle

import matplotlib.pyplot as plt
from plot_data import plot_score_hist, plot



st.title("Lifting the Veil")
# text = st.text_input()
agents = st.multiselect("Agents", options=["DrQ_eps", "DER"])
print(agents)
hyperparameter = st.radio("Hyperparameter", 
                        options=["Batch Size", 
                        "Number of Dense Layers", 
                        "Number of Convolutional Layers",
                        "Reward Clipping"]
                        )

experiments_mapping = { "Activation Function": "layer_funct",
                        "Adam's epsilon": "epsilon",
                        "Batch Size": "batch_sizes",
                        "Convolutional Normalization": "convs_normalizations", 
                        "Dense Normalization": "normalizations",
                        "Discount Factor": "gammas",
                        "Learning Rate": "learning_rate",
                        "Number of Atoms": "num_atoms", 
                        "Number of Convolutional Layers": "convs", 
                        "Number of Dense Layers": "depths",
                        "Reward Clipping": "clip_rewards"
                    }

hyp = experiments_mapping[hyperparameter]

with open(f'data/{hyp}.pickle', mode='rb') as f:
    data = pickle.load(f)
for ag in agents:
    fig = plot_score_hist(data[f'{ag}_{hyp}'], bins=4, N=6, figsize=(28, 11))
    fig.subplots_adjust(hspace=0.85, wspace=0.17)
    st.pyplot(fig)

    fig2 = plot(data[f'{ag}_{hyp}'])
    st.pyplot(fig2)