import streamlit as st

import pickle

from plot_data import plot_score_hist, plot

@st.experimental_memo
def plot_agent(ag, hyp, data):
    fig = plot_score_hist(data[f'{ag}_{hyp}'], bins=4, N=6, figsize=(28, 11))
    fig.subplots_adjust(hspace=0.85, wspace=0.17)
    fig2 = plot(data[f'{ag}_{hyp}'])
    return fig, fig2

st.title("Lifting the Veil")
# text = st.text_input()
agents = st.multiselect("Agents", options=["DrQ_eps", "DER"])

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
hyperparameter = st.radio("Hyperparameter", options=experiments_mapping.keys())
hyp = experiments_mapping[hyperparameter]

with open(f'data/{hyp}.pickle', mode='rb') as f:
    data = pickle.load(f)

for ag in agents:
    fig, fig2 = plot_agent(ag, hyp, data)
    st.pyplot(fig)
    st.pyplot(fig2)