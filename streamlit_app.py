import streamlit as st

import pickle

from plot_data import plot_score_hist, plot, plot_human_normalized

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
                        "Convolutional Normalization": "convs_normalization", 
                        "Dense Normalization": "normalizations",
                        "Discount Factor": "gammas",
                        "Learning Rate": "learning_rate",
                        "Minimum Replay History": "min_replay_history",
                        "Number of Atoms": "num_atoms", 
                        "Number of Convolutional Layers": "convs", 
                        "Number of Dense Layers": "depths",
                        "Reward Clipping": "clip_rewards",
                        "Update Horizon": "update_horizont",
                        "Weight Decay": "weightdecay",
                    }
hyperparameter = st.radio("Hyperparameter", options=experiments_mapping.keys())
hyp = experiments_mapping[hyperparameter]

with open(f'data/final_perf/{hyp}.pickle', mode='rb') as f:
    data = pickle.load(f)

with open(f'data/human_normalized_curve/{hyp}.pickle', mode='rb') as f:
    data2 = pickle.load(f)


for ag in agents:
    if ag == "DrQ_eps" and hyp == "num_atoms":
        continue
    fig, fig2 = plot_agent(ag, hyp, data)
    fig3 = plot_human_normalized(data2[f'{ag}_{hyp}'])
    st.pyplot(fig)
    st.pyplot(fig2)
    st.pyplot(fig3)