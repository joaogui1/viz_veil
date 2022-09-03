import streamlit as st

import pickle

from plot_data import plot_score_hist, plot, plot_human_normalized

plot = st.experimental_memo(plot)
plot_human_normalized = st.experimental_memo(plot_human_normalized)


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
                        "Update Horizon": "update_horizon",
                        "Weight Decay": "weightdecay",
                    }
hyperparameter = st.radio("Hyperparameter", options=experiments_mapping.keys())
hyp = experiments_mapping[hyperparameter]

with open(f'data/final_perf/{hyp}.pickle', mode='rb') as f:
    data = pickle.load(f)

try:
    with open(f'data/human_normalized_curve/{hyp}.pickle', mode='rb') as f:
        data2 = pickle.load(f)
except:
    data2 = None


for ag in agents:
    if ag == "DrQ_eps" and hyp == "num_atoms":
        continue
    fig = plot(data[f'{ag}_{hyp}'])
    st.pyplot(fig)
    if data2 is not None:
        fig2 = plot_human_normalized(data2[f'{ag}_{hyp}'])
        st.pyplot(fig2)