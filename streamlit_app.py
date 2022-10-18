import streamlit as st

import pickle

from plot_data import plot, plot_human_normalized, plot_all_games

plot = st.experimental_memo(plot)
plot_human_normalized = st.experimental_memo(plot_human_normalized)

st.set_page_config(layout="wide")

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
                        "Replay Capacity": "replay_capacity",
                        "Reward Clipping": "clip_rewards",
                        "Target Update Period": "target_update_periods",
                        "Update Horizon": "update_horizon",
                        "Update Period": "update_periods",
                        "Weight Decay": "weightdecay",
                        "Width": "widths",
                    }
hyperparameter = st.radio("Hyperparameter", options=experiments_mapping.keys())
hyp = experiments_mapping[hyperparameter]
# if hyp in ["gammas", "layer_funct", "normalization"]:
#     shim = "40M_experiments"
# else:
shim = "100k_experiments"
with open(f'data/{shim}/final_perf/{hyp}.pickle', mode='rb') as f:
    data = pickle.load(f)

try:
    with open(f'data/{shim}/human_normalized_curve/{hyp}.pickle', mode='rb') as f:
        data2 = pickle.load(f)
except:
    data2 = None

with open(f'data/100k_experiments/curves_all_games/{hyp}.pickle', mode='rb') as f:
    data3 = pickle.load(f)

col1, col2 = st.columns(2)
ag_col = {"DrQ_eps": col1, "DER": col2}


col1.subheader('DrQ Epsilon')

col2.subheader('DER')

for ag in agents:
    if ag == "DrQ_eps" and hyp == "num_atoms":
        continue
    fig = plot(data[f'{ag}_{hyp}'])
    ag_col[ag].pyplot(fig)
    if data2 is not None:
        fig2 = plot_human_normalized(data2[f'{ag}_{hyp}'])
        ag_col[ag].pyplot(fig2)

    fig3 = plot_all_games(data3[f'{ag}_{hyp}'])
    ag_col[ag].pyplot(fig3)
