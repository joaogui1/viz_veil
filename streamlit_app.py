import streamlit as st

import PIL
PIL.Image.MAX_IMAGE_PIXELS = 900000000

from plot_data import plot, plot_human_normalized, plot_all_games

plot = st.experimental_memo(plot)
plot_human_normalized = st.experimental_memo(plot_human_normalized)

st.set_page_config(layout="wide", page_title="Lifting the Veil")
st.sidebar.markdown("# Main page 🌈")

st.title("Lifting the Veil")
# text = st.text_input()
# agents = st.multiselect("Agents", options=["DrQ_eps", "DER"])
agents = ["DrQ_eps", "DER"]

experiments_mapping = { "Activation Function (40M)": "layer_funct",
                        "Adam's epsilon (40M)": "epsilon",
                        "Batch Size (40M)": "batch_sizes",
                        "Convolutional Normalization (40M)": "convs_normalization", 
                        "Dense Normalization (40M)": "normalizations",
                        "Discount Factor (40M)": "gammas",
                        "Learning Rate (40M)": "learning_rate",
                        "Minimum Replay History (40M)": "min_replay_history",
                        "Number of Atoms (40M)": "num_atoms", 
                        "Number of Convolutional Layers (40M)": "convs", 
                        "Number of Dense Layers (40M)": "depths",
                        "Replay Capacity (100k)": "replay_capacity",
                        "Reward Clipping (40M)": "clip_rewards",
                        "Target Update Period (40M)": "target_update_periods",
                        "Update Horizon (40M)": "update_horizon",
                        "Update Period (40M)": "update_periods",
                        "Weight Decay (40M)": "weightdecay",
                        "Width (40M)": "widths",
                    }
hyperparameter = st.radio("Hyperparameter", options=experiments_mapping.keys())
hyp = experiments_mapping[hyperparameter]
if hyp == "replay_capacity":
    shim = "40M_experiments"
    shim2 = "split"
else:
    shim = "100k_experiments"
    shim2 = "100k_experiments"

st.header(hyperparameter)

fig1_path = f"figures/{shim}/IQM/{hyperparameter.split('(')[0][:-1]}"
# with open(f'data/{shim}/final_perf/{hyp}.pickle', mode='rb') as f:
    # data = pickle.load(f)

fig2_path = f"figures/{shim2}/HNS/{hyperparameter.split('(')[0][:-1]}"
# try:
#     with open(f'data/{shim}/human_normalized_curve/{hyp}.pickle', mode='rb') as f:
#         data2 = pickle.load(f)
# except:
#     data2 = None

# with open(f'data/{shim}/curves_all_games/{hyp}.pickle', mode='rb') as f:
#     data3 = pickle.load(f)

# fig3_path = f"figures/{shim}/all_games/{hyperparameter}"

col1, col2 = st.columns(2)
ag_col = {"DrQ_eps": col1, "DER": col2}

col1.subheader('DrQ(ε)')

col2.subheader('DER')

for ag in agents:
    if ag == "DrQ_eps" and hyp == "num_atoms":
        continue
    # fig = plot(data[f'{ag}_{hyp}'])
    # ag_col[ag].pyplot(fig)
    for shim in ["100k_experiments", "40M_experiments"]:
        fig1_path = f"figures/{shim}/IQM/{hyperparameter.split('(')[0][:-1]}" 
        try:
            ag_col[ag].image(fig1_path+f"/{ag}.png")
        except:
            pass


    # if data2 is not None:
    #     fig2 = plot_human_normalized(data2[f'{ag}_{hyp}'])
    #     ag_col[ag].pyplot(fig2)
    try:
        ag_col[ag].image(fig2_path+f"/{ag}.png")
    except:
        pass
    

    # fig3 = plot_all_games(data3[f'{ag}_{hyp}'])
    # ag_col[ag].pyplot(fig3)
    # ag_col[ag].image(fig3_path+f"/{ag}.png")

