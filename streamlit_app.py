import streamlit as st

import PIL
PIL.Image.MAX_IMAGE_PIXELS = 900000000

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
if hyp in ["gammas", "layer_funct", "convs_normalization", "min_replay_history", "num_atoms", "update_horizon"]:
    shim = "40M_experiments"
    shim2 = "split"
else:
    shim = "100k_experiments"
    shim2 = "100k_experiments"

fig1_path = f"figures/{shim}/IQM/{hyperparameter}"
# with open(f'data/{shim}/final_perf/{hyp}.pickle', mode='rb') as f:
    # data = pickle.load(f)

fig2_path = f"figures/{shim2}/HNS/{hyperparameter}"
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

col1.subheader('DrQ Epsilon')

col2.subheader('DER')

for ag in agents:
    if ag == "DrQ_eps" and hyp == "num_atoms":
        continue
    # fig = plot(data[f'{ag}_{hyp}'])
    # ag_col[ag].pyplot(fig)
    ag_col[ag].image(fig1_path+f"/{ag}.png")


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

