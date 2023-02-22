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

experiments_mapping = { 
                        "Adam's ε": "epsilon",
                        "Batch Size": "batch_sizes",
                        "Conv. Activation Function": "layer_funct_conv",
                        "Convolutional Normalization": "normalizations_convs", 
                        "Convolutional Width": "CNN_widths",
                        "Dense Activation Function": "layer_funct_dense",
                        "Dense Normalization": "normalizations",
                        "Dense Width": "widths",
                        "Discount Factor": "gammas",
                        "Exploration ε": "eps_train",
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
hyperparameter = st.radio("Hyperparameter", options=experiments_mapping.keys())
hyp = experiments_mapping[hyperparameter]
shim = "40M_experiments"
shim2 = "split"
    

st.header(hyperparameter)

fig1_path = f"figures/{shim}/IQM/{hyperparameter}"
fig2_path = f"figures/{shim2}/HNS/{hyperparameter}"


col1, col2 = st.columns(2)
ag_col = {"DrQ_eps": col1, "DER": col2}

col1.subheader('DrQ(ε)')

col2.subheader('DER')

for ag in agents:
    if ag == "DrQ_eps" and hyp == "num_atoms":
        continue

    for shim in ["100k_experiments", "40M_experiments"]:
        fig1_path = f"figures/{shim}/IQM/{hyperparameter}" 
        try:
            ag_col[ag].image(fig1_path+f"/{ag}.png")
        except:
            pass

    try:
        ag_col[ag].image(fig2_path+f"/{ag}.png")
    except:
        pass

