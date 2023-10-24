import streamlit as st

from plot_data import plot_game, experiments_mapping
from utils import THIS_METRIC

plot_game = st.experimental_memo(plot_game)

st.set_page_config(layout="wide")

st.title("Lifting the Veil")
agents = ["DrQ_eps", "DER"]

hparam = experiments_mapping[st.radio("Hyperparameter", options=list(experiments_mapping.keys()))]

col1, col2 = st.columns(2)
ag_col = {"DrQ_eps": col1, "DER": col2}

col1.subheader('DrQ(ε) 100k')

col2.subheader('DER 100k')

if hparam is not None:
    main_path = f"figures/100k_experiments/hparam_comparison/{hparam}"
    for ag in agents:
        ag_col[ag].image(main_path+f"/{ag}.png")

col1.subheader('DrQ(ε) 40M')

col2.subheader('DER 40M')

if hparam is not None:
    col1.subheader(f'DrQ(ε) 40M: {THIS_METRIC["DrQ_eps"][hparam]}')

    col2.subheader(f'DER 40M: {THIS_METRIC["DER"][hparam]}')
    main_path = f"figures/40M_experiments/hparam_comparison/{hparam}"
    for ag in agents:
        ag_col[ag].image(main_path+f"/{ag}.png")