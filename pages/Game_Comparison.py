import streamlit as st

from plot_data import plot_game, ATARI_100K_GAMES

plot_game = st.experimental_memo(plot_game)

st.set_page_config(layout="wide")

st.title("Lifting the Veil")
agents = st.multiselect("Agents", options=["DrQ_eps", "DER"])

game = st.radio("Game", options=ATARI_100K_GAMES)


col1, col2 = st.columns(2)
ag_col = {"DrQ_eps": col1, "DER": col2}

col1.subheader('DrQ Epsilon')

col2.subheader('DER')
if game is not None:
    for ag in agents:
        fig = plot_game(ag, game)
        ag_col[ag].pyplot(fig)
