import glob
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pingouin as pg

from plot_data import ATARI_100K_GAMES
from utils import get_game_rankings


def get_metric(data):
    rankings = get_game_rankings(data)
    rankings = {game:
            { 
                tup[0]: tup[1]
                for tup in rankings[game]
            }
            for game in rankings
           }
    temp = pd.DataFrame(rankings).values
    long_form = temp.reshape(-1)
    games_column = np.array(len(data.keys())*ATARI_100K_GAMES)
    hp_column = np.array(list(data.keys())).repeat(len(ATARI_100K_GAMES))
    df = pd.DataFrame([hp_column, games_column, long_form], index=["hyperparameter", "game", "ratings"]).T
    return df.drop(columns=["game"]).groupby(by="hyperparameter").agg("std")["ratings"].mean()
