import os
import pickle

import numpy as np

from rliable import library as rly
from rliable import metrics

from plot_data import ATARI_100K_GAMES

N_GAMES = len(ATARI_100K_GAMES)

def get_int_estimate(data):
    return rly.get_interval_estimates(
        data, 
        lambda x: np.array([metrics.aggregate_iqm(x)]),
        reps=50
    )[1]

def flatten_interval_dict(d):
    # return [(k, v) for k, v in d.items()]
    return [(k, v[0, 0], v[1, 0]) for k, v in d.items()]

def interval_to_ranking(scores):
    rank_by_mean = sorted(scores, key=lambda x: -(x[1] + x[2]))
    ranking = [1]
    for idx, interval in enumerate(rank_by_mean[1:], start=1):
        ranking.append(ranking[idx - 1] + (interval[2] < rank_by_mean[idx-1][1]))
    return list(zip([mr[0] for mr in rank_by_mean], ranking))

def iqm_to_ranking(scores):
    rank_by_mean = sorted(scores, key=lambda x: -x[1])
    return list(zip([mr[0] for mr in rank_by_mean], range(len(rank_by_mean))))
    
def get_game_rankings(data):
    transposed_data = {
        game: 
            {
                hp.split('_')[-1]: data[hp][:, idx].reshape(1, -1)
                for hp in data.keys()
            } 
            for idx, game in enumerate(ATARI_100K_GAMES)
        }
    game_intervals = {game: 
                      flatten_interval_dict(
                        get_int_estimate(transposed_data[game])) 
                        for game in transposed_data.keys()}
    
    return {game: interval_to_ranking(intervals)
                for game, intervals in game_intervals.items()}

def kendall_w(data):
    rankings = get_game_rankings(data)
    total_rankings = {hp: 
                      sum(rankings[game][idx][1] for game in rankings.keys())
                      for idx, (hp, _) in enumerate(rankings[ATARI_100K_GAMES[0]])}
    mean_ranking = np.mean(list(total_rankings.values()))
    s = sum((mean_ranking - np.array(list(total_rankings.values())))**2)
    num_hps = len(total_rankings)
    return 12*s/(N_GAMES*N_GAMES*(num_hps**3 - num_hps))



if __name__ == "__main__":
    with open(f'data/40M_experiments/final_perf/batch_sizes.pickle', mode='rb') as f:
        data_batch = pickle.load(f)
    rankings = get_game_rankings(data_batch['DER_batch_sizes'])
    print("rankings:", rankings['Alien'])
    print(kendall_w(data_batch['DER_batch_sizes']))