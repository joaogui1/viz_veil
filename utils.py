from collections import namedtuple
import pickle

import numpy as np
import pandas as pd

from rliable import library as rly
from rliable import metrics

from plot_data import ATARI_100K_GAMES

N_GAMES = len(ATARI_100K_GAMES)

def area_under_the_curve(data):
    return  data[:, :, 0]/2 + data[:, :, 1:-1].sum(axis=2) + data[:, :, -1]/2

Interval = namedtuple("Interval", "hparam lo hi")

def get_int_estimate(data):
    # reliable = rly.get_interval_estimates(
    #     data, 
    #     lambda x: np.array([metrics.aggregate_iqm(x)]),
    #     reps=50
    # )
    # print(reliable)
    # return rly.get_interval_estimates(
    #     data, 
    #     lambda x: np.array([metrics.aggregate_iqm(x)]),
    #     reps=50
    # )[1]
    # import pdb; pdb.set_trace()
    return [Interval(k, np.mean(v) - np.std(v), np.mean(v) + np.std(v)) for k, v in data.items()]

def get_iqm_estimate(data):
    return rly.get_interval_estimates(
        data, 
        lambda x: np.array([metrics.aggregate_iqm(x)]),
        reps=50
    )[0]

def flatten_interval_dict(d):
    return [Interval(k, v[0, 0], v[1, 0]) for k, v in d.items()]

def flatten_iqm_dict(d):
    return [(k, v) for k, v in d.items()]

def interval_to_ranking(scores):
    rank_by_high = sorted(scores, key=lambda x: -x.hi)
    rank_by_high.append(Interval(-1, -100, -100)) # Sentinel
    
    ranks = np.ones((len(scores),))
    for idx, interval in enumerate(rank_by_high):
        begin_equal = 0
        for comp_idx in range(0, idx+1):
            if rank_by_high[comp_idx].lo <= interval.hi:
                begin_equal = comp_idx
                break
        
        for comp_idx in range(idx+1, len(rank_by_high)):
            if rank_by_high[comp_idx].hi < interval.lo: # comp_idx is smaller than interval
                ranks[idx] = (begin_equal + 1 + comp_idx)/2
                break


    # for idx, interval in enumerate(rank_by_high):
    #     if interval.hi <= rank_by_high[idx - 1].lo: # statistically different
    #         ranks[begin_equal:idx] = (begin_equal + 1 + idx - 1 + 1)/2 # average rank
    #         begin_equal = idx
    return list(zip([mr.hparam for mr in rank_by_high], ranks))

def iqm_to_ranking(scores):
    rank_by_mean = sorted(scores, key=lambda x: -x[1])
    return list(zip([mr[0] for mr in rank_by_mean], range(len(rank_by_mean))))
    
def get_game_rankings(data):
    if len(list(data.values())[0].shape) == 3:
        data = {k: area_under_the_curve(v) for k, v in data.items()}
    transposed_data = {
        game: 
            {
                hp.split('_')[-1]: data[hp][:, idx].reshape(1, -1)
                for hp in data.keys()
            } 
            for idx, game in enumerate(ATARI_100K_GAMES)
        }
    game_intervals = {game: 
                      #flatten_interval_dict(
                        get_int_estimate(transposed_data[game])#) 
                        for game in transposed_data.keys()}
    # import pdb; pdb.set_trace()
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
    print(s, N_GAMES)
    return 12*s/(N_GAMES*N_GAMES*(num_hps**3 - num_hps))

def span(row):
    return np.ptp(row) + 1
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
    hp_column = np.array(list(data.keys())).repeat(len(ATARI_100K_GAMES))
    df = pd.DataFrame([hp_column, long_form], 
                      index=["hyperparameter", "ratings"]).T
    df_deviations = df.groupby(by="hyperparameter").agg(np.ptp)["ratings"]
    df_deviations = (df_deviations/(len(df_deviations)-1))

    return (df_deviations.mean(), df_deviations.std())


if __name__ == "__main__":
    with open(f'data/40M_experiments/human_normalized_curve/layer_funct_conv.pickle', mode='rb') as f:
        data = pickle.load(f)
    rankings = get_game_rankings(data['DER_layer_funct_conv'])
    print("rankings:", rankings['Alien'])
    print(kendall_w(data['DER_layer_funct_conv']))


THIS_METRIC = {
    'DrQ_eps': 
               {"Adam's ε": (0.8333333333333334, 0.14433756729740646), 
                'Batch Size': (0.7083333333333334, 0.25),
                'Conv. Activation Function': (0.18333333333333335, 0.1329160135825126),
                'Convolutional Normalization': (0.8333333333333334, 0.14433756729740646),
                'Convolutional Width': (0.7, 0.14252192813739223),
                'Dense Activation Function': (0.5166666666666667, 0.09831920802501748),
                'Dense Width': (0.75, 0.09622504486493767),
                'Discount Factor': (0.8333333333333334, 0.14433756729740646),
                'Exploration ε': (0.4, 0.05590169943749474),
                'Learning Rate': (0.5, 0.19764235376052372),
                'Minimum Replay History': (0.3333333333333333, 0.0),
                'Number of Convolutional Layers': (0.75, 0.25),
                'Number of Dense Layers': (0.6666666666666666, 0.14433756729740646),
                'Replay Capacity': (0.5416666666666666, 0.08333333333333331),
                'Reward Clipping': (1.0, 0.0),
                'Target Update Period': (0.075, 0.06846531968814577),
                'Update Horizon': (0.775, 0.22360679774997896),
                'Update Period': (0.6666666666666666, 0.23570226039551584),
                'Weight Decay': (0.575, 0.11180339887498948)},
    'DER': 
            {"Adam's ε": (0.5833333333333334, 0.14433756729740646),
             'Batch Size': (0.6666666666666666, 0.23570226039551584),
             'Conv. Activation Function': (0.19999999999999998, 0.10954451150103323),
             'Convolutional Normalization': (0.5833333333333334, 0.14433756729740646),
             'Convolutional Width': (0.55, 0.2091650066335189),
             'Dense Activation Function': (0.4333333333333333, 0.0816496580927726),
             'Dense Width': (0.7083333333333334, 0.25),
             'Discount Factor': (0.9166666666666666, 0.14433756729740646),
             'Exploration ε': (0.175, 0.06846531968814576),
             'Learning Rate': (0.45, 0.2592055169165965),
             'Minimum Replay History': (0.20833333333333331, 0.08333333333333333),
             'Number of Atoms': (0.625, 0.28463752127665554),
             'Number of Convolutional Layers': (0.75, 0.25),
             'Number of Dense Layers': (0.5833333333333334, 0.14433756729740646),
             'Replay Capacity': (0.4583333333333333, 0.08333333333333334),
             'Reward Clipping': (1.0, 0.0),
             'Target Update Period': (0.25, 0.08838834764831845),
             'Update Horizon': (0.7, 0.14252192813739223),
             'Update Period': (0.5833333333333333, 0.09622504486493762),
             'Weight Decay': (0.7, 0.06846531968814577)}
    }