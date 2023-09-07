from collections import namedtuple
import os
import pickle

import numpy as np

from rliable import library as rly
from rliable import metrics

from plot_data import ATARI_100K_GAMES

N_GAMES = len(ATARI_100K_GAMES)

Interval = namedtuple("Interval", "hparam lo hi")

def iqm(values):
    n = len(values)
    return np.mean(np.sort(values)[n//4:-n//4])

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



def bootsrapped_ranking(scores, reps=100):
    bootstrap_scores = {
        hp: np.array(
                [
                iqm(np.random.choice(sc[0], size=10, replace=True))
                for _ in range(reps)
                ]
            )
            for hp, sc in scores.items()
        }
    sample_rankings = {hp: [] for hp in scores.keys()}
    for idx in range(reps):
        performances = np.array([bootstrap_scores[hp][idx] for hp in sample_rankings])
        rankings = np.argsort(np.argsort(-performances))
        for idy, hp in enumerate(sample_rankings.keys()):
            sample_rankings[hp].append(rankings[idy])
    return {hp: 1+np.median(sc) for hp, sc in sample_rankings.items()}


def get_elos(scores):
    for hp, sc in scores.items():
        for op_hp, op_sc in scores.items():
            if hp == op_hp:
                continue
            perf = np.random.choice(sc, size=10, replace=True)
            op_perf = np.random.choice(op_sc, size=10, replace=True)
            results = np.sign(perf - op_perf) # 1 for win, 0 for draw, -1 for lose


def elo_ranking(scores):
    elos = get_elos(scores)
    return sorted(elos)

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
    return {game: 
                      #flatten_interval_dict(
                        bootsrapped_ranking(transposed_data[game])#) 
                        for game in transposed_data.keys()}
    import pdb; pdb.set_trace()
    exit()
    return {game: interval_to_ranking(intervals)
                for game, intervals in game_intervals.items()}

def kendall_w(data):
    rankings = get_game_rankings(data)
    # import pdb; pdb.set_trace()
    total_rankings = {hp: 
                      np.std([rankings[game][hp] for game in rankings.keys()])
                      for hp in rankings[ATARI_100K_GAMES[0]]}
    return np.mean(list(total_rankings.values()))

    mean_ranking = np.mean(list(total_rankings.values()))
    
    s = sum((mean_ranking - np.array(list(total_rankings.values())))**2)
    num_hps = len(total_rankings)
    print(s, N_GAMES)
    return 12*s/(N_GAMES*N_GAMES*(num_hps**3 - num_hps))



if __name__ == "__main__":
    with open(f'data/40M_experiments/final_perf/layer_funct_conv.pickle', mode='rb') as f:
        data = pickle.load(f)
    rankings = get_game_rankings(data['DER_layer_funct_conv'])
    print("rankings:", rankings['Alien'])
    print("rankings:", rankings)
    print(kendall_w(data['DER_layer_funct_conv']))