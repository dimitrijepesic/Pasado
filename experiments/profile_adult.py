"""Profiles the Section 5.5 workload in a single process. adult_script.py uses a
multiprocessing pool, which cProfile cannot see into.

Run from Section_5_5: python ../experiments/profile_adult.py
Raw stats go to results/profile_adult.prof."""

import cProfile
import os
import pickle
import pstats

import sys

# Moved from Section_5_5/ to experiments/ on 2026-09-15; adult_eval still lives there.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Section_5_5"))
from adult_eval import adult_interval, adult_affine, adult_pasado, continuous_idx

N_INPUTS = 3
EPS_VALUES = [0.05, 0.25, 0.45]


def run(inputs, cls):
    for eps_value in EPS_VALUES:
        for n, inputs_ in enumerate(inputs):
            inputs_ = list(inputs_)
            eps = [0.] * len(inputs_)
            for i_ in continuous_idx:
                inputs_[i_] = 0.  # Mean.
                eps[i_] = eps_value
            adult_interval(inputs_, eps, cls, return_count=True)
            adult_affine(inputs_, eps, cls, return_count=True)
            adult_pasado(inputs_, eps, cls, return_count=True)
            print(f"done eps={eps_value} input={n}", flush=True)


if __name__ == '__main__':
    with open('saved/inputs.sav', 'rb') as f:
        inputs = pickle.load(f)[:N_INPUTS]
    with open('saved/cls.sav', 'rb') as f:
        cls = pickle.load(f)

    profiler = cProfile.Profile()
    profiler.enable()
    run(inputs, cls)
    profiler.disable()

    os.makedirs('results', exist_ok=True)
    profiler.dump_stats('results/profile_adult.prof')

    stats = pstats.Stats(profiler).strip_dirs()
    print(f"\nTotal: {stats.total_tt:.2f}s, {stats.total_calls} function calls")
    stats.sort_stats('cumulative').print_stats(20)
    stats.sort_stats('tottime').print_stats(20)
