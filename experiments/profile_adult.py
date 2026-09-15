"""Profile the Section 5.5 workload under cProfile.

adult_script.py farms the same tasks out to a multiprocessing.Pool, which
cProfile cannot see into (it would only measure the parent waiting on the
pool). This script therefore runs a small subset of the same tasks
in-process: for each (input, eps) pair it evaluates all three abstract
domains (Interval, Zonotope/Affine, Pasado), exactly like wrapper() in
adult_script.py.

Run from Section_5_5:  ../.venv/Scripts/python.exe profile_adult.py
Raw stats are dumped to results/profile_adult.prof (open with snakeviz/pstats).
"""

import cProfile
import os
import pickle
import pstats

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
