#!/usr/bin/env python
"""Summarise the `evaluating_<setting>_seed_<seed>.txt` files written by eval_unity.py.

Rates are pooled over all robot episodes of all layouts and robots.
"""
import argparse
import re
from pathlib import Path

import numpy as np

NAMES = {'iid': 'In-distribution', 'ood': 'Unseen layouts',
         'ood_faster': 'Rushing humans', 'ood_denser': 'Denser crowds', 'ood_rooms': 'Four rooms'}


def name(exp):
    # <setting> or <setting>_n<N> (run with NUM_ROBOTS=N)
    fam, _, n = exp.rpartition('_n') if re.fullmatch(r'.+_n\d+', exp) else (exp, '', '')
    return NAMES.get(fam, fam) + (f' (N={n})' if n else '')


def read_counts(text, key):
    block = re.search(key + r": (\[\[.*?\]\])", text, re.S).group(1)
    return np.array([[float(x) for x in row.split()]
                     for row in re.findall(r"\[([^\[\]]+)\]", block)])


def read_metric(text, key):
    return float(np.mean([float(x) for x in re.findall(r"\b" + key + r": ([-\d.naninf]+)", text)]))


def summarise(path):
    text = path.read_text()
    text = text[text.rindex('total_num_steps'):]
    n = read_counts(text, 'end counts').sum()
    rate = {k: read_counts(text, k + '_count').sum() / n
            for k in ['success', 'obstacle_collision', 'npc_collision', 'robot_collision', 'timeout']}
    sr = rate['success']
    return dict(n=int(n), sr=sr, se=np.sqrt(sr * (1 - sr) / n), obs=rate['obstacle_collision'],
                npc=rate['npc_collision'], rob=rate['robot_collision'], to=rate['timeout'],
                nt=read_metric(text, 'ANT'), pl=read_metric(text, 'PL'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('families', nargs='*', default=list(NAMES))
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--result_dir', type=str,
                        default=str(Path(__file__).resolve().parents[3] / 'results'))
    args = parser.parse_args()

    print(f"{'Setting':22s} {'episodes':>8s} {'SR':>13s} {'CR-obstacle':>11s} {'CR-human':>8s} "
          f"{'CR-robot':>8s} {'TR':>6s} {'NT (s)':>7s} {'PL (m)':>7s}")
    for fam in args.families:
        path = Path(args.result_dir) / f'evaluating_{fam}_seed_{args.seed}.txt'
        if not path.exists() or 'ANT' not in path.read_text():
            print(f"{name(fam):22s} missing ({path})")
            continue
        r = summarise(path)
        print(f"{name(fam):22s} {r['n']:8d} {r['sr']:7.3f}±{r['se']:.3f} {r['obs']:11.3f} "
              f"{r['npc']:8.3f} {r['rob']:8.3f} {r['to']:6.3f} {r['nt']:7.2f} {r['pl']:7.2f}")
