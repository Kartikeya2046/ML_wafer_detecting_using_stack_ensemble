"""Stage 2 C3: MLR stacker weights per class and per base learner (the paper's Fig. 5, for our stacks).

Usage:  python mlr_weights.py NAME [NAME2 ...]
  NAME: an MLR stack_tune.py trial with final=1 runs (data/stack_NAME_b*_s*.pkl, 'coef' saved).
The ridge maps the concatenated base-learner probabilities [p_1 | p_2 | ...] to 9 class scores; coef is (9, 9 * L).
The weight that learner l gets for class c is its diagonal coefficient coef[c, l * 9 + c] (its own vote for c),
averaged over runs - as in the paper's weight-matrix discussion. Off-diagonal mass is reported as a share.
Figure -> figures/mlr_weights_<NAME>.png; table printed.
"""
import sys, glob
import numpy as np
from common import *

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.makedirs(P('figures'), exist_ok=True)
COL = ['#3b6ea5', '#c0392b', '#2e8b57', '#8e44ad']

for name in sys.argv[1:]:
    fs = [f for f in sorted(glob.glob(P('data', f'stack_{name}_b*_s*.pkl')))
          if os.path.basename(f)[len(f'stack_{name}_'):].count('_') == 1]
    runs = [ld(os.path.basename(f)) for f in fs]
    C = np.array([c for r in runs for c in r['coef']])  # (runs, 9, 9L)
    learners = [s.split('+')[0] for s in runs[0]['cfg']['inputs'].split(',')]
    L = len(learners)
    W = np.array([[C[:, c, l * 9 + c] for l in range(L)] for c in range(9)])  # (9 classes, L, runs)
    off = np.abs(C).sum((1, 2)) - np.abs(W).sum((0, 1))
    print(f'{name}: {len(C)} runs, learners {learners}, off-diagonal |coef| share {(off / np.abs(C).sum((1, 2))).mean():.1%}')
    print('  class      ' + ''.join(f'{l[:18]:>20s}' for l in learners))
    for c in range(9):
        print(f'  {NAMES[c]:10s} ' + ''.join(f'{W[c, l].mean():12.3f} ± {W[c, l].std():.3f}' for l in range(L)))
    fig, a = plt.subplots(figsize=(9, 4))
    w = 0.8 / L
    for l in range(L):
        a.bar(np.arange(9) + (l - (L - 1) / 2) * w, W[:, l].mean(1), w, yerr=W[:, l].std(1), color=COL[l % 4],
              label=learners[l], capsize=2)
    a.axhline(0, c='#444', lw=0.8)
    a.set_xticks(range(9)); a.set_xticklabels(NAMES, rotation=30)
    a.set_ylabel('MLR weight on own class vote'); a.set_title(f'Stacker weight per class and base learner — {name}')
    a.legend(fontsize=8); fig.tight_layout()
    fig.savefig(P('figures', f'mlr_weights_{name}.png'), dpi=150); plt.close(fig)
