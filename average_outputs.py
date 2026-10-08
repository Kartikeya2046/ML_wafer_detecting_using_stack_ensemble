"""Stage 2b: average several base-learner runs (seeds) into one ensemble input for stack_tune.py.

Usage:  python average_outputs.py OUT IN1 IN2 ...   (names without the _outputs.pkl suffix)
Writes data/OUT_outputs.pkl with train_prob / test_prob = mean of the inputs' probabilities, and prints each input's
and the ensemble's B_val macro-F1 (test only with test=1 as the last argument: selection must not look at test). All inputs must follow the same protocol (fit A, early-stop B_fit), so the
ensemble's outputs on B stay honest.
"""
import sys
import numpy as np
from common import *

show_test = sys.argv[-1] == 'test=1'
out, ins = sys.argv[1], [a for a in sys.argv[2:] if a != 'test=1']
y, yt = labels()
A, B_fit, B_val = splits(y)
O = [ld(f'{n}_outputs.pkl') for n in ins]
for n, o in zip(ins, O):
    print(f'{n:32s} B_val {f1s(y[B_val], o["train_prob"][B_val]).mean():.4f}' + (f'  test {f1s(yt, o["test_prob"]).mean():.4f}' if show_test else ''))
e = {k: np.mean([o[k] for o in O], 0).astype(np.float32) for k in ('train_prob', 'test_prob')}
e['members'] = ins
dump(e, f'{out}_outputs.pkl')
print(f'{out:32s} B_val {f1s(y[B_val], e["train_prob"][B_val]).mean():.4f}' + (f'  test {f1s(yt, e["test_prob"]).mean():.4f}' if show_test else '') + f'  (mean of {len(ins)})')
