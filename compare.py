"""Stage 2 statistics (stage2_plan.md rule 4): is trial X really better than trial Y?

Usage:  python compare.py X Y [test=1] [boot=2000] [label=...]
  X, Y: a stack_tune.py trial name, or out:<name> for a base learner (data/<name>_outputs.pkl).
  Default evaluation set: B (pooled 2-fold B-cv predictions, data/stack_cv_<NAME>_b*_s*.pkl);
  B_val if an out: input is involved (base learners early-stop on B_fit). test=1 -> the 10,000 test wafers
  (data/stack_<NAME>_b*_s*.pkl from final=1 runs).
Prints mean macro-F1 of X and Y, the mean difference X - Y,
  - paired t-test across runs (scipy.stats.ttest_rel; runs paired by (base seed, seed) - needs equal run counts),
  - paired bootstrap 95% CI over wafers: resample wafers, macro-F1 averaged over runs, difference X - Y.
Appends to logs/compare.jsonl.
"""
import sys, glob, re, json
import numpy as np
from scipy.stats import ttest_rel
from common import *

a, b = sys.argv[1], sys.argv[2]
kv = dict(s.split('=', 1) for s in sys.argv[3:])
test, nboot, label = int(kv.get('test', 0)), int(kv.get('boot', 2000)), kv.get('label', '')
y, yt = labels()
A, B_fit, B_val = splits(y)


def runs(spec):
    """{(b, s): prob over the full train (cv) or test set}, keyed for pairing."""
    if spec.startswith('out:'):
        o = ld(f'{spec[4:]}_outputs.pkl')
        return {(0, 0): o['test_prob'] if test else o['train_prob']}
    pre = 'stack_' if test else 'stack_cv_'
    pat = f'{pre}{spec}_b*_s*.pkl'
    out = {}
    for f in sorted(glob.glob(P('data', pat))):
        m = re.fullmatch(f'{pre}{re.escape(spec)}_b(\\d+)_s(\\d+)\\.pkl', os.path.basename(f))
        if not m:  # the glob also matches longer trial names (spec_xyz_b0_s0)
            continue
        d = ld(os.path.basename(f))
        if test:
            out[int(m[1]), int(m[2])] = d['test_prob']
        else:
            full = np.zeros((len(y), 9), np.float32); full[CUT:] = d['B_prob']
            out[int(m[1]), int(m[2])] = full
    assert out, f'no runs found for {spec} ({pat})'
    return out


RA, RB = runs(a), runs(b)
idx = np.arange(len(yt)) if test else (B_val if a.startswith('out:') or b.startswith('out:') else np.arange(CUT, len(y)))
yy = (yt if test else y)[idx]
PA = np.array([RA[k][idx].argmax(1) for k in sorted(RA)])
PB = np.array([RB[k][idx].argmax(1) for k in sorted(RB)])


def macro(yv, pred):
    """macro-F1 of each run (rows of pred) from confusion counts; fast enough for the bootstrap."""
    out = []
    for p in pred:
        cm = np.bincount(yv * 9 + p, minlength=81).reshape(9, 9)
        tp = np.diag(cm); den = cm.sum(0) + cm.sum(1)
        out.append(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0).mean())
    return np.array(out)


fa, fb = macro(yy, PA), macro(yy, PB)
diff = fa.mean() - fb.mean()
p = ttest_rel(fa, fb).pvalue if len(fa) == len(fb) and len(fa) > 1 and sorted(RA) == sorted(RB) else float('nan')
rng = np.random.default_rng(0)
bs = np.empty(nboot)
for i in range(nboot):
    r = rng.integers(0, len(yy), len(yy))
    bs[i] = macro(yy[r], PA[:, r]).mean() - macro(yy[r], PB[:, r]).mean()
lo, hi = np.percentile(bs, [2.5, 97.5])
rec = dict(a=a, b=b, test=test, label=label, n_eval=len(yy), runs=[len(fa), len(fb)],
           a_mean=fa.mean(), a_std=fa.std(), b_mean=fb.mean(), b_std=fb.std(), diff=diff, p_ttest=p, ci95=[lo, hi], boot=nboot)
with open(P('logs', 'compare.jsonl'), 'a') as f:
    f.write(json.dumps(rec) + '\n')
print(f"{'TEST' if test else 'B' if len(yy) > len(B_val) + 1 else 'B_val'} (n={len(yy)})  "
      f"{a}: {fa.mean():.4f} ± {fa.std():.4f} [{len(fa)}]   {b}: {fb.mean():.4f} ± {fb.std():.4f} [{len(fb)}]\n"
      f"  diff {diff:+.4f}   paired t-test p = {p:.4f}   bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]", flush=True)
