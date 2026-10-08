"""Stage 2 C2: temperature scaling + reject option for a stack (stage2_plan.md step 3).

Usage:  python calibrate.py NAME [label=...]
  NAME: a stack_tune.py trial that has both B-cv runs (data/stack_cv_NAME_b*_s*.pkl) and final test runs
  (data/stack_NAME_b*_s*.pkl, same name, final=1).
Per run (paired by (base seed, seed)):
  - scores: MLR -> the ridge outputs; FNN -> log of its softmax probabilities. Calibrated prob = softmax(scores / T),
    one T per run fitted by log-loss on the pooled 2-fold B predictions, then applied to test unchanged.
    "Before" = the uncalibrated probabilities: FNN its own softmax; MLR the ridge outputs clipped to [0, 1], renormalised.
  - ECE (15 equal-width confidence bins) on test, before / after.
  - Reject rule: confidence = max calibrated probability, one global threshold. Thresholds for target coverage
    100/98/95/90/85/80% are chosen on B and applied to test (so test coverage is close to, not exactly, the target);
    also the B threshold reaching 99% / 99.5% B accuracy, and per-class rejection rates on test.
    Macro-F1 on accepted wafers averages over the classes present among them (a fully rejected class is not a 0).
Figures -> figures/reliability_<NAME>.png, figures/reject_curve_<NAME>.png, figures/rejection_by_class_<NAME>.png.
Numbers -> data/calib_<NAME>.pkl, logs/calibrate.jsonl (mean ± std over runs).
"""
import sys, glob, re, json
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from common import *

name = sys.argv[1]
kv = dict(s.split('=', 1) for s in sys.argv[2:])
y, yt = labels()
yB = y[CUT:]
COV = [1.0, 0.98, 0.95, 0.90, 0.85, 0.80]
ACC = [0.99, 0.995]
os.makedirs(P('figures'), exist_ok=True)


def load(pre, key):
    out = {}
    for f in glob.glob(P('data', f'{pre}{name}_b*_s*.pkl')):
        m = re.fullmatch(f'{pre}{re.escape(name)}_b(\\d+)_s(\\d+)\\.pkl', os.path.basename(f))
        if m:
            d = ld(os.path.basename(f)); out[int(m[1]), int(m[2])] = (d[key], d['cfg'])
    return out


CV, TE = load('stack_cv_', 'B_prob'), load('stack_', 'test_prob')
keys = sorted(set(CV) & set(TE))
assert keys, f'need both B-cv and final runs for {name}'
meta = TE[keys[0]][1]['meta']
scores = (lambda p: p.astype(np.float64)) if meta == 'mlr' else (lambda p: np.log(np.clip(p, 1e-12, 1)))
# uncalibrated probabilities: FNN its own softmax; MLR the ridge outputs clipped to [0, 1] and renormalised
raw = ((lambda p: np.clip(p, 1e-6, 1) / np.clip(p, 1e-6, 1).sum(1, keepdims=True)) if meta == 'mlr'
       else (lambda p: p.astype(np.float64)))


def ece(prob, yv, bins=15):
    c, ok = prob.max(1), prob.argmax(1) == yv
    e = np.minimum((c * bins).astype(int), bins - 1)
    return sum(abs(ok[e == i].mean() - c[e == i].mean()) * (e == i).mean() for i in range(bins) if (e == i).any())


def reliability(prob, yv, bins=15):
    c, ok = prob.max(1), prob.argmax(1) == yv
    e = np.minimum((c * bins).astype(int), bins - 1)
    return np.array([[c[e == i].mean(), ok[e == i].mean(), (e == i).sum()] if (e == i).any() else [np.nan] * 2 + [0]
                     for i in range(bins)])


def metrics(prob, yv, accept):
    a = accept & True
    pred = prob.argmax(1)
    return dict(coverage=a.mean(), acc=(pred[a] == yv[a]).mean() if a.any() else np.nan,
                macro=f1s(yv[a], pred[a])[np.unique(yv[a])].mean() if a.any() else np.nan)  # classes present among accepted


R = []
for k in keys:
    sB, sT = scores(CV[k][0]), scores(TE[k][0])
    nll = lambda T: -log_softmax(sB / T, axis=1)[np.arange(len(yB)), yB].mean()
    T = minimize_scalar(nll, bounds=(0.01, 100), method='bounded').x
    pB, pT0, pT = softmax(sB / T, axis=1), raw(TE[k][0]), softmax(sT / T, axis=1)
    cB, cT = pB.max(1), pT.max(1)
    r = dict(key=k, T=T, nll_B_before=nll(1.0), nll_B_after=nll(T), ece_before=ece(pT0, yt), ece_after=ece(pT, yt),
             ece_B_before=ece(raw(CV[k][0]), yB), ece_B_after=ece(pB, yB),
             rel_before=reliability(pT0, yt), rel_after=reliability(pT, yt), table=[], acc_targets=[])
    for c in COV:
        th = -np.inf if c >= 1 else np.quantile(cB, 1 - c)
        r['table'].append(dict(target=c, threshold=th, **metrics(pT, yt, cT >= th)))
    # B threshold reaching the target accuracy on B (lowest threshold = highest coverage that does)
    order = np.argsort(-cB); okB = (pB.argmax(1) == yB)[order]
    run_acc = np.cumsum(okB) / np.arange(1, len(okB) + 1)
    for a in ACC:
        n = np.nonzero(run_acc >= a)[0].max() + 1 if (run_acc >= a).any() else 0
        th = cB[order][n - 1] if n else np.inf
        r['acc_targets'].append(dict(target_acc=a, threshold=th, B_coverage=n / len(cB), **metrics(pT, yt, cT >= th)))
    # risk-coverage curve on test (oracle ordering by test confidence), and per-class rejection at 95% / 90% targets
    o = np.argsort(-cT); okT = (pT.argmax(1) == yt)[o]
    r['curve_cov'] = np.arange(1, len(o) + 1) / len(o); r['curve_acc'] = np.cumsum(okT) / np.arange(1, len(o) + 1)
    r['reject_by_class'] = {t['target']: [float((cT[yt == c] < t['threshold']).mean()) for c in range(9)]
                            for t in r['table'] if t['target'] in (0.95, 0.90)}
    R.append(r)

ms = lambda v: (float(np.nanmean(v)), float(np.nanstd(v)))
summ = dict(name=name, label=kv.get('label', ''), meta=meta, runs=len(R), T=ms([r['T'] for r in R]),
            ece_before=ms([r['ece_before'] for r in R]), ece_after=ms([r['ece_after'] for r in R]),
            ece_B_before=ms([r['ece_B_before'] for r in R]), ece_B_after=ms([r['ece_B_after'] for r in R]),
            table=[{q: (ms([r['table'][i][q] for r in R]) if q != 'target' else c) for q in ['target', 'coverage', 'acc', 'macro']}
                   for i, c in enumerate(COV)],
            acc_targets=[{q: (ms([r['acc_targets'][i][q] for r in R]) if q != 'target_acc' else a)
                          for q in ['target_acc', 'B_coverage', 'coverage', 'acc', 'macro']} for i, a in enumerate(ACC)],
            reject_by_class={t: np.mean([r['reject_by_class'][t] for r in R], 0).tolist() for t in (0.95, 0.90)})
dump(dict(summary=summ, runs=R), f'calib_{name}.pkl')
with open(P('logs', 'calibrate.jsonl'), 'a') as f:
    f.write(json.dumps(summ) + '\n')

# ---- figures ----
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for a, key, ttl in [(ax[0], 'rel_before', 'before (uncalibrated)'), (ax[1], 'rel_after', f'after (T = {summ["T"][0]:.2f})')]:
    rel = R[0][key]; m = rel[:, 2] > 0
    a.plot([0, 1], [0, 1], ls='--', c='#888', lw=1)
    a.bar(rel[m, 0], rel[m, 1], width=1 / 15 * 0.9, color='#3b6ea5', alpha=0.85, label='accuracy per bin')
    a.set_title(f'{ttl}   ECE {summ["ece_before" if key == "rel_before" else "ece_after"][0]:.4f}')
    a.set_xlabel('confidence (max probability)'); a.set_xlim(0, 1); a.set_ylim(0, 1)
ax[0].set_ylabel('accuracy'); fig.suptitle(f'Reliability on test — {name}'); fig.tight_layout()
fig.savefig(P('figures', f'reliability_{name}.png'), dpi=150); plt.close(fig)

fig, a = plt.subplots(figsize=(6, 4.4))
grid = np.linspace(0.5, 1, 501)
curves = np.array([np.interp(grid, r['curve_cov'], r['curve_acc']) for r in R])
a.plot(grid * 100, curves.mean(0) * 100, c='#3b6ea5', lw=2)
for t in summ['table']:
    a.plot(t['coverage'][0] * 100, t['acc'][0] * 100, 'o', c='#c0392b', ms=5)
a.set_xlabel('coverage: % of wafers auto-classified'); a.set_ylabel('accuracy on accepted wafers (%)')
a.set_title(f'Risk–coverage on test — {name}'); a.grid(alpha=0.3); fig.tight_layout()
fig.savefig(P('figures', f'reject_curve_{name}.png'), dpi=150); plt.close(fig)

fig, a = plt.subplots(figsize=(8, 4))
w = 0.38
for i, (t, col) in enumerate([(0.95, '#3b6ea5'), (0.90, '#c0392b')]):
    a.bar(np.arange(9) + (i - 0.5) * w, np.array(summ['reject_by_class'][t]) * 100, w, color=col, label=f'target coverage {t:.0%}')
a.set_xticks(range(9)); a.set_xticklabels(NAMES, rotation=30); a.set_ylabel('% of class routed to a human')
a.legend(); a.set_title(f'Rejected wafers by true class (test) — {name}'); fig.tight_layout()
fig.savefig(P('figures', f'rejection_by_class_{name}.png'), dpi=150); plt.close(fig)

f = lambda v: f'{v[0]:.4f} ± {v[1]:.4f}'
print(f'{name} ({meta}, {len(R)} runs)  T {f(summ["T"])}   test ECE {f(summ["ece_before"])} -> {f(summ["ece_after"])}')
for t in summ['table']:
    print(f'  target {t["target"]:.0%}: coverage {f(t["coverage"])}  acc {f(t["acc"])}  macro-F1 {f(t["macro"])}')
for t in summ['acc_targets']:
    print(f'  B acc >= {t["target_acc"]:.1%}: test coverage {f(t["coverage"])}  test acc {f(t["acc"])}')
print('  rejected by class @95%:', ' '.join(f'{n} {v:.0%}' for n, v in zip(NAMES, summ['reject_by_class'][0.95])))
