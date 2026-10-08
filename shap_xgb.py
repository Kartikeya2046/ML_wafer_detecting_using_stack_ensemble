"""Stage 2 C3 (optional): SHAP on the 59 handcrafted features, via XGBoost's exact TreeSHAP (pred_contribs).

Usage:  run_cpu.sh shap_xgb.py [name=d6_lr05_cw1]
Refits the selected XGB config exactly as xgb_tune.py does (deterministic: hist, no subsampling) and checks its test
probabilities against data/xgb_<name>_outputs.pkl, then computes per-class mean |SHAP| on the test set.
Feature names follow the reference repo's order: 13 density regions, 40 Radon (20 mean + 20 std), 6 geometry.
Figure -> figures/shap_xgb_<name>.png (top features per class); numbers -> data/shap_xgb_<name>.pkl.
"""
import sys
import numpy as np
import xgboost as xgb
from common import *

kv = dict(s.split('=', 1) for s in sys.argv[1:])
name = kv.get('name', 'd6_lr05_cw1')
o = ld(f'xgb_{name}_outputs.pkl'); cfg = o['cfg']
Xtr, Xte = ld('X_MFE.pkl')
y, yt = labels()
A, B_fit, B_val = splits(y)
w = class_weights(y[A], cfg['cw'])
m = xgb.XGBClassifier(n_estimators=o['best_iteration'] + 1, max_depth=cfg['depth'], learning_rate=cfg['lr'],
                      tree_method='hist', n_jobs=16, random_state=0, objective='multi:softprob')
m.fit(Xtr[A], y[A], sample_weight=np.array([w[c] for c in y[A]]))
pt = m.predict_proba(Xte)
print('max |diff| vs saved test probs:', float(np.abs(pt - o['test_prob']).max()))

FEAT = ([f'density_{i + 1}' for i in range(13)] + [f'radon_mean_{i + 1}' for i in range(20)] +
        [f'radon_std_{i + 1}' for i in range(20)] + ['geom_area', 'geom_perimeter', 'geom_major_axis',
                                                    'geom_minor_axis', 'geom_eccentricity', 'geom_solidity'])
assert len(FEAT) == Xte.shape[1]
S = m.get_booster().predict(xgb.DMatrix(Xte), pred_contribs=True)  # (n, 9, 60): last column = bias
S = np.abs(S[:, :, :-1])
imp = np.array([S[yt == c, c].mean(0) for c in range(9)])  # mean |SHAP| for the true class, per class
dump(dict(imp=imp, features=FEAT), f'shap_xgb_{name}.pkl')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 3, figsize=(13, 10))
for c, a in enumerate(ax.flat):
    top = np.argsort(-imp[c])[:8][::-1]
    grp = ['#3b6ea5' if FEAT[i].startswith('density') else '#c0392b' if FEAT[i].startswith('radon') else '#2e8b57' for i in top]
    a.barh([FEAT[i] for i in top], imp[c, top], color=grp)
    a.set_title(NAMES[c], fontsize=10); a.tick_params(labelsize=7)
fig.suptitle(f'XGBoost ({name}) — top handcrafted features per class, mean |SHAP| on test '
             '(blue density, red Radon, green geometry)', fontsize=10)
fig.tight_layout(); os.makedirs(P('figures'), exist_ok=True)
fig.savefig(P('figures', f'shap_xgb_{name}.png'), dpi=140)
for c in range(9):
    print(f'{NAMES[c]:10s}', ', '.join(FEAT[i] for i in np.argsort(-imp[c])[:4]))
