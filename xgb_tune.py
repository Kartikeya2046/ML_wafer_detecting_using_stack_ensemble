"""Stage 2 C1: XGBoost on the 59 handcrafted features as a third base learner (stage2_plan.md step 2).

Usage:  run_cpu.sh xgb_tune.py NAME [key=value ...]
  keys: depth=6 lr=0.1 cw=0 (class-weight power: 1 balanced, 0 none) rounds=5000 es=50 nthread=4 device=cpu
Fits on A, early-stops (number of trees) on B_fit, scores standalone macro-F1 on B_val. Same protocol as
mfe_tune.py, so its outputs on B are honest and stack_tune.py can use them with inputs=...,xgb_<NAME> mode=B.
Softmax for all train + test -> data/xgb_<NAME>_outputs.pkl. Results append to logs/xgb_trials.jsonl.
(The old prototype data/xgb_outputs.pkl used 5-fold OOF over all of train - not used, breaks stage2 rule 2.)
"""
import sys, json, time
import numpy as np
import xgboost as xgb
from common import *

name, kv = sys.argv[1], dict(s.split('=') for s in sys.argv[2:])
cfg = dict(depth=6, lr=0.1, cw=0.0, rounds=5000, es=50, nthread=4, device='cpu')
cfg.update({k: type(cfg[k])(v) for k, v in kv.items()})

Xtr, Xte = ld('X_MFE.pkl')
y, yt = labels()
A, B_fit, B_val = splits(y)
w = class_weights(y[A], cfg['cw'])

t0 = time.time()
m = xgb.XGBClassifier(n_estimators=cfg['rounds'], max_depth=cfg['depth'], learning_rate=cfg['lr'],
                      tree_method='hist', device=cfg['device'], n_jobs=cfg['nthread'], random_state=0,
                      objective='multi:softprob', eval_metric='mlogloss', early_stopping_rounds=cfg['es'])
m.fit(Xtr[A], y[A], sample_weight=np.array([w[c] for c in y[A]]), eval_set=[(Xtr[B_fit], y[B_fit])], verbose=False)
ptr, pte = m.predict_proba(Xtr).astype(np.float32), m.predict_proba(Xte).astype(np.float32)
dump({'train_prob': ptr, 'test_prob': pte, 'cfg': cfg, 'best_iteration': int(m.best_iteration)}, f'xgb_{name}_outputs.pkl')
F = f1s(y[B_val], ptr[B_val])
rec = dict(name=name, cfg=cfg, macro=F.mean(), per_class=F.tolist(), trees=int(m.best_iteration) + 1,
           minutes=(time.time() - t0) / 60)
with open(P('logs', 'xgb_trials.jsonl'), 'a') as f:
    f.write(json.dumps(rec) + '\n')
print(f"{name:20s} B_val macro-F1 {F.mean():.4f}  NearFull {F[7]:.3f}  trees {rec['trees']}  {rec['minutes']:.1f} min", flush=True)
