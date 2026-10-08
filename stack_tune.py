"""Stacking meta-learner: tuning trials (Part B2) and final test runs (Part B1/B5).

Usage:  run_cpu.sh stack_tune.py NAME [key=value ...] [final=1]
  keys (defaults = notebook cell 14 + class-weight fix):
    inputs=mfe,cnn  hidden=10  lr=1e-4  batch=32  cw=1.0 (class-weight power: 1 balanced, .5 sqrt, 0 none)
    mode=A   A  = fit on A (base preds in-sample, as in the paper repo), early-stop on held-out part
             B  = fit on B only (base preds honest there - base learners never fitted on B)
             AB = fit on A + part of B
    monitor=loss|f1   patience=20   dropout=0   l2=0   seeds=5
    base_seeds=1   >1: repeat with the first input's other seeds (<name>_s1, _s2 ...), i.e. score the stack over base-learner seeds
                   an input written <name>+k uses seed (b + k) mod base_seeds in round b (e.g. a 2nd MFE seed as 3rd learner)
    meta=fnn|mlr   mlr = paper's proposed meta-learner: ridge regression on one-hot targets, alpha=0.1 (eq. 5)
                   dt = paper's Stacking-DT baseline (sklearn default decision tree)
  Tuning score: 2-fold cross-fit over B (B_fit <-> B_val), predictions pooled over all of B.
  final=1: fit with the full protocol, score on the 10,000 test wafers, save probs.
Results append to logs/stack_trials.jsonl. Saved per (base seed b, seed s): tuning -> pooled B predictions
data/stack_cv_<NAME>_b<b>_s<s>.pkl (for compare.py / calibrate.py); final -> data/stack_<NAME>_b<b>_s<s>.pkl.
MLR runs also save the ridge coefficients ('coef', 9 x n_inputs*9 per fit; 2 fits for the 2-fold B-cv).
"""
import sys, json, time, itertools
import numpy as np
from sklearn.model_selection import train_test_split
from common import *
import tensorflow as tf

name, kv = sys.argv[1], dict(s.split('=') for s in sys.argv[2:])
cfg = dict(inputs='mfe,cnn', hidden=10, lr=1e-4, batch=32, cw=1.0, mode='A', monitor='loss',
           patience=20, dropout=0.0, l2=0.0, seeds=5, final=0, meta='fnn', alpha=0.1, base_seeds=1)
cfg.update({k: type(cfg[k])(v) for k, v in kv.items()})
SRC = {'mfe': 'mfe_fnn_outputs.pkl', 'cnn': 'cnn_outputs.pkl', 'xgb': 'xgb_outputs.pkl'}

y, yt = labels()
A, B_fit, B_val = splits(y)


def load_inputs(bs):
    names = []
    for i, s in enumerate(cfg['inputs'].split(',')):
        s, k = s.split('+') if '+' in s else (s, 0 if i == 0 else None)
        b = None if k is None else (bs + int(k)) % cfg['base_seeds']
        names.append(s + (f'_s{b}' if b else ''))
    outs = [ld(SRC.get(s, f'{s}_outputs.pkl')) for s in names]
    return (np.hstack([o['train_prob'] for o in outs]).astype(np.float32),
            np.hstack([o['test_prob'] for o in outs]).astype(np.float32))
oh = lambda v: tf.keras.utils.to_categorical(v, 9)


class F1Stop(tf.keras.callbacks.Callback):
    """Early stopping on macro-F1 of the early-stopping set, restoring the best weights."""
    def __init__(self, Xe, ye, patience):
        super().__init__(); self.Xe, self.ye, self.p = Xe, ye, patience
        self.best, self.wait, self.w = -1, 0, None

    def on_epoch_end(self, epoch, logs=None):
        f = f1s(self.ye, self.model.predict(self.Xe, batch_size=4096, verbose=0)).mean()
        if f > self.best:
            self.best, self.wait, self.w = f, 0, self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.p:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.model.set_weights(self.w)


def fit_predict(fit_idx, es_idx, Xp, seed):
    if cfg['meta'] == 'dt':  # paper's Stacking-DT baseline: scikit-learn DecisionTreeClassifier, default hyperparameters
        from sklearn.tree import DecisionTreeClassifier
        t = DecisionTreeClassifier(random_state=seed).fit(X[fit_idx], y[fit_idx])
        return t.predict_proba(Xp), 0
    if cfg['meta'] == 'mlr':  # deterministic, no early stopping; cw>0 -> class-weighted least squares
        from sklearn.linear_model import Ridge
        w = class_weights(y[fit_idx], cfg['cw'])
        r = Ridge(alpha=cfg['alpha'], fit_intercept=False).fit(X[fit_idx], oh(y[fit_idx]), sample_weight=np.array([w[c] for c in y[fit_idx]]))
        coefs.append(r.coef_.astype(np.float32))
        return r.predict(Xp), 0
    tf.keras.utils.set_random_seed(seed)
    reg = tf.keras.regularizers.l2(cfg['l2']) if cfg['l2'] else None
    m = tf.keras.Sequential([tf.keras.layers.Input((X.shape[1],)),
                             tf.keras.layers.Dense(cfg['hidden'], activation='relu', kernel_regularizer=reg),
                             tf.keras.layers.Dropout(cfg['dropout']),
                             tf.keras.layers.Dense(9, activation='softmax')])
    # steps_per_execution only batches the Python->graph calls; the optimisation is unchanged
    m.compile(tf.keras.optimizers.Adam(cfg['lr']), 'categorical_crossentropy', steps_per_execution=64)
    cb = (F1Stop(X[es_idx], y[es_idx], cfg['patience']) if cfg['monitor'] == 'f1' else
          tf.keras.callbacks.EarlyStopping(patience=cfg['patience'], restore_best_weights=True))
    h = m.fit(X[fit_idx], oh(y[fit_idx]), validation_data=(X[es_idx], oh(y[es_idx])), epochs=1000,
              batch_size=cfg['batch'], class_weight=class_weights(y[fit_idx], cfg['cw']), callbacks=[cb], verbose=0)
    return m.predict(Xp, batch_size=4096, verbose=0), len(h.history['loss'])


def plan(held):
    """(fit_idx, es_idx) for the given held-out pool, according to mode."""
    if cfg['mode'] == 'A':
        return A, held
    f, e = train_test_split(held, test_size=0.2, random_state=0, stratify=y[held])
    return (f, e) if cfg['mode'] == 'B' else (np.concatenate([A, f]), e)


t0, rows, eps = time.time(), [], []
for bs, seed in itertools.product(range(cfg['base_seeds']), range(cfg['seeds'])):
    X, Xt = load_inputs(bs)
    coefs = []
    if cfg['final']:
        fi, ei = plan(np.concatenate([B_fit, B_val]))
        p, ep = fit_predict(fi, ei, Xt, seed); yy = yt
        rows.append(f1s(yt, p)); eps.append(ep)
        dump({'test_prob': p, 'cfg': cfg, 'coef': coefs}, f'stack_{name}_b{bs}_s{seed}.pkl')
    else:
        pool = np.zeros((len(y), 9), np.float32)
        for h1, h2 in [(B_fit, B_val), (B_val, B_fit)]:
            fi, ei = plan(h1)
            pool[h2], ep = fit_predict(fi, ei, X[h2], seed); eps.append(ep)
        rows.append(f1s(y[CUT:], pool[CUT:]))
        dump({'B_prob': pool[CUT:], 'cfg': cfg, 'coef': coefs}, f'stack_cv_{name}_b{bs}_s{seed}.pkl')
F = np.array(rows); mac = F.mean(1)
rec = dict(name=name, cfg=cfg, macro_mean=mac.mean(), macro_std=mac.std(), macro_seeds=mac.tolist(),
           per_class=F.mean(0).tolist(), epochs=eps, minutes=(time.time() - t0) / 60)
with open(P('logs', 'stack_trials.jsonl'), 'a') as f:
    f.write(json.dumps(rec) + '\n')
print(f"{name:24s} {'TEST' if cfg['final'] else 'B-cv'} macro-F1 {mac.mean():.4f} ± {mac.std():.4f}  "
      f"NearFull {F[:, 7].mean():.3f}  epochs {eps}  {rec['minutes']:.1f} min", flush=True)
