"""MFE + FNN base learner tuning (Part B3).

Usage:  run_cpu.sh mfe_tune.py NAME [key=value ...]
  keys (defaults = paper Table 3 / notebook cell 4):
    width=128 depth=2 act=tanh dropout=0 lr=1e-4 batch=32 cw=1.0 patience=20 seeds=3
    save=1: also save each seed's model (models/mfe_<NAME>[_s<seed>].keras) and the feature scaling (models/mfe_<NAME>_scale.npz)
Fits on A, early-stops on B_fit, scores standalone macro-F1 on B_val. Each seed's softmax (all train + test)
is saved to data/mfe_<NAME>[_s<seed>]_outputs.pkl; stack_tune.py uses it via inputs=mfe_<NAME>,cnn
(add base_seeds=3 there to average the stack over the 3 MFE seeds).
Results append to logs/mfe_trials.jsonl.
"""
import sys, json, time
import numpy as np
from common import *
import tensorflow as tf

name, kv = sys.argv[1], dict(s.split('=') for s in sys.argv[2:])
cfg = dict(width=128, depth=2, act='tanh', dropout=0.0, lr=1e-4, batch=32, cw=1.0, patience=20, seeds=3, save=0)
cfg.update({k: type(cfg[k])(v) for k, v in kv.items()})

Xtr, Xte = ld('X_MFE.pkl')
y, yt = labels()
A, B_fit, B_val = splits(y)
mu, sd = Xtr[A].mean(0), Xtr[A].std(0)
sd[sd == 0] = 1
Xtr, Xte = ((Xtr - mu) / sd).astype(np.float32), ((Xte - mu) / sd).astype(np.float32)
if cfg['save']:
    np.savez(P('models', f'mfe_{name}_scale.npz'), mu=mu, sd=sd)
oh = lambda v: tf.keras.utils.to_categorical(v, 9)

t0, rows, eps = time.time(), [], []
for seed in range(cfg['seeds']):
    tf.keras.utils.set_random_seed(seed)
    m = tf.keras.Sequential([tf.keras.layers.Input((59,))] +
                            sum([[tf.keras.layers.Dense(cfg['width'], activation=cfg['act']),
                                  tf.keras.layers.Dropout(cfg['dropout'])] for _ in range(cfg['depth'])], []) +
                            [tf.keras.layers.Dense(9, activation='softmax')])
    m.compile(tf.keras.optimizers.Adam(cfg['lr']), 'categorical_crossentropy', steps_per_execution=64)
    h = m.fit(Xtr[A], oh(y[A]), validation_data=(Xtr[B_fit], oh(y[B_fit])), epochs=1000, batch_size=cfg['batch'],
              class_weight=class_weights(y[A], cfg['cw']), verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=cfg['patience'], restore_best_weights=True)])
    eps.append(len(h.history['loss']))
    ptr = m.predict(Xtr, batch_size=4096, verbose=0)
    rows.append(f1s(y[B_val], ptr[B_val]))
    if cfg['save']:
        m.save(P('models', f'mfe_{name}.keras' if seed == 0 else f'mfe_{name}_s{seed}.keras'))
    dump({'train_prob': ptr, 'test_prob': m.predict(Xte, batch_size=4096, verbose=0), 'cfg': cfg},
         f'mfe_{name}_outputs.pkl' if seed == 0 else f'mfe_{name}_s{seed}_outputs.pkl')
F = np.array(rows); mac = F.mean(1)
rec = dict(name=name, cfg=cfg, macro_mean=mac.mean(), macro_std=mac.std(), macro_seeds=mac.tolist(),
           per_class=F.mean(0).tolist(), epochs=eps, minutes=(time.time() - t0) / 60)
with open(P('logs', 'mfe_trials.jsonl'), 'a') as f:
    f.write(json.dumps(rec) + '\n')
print(f"{name:16s} B_val macro-F1 {mac.mean():.4f} ± {mac.std():.4f}  epochs {eps}  {rec['minutes']:.1f} min", flush=True)
