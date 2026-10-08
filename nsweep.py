"""Paper's training-size sweep (Table 7 / Fig. 3, N in {500, 5000, 50000}) with our Stage 1 recipes.

Usage:  run_cpu.sh nsweep.py mfe N REP        MFE-FNN on the replicate's subset (CPU)
        run_gpu.sh cnn_train.py --tag nsN_rREP --subset N --rep REP --seed REP --imagenet --cw 0 --fp32 --aug   (CNN, GPU)
        run_cpu.sh nsweep.py stack            stackers + table for every (N, REP) with both base learners done
Protocol = paper 4.2: a random sample of N training wafers per replicate (`common.subset`), 80% for backpropagation,
20% for early stopping; the meta-classifier is fitted on the base classifiers' outputs for the 80% part (paper protocol,
as the Stage 1 A-mode headline); all models scored on the same 10,000 test wafers.
Base-learner recipes are our frozen Stage 1 ones (MFE-FNN Table 3 no class weights; VGG16 ImageNet, no cw, aug, fp32).
Outputs: data/nsweep/*.pkl; table -> nsweep_results.md; records -> logs/nsweep.jsonl.
"""
import sys, glob, json, re
import numpy as np
from common import *

y, yt = labels()
PAPER = {  # Table 7 F1macro: MFE+FNN, CNN, MultiNN, Stacking-DT, Stacking-FNN, Stacking-MLR
    500: ['0.5558 ± 0.0376', '0.4937 ± 0.0642', '0.5371 ± 0.0902', '0.5179 ± 0.0368', '0.5085 ± 0.0662', '0.5872 ± 0.0542'],
    5000: ['0.6983 ± 0.0371', '0.6954 ± 0.0420', '0.6969 ± 0.0309', '0.7217 ± 0.0220', '0.7350 ± 0.0556', '0.7599 ± 0.0202'],
    50000: ['0.8310 ± 0.0109', '0.8397 ± 0.0213', '0.8201 ± 0.0174', '0.8415 ± 0.0116', '0.8746 ± 0.0088', '0.8686 ± 0.0078'],
    162946: ['0.8599 ± 0.0117', '0.8679 ± 0.0126', '0.8455 ± 0.0170', '0.8789 ± 0.0094', '0.8991 ± 0.0096', '0.8949 ± 0.0121']}
os.makedirs(P('data', 'nsweep'), exist_ok=True)


def mfe(n, rep):
    import tensorflow as tf
    fit, es = subset(n, rep)
    Xtr, Xte = ld('X_MFE.pkl')
    mu, sd = Xtr[fit].mean(0), Xtr[fit].std(0); sd[sd == 0] = 1
    Xtr, Xte = ((Xtr - mu) / sd).astype(np.float32), ((Xte - mu) / sd).astype(np.float32)
    oh = lambda v: tf.keras.utils.to_categorical(v, 9)
    tf.keras.utils.set_random_seed(rep)
    m = tf.keras.Sequential([tf.keras.layers.Input((59,)), tf.keras.layers.Dense(128, activation='tanh'),
                             tf.keras.layers.Dense(128, activation='tanh'), tf.keras.layers.Dense(9, activation='softmax')])
    m.compile(tf.keras.optimizers.Adam(1e-4), 'categorical_crossentropy')
    h = m.fit(Xtr[fit], oh(y[fit]), validation_data=(Xtr[es], oh(y[es])), epochs=1000, batch_size=32, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
    dump({'fit_idx': fit, 'fit_prob': m.predict(Xtr[fit], batch_size=4096, verbose=0),
          'test_prob': m.predict(Xte, batch_size=4096, verbose=0), 'epochs': len(h.history['loss'])},
         f'nsweep/mfe_N{n}_r{rep}.pkl')
    print(f'mfe N={n} rep {rep}: epochs {len(h.history["loss"])}', flush=True)


def stack():
    import tensorflow as tf
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeClassifier
    oh = lambda v: tf.keras.utils.to_categorical(v, 9)
    res = {}
    for f in sorted(glob.glob(P('data', 'nsweep', 'cnn_N*_r*.pkl'))):
        n, rep = map(int, re.search(r'cnn_N(\d+)_r(\d+)\.pkl', f).groups())
        if not os.path.exists(P('data', 'nsweep', f'mfe_N{n}_r{rep}.pkl')):
            continue
        M, C = ld(f'nsweep/mfe_N{n}_r{rep}.pkl'), ld(f'nsweep/cnn_N{n}_r{rep}.pkl')
        assert (M['fit_idx'] == C['fit_idx']).all()
        fit = M['fit_idx']
        X, Xt = np.hstack([M['fit_prob'], C['fit_prob']]), np.hstack([M['test_prob'], C['test_prob']])
        r = Ridge(alpha=0.1, fit_intercept=False).fit(X, oh(y[fit]))
        t = DecisionTreeClassifier(random_state=rep).fit(X, y[fit])
        tf.keras.utils.set_random_seed(rep)  # paper's Stacking-FNN: 18 -> 10 -> 9, early stopping on 20% of the fit part
        k = int(0.8 * len(fit))
        nn = tf.keras.Sequential([tf.keras.layers.Input((18,)), tf.keras.layers.Dense(10, activation='relu'),
                                  tf.keras.layers.Dense(9, activation='softmax')])
        nn.compile(tf.keras.optimizers.Adam(1e-4), 'categorical_crossentropy')
        nn.fit(X[:k], oh(y[fit][:k]), validation_data=(X[k:], oh(y[fit][k:])), epochs=1000, batch_size=32, verbose=0,
               callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
        sc = {'MFE+FNN': M['test_prob'], 'CNN': C['test_prob'], 'Stacking-DT': t.predict_proba(Xt),
              'Stacking-FNN': nn.predict(Xt, verbose=0), 'Stacking-MLR': r.predict(Xt)}
        res[n, rep] = {k_: (f1s(yt, p).mean(), (p.argmax(1) == yt).mean()) for k_, p in sc.items()}
        with open(P('logs', 'nsweep.jsonl'), 'a') as h:
            h.write(json.dumps(dict(N=n, rep=rep, **{k_: v for k_, v in res[n, rep].items()})) + '\n')
    models = ['MFE+FNN', 'CNN', 'Stacking-DT', 'Stacking-FNN', 'Stacking-MLR']
    pidx = [0, 1, 3, 4, 5]
    L = ['# Training-size sweep — ours vs paper Table 7 (macro-F1 on the 10,000 test wafers)', '',
         'Built by `nsweep.py stack`. Ours: mean ± std over replicates (random training subsets, paper protocol); '
         'paper: 10 replications. N = 162,946 row of ours = Stage 1 frozen (`results_comparison.md`; stackers there use split A/B, see tuning_log.md).', '',
         '| N | reps | ' + ' | '.join(f'{m} ours | paper' for m in models) + ' |', '|---|---|' + '---|---|' * len(models)]
    for n in sorted({k[0] for k in res}):
        R = [res[k] for k in res if k[0] == n]
        cells = [f'{np.mean([r[m][0] for r in R]):.4f} ± {np.std([r[m][0] for r in R]):.4f} | {PAPER[n][i]}' for m, i in zip(models, pidx)]
        L.append(f'| {n:,} | {len(R)} | ' + ' | '.join(cells) + ' |')
    L.append(f'| 162,946 | 1–5 | 0.8572 ± 0.0062 | {PAPER[162946][0]} | 0.8777 | {PAPER[162946][1]} | 0.8782 ± 0.0065 | '
             f'{PAPER[162946][3]} | 0.9041 ± 0.0028 (tuned) | {PAPER[162946][4]} | 0.8967 ± 0.0060 | {PAPER[162946][5]} |')
    open(P('nsweep_results.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    if sys.argv[1] == 'mfe':
        mfe(int(sys.argv[2]), int(sys.argv[3]))
    else:
        stack()
