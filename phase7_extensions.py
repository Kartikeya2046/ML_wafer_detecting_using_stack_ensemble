"""Original extensions: (1) 3-learner stack (+XGBoost on MFE features), (2) calibrated reject option.
Meta-learner uses balanced class weights (the paper's repo trains it unweighted; that collapses rare classes here)."""
import os, pickle, numpy as np, xgboost as xgb, tensorflow as tf
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

D = os.path.dirname(os.path.abspath(__file__)); P = lambda *a: os.path.join(D, *a)
NAMES = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full','none']
ld = lambda n: pickle.load(open(P('data', n), 'rb'))
m, c = ld('mfe_fnn_outputs.pkl'), ld('cnn_outputs.pkl')
Xtr, Xte = ld('X_MFE.pkl'); y, yt = m['y_train'], m['y_test']
cw = compute_class_weight('balanced', classes=np.arange(9), y=y); sw = cw[y]

# ---- 1. third base learner: XGBoost on the 59 handcrafted features; OOF probs for train (no in-sample leakage)
if not os.path.exists(P('data', 'xgb_outputs.pkl')):
    mk = lambda: xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method='hist', n_jobs=-1, random_state=0)
    oof = np.zeros((len(y), 9), np.float32)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y):
        oof[va] = mk().fit(Xtr[tr], y[tr], sample_weight=sw[tr]).predict_proba(Xtr[va]); print('fold done', flush=True)
    te = mk().fit(Xtr, y, sample_weight=sw).predict_proba(Xte)
    pickle.dump({'train_prob': oof, 'test_prob': te}, open(P('data', 'xgb_outputs.pkl'), 'wb'))
x = ld('xgb_outputs.pkl')
print('XGB standalone macro-F1 %.4f' % f1_score(yt, x['test_prob'].argmax(1), average='macro'), flush=True)

# ---- 2. stack with 2 learners (paper) vs 3 learners (ours), 5 seeds each
def stack(Xa, Xb, seed):
    tf.keras.utils.set_random_seed(seed)
    net = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(Xa.shape[1],)),
        tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(9, activation='softmax')])
    net.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy')
    net.fit(Xa, tf.keras.utils.to_categorical(y, 9), validation_split=0.2, epochs=1000, batch_size=32,
            class_weight=dict(enumerate(cw)), verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
    return net.predict(Xb, verbose=0)
cfg = {'Stack-2 (paper: MFE+CNN)': ([m, c]), 'Stack-3 (ours: MFE+CNN+XGB)': ([m, c, x])}
res, probs = {}, {}
for k, srcs in cfg.items():
    A = np.hstack([s['train_prob'] for s in srcs]); B = np.hstack([s['test_prob'] for s in srcs])
    ps = [stack(A, B, s) for s in range(5)]
    f = np.array([f1_score(yt, p.argmax(1), average=None, labels=range(9)) for p in ps])
    res[k] = f; probs[k] = np.mean(ps, 0); print(k, 'macro %.4f +- %.4f' % (f.mean(1).mean(), f.mean(1).std()), flush=True)

# ---- 3. calibration (temperature, 2-fold cross-fit on test) + reject option, on the 3-learner ensemble
def ece(p, yy, bins=10):
    conf, ok = p.max(1), p.argmax(1) == yy; e = 0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        s = (conf > lo) & (conf <= lo + 1 / bins)
        if s.any(): e += s.mean() * abs(ok[s].mean() - conf[s].mean())
    return e
pr = probs['Stack-3 (ours: MFE+CNN+XGB)']; lg = np.log(np.clip(pr, 1e-9, 1))
idx = np.random.RandomState(0).permutation(len(yt)); halves = [idx[:5000], idx[5000:]]
cal = np.zeros_like(pr)
for a, b in [(0, 1), (1, 0)]:
    T = minimize_scalar(lambda t: -np.log(np.clip(softmax(lg[halves[a]] / t, 1)[np.arange(5000), yt[halves[a]]], 1e-9, 1)).mean(), bounds=(.05, 10), method='bounded').x
    cal[halves[b]] = softmax(lg[halves[b]] / T, 1)
print('ECE raw %.4f -> calibrated %.4f' % (ece(pr, yt), ece(cal, yt)), flush=True)
order = np.argsort(-cal.max(1)); cov = np.array([1.0, .98, .95, .9, .85, .8])
rows = []
for cv in cov:
    k = order[:int(cv * len(yt))]; pred = cal[k].argmax(1)
    rows.append((cv, accuracy_score(yt[k], pred), f1_score(yt[k], pred, average='macro', labels=range(9))))
xs = np.linspace(.5, 1, 51); plt.figure(figsize=(5, 3.5))
plt.plot(xs, [1 - (cal[order[:int(v * len(yt))]].argmax(1) != yt[order[:int(v * len(yt))]]).mean() for v in xs])
plt.xlabel('coverage (fraction auto-classified)'); plt.ylabel('accuracy on accepted'); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(P('reject_curve.png'), dpi=150)

# ---- report
o = ['# Extensions Results (original contribution)\n', '## 1. Per-class F1, mean of 5 seeds (test n=10,000)\n',
     '| Class | MFE+FNN | CNN | XGB (new) | ' + ' | '.join(cfg) + ' |', '|---|---|---|---|---|---|']
base = [f1_score(yt, s['test_prob'].argmax(1), average=None, labels=range(9)) for s in (m, c, x)]
for i, n in enumerate(NAMES): o.append(f'| {n} | ' + ' | '.join('%.4f' % b[i] for b in base) + ' | ' + ' | '.join('%.4f' % res[k][:, i].mean() for k in cfg) + ' |')
o.append('| **Macro** | ' + ' | '.join('**%.4f**' % b.mean() for b in base) + ' | ' + ' | '.join('**%.4f ± %.4f**' % (res[k].mean(1).mean(), res[k].mean(1).std()) for k in cfg) + ' |')
o += ['\n## 2. Calibration and reject option (3-learner stack)\n', f'ECE: {ece(pr, yt):.4f} raw -> {ece(cal, yt):.4f} after temperature scaling (2-fold cross-fit).\n',
      '| Coverage | Accuracy | Macro-F1 (accepted) |', '|---|---|---|'] + ['| %.0f%% | %.4f | %.4f |' % (a * 100, b, f) for a, b, f in rows]
open(P('extensions_results.md'), 'w').write('\n'.join(o)); pickle.dump({'res': res, 'probs': probs, 'cal': cal}, open(P('data', 'extension_outputs.pkl'), 'wb'))
print('\n'.join(o))
