"""Final best pipeline (Stage 2b), runnable from saved models only:
    5 VGG16 CNNs (ImageNet init, aug, no class weights; seeds 0-4), each averaged over the 8 symmetries (TTA)
  + XGBoost on the 59 handcrafted features (depth 6, lr 0.05, balanced class weights)
  -> MLR stacker (ridge, alpha 0.1, fitted on split B) -> class scores; temperature softmax -> probabilities;
  optional reject option (send low-confidence wafers to a human).

Usage:  run_gpu.sh predict.py [verify=1] [coverage=0.95]
  Reads the preprocessed test set (data/X_CNN.pkl, data/X_MFE.pkl - made by phase2_data.py from LSWMD), writes
  data/final_predictions.pkl. verify=1 checks the result against the saved Stage 2b test outputs.
Model files: models/cnn_imnet_cw0_aug[_s1.._s4].keras (not in git: 170 MB each), models/xgb_d6_lr05_cw1.json,
models/final_pipeline.npz (MLR coefficients, temperature, reject thresholds) - written by `predict.py export=1`.
"""
import sys
import numpy as np
from common import *

kv = dict(s.split('=', 1) for s in sys.argv[1:])
CNNS = ['cnn_imnet_cw0_aug'] + [f'cnn_imnet_cw0_aug_s{s}' for s in range(1, 5)]
XGB = 'xgb_d6_lr05_cw1'

if kv.get('export'):  # collect the stacker + calibration numbers chosen in Stage 2b into one file
    st, cal = ld('stack_e5_ctlA_mlr_b0_s0.pkl'), ld('calib_e5_ctlA_mlr.pkl')
    r = cal['runs'][0]
    np.savez(P('models', 'final_pipeline.npz'), coef=st['coef'][0], T=r['T'],
             coverage=np.array([t['target'] for t in r['table']]), threshold=np.array([t['threshold'] for t in r['table']]),
             inputs=np.array(st['cfg']['inputs'].split(',')))
    print('wrote models/final_pipeline.npz:', st['cfg']['inputs'], 'T', round(r['T'], 4))
    raise SystemExit

gpu_setup()
import tensorflow as tf
import xgboost as xgb
from scipy.special import softmax
L = tf.keras.layers
F = np.load(P('models', 'final_pipeline.npz'))


def cnn():
    x = inp = L.Input((64, 64, 1))
    x = L.Lambda(lambda t: tf.repeat(t, 3, axis=-1))(x)
    x = tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=(64, 64, 3))(x)
    return tf.keras.Model(inp, L.Dense(9, activation='softmax', dtype='float32')(L.GlobalAveragePooling2D()(x)))


_, Xc = ld('X_CNN.pkl')
_, Xm = ld('X_MFE.pkl')
_, yt = labels()
X = (Xc.reshape(-1, 64, 64, 1).astype(np.float32) - 0.5) * 2.0  # same input scaling as training
syms = [np.rot90(np.flip(X, 2) if k >= 4 else X, k % 4, axes=(1, 2)) for k in range(8)]
p_cnn = 0
for name in CNNS:
    m = cnn(); m.load_weights(P('models', f'{name}.keras'))  # weights only: the Lambda cannot be unmarshalled across Pythons
    p_cnn += np.mean([m.predict(s, batch_size=128, verbose=0) for s in syms], 0) / len(CNNS)
    tf.keras.backend.clear_session()
b = xgb.XGBClassifier(); b.load_model(P('models', f'{XGB}.json'))
p_xgb = b.predict_proba(Xm)
scores = np.hstack([p_cnn, p_xgb]) @ F['coef'].T
prob = softmax(scores / float(F['T']), axis=1)
pred, conf = prob.argmax(1), prob.max(1)
cov = float(kv.get('coverage', 0.95))
th = float(F['threshold'][np.argmin(np.abs(F['coverage'] - cov))])
acc = conf >= th
dump({'scores': scores, 'prob': prob, 'pred': pred, 'accepted': acc}, 'final_predictions.pkl')
print(f'test macro-F1 {f1s(yt, pred).mean():.4f}  accuracy {(pred == yt).mean():.4f}')
print(f'reject option (target {cov:.0%}): auto-classified {acc.mean():.1%}, accuracy on them {(pred[acc] == yt[acc]).mean():.4f}, '
      f'{(~acc).sum()} wafers to a human')
if kv.get('verify'):
    ref = ld('stack_e5_ctlA_mlr_b0_s0.pkl')['test_prob']
    print(f'verify vs saved Stage 2b scores: max |diff| {np.abs(scores - ref).max():.2e}, same class {(pred == ref.argmax(1)).mean():.4f}')
