"""Shared helpers for Stage 1 tuning (implementation_plan.md Part B).

Split used everywhere in Part B (fixed, see tuning_log.md):
  A  = train[:130356]  -> base learners are fitted on this (Keras validation_split=0.2 kept the first 80%)
  B  = train[130356:]  -> base learners only early-stop on this, so their predictions here are honest
  B is split once, stratified, into B_fit (early stopping / stacker fitting) and B_val (tuning score).
  The 10,000 test wafers are touched only for final reported numbers.
"""
import os, pickle
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

D = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(D, *a)
NAMES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full', 'none']
N_TRAIN, CUT = 162946, 130356  # CUT = int(0.8 * N_TRAIN) - Keras validation_split boundary


def ld(name):
    with open(P('data', name), 'rb') as f:
        return pickle.load(f)


def dump(obj, name):
    with open(P('data', name), 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def labels():
    return ld('y.pkl')  # (y_train, y_test)


def splits(y):
    """Index arrays into the training set: A, B_fit, B_val."""
    A = np.arange(CUT)
    B = np.arange(CUT, len(y))
    B_fit, B_val = train_test_split(B, test_size=0.5, random_state=0, stratify=y[B])
    return A, np.sort(B_fit), np.sort(B_val)


def class_weights(y, power=1.0):
    """power=1 balanced, 0.5 sqrt-balanced, 0 none."""
    if power == 0:  # no class weights (the frozen recipe); also safe when a small subset misses a class
        return {c: 1.0 for c in range(9)}
    present = np.unique(y)
    w = dict(zip(present, compute_class_weight('balanced', classes=present, y=y) ** power))
    return {c: w.get(c, 1.0) for c in range(9)}


def f1s(y, prob):
    pred = prob.argmax(1) if prob.ndim == 2 else prob
    return f1_score(y, pred, average=None, labels=range(9), zero_division=0)


def gpu_setup(mixed=False):
    import tensorflow as tf
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    if mixed:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    return tf.config.list_physical_devices('GPU')


def subset(n, rep):
    """Training-size sweep (paper 4.2: N in {500, 5000, 50000}): a random sample of n training wafers for replicate rep,
    split 80/20 into (fit, es) - fit for backpropagation, es for early stopping, as in the paper."""
    idx = np.random.default_rng(1000 * rep + 7).choice(N_TRAIN, n, replace=False)
    k = int(0.8 * n)
    return np.sort(idx[:k]), np.sort(idx[k:])
