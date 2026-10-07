"""CNN base learner (VGG16-style, paper Table 4) - GPU training for Part B4.

Usage:  run_gpu.sh cnn_train.py --tag NAME [--aug] [--smooth 0.1] [--plateau] [--batch 32] [--fp32] [--imagenet] [--cw 0]
        run_gpu.sh cnn_train.py --bench
Fits on A, early-stops on B_fit, saves softmax for train (all 162,946) + test to data/cnn_<tag>_outputs.pkl.
"""
import argparse, time
import numpy as np
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--tag'); ap.add_argument('--aug', action='store_true'); ap.add_argument('--smooth', type=float, default=0.0)
ap.add_argument('--plateau', action='store_true'); ap.add_argument("--batch", type=int, default=32)  # 128 was 1.5x faster but cost ~4 pts macro-F1 (run ctrl)
ap.add_argument('--fp32', action='store_true'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--bench', action='store_true')
ap.add_argument('--imagenet', action='store_true')  # paper 4.3: VGG pre-trained on ImageNet
ap.add_argument('--cw', type=float, default=1.0)  # class-weight power: 1 balanced, 0 none (paper)
a = ap.parse_args()

print('GPUs:', gpu_setup(mixed=not a.fp32), flush=True)
import tensorflow as tf
L = tf.keras.layers


def build():
    x = inp = L.Input((64, 64, 1))
    x = L.Lambda(lambda t: tf.repeat(t, 3, axis=-1))(x)
    if a.imagenet:  # same conv stack as below, ImageNet weights
        x = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(64, 64, 3))(x)
    for n, f in [] if a.imagenet else [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]:
        for _ in range(n):
            x = L.Conv2D(f, 3, padding='same', activation='relu')(x)
        x = L.MaxPooling2D(2)(x)
    x = L.GlobalAveragePooling2D()(x)
    out = L.Dense(9, activation='softmax', dtype='float32')(x)  # float32 head for mixed precision stability
    return tf.keras.Model(inp, out)


def augment(x, *rest):
    # per-sample flip + 90-degree rotation (the 8 symmetries of the square), vectorised over the batch
    n = tf.shape(x)[0]
    rots = tf.stack([tf.image.rot90(x, i) for i in range(4)], axis=1)  # (n, 4, 64, 64, 1)
    x = tf.gather(rots, tf.random.uniform([n], 0, 4, tf.int32), batch_dims=1)
    x = tf.where(tf.random.uniform([n, 1, 1, 1]) < 0.5, tf.reverse(x, [2]), x)
    return (x, *rest)


def ds(idx, src, mode='pred', aug=False):
    """Batches gathered by index from one CPU-resident image store (no per-split copies of the 1.3 GB array).
    mode: 'train' -> (x, y, w) shuffled, 'val' -> (x, y), 'pred' -> x."""
    def get(i):
        x = (tf.cast(tf.gather(src, i), tf.float32) - 0.5) * 2.0  # stored float16 in [0,1] -> [-1,1] as in the paper repo
        if mode == 'pred':
            return x
        return (x, tf.gather(YT, i)) + ((tf.gather(WT, i),) if mode == 'train' else ())
    d = tf.data.Dataset.from_tensor_slices(np.asarray(idx, np.int64))
    if mode == 'train':
        d = d.shuffle(len(idx), seed=a.seed, reshuffle_each_iteration=True)
    d = d.batch(a.batch).map(get, num_parallel_calls=tf.data.AUTOTUNE)
    if aug:
        d = d.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return d.prefetch(tf.data.AUTOTUNE)


Xtr, Xte = ld('X_CNN.pkl')
y, yt = labels()
A, B_fit, B_val = splits(y)
cw = class_weights(y[A], a.cw)
with tf.device('/CPU:0'):  # Variables are captured by reference in tf.data (constants would hit the 2 GB graph limit)
    XT = tf.Variable(Xtr.reshape(-1, 64, 64, 1), trainable=False)  # float16, the only copy
    XE = tf.Variable(Xte.reshape(-1, 64, 64, 1), trainable=False)
    YT = tf.constant(tf.keras.utils.to_categorical(y, 9))
    WT = tf.constant(np.array([cw[c] for c in y], np.float32))
del Xtr, Xte

if a.bench:
    for bs, mp in [(32, False), (32, True), (64, True), (128, True)]:
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy('mixed_float16' if mp else 'float32')
        a.batch = bs
        m = build(); m.compile(tf.keras.optimizers.Adam(1e-4), 'categorical_crossentropy')
        d = ds(np.arange(bs * 160), XT, 'val')
        m.fit(d.take(20), verbose=0)
        t = time.time(); m.fit(d.skip(20), verbose=0); dt = time.time() - t
        print(f'batch {bs:4d} mixed={mp}: {140 / dt:6.1f} steps/s, {140 * bs / dt:7.0f} img/s, '
              f'est epoch(130k) {len(A) / (140 * bs / dt):5.0f}s', flush=True)
    raise SystemExit

tf.keras.utils.set_random_seed(a.seed)
model = build()
opt = tf.keras.optimizers.Adam(1e-4 * (a.batch / 32) ** 0.5)  # sqrt LR scaling for larger batch (no BN, so linear is risky)
if not a.fp32:
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
model.compile(opt, tf.keras.losses.CategoricalCrossentropy(label_smoothing=a.smooth), metrics=['accuracy'])

cbs = [tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
       tf.keras.callbacks.CSVLogger(P('logs', f'cnn_{a.tag}.csv'))]
if a.plateau:
    cbs.append(tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6))
t0 = time.time()
model.fit(ds(A, XT, 'train', aug=a.aug), validation_data=ds(B_fit, XT, 'val'),
          epochs=1000, callbacks=cbs, verbose=2)
print(f'train time {(time.time() - t0) / 3600:.2f} h', flush=True)

pred = lambda src: model.predict(ds(np.arange(src.shape[0]), src), verbose=0).astype(np.float32)
out = {'train_prob': pred(XT), 'test_prob': pred(XE), 'args': vars(a)}
model.save(P('models', f'cnn_{a.tag}.keras'))
dump(out, f'cnn_{a.tag}_outputs.pkl')
print(f'[{a.tag}] B_val macro-F1 {f1s(y[B_val], out["train_prob"][B_val]).mean():.4f}', flush=True)
