"""Stage 2 C3 (optional): Grad-CAM heatmaps for the frozen CNN (stage2_plan.md step 4).

Usage:  run_gpu.sh gradcam.py [tag=imnet_cw0_aug] [layer=block4_conv3] [n=3]
For each class: n correctly classified test wafers (highest confidence) + up to 2 misclassified ones.
Layer: block4_conv3 (8x8 maps) - the last conv layer block5_conv3 is only 4x4 on 64x64 inputs, too coarse to read.
Heatmap = ReLU(sum_k mean(dy_c/dA_k) * A_k) for the predicted class c, upsampled to 64x64.
Figure -> figures/gradcam_<tag>.png.
"""
import sys
import numpy as np
from common import *
kv = dict(s.split('=', 1) for s in sys.argv[1:])
tag, layer, n = kv.get('tag', 'imnet_cw0_aug'), kv.get('layer', 'block4_conv3'), int(kv.get('n', 3))
gpu_setup()
import tensorflow as tf
L = tf.keras.layers

x = inp = L.Input((64, 64, 1))
x = L.Lambda(lambda t: tf.repeat(t, 3, axis=-1))(x)
vgg = tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=(64, 64, 3))
x = L.GlobalAveragePooling2D()(vgg(x))
model = tf.keras.Model(inp, L.Dense(9, activation='softmax', dtype='float32')(x))
model.load_weights(P('models', f'cnn_{tag}.keras'))  # load_model cannot unmarshal the Lambda across Python versions
names = [l.name for l in vgg.layers]
head_layers = vgg.layers[names.index(layer) + 1:]
conv = tf.keras.Model(vgg.input, vgg.get_layer(layer).output)
gap, dense = model.layers[-2], model.layers[-1]


def cam(xb):
    x3 = tf.repeat(tf.constant(xb, tf.float32), 3, axis=-1)
    with tf.GradientTape() as t:
        A = conv(x3); t.watch(A); h = A
        for l in head_layers:
            h = l(h)
        p = dense(gap(h))
        c = tf.argmax(p, 1)
        score = tf.gather(p, c, batch_dims=1)
    g = t.gradient(score, A)
    m = tf.nn.relu(tf.reduce_sum(tf.reduce_mean(g, (1, 2), keepdims=True) * A, -1))
    m = tf.image.resize(m[..., None], (64, 64), 'bilinear')[..., 0].numpy()
    return m / (m.max((1, 2), keepdims=True) + 1e-8), p.numpy()


_, Xte = ld('X_CNN.pkl')
_, yt = labels()
X = (Xte.reshape(-1, 64, 64, 1).astype(np.float32) - 0.5) * 2.0
prob = model.predict(X, batch_size=512, verbose=0)
pred, conf = prob.argmax(1), prob.max(1)
rows = []
for c in range(9):
    ok = np.nonzero((yt == c) & (pred == c))[0]; bad = np.nonzero((yt == c) & (pred != c))[0]
    rows.append(list(ok[np.argsort(-conf[ok])][:n]) + list(bad[np.argsort(-conf[bad])][:2]))
sel = np.array([i for r in rows for i in r])
H, Pp = cam(X[sel])
hm = dict(zip(sel, H))

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ncol = n + 2
fig, ax = plt.subplots(9, ncol, figsize=(1.7 * ncol, 1.75 * 9))
for c, r in enumerate(rows):
    for j in range(ncol):
        a = ax[c, j]; a.set_xticks([]); a.set_yticks([])
        if j < len(r):
            i = r[j]
            a.imshow(Xte[i].reshape(64, 64).astype(np.float32), cmap='gray_r', vmin=0, vmax=1)
            a.imshow(hm[i], cmap='jet', alpha=0.45, vmin=0, vmax=1)
            a.set_title(('' if pred[i] == c else 'as ' + NAMES[pred[i]] + ' ') + f'{conf[i]:.2f}', fontsize=7,
                        color='#222' if pred[i] == c else '#c0392b')
        else:
            a.axis('off')
    ax[c, 0].set_ylabel(NAMES[c], fontsize=9)
fig.suptitle(f'Grad-CAM ({layer}) on test wafers — {n} correct (left) + misclassified (red)', fontsize=10)
fig.tight_layout()
os.makedirs(P('figures'), exist_ok=True)
fig.savefig(P('figures', f'gradcam_{tag}.png'), dpi=150)
print('saved', P('figures', f'gradcam_{tag}.png'), '| test macro-F1 (plain, no TTA)', round(f1s(yt, prob).mean(), 4))
