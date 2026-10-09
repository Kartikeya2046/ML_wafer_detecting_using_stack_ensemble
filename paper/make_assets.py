"""Build every table (paper/tables/*.tex) and figure (paper/figures/*.pdf) of the paper from saved outputs and logs.
Run from the project root:  .venv/bin/python paper/make_assets.py
No number in the paper is typed by hand: tables come from data/*.pkl and logs/*.jsonl, as in extensions_results.py.
"""
import os, sys, json, glob, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

OUT = P('paper')
os.makedirs(os.path.join(OUT, 'tables'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'figures'), exist_ok=True)
y, yt = labels()
A, B_fit, B_val = splits(y)
SHORT = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full', 'None']
# Kang & Kang (2021), Table 7 (macro, micro) and Table 8 (per-class, N = 162,946), from journal.pdf
PAPER7 = {  # N: model -> (macro mean, macro std, micro mean, micro std)
    500: {'MFE+FNN': (.5558, .0376, .9415, .0044), 'CNN': (.4937, .0642, .9292, .0092), 'MultiNN': (.5371, .0902, .9352, .0084),
          'Stacking-DT': (.5179, .0368, .9340, .0040), 'Stacking-FNN': (.5085, .0662, .9445, .0030), 'Stacking-MLR': (.5872, .0542, .9462, .0042)},
    5000: {'MFE+FNN': (.6983, .0371, .9600, .0011), 'CNN': (.6954, .0420, .9607, .0027), 'MultiNN': (.6969, .0309, .9583, .0020),
           'Stacking-DT': (.7217, .0220, .9607, .0025), 'Stacking-FNN': (.7350, .0556, .9675, .0024), 'Stacking-MLR': (.7599, .0202, .9664, .0018)},
    50000: {'MFE+FNN': (.8310, .0109, .9712, .0010), 'CNN': (.8397, .0213, .9731, .0023), 'MultiNN': (.8201, .0174, .9699, .0015),
            'Stacking-DT': (.8415, .0116, .9720, .0021), 'Stacking-FNN': (.8746, .0088, .9779, .0011), 'Stacking-MLR': (.8686, .0078, .9772, .0013)},
    162946: {'MFE+FNN': (.8599, .0117, .9741, .0010), 'CNN': (.8679, .0126, .9775, .0014), 'MultiNN': (.8455, .0170, .9728, .0015),
             'Stacking-DT': (.8789, .0094, .9757, .0011), 'Stacking-FNN': (.8991, .0096, .9808, .0010), 'Stacking-MLR': (.8949, .0121, .9801, .0010)}}
PAPER8_MLR = [0.9424, 0.8969, 0.8492, 0.9823, 0.7950, 0.8979, 0.7928, 0.9053, 0.9918]


def jl(f):
    with open(P('logs', f)) as h:
        return [json.loads(l) for l in h if l.strip()]


TR = {}
for r in jl('stack_trials.jsonl'):
    TR[r['name'], r['cfg']['final']] = r
CMP = {}
for r in jl('compare.jsonl'):
    CMP[r['a'], r['b'], r['test']] = r


def runs(name):  # test probabilities of a final stack_tune.py trial
    fs = [f for f in sorted(glob.glob(P('data', f'stack_{name}_b*_s*.pkl'))) if re.fullmatch(f'stack_{re.escape(name)}_b\\d+_s\\d+\\.pkl', os.path.basename(f))]
    return [ld(os.path.basename(f))['test_prob'] for f in fs]


def score(probs):  # (macro mean, macro std, micro mean, micro std) over runs
    ma = [f1s(yt, p).mean() for p in probs]; mi = [(p.argmax(1) == yt).mean() for p in probs]
    return np.mean(ma), np.std(ma), np.mean(mi), np.std(mi)


def pm(m, s, n=1, d=4):
    return f'{m:.{d}f}' if n == 1 else f'{m:.{d}f} $\\pm$ {s:.{d}f}'


def write(name, s):
    open(os.path.join(OUT, 'tables', name), 'w').write(s)


# ---------- Table: reproduction at N = 162,946 ----------
mfe = [ld(f)['test_prob'] for f in ['mfe_x3_cw0_outputs.pkl', 'mfe_x3_cw0_s1_outputs.pkl', 'mfe_x3_cw0_s2_outputs.pkl']]
REPRO = [('MFE+FNN', mfe, '3 seeds'), ('CNN', [ld('cnn_imnet_cw0_aug_outputs.pkl')['test_prob']], '1 run'),
         ('MultiNN', [ld('cnn_multinn_outputs.pkl')['test_prob']], '1 run'),
         ('Stacking-DT', runs('FINAL_stack_dt'), '3 seeds'), ('Stacking-FNN', runs('FINAL_stack_fnn'), '5 seeds'),
         ('Stacking-MLR', runs('FINAL_stack_mlr'), '3 seeds')]
rows = []
for name, probs, nn in REPRO:
    m, s, mi, si = score(probs); pp = PAPER7[162946][name]; k = len(probs)
    rows.append(f'{name}{"$^\\dagger$" if name == "Stacking-FNN" else ""} & {pm(m, s, k)} & {pm(pp[0], pp[1], 2)} & {pm(mi, si, k)} & {pm(pp[2], pp[3], 2)} & {nn} \\\\')
write('repro.tex', '\n'.join(rows) + '\n')

# ---------- Table: training-size sweep ----------
NS = {}
for r in jl('nsweep.jsonl'):
    NS.setdefault(r['N'], []).append(r)
MODELS = ['MFE+FNN', 'CNN', 'Stacking-DT', 'Stacking-FNN', 'Stacking-MLR']
rows = []
for n in sorted(NS):
    R = NS[n]
    cells = []
    for mname in MODELS:
        v = [r[mname][0] for r in R]
        cells.append(f'{np.mean(v):.3f} / {PAPER7[n][mname][0]:.3f}')
    rows.append(f'{n:,} & {len(R)} & ' + ' & '.join(cells) + ' \\\\')
stage1 = {'MFE+FNN': score(mfe)[0], 'CNN': score(REPRO[1][1])[0], 'Stacking-DT': score(REPRO[3][1])[0],
          'Stacking-FNN': score(REPRO[4][1])[0], 'Stacking-MLR': score(REPRO[5][1])[0]}
rows.append('162,946 & 1--5 & ' + ' & '.join(f'{stage1[m]:.3f} / {PAPER7[162946][m][0]:.3f}' for m in MODELS) + ' \\\\')
write('sweep.tex', '\n'.join(rows) + '\n')

# ---------- Table: extensions (ablation path) ----------
EXT = [('Stack-2: MFE-FNN + CNN', 'S2', 'baseline (split-B meta-training)'),
       ('+ TTA on the CNN', 'S2tta', 'C1'),
       ('+ XGBoost as third learner', 'S3', 'C2'),
       ('Control: XGBoost replaces MFE-FNN', 'ctlA', 'C2'),
       ('Control: 2nd MFE-FNN seed as third learner', 'ctlB', 'C2'),
       ('MFE-FNN + CNN$\\times$5', 'e5_S2', 'C3'),
       ('MFE-FNN + CNN$\\times$5 + XGBoost', 'e5_S3', 'C3'),
       ('\\textbf{CNN$\\times$5 + XGBoost (final)}', 'e5_ctlA', 'C3')]
rows = []
for lab, k, tag in EXT:
    bm, tm, tf_ = TR[f'{k}_mlr', 0], TR[f'{k}_mlr', 1], TR[f'{k}_fnn', 1]
    nb = len(bm['macro_seeds'])
    rows.append(f'{lab} & {pm(bm["macro_mean"], bm["macro_std"], nb)} & {pm(tm["macro_mean"], tm["macro_std"], len(tm["macro_seeds"]))} & '
                f'{pm(tf_["macro_mean"], tf_["macro_std"], len(tf_["macro_seeds"]))} \\\\')
write('extensions.tex', '\n'.join(rows) + '\n')

# ---------- Table: paired statistics ----------
STATS = [('S2tta_mlr', 'S2_mlr', 0, 'TTA (MLR)'), ('S2tta_fnn', 'S2_fnn', 0, 'TTA (FNN)'),
         ('S3_mlr', 'S2tta_mlr', 0, 'XGBoost as 3rd learner (MLR)'), ('S3_fnn', 'S2tta_fnn', 0, 'XGBoost as 3rd learner (FNN)'),
         ('S3_mlr', 'ctlB_mlr', 0, 'XGBoost vs.\\ 2nd MFE-FNN seed (MLR)'), ('S3_fnn', 'ctlB_fnn', 0, 'XGBoost vs.\\ 2nd MFE-FNN seed (FNN)'),
         ('e5_ctlA_mlr', 'ctlA_mlr', 0, 'CNN$\\times$5 vs.\\ CNN$\\times$1 (final stack)'),
         ('e5_S3_mlr', 'out:cnn_ens5_tta', 0, 'adding MFE-FNN + XGB to CNN$\\times$5$^\\ddagger$'),
         ('e5_ctlA_mlr', 'FINAL_stack_mlr', 1, 'final vs.\\ reproduced Stacking-MLR'),
         ('e5_ctlA_mlr', 'S2tta_mlr', 1, 'final vs.\\ Stack-2 + TTA (MLR)')]
rows = []
for a, b, t, lab in STATS:
    r = CMP[a, b, t]
    p = '--' if r['p_ttest'] != r['p_ttest'] else f'{r["p_ttest"]:.3f}'
    rows.append(f'{lab} & {"test" if t else ("$B_{val}$" if b.startswith("out:") else "$B$")} & {r["diff"]:+.4f} & {p} & [{r["ci95"][0]:+.4f}, {r["ci95"][1]:+.4f}] \\\\')
write('stats.tex', '\n'.join(rows) + '\n')

# ---------- Table: reject option (final pipeline) ----------
cal = ld('calib_e5_ctlA_mlr.pkl')['summary']
rows = [f'{t["target"]:.0%} & {t["coverage"][0]:.3f} & {t["acc"][0]:.4f} & {t["macro"][0]:.4f} \\\\'.replace('%', '\\%')
        for t in cal['table']]
write('reject.tex', '\n'.join(rows) + '\n')
CALIB = dict(T=cal['T'][0], ece_raw=cal['ece_before'][0], ece_T=cal['ece_after'][0])

# ---------- Table: per-class F1 (appendix) ----------
fin = runs('e5_ctlA_mlr')[0]; rep = runs('FINAL_stack_mlr')
cnt = np.bincount(yt, minlength=9)
F_rep = np.mean([f1s(yt, p) for p in rep], 0); F_fin = f1s(yt, fin)
rows = [f'{SHORT[c]} & {cnt[c]} & {PAPER8_MLR[c]:.4f} & {F_rep[c]:.4f} & {F_fin[c]:.4f} \\\\' for c in range(9)]
rows.append(f'\\midrule Macro & 10,000 & {np.mean(PAPER8_MLR):.4f} & {F_rep.mean():.4f} & {F_fin.mean():.4f} \\\\')
write('perclass.tex', '\n'.join(rows) + '\n')

# ---------- Table: CNN seeds (appendix) ----------
rows, bv, te = [], [], []
for k, s in enumerate(['cnn_imnet_cw0_aug'] + [f'cnn_imnet_cw0_aug_s{i}' for i in range(1, 5)]):
    o, t = ld(f'{s}_outputs.pkl'), ld(f'{s}_tta_outputs.pkl')
    bv.append(f1s(y[B_val], t['train_prob'][B_val]).mean()); te.append(f1s(yt, t['test_prob']).mean())
    rows.append(f'{k} & {f1s(y[B_val], o["train_prob"][B_val]).mean():.4f} & {bv[-1]:.4f} & {te[-1]:.4f} \\\\')
e5 = ld('cnn_ens5_tta_outputs.pkl')
rows.append(f'\\midrule mean $\\pm$ std & & {np.mean(bv):.4f} $\\pm$ {np.std(bv):.4f} & {np.mean(te):.4f} $\\pm$ {np.std(te):.4f} \\\\')
rows.append(f'ensemble (5) & & {f1s(y[B_val], e5["train_prob"][B_val]).mean():.4f} & {f1s(yt, e5["test_prob"]).mean():.4f} \\\\')
write('seeds.tex', '\n'.join(rows) + '\n')

# ---------- Table: XGBoost grid (appendix) ----------
XG = {r['name']: r for r in jl('xgb_trials.jsonl') if re.fullmatch(r'd\d_lr\d+_cw\d', r['name'])}
rows = []
for k in sorted(XG, key=lambda k: -TR[f'S3sel_{k}', 0]['macro_mean']):
    r, s = XG[k], TR[f'S3sel_{k}', 0]
    d, lr, cw = re.fullmatch(r'd(\d)_lr(\d+)_cw(\d)', k).groups()
    rows.append(f'{d} & 0.{lr} & {"balanced" if cw == "1" else "none"} & {r["trees"]} & {r["macro"]:.4f} & '
                f'{pm(s["macro_mean"], s["macro_std"], 5)} \\\\')
write('xgbgrid.tex', '\n'.join(rows) + '\n')

# ---------- macros with single numbers used in the text ----------
fin_r = TR['e5_ctlA_mlr', 1]
M = {'FinalMacro': f'{fin_r["macro_mean"]:.4f}', 'FinalAcc': f'{(fin.argmax(1) == yt).mean() * 100:.2f}',
     'ReproMLR': f'{score(rep)[0]:.4f}', 'PaperMLR': '0.8949',
     'CalT': f'{CALIB["T"]:.3f}', 'EceRaw': f'{CALIB["ece_raw"]:.4f}', 'EceT': f'{CALIB["ece_T"]:.4f}',
     'RejCov': f'{cal["table"][2]["coverage"][0] * 100:.1f}', 'RejAcc': f'{cal["table"][2]["acc"][0] * 100:.2f}',
     'RejMacro': f'{cal["table"][2]["macro"][0]:.4f}', 'SeedStd': f'{np.std(bv):.3f}',
     'GainOverRepro': f'{CMP["e5_ctlA_mlr", "FINAL_stack_mlr", 1]["diff"]:+.4f}'}
open(os.path.join(OUT, 'tables', 'numbers.tex'), 'w').write(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in M.items()))

# ---------- figures ----------
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.linewidth': 0.6, 'axes.edgecolor': '#52514e',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#0b0b0b',
                     'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
C = ['#2a78d6', '#eb6834', '#1baf7a']  # validated categorical slots 1-3 (all-pairs, light)
GRID = dict(color='#e4e3df', lw=0.5)

# Fig: training-size sweep, ours (solid, filled) vs paper (dashed, hollow)
fig, ax = plt.subplots(figsize=(3.4, 2.4))
Ns = sorted(NS) + [162946]
for (mname, col, mk) in zip(['MFE+FNN', 'CNN', 'Stacking-MLR'], C, ['o', 's', '^']):
    ours = [np.mean([r[mname][0] for r in NS[n]]) for n in sorted(NS)] + [stage1[mname]]
    sd = [np.std([r[mname][0] for r in NS[n]]) for n in sorted(NS)] + [0]
    pap = [PAPER7[n][mname][0] for n in Ns]
    ax.errorbar(Ns, ours, yerr=sd, color=col, marker=mk, ms=4, lw=1.4, capsize=2, elinewidth=0.7, label=f'{mname} (ours)')
    ax.plot(Ns, pap, color=col, marker=mk, ms=4, lw=1.0, ls='--', mfc='white', label=f'{mname} (paper)')
ax.set_xscale('log'); ax.set_xticks(Ns); ax.set_xticklabels(['500', '5k', '50k', '163k'])
ax.set_xlabel('training-set size $N$'); ax.set_ylabel('macro-F1 (test)')
ax.yaxis.grid(True, **GRID); ax.set_axisbelow(True)
h, l = ax.get_legend_handles_labels(); o = [0, 2, 4, 1, 3, 5]
ax.legend([h[i] for i in o], [l[i] for i in o], fontsize=6, frameon=False, ncol=2, loc='lower right')
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, 'figures', 'sweep.pdf')); plt.close(fig)

# Fig: risk-coverage of the final pipeline (single series) with the B-chosen operating points
r0 = ld('calib_e5_ctlA_mlr.pkl')['runs'][0]
fig, ax = plt.subplots(figsize=(3.4, 2.2))
m = r0['curve_cov'] >= 0.5
ax.plot(r0['curve_cov'][m] * 100, r0['curve_acc'][m] * 100, color=C[0], lw=1.6)
for t in r0['table']:
    ax.plot(t['coverage'] * 100, t['acc'] * 100, 'o', color=C[0], ms=4, mec='white', mew=0.8)
    off = {1.0: (-14, 0), 0.95: (-22, -42), 0.90: (-34, -16)}
    if t['target'] in off:
        ax.annotate(f'{t["coverage"] * 100:.1f}% → {t["acc"] * 100:.2f}%', (t['coverage'] * 100, t['acc'] * 100),
                    textcoords='offset points', xytext=off[t['target']], ha='right', va='center', fontsize=6.5, color='#52514e',
                    arrowprops=dict(arrowstyle='-', lw=0.5, color='#8a8984', shrinkA=1, shrinkB=3))
ax.set_xlabel('coverage (% of wafers classified automatically)'); ax.set_ylabel('accuracy on accepted (%)')
ax.set_ylim(98.0, 100.1)
ax.yaxis.grid(True, **GRID); ax.set_axisbelow(True)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, 'figures', 'reject.pdf')); plt.close(fig)

# Fig: MLR weights per class, MFE-FNN vs CNN (the paper's Fig. 5 for our split-B stack with TTA)
st = [ld(os.path.basename(f)) for f in sorted(glob.glob(P('data', 'stack_S2tta_mlr_b*_s0.pkl')))]
Cf = np.array([c for r in st for c in r['coef']])
W = np.array([[Cf[:, c, l * 9 + c] for l in range(2)] for c in range(9)])
fig, ax = plt.subplots(figsize=(3.4, 2.0))
xw = 0.38
for l, (lab, col) in enumerate([('MFE-FNN', C[0]), ('CNN', C[1])]):
    ax.bar(np.arange(9) + (l - 0.5) * (xw + 0.02), W[:, l].mean(1), xw, yerr=W[:, l].std(1), color=col,
           error_kw=dict(lw=0.6, capsize=1.5, ecolor='#52514e'), label=lab)
ax.axhline(0, color='#52514e', lw=0.6)
ax.set_xticks(range(9)); ax.set_xticklabels(SHORT, rotation=35, ha='right')
ax.set_ylabel('MLR weight'); ax.yaxis.grid(True, **GRID); ax.set_axisbelow(True)
ax.legend(fontsize=6.5, frameon=False, ncol=2, loc='upper left')
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, 'figures', 'weights.pdf')); plt.close(fig)


# Fig (slides): one example test wafer map per class, the most confidently correct one under the final pipeline
_, Xte_img = ld('X_CNN.pkl')
fig, axs = plt.subplots(1, 9, figsize=(9, 1.35))
for c, a in enumerate(axs):
    idx = np.nonzero((yt == c) & (fin.argmax(1) == c))[0]
    i = idx[np.argmax(fin[idx, c])]
    a.imshow(Xte_img[i].reshape(64, 64).astype(np.float32), cmap='Greys', vmin=0, vmax=1)
    a.set_title(SHORT[c], fontsize=8); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_color('#c3c2b7'); sp.set_linewidth(0.5)
fig.tight_layout(pad=0.2); fig.savefig(os.path.join(OUT, 'figures', 'examples.pdf')); plt.close(fig)
fig, axs = plt.subplots(3, 3, figsize=(3.3, 3.55))  # same maps, 3x3 for a single paper column
for c, a in enumerate(axs.flat):
    idx = np.nonzero((yt == c) & (fin.argmax(1) == c))[0]
    i = idx[np.argmax(fin[idx, c])]
    a.imshow(Xte_img[i].reshape(64, 64).astype(np.float32), cmap='Greys', vmin=0, vmax=1)
    a.set_title(SHORT[c], fontsize=8, pad=2); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_color('#c3c2b7'); sp.set_linewidth(0.5)
fig.tight_layout(pad=0.2, h_pad=0.4, w_pad=0.3); fig.savefig(os.path.join(OUT, 'figures', 'examples_grid.pdf')); plt.close(fig)
# Grad-CAM figure (appendix): reuse the rendered panel
import shutil
shutil.copy(P('figures', 'gradcam_imnet_cw0_aug.png'), os.path.join(OUT, 'figures', 'gradcam.png'))
print('tables:', sorted(os.listdir(os.path.join(OUT, 'tables'))))
print('figures:', sorted(os.listdir(os.path.join(OUT, 'figures'))))
print(M)
