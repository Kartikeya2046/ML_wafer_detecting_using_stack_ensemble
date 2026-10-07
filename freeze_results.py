"""Part B5: build results_comparison.md from the saved test-set outputs (no training). Run: python freeze_results.py"""
import glob, os
import numpy as np
from sklearn.metrics import accuracy_score
from common import *

y, yt = labels()
SHORT = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full', 'None']

# Paper (Kang & Kang 2021), N = 162,946, mean ± std over 10 replicates.
PAPER_T7 = {  # model: (F1macro, F1micro)
    'MFE+FNN': ('0.8599 ± 0.0117', '0.9741 ± 0.0010'), 'CNN': ('0.8679 ± 0.0126', '0.9775 ± 0.0014'),
    'Stacking-FNN': ('0.8991 ± 0.0096', '0.9808 ± 0.0010'), 'Stacking-MLR': ('0.8949 ± 0.0121', '0.9801 ± 0.0010')}
PAPER_T8 = {  # last block of Table 8 (printed there as "162,496")
    'MFE+FNN':      [0.9323, 0.8615, 0.8051, 0.9667, 0.7406, 0.8928, 0.6451, 0.9053, 0.9899],
    'CNN':          [0.9326, 0.8760, 0.8301, 0.9794, 0.7715, 0.8653, 0.7737, 0.7920, 0.9909],
    'Stacking-FNN': [0.9455, 0.8843, 0.8562, 0.9824, 0.8059, 0.9033, 0.8083, 0.9142, 0.9923],
    'Stacking-MLR': [0.9424, 0.8969, 0.8492, 0.9823, 0.7950, 0.8979, 0.7928, 0.9053, 0.9918]}


def runs(*names):
    """Test-prob arrays: data/<name> files, or stack_<name>_*.pkl globs."""
    out = []
    for n in names:
        files = sorted(glob.glob(P('data', n))) or sorted(glob.glob(P('data', f'stack_{n}_*.pkl')))
        assert files, n
        out += [ld(os.path.basename(f))['test_prob'] for f in files]
    return out


def stats(probs):
    F = np.array([f1s(yt, p) for p in probs])
    mic = np.array([accuracy_score(yt, p.argmax(1)) for p in probs])
    fmt = lambda v: f'{v.mean():.4f}' + (f' ± {v.std():.4f}' if len(v) > 1 else '')
    return fmt(F.mean(1)), fmt(mic), F.mean(0), len(probs)


ROWS = [  # (model, stage, outputs)
    ('MFE+FNN', 'original (balanced cw)', runs('mfe_fnn_outputs.pkl')),
    ('MFE+FNN', 'tuned (no cw)', runs('mfe_x3_cw0_outputs.pkl', 'mfe_x3_cw0_s1_outputs.pkl', 'mfe_x3_cw0_s2_outputs.pkl')),
    ('CNN', 'original (scratch, balanced cw)', runs('cnn_outputs.pkl')),
    ('CNN', 'tuned (ImageNet init, no cw, aug)', runs('cnn_imnet_cw0_aug_outputs.pkl')),
    ('Stacking-FNN', 'before fix (no cw, = notebook/repo)', runs('B1_paper_unweighted')),
    ('Stacking-FNN', 'after fix (balanced cw) — B1 baseline', runs('B1_fixed_baseline')),
    ('Stacking-FNN', 'tuned (h64, honest B inputs, no cw)', runs('FINAL_stack_fnn')),
    ('Stacking-MLR', 'original base learners', runs('final_mlr_A')),
    ('Stacking-MLR', '**tuned (frozen pipeline)**', runs('FINAL_stack_mlr')),
]

o = ['# Results Comparison — Stage 1 frozen (implementation_plan.md B5)', '',
     'Test set: 10,000 wafers, untouched by any tuning decision (all selection on the training split B, see tuning_log.md).',
     'Paper: Kang & Kang 2021, Table 7 / Table 8, N = 162,946, mean ± std over 10 replicates (different splits and seeds).',
     'Ours: mean ± std over the runs listed in the n column (seeds of the stacker, or of the MFE-FNN for MLR, which is deterministic).', '',
     '## Macro- and micro-F1', '',
     '| Model | Ours: stage | n | **F1 macro** | F1 micro (= accuracy) | Paper F1 macro | Paper F1 micro |',
     '|---|---|---|---|---|---|---|']
tuned = {}
for model, stage, probs in ROWS:
    mac, mic, pc, n = stats(probs)
    if 'tuned' in stage:
        tuned[model] = pc
    o.append(f'| {model} | {stage} | {n} | {mac} | {mic} | {PAPER_T7[model][0]} | {PAPER_T7[model][1]} |')

o += ['', '## Per-class F1 — tuned pipeline vs paper (Table 8, N = 162,946)', '',
      '| Class | test n | MFE+FNN ours | paper | CNN ours | paper | Stack-FNN ours | paper | **Stack-MLR ours** | paper |',
      '|---|---|---|---|---|---|---|---|---|---|']
cnt = np.bincount(yt, minlength=9)
for i, c in enumerate(SHORT):
    cells = ' | '.join(f'{tuned[m][i]:.4f} | {PAPER_T8[m][i]:.4f}' for m in PAPER_T8)
    o.append(f'| {c} | {cnt[i]} | {cells} |')
o.append('| **Macro** | | ' + ' | '.join(f'**{tuned[m].mean():.4f}** | {np.mean(PAPER_T8[m]):.4f}' for m in PAPER_T8) + ' |')
o += ['', 'Near-full has only 9 test wafers: one wafer moves its F1 by ~0.1 and the macro-F1 by ~0.01.',
      'Frozen files: `data/mfe_x3_cw0[_s1,_s2]_outputs.pkl`, `data/cnn_imnet_cw0_aug_outputs.pkl` (+ `models/cnn_imnet_cw0_aug.keras`), '
      'stacker outputs `data/stack_FINAL_stack_{mlr,fnn}_*.pkl`. Regenerate this file with `python freeze_results.py`.']
open(P('results_comparison.md'), 'w', encoding='utf-8').write('\n'.join(o) + '\n')
print('\n'.join(o))
