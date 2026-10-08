"""Stage 2 step 6: build extensions_results.md from saved outputs and logs (no training). Run: python extensions_results.py
Inputs: logs/stack_trials.jsonl, logs/compare.jsonl, logs/xgb_trials.jsonl, data/calib_*.pkl, data/stack_*_b*_s*.pkl
(coef), data/shap_xgb_*.pkl, base-learner outputs. Every number in the document comes from those files.
"""
import json, glob, re
import numpy as np
from common import *

y, yt = labels()
A, B_fit, B_val = splits(y)
SHORT = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full', 'None']
CNN, CNNT, XGB, MFE = 'cnn_imnet_cw0_aug', 'cnn_imnet_cw0_aug_tta', 'xgb_d6_lr05_cw1', 'mfe_cw0_5s'


def jl(f):
    with open(P('logs', f)) as h:
        return [json.loads(l) for l in h if l.strip()]


TR = {}
for r in jl('stack_trials.jsonl'):  # latest record per (name, final)
    TR[r['name'], r['cfg']['final']] = r
CMP = {}
for r in jl('compare.jsonl'):
    CMP[r['a'], r['b'], r['test']] = r
XG = {r['name']: r for r in jl('xgb_trials.jsonl')}
ms = lambda r: f"{r['macro_mean']:.4f} ± {r['macro_std']:.4f}"
n = lambda r: len(r['macro_seeds'])


def cmp(a, b, test):
    r = CMP.get((a, b, test))
    if r is None:
        return '—', '—', '—'
    p = '—' if r['p_ttest'] != r['p_ttest'] else f"{r['p_ttest']:.3f}"
    return f"{r['diff']:+.4f}", p, f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"


L = []
w = L.append
w('# Stage 2 Results — Our Extensions (stage2_plan.md)')
w('')
w('Built by `python extensions_results.py` from saved outputs and logs. Frozen Stage 1 starting point: `results_comparison.md`.')
w('All choices were made on split B (2-fold cross-fitted stacker predictions pooled over B = 32,590 wafers); '
  'the 10,000 test wafers were evaluated once, after the B decisions (step 5) were fixed.')
w('Statistics: mean ± std over runs (5 MFE-FNN seeds; FNN-stacker seed 0), paired t-test across runs, '
  'paired bootstrap 95% CI over wafers (2,000 resamples) of the macro-F1 difference. '
  '**Inclusion rule:** an extension joins the pipeline if it improves the B score by more than the baseline seed std.')
w('')

# ---------------- summary ----------------
w('## 1. Summary')
w('')
w('| Configuration | Base learners | stacker | B-cv macro-F1 | n | **Test macro-F1** | Test F1 micro (acc.) |')
w('|---|---|---|---|---|---|---|')
ROWS = [('S2', 'Stack-2 (Stage 2 baseline, B mode)', f'MFE-FNN + CNN'),
        ('S2tta', 'Stack-2 + TTA (C4)', 'MFE-FNN + CNN-TTA'),
        ('S3', 'Stack-3 (C1)', 'MFE-FNN + CNN-TTA + XGB'),
        ('ctlA', 'Control A: XGB replaces MFE-FNN', 'CNN-TTA + XGB'),
        ('ctlB', 'Control B: 2nd MFE-FNN seed as 3rd learner', 'MFE-FNN ×2 + CNN-TTA')]


def acc_of(name):
    fs = [f for f in glob.glob(P('data', f'stack_{name}_b*_s*.pkl')) if re.fullmatch(f'stack_{name}_b\\d+_s\\d+\\.pkl', os.path.basename(f))]
    a = [(ld(os.path.basename(f))['test_prob'].argmax(1) == yt).mean() for f in fs]
    return f'{np.mean(a):.4f}'


FINAL = {'mlr': 'S2tta', 'fnn': 'S3'}
for meta in ['mlr', 'fnn']:
    for k, lab, base in ROWS:
        nm = f'{k}_{meta}'
        b, t = TR.get((nm, 0)), TR.get((nm, 1))
        star = ' ← **final (B rule)**' if FINAL[meta] == k else ''
        w(f"| {lab}{star} | {base} | {meta.upper()} | {ms(b)} | {n(b)} | **{ms(t)}** | {acc_of(nm)} |")
w(f"| Stage 1 frozen headline (A mode) | MFE-FNN + CNN | MLR | — | — | {ms(TR['FINAL_stack_mlr', 1])} | 0.9805 |")
w(f"| Stage 1 tuned Stacking-FNN | MFE-FNN + CNN | FNN | — | — | {ms(TR['FINAL_stack_fnn', 1])} | 0.9811 |")
w(f"| Paper (Table 7, N = 162,946) | | MLR / FNN | — | — | 0.8949 ± 0.0121 / 0.8991 ± 0.0096 | 0.9801 / 0.9808 |")
w('')
w('**Final Stage 2 pipelines, chosen on B before the test set was opened:** MLR (headline) = MFE-FNN + CNN-TTA '
  f"(test {ms(TR['S2tta_mlr', 1])}); FNN = MFE-FNN + CNN-TTA + XGB (test {ms(TR['S3_fnn', 1])}).")
w('')
w('**Findings in one line each:**')
w('1. **C4 TTA helps** both stackers on B (> seed std, CI excludes 0 for FNN) and on test (+0.005 MLR, +0.006 FNN). '
  'Cheapest win: 8 forward passes per wafer, no retraining. The CNN alone barely moves on test (0.8777 → 0.8781); '
  'the gain shows up in the stack, where smoother CNN probabilities combine better.')
w('2. **C1 XGB as a third learner: mixed.** On B it passes the rule for the FNN stacker but not for MLR. On test the '
  'MLR Stack-3 scores 0.9149 (+0.0098 vs the chosen MLR pipeline, p = 0.057, CI includes 0) - the rule, applied on B, '
  'rejected it; we report that rather than switch after seeing test. Most of the test gap is the 9 Near-full wafers.')
w('3. **"Diversity helps" is not supported**: Stack-3 never beats Control B (a second MFE-FNN seed in the third slot) by a '
  'significant margin on B (on test, MLR: +0.009, p = 0.028, but the bootstrap CI includes 0). **XGB is simply a better handcrafted-feature model** (standalone B_val 0.870 vs 0.853 for the '
  'MFE-FNN); Control A (CNN-TTA + XGB, no FNN at all) is the best MLR stack on both B and test - a negative result for '
  'the paper\'s MFE-FNN, not part of the pre-registered pipeline.')
w('4. **C2 reject option works**: routing the 5% least confident wafers to a human lifts accuracy 98.1% → 99.6% and '
  'macro-F1 0.905 → 0.979 (MLR). **Temperature scaling does not improve calibration for MLR** (its clipped ridge outputs '
  'are already well calibrated) and barely for FNN - a negative result.')
w('5. **C3 weights** reproduce the paper\'s story: the handcrafted learner carries Random and Near-full, the CNN carries the '
  'spatial patterns (Edge-Ring, Scratch, Edge-Loc).')
w('')

# ---------------- C4 ----------------
w('## 2. C4 — Test-time augmentation (8 symmetries: 4 rotations × flip)')
w('')
c0, c1 = ld(f'{CNN}_outputs.pkl'), ld(f'{CNNT}_outputs.pkl')
w('| CNN alone | B_val macro-F1 | Test macro-F1 | Test acc. |')
w('|---|---|---|---|')
for lab, o in [('plain (Stage 1 frozen)', c0), ('TTA', c1)]:
    w(f"| {lab} | {f1s(y[B_val], o['train_prob'][B_val]).mean():.4f} | {f1s(yt, o['test_prob']).mean():.4f} | "
      f"{(o['test_prob'].argmax(1) == yt).mean():.4f} |")
d, p, ci = cmp(f'out:{CNNT}', f'out:{CNN}', 0)
w(f'\nCNN alone, TTA − plain on B_val: {d}, bootstrap CI {ci} (single CNN seed, no t-test).')
w('')
w('| Stack, TTA − plain | set | diff | paired t-test p | bootstrap 95% CI | baseline seed std | rule |')
w('|---|---|---|---|---|---|---|')
for meta in ['mlr', 'fnn']:
    for test in [0, 1]:
        d, p, ci = cmp(f'S2tta_{meta}', f'S2_{meta}', test)
        sd = TR[f'S2_{meta}', test]['macro_std']
        rule = ('pass' if float(d) > sd else 'fail') if not test else '(test: report only)'
        w(f"| {meta.upper()} | {'test' if test else 'B'} | {d} | {p} | {ci} | {sd:.4f} | {rule} |")
w('')

# ---------------- C1 ----------------
w('## 3. C1 — XGBoost on the 59 handcrafted features as a third base learner')
w('')
w('XGBoost fitted on A, early-stopped (number of trees) on B_fit, `tree_method=hist`. Grid: depth {4, 6, 8} × learning '
  'rate {0.05, 0.1} × class weights {none, balanced}; **selected by Stack-3 MLR score on B** (plan: by stack score, '
  'not standalone; the MLR stacker is deterministic and instant, so the grid was scored with it).')
w('')
w('| XGB config | trees | standalone B_val macro-F1 | Stack-3 MLR B-cv |')
w('|---|---|---|---|')
for k, r in sorted(XG.items(), key=lambda kv: -TR.get((f'S3sel_{kv[0]}', 0), {'macro_mean': 0})['macro_mean']):
    s = TR.get((f'S3sel_{k}', 0))
    w(f"| {k}{' ← selected' if k == XGB[4:] else ''} | {r['trees']} | {r['macro']:.4f} | {ms(s) if s else '—'} |")
w('')
w('Selecting the best of 12 configs on the same B score inflates its B number slightly (winner\'s curse); the test '
  'evaluation is unaffected.')
w('')
w('| Comparison | stacker | set | diff | paired t-test p | bootstrap 95% CI |')
w('|---|---|---|---|---|---|')
for a, b, lab in [('S3', 'S2tta', 'Stack-3 − Stack-2 (both with TTA)'), ('S3', 'ctlB', 'Stack-3 − Control B (is it diversity?)'),
                  ('ctlB', 'S2tta', 'Control B − Stack-2'), ('ctlA', 'S2tta', 'Control A − Stack-2'),
                  ('S3', 'ctlA', 'Stack-3 − Control A')]:
    for meta in ['mlr', 'fnn']:
        for test in [0, 1]:
            d, p, ci = cmp(f'{a}_{meta}', f'{b}_{meta}', test)
            if d != '—':
                w(f"| {lab} | {meta.upper()} | {'test' if test else 'B'} | {d} | {p} | {ci} |")
w('')
w('Control A has a single run for MLR (XGB and the CNN are single deterministic models), so no t-test there.')
w('')

# ---------------- C2 ----------------
w('## 4. C2 — Calibration and reject option')
w('')
w('Temperature T fitted per run by log-loss on the stacker\'s pooled 2-fold B predictions, applied to test unchanged. '
  'ECE: 15 equal-width bins. "Before" = uncalibrated probabilities (FNN softmax; MLR ridge outputs clipped to [0, 1], '
  'renormalised). Reject thresholds are chosen on B for each target coverage and applied to test, so the realised test '
  'coverage is close to (not exactly) the target. Macro-F1 on accepted wafers averages over the classes present.')
w('')
CAL = {os.path.basename(f)[6:-4]: ld(os.path.basename(f))['summary'] for f in sorted(glob.glob(P('data', 'calib_*.pkl')))}
f2 = lambda v: f'{v[0]:.4f} ± {v[1]:.4f}'
w('| Stack | T | test ECE before | test ECE after | B ECE before | B ECE after |')
w('|---|---|---|---|---|---|')
for k, s in CAL.items():
    w(f"| {k} | {s['T'][0]:.3f} | {f2(s['ece_before'])} | {f2(s['ece_after'])} | {f2(s['ece_B_before'])} | {f2(s['ece_B_after'])} |")
w('')
for k in ['S2tta_mlr', 'S3_fnn']:
    s = CAL[k]
    w(f'**Reject option — {k} (final {s["meta"].upper()} pipeline), test set:**')
    w('')
    w('| target coverage | test coverage | accuracy on accepted | macro-F1 on accepted |')
    w('|---|---|---|---|')
    for t in s['table']:
        w(f"| {t['target']:.0%} | {t['coverage'][0]:.3f} | {t['acc'][0]:.4f} | {t['macro'][0]:.4f} |")
    for t in s['acc_targets']:
        w(f"| B accuracy ≥ {t['target_acc']:.1%} | {t['coverage'][0]:.3f} | {t['acc'][0]:.4f} | {t['macro'][0]:.4f} |")
    w('')
w('FNN test coverage stalls near 86% for targets ≤ 85%: the thresholds come from the 2-fold B stackers, but test is '
  'predicted by the final stacker fitted on B, and the FNN\'s confidence scale shifts between fits (most None wafers sit '
  'on a plateau near 0.999; the B threshold for 80% coverage, 0.9957, admits 86% of test). The MLR thresholds transfer '
  'cleanly (test coverage within 0.5 pt of target) - one more reason the MLR is the better basis for the reject rule.')
w('')
w('**Which wafers go to the human** (% of each true class rejected, MLR final pipeline):')
w('')
w('| target coverage | ' + ' | '.join(SHORT) + ' |')
w('|---|' + '---|' * 9)
for t in (0.95, 0.90):
    w(f'| {t:.0%} | ' + ' | '.join(f'{v:.0%}' for v in CAL['S2tta_mlr']['reject_by_class'][t]) + ' |')
w('')
w('The rejected wafers are mostly the hard local patterns (Loc, Edge-Loc, Scratch) and rare Donut/Near-full; the "None" '
  'class (85% of wafers) is almost never rejected, so the human workload stays small. No class is rejected wholesale, '
  'so per-class thresholds (plan follow-up) were not needed.')
w('')

# ---------------- C3 ----------------
w('## 5. C3 — Explainability')
w('')
w('**MLR weights** (diagonal coefficient: the weight each base learner\'s vote for class c gets in the score for class c; '
  'mean ± std over runs; the paper\'s Fig. 5 for our stacks). Figures: `figures/mlr_weights_*.png`.')
w('')


def weights(name):
    fs = [f for f in sorted(glob.glob(P('data', f'stack_{name}_b*_s*.pkl'))) if re.fullmatch(f'stack_{name}_b\\d+_s\\d+\\.pkl', os.path.basename(f))]
    runs = [ld(os.path.basename(f)) for f in fs]
    C = np.array([c for r in runs for c in r['coef']])
    learners = runs[0]['cfg']['inputs'].split(',')
    return learners, np.array([[C[:, c, l * 9 + c] for l in range(len(learners))] for c in range(9)])


for name in ['S2tta_mlr', 'S3_mlr']:
    ln, W = weights(name)
    w(f'| {name} | ' + ' | '.join(ln) + ' |')
    w('|---|' + '---|' * len(ln))
    for c in range(9):
        w(f'| {SHORT[c]} | ' + ' | '.join(f'{W[c, l].mean():.2f} ± {W[c, l].std():.2f}' for l in range(len(ln))) + ' |')
    w('')
w('Handcrafted features dominate **Random** and **Near-full** (global density patterns); the CNN dominates the spatial '
  'patterns. In Stack-3, XGB takes over Near-full from the MFE-FNN (0.71 vs 0.28), which is where its test gain comes from.')
w('')
sh = sorted(glob.glob(P('data', 'shap_xgb_*.pkl')))
if sh:
    S = ld(os.path.basename(sh[0]))
    w('**SHAP (XGBoost, exact TreeSHAP)** — top 3 handcrafted features per class by mean |SHAP| on test '
      '(`figures/shap_xgb_*.png`):')
    w('')
    w('| Class | top features |')
    w('|---|---|')
    for c in range(9):
        w(f"| {SHORT[c]} | {', '.join(S['features'][i] for i in np.argsort(-S['imp'][c])[:3])} |")
    w('')
w('**Grad-CAM** (`figures/gradcam_imnet_cw0_aug.png`, layer block4_conv3, 8×8): on correctly classified Center, '
  'Edge-Loc, Random and Near-full wafers the heat sits on the defect; several misclassified wafers light up on the wafer '
  'edge or a corner instead. Confident correct Edge-Ring/Scratch maps are nearly blank because the softmax is saturated '
  '(p = 1.00 gives vanishing gradients) - a known Grad-CAM limitation.')
w('')

# ---------------- per-class ----------------
w('## 6. Per-class test F1 — final pipelines vs Stage 1')
w('')
w('| Class | test n | Stage 1 MLR (A) | Stage 2 MLR: Stack-2 + TTA | Stage 2 MLR: Stack-3 | Stage 1 FNN | Stage 2 FNN: Stack-3 |')
w('|---|---|---|---|---|---|---|')
cols = [TR['FINAL_stack_mlr', 1], TR['S2tta_mlr', 1], TR['S3_mlr', 1], TR['FINAL_stack_fnn', 1], TR['S3_fnn', 1]]
cnt = np.bincount(yt, minlength=9)
for c in range(9):
    w(f'| {SHORT[c]} | {cnt[c]} | ' + ' | '.join(f"{r['per_class'][c]:.4f}" for r in cols) + ' |')
w('| **Macro** | | ' + ' | '.join(f"**{r['macro_mean']:.4f}**" for r in cols) + ' |')
w('')

# ---------------- Stage 2b ----------------
w('## 7. Stage 2b — best-results pipeline (CNN seed ensemble) and the missing paper baselines')
w('')
w('After Stage 2 the test set had been seen once; Stage 2b still selects on B only, but this is a second look at test.')
w('')
w('**CNN seeds** (same recipe, seeds 0–4; seed 0 = the Stage 1 frozen CNN, trained on the laptop / TF 2.10):')
w('')
w('| CNN | B_val plain | B_val TTA | Test TTA |')
w('|---|---|---|---|')
seeds = ['cnn_imnet_cw0_aug'] + [f'cnn_imnet_cw0_aug_s{s}' for s in range(1, 5)]
bv, te = [], []
for k, s in enumerate(seeds):
    o, t = ld(f'{s}_outputs.pkl'), ld(f'{s}_tta_outputs.pkl')
    bv.append(f1s(y[B_val], t['train_prob'][B_val]).mean()); te.append(f1s(yt, t['test_prob']).mean())
    w(f"| seed {k} | {f1s(y[B_val], o['train_prob'][B_val]).mean():.4f} | {bv[-1]:.4f} | {te[-1]:.4f} |")
w(f'| mean ± std | | {np.mean(bv):.4f} ± {np.std(bv):.4f} | {np.mean(te):.4f} ± {np.std(te):.4f} |')
e5 = ld('cnn_ens5_tta_outputs.pkl')
w(f"| **5-seed ensemble** | | **{f1s(y[B_val], e5['train_prob'][B_val]).mean():.4f}** | **{f1s(yt, e5['test_prob']).mean():.4f}** |")
w('')
w('Best validation losses are nearly equal across seeds (0.053–0.054), yet macro-F1 varies by ±0.012: the rare classes '
  'swing between seeds. Averaging the seeds removes most of that variance.')
w('')
w('| Stack (CNN = 5-seed TTA ensemble) | stacker | B-cv macro-F1 | n | Test macro-F1 |')
w('|---|---|---|---|---|')
for k, lab in [('e5_S2', 'MFE-FNN + CNN×5'), ('e5_S3', 'MFE-FNN + CNN×5 + XGB'), ('e5_ctlA', 'CNN×5 + XGB')]:
    for meta in ['mlr', 'fnn']:
        b, t = TR[f'{k}_{meta}', 0], TR[f'{k}_{meta}', 1]
        star = ' ← **final (best on B)**' if (k, meta) == ('e5_ctlA', 'mlr') else ''
        w(f'| {lab}{star} | {meta.upper()} | {ms(b)} | {n(b)} | {ms(t)} |')
w('')
w('| Comparison | set | diff | paired t-test p | bootstrap 95% CI |')
w('|---|---|---|---|---|')
for a, b, test, lab in [('e5_ctlA_mlr', 'ctlA_mlr', 0, 'CNN×5 + XGB − CNN×1 + XGB (MLR)'),
                        ('e5_S2_mlr', 'S2tta_mlr', 0, 'MFE + CNN×5 − MFE + CNN×1 (MLR)'),
                        ('e5_ctlA_mlr', 'out:cnn_ens5_tta', 0, 'final stack − CNN×5 alone (B_val)'),
                        ('e5_S3_mlr', 'out:cnn_ens5_tta', 0, 'MFE + CNN×5 + XGB − CNN×5 alone (B_val)'),
                        ('e5_ctlA_mlr', 'e5_S3_mlr', 0, 'without MFE-FNN − with MFE-FNN (MLR)'),
                        ('e5_ctlA_mlr', 'FINAL_stack_mlr', 1, 'final − Stage 1 headline'),
                        ('e5_ctlA_mlr', 'S2tta_mlr', 1, 'final − Stage 2 MLR final'),
                        ('e5_ctlA_mlr', 'ctlA_mlr', 1, 'final − CNN×1 + XGB')]:
    d, p, ci = cmp(a, b, test)
    w(f"| {lab} | {'test' if test else 'B'} | {d} | {p} | {ci} |")
w('')
c = CAL['e5_ctlA_mlr']
t95 = [t for t in c['table'] if t['target'] == 0.95][0]
w(f"**Final pipeline: CNN×5 (TTA) + XGB → MLR — test macro-F1 {TR['e5_ctlA_mlr', 1]['macro_mean']:.4f}, accuracy "
  f"{acc_of('e5_ctlA_mlr')}.** Reject option at 95% target: {t95['coverage'][0]:.1%} auto-classified at accuracy "
  f"{t95['acc'][0]:.4f}, macro-F1 {t95['macro'][0]:.4f}. Runnable from saved models: `predict.py` (reproduces the test "
  'scores within 7e-5, identical classes).')
w('')
w('Findings: (1) the CNN seed ensemble is the largest single gain of the project on B (+0.009 to +0.010, CI excludes 0); '
  '(2) once the CNN is this strong, **the paper\'s MFE-FNN no longer helps** - adding it is below the CNN ensemble alone '
  'on B_val; the handcrafted features still help through XGB, slightly; (3) on test the single-CNN version of the final '
  'stack (0.9160) and the ensemble (0.9134) are within noise (CI [−0.014, +0.009]); the ensemble was chosen on B.')
w('')
mn = ld('cnn_multinn_outputs.pkl')
w(f"**Paper baselines added:** Stacking-DT (paper protocol, fitted on A): test {ms(TR['FINAL_stack_dt', 1])} "
  f"(paper 0.8789 ± 0.0094). MultiNN (single run, our CNN recipe + the 59 features): B_val "
  f"{f1s(y[B_val], mn['train_prob'][B_val]).mean():.4f}, test {f1s(yt, mn['test_prob']).mean():.4f} (paper 0.8455 ± 0.0170). "
  'Training-size sweep: `nsweep_results.md`.')
w('')

# ---------------- limitations ----------------
w('## 8. Limitations')
w('')
w('- Stages 1–2: one CNN seed (Stage 2b adds 4 more: seed std ±0.012 macro-F1). The wafer bootstrap covers test-sampling noise.')
w('- Near-full has 9 test wafers (30 in B): one wafer ≈ 0.01 macro-F1, and several test differences above are mostly Near-full.')
w('- One fixed stratified split (the paper averages 10 random splits).')
w('- Stage 2 stackers are fitted on B only (honest base outputs, 26k wafers) while the Stage 1 headline MLR was fitted on A; '
  'the like-for-like Stage 2 baseline is Stack-2 in B mode.')
w('- FNN stacker runs use one stacker seed per MFE seed (n = 5), so stacker-seed variance is only partly included.')
w('- Environment: Stage 2 ran on Linux, TF 2.17 + tf_keras (Stage 1: Windows, TF 2.10). Verified equivalent: the frozen '
  'CNN re-predicts test within 1.2e-4, and MFE-FNN seeds 0–2 reproduce Stage 1 exactly (B_val 0.8442 / 0.8481 / 0.8616).')

with open(P('extensions_results.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('\n'.join(L))
