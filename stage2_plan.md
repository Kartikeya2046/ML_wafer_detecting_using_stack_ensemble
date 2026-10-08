# Stage 2 Plan — Original Contribution (implementation_plan.md Part C)

**Status:** written 2026-10-07 from a question-by-question design session with the user (grilling skill, 16 decisions,
all settled — see "Decision record" at the bottom). **Signed off 2026-10-08 and implemented the same day** (branch `stage2`).
Results: `extensions_results.md`; deviations from this plan are listed in `tuning_log.md` "Stage 2".

**Goal:** a *stronger and more trustworthy* stack on top of the frozen Stage 1 pipeline:
accuracy (C4 test-time augmentation, C1 third base learner) + reliability (C2 calibration and reject option) +
interpretability (C3 MLR weight bars; Grad-CAM / SHAP only if time).

**Budget:** ~1 week, CPU-mostly. GPU only for short inference jobs (C4, optional Grad-CAM). **No new CNN training.**

---

## 0. Starting point (frozen Stage 1, git tag `stage1-frozen`)

| Component | Frozen choice | Files |
|---|---|---|
| MFE+FNN | paper arch 59→128→128→9 tanh, Adam 1e-4, batch 32, **no class weights** | `data/mfe_x3_cw0[_s1,_s2]_outputs.pkl` |
| CNN | VGG16 ImageNet init, no class weights, flip/rot90 aug, fp32 | `data/cnn_imnet_cw0_aug_outputs.pkl`, `models/cnn_imnet_cw0_aug.keras` (not in git, ~170 MB) |
| Stacker (headline) | MLR = ridge, α=0.1, no intercept, no cw, fitted on A | test macro-F1 **0.8967 ± 0.0060** |
| Stacker (also reported) | FNN h64, fitted on B, no cw | test macro-F1 **0.9041 ± 0.0028** |

Split (unchanged, `common.py:splits`): A = train[:130356] (base learners fitted here), B = train[130356:]
(base learners only early-stopped here → honest outputs), B = B_fit ∪ B_val (stratified halves). Test = 10,000 wafers.

## 1. Rules for all of Stage 2

1. **All choices on B; the test set is touched once, at the very end** (step 6).
2. **All Stage 2 stackers are fitted on B** (`mode=B`), where every base learner's outputs are honest. The Stage 1
   A-mode headline (0.8967) stays as-is; Stage 2's own baseline is **Stack-2 in B mode** (like-for-like).
3. **Both stackers everywhere:** MLR (headline) and the tuned FNN.
4. **Statistics for every comparison:** mean ± std over seeds, **paired t-test across seeds** (as the paper does),
   **paired bootstrap 95% CI over wafers** on the macro-F1 difference.
5. **Inclusion rule:** an extension joins the final pipeline if it improves the B score by more than the seed std;
   its bootstrap CI is reported alongside. **Extensions that fail are reported as negative results**, never dropped.
6. **Seeds:** MFE-FNN goes to **5 seeds** (MLR comparisons then have n = 5). The CNN stays one seed (a retrain is 2+ h)
   — stated as a limitation; the wafer bootstrap covers test-sampling noise.
7. Work on branch **`stage2`**, created from tag `stage1-frozen`. Log every trial in `tuning_log.md` (new "Stage 2"
   section) and `logs/*.jsonl`, as in Stage 1.

## 2. Order of work

### Step 0 — prerequisites (CPU, ~30 min)
- `mfe_tune.py cw0_5s cw=0 seeds=5` → `data/mfe_cw0_5s[_s1.._s4]_outputs.pkl` (same config as the frozen MFE-FNN, 5 seeds).
- Small script changes:
  - `stack_tune.py`: save the 2-fold **pooled B predictions** of each trial (needed by C2 and by the t-test/bootstrap);
    for `meta=mlr`, save the ridge **coefficients** (needed by C3); let `base_seeds` also apply a seed suffix to a
    named input other than the first (needed by C1 control B).
  - New `compare.py`: paired t-test (`scipy.stats.ttest_rel`) across seeds + paired bootstrap (≥ 2,000 resamples of
    wafers) of the macro-F1 difference between two trials; prints mean diff, p-value, 95% CI.
- Baseline: **Stack-2 (B mode)** = `inputs=mfe_cw0_5s,cnn_imnet_cw0_aug mode=B`, MLR (`base_seeds=5`) and FNN (h64, cw0).

### Step 1 — C4: test-time augmentation (GPU inference ~30 min, then CPU)
- Add a predict-only `--tta` mode to `cnn_train.py`: load `models/cnn_imnet_cw0_aug.keras`, average softmax over the
  8 symmetries (4 rotations × optional flip) for **all train wafers and the test set** (≈1.4 M forward passes, fp32),
  save `data/cnn_imnet_cw0_aug_tta_outputs.pkl`. Run detached (see context.md §8, memory guard).
- Report: CNN alone with/without TTA (B_val), Stack-2 with/without TTA (B-cv), with `compare.py` statistics.
- If TTA passes the inclusion rule, the TTA CNN becomes the CNN input for every later step.

### Step 2 — C1: third base learner, XGBoost on the 59 handcrafted features (CPU, ~half a day)
- New `xgb_tune.py`: fit on **A**, early stopping on **B_fit** (number of trees), predict all train + test; save
  `data/xgb_<name>_outputs.pkl`. Grid (~20 configs): `max_depth` {4, 6, 8} × `learning_rate` {0.05, 0.1} ×
  class weights {none, balanced}, `tree_method=hist`. (The old prototype output `data/xgb_outputs.pkl` used 5-fold OOF
  over all of train with balanced weights — **not** used; it does not follow rule 2.)
- Select the XGB config by **stack score** (Stack-3, B-cv), not standalone score (lesson from Stage 1 B3).
- **Stack-3** = MFE-FNN + CNN + XGB, MLR and FNN, B mode, vs Stack-2 (B mode).
- **Control A:** XGB *replaces* the MFE-FNN (Stack = XGB + CNN) — is XGB simply a better feature model?
- **Control B:** a **second MFE-FNN seed** in the third slot instead of XGB — does any third model help (seed
  ensembling), or specifically a *different* one?
- Claim "diversity helps" only if Stack-3 beats both Stack-2 and control B.

### Step 3 — C2: calibration + reject option (CPU, ~half a day)
- New `calibrate.py`, applied to the final stack (MLR and FNN):
  - **Temperature scaling:** `softmax(scores / T)`, one T fitted by log-loss on the stack's **pooled 2-fold B
    predictions**, then applied to test unchanged. (For MLR this also turns ridge scores into probabilities.)
  - **ECE** (15 bins) and **reliability diagrams**, before/after.
  - **Reject rule:** confidence = max probability, one global threshold.
  - **Reports:** risk–coverage curve (`reject_curve.png`); table at coverage 100/98/95/90/85/80% (accuracy + macro-F1
    on accepted wafers); coverage needed to reach 99% and 99.5% accuracy; **per-class rejection breakdown** (which
    defect classes get routed to the human). Per-class thresholds only as a follow-up if the breakdown shows a
    class being rejected wholesale.

### Step 4 — C3: explainability
- **Core:** MLR weight bars per class and per learner (the paper's Fig. 5, for our final stack) → `mlr_weights.png`.
  Explains the C1 result: if XGB helps, the bars show for which classes.
- **If time:** Grad-CAM heatmaps for a few CNN examples per class (incl. misclassified) — short GPU job; then SHAP on
  the 59 handcrafted features (MFE-FNN or XGB).

### Step 5 — freeze the Stage 2 pipeline on B
Apply the inclusion rule to C4 and C1; fix the final stack (MLR headline + FNN), its temperature and the reject table.

### Step 6 — single test-set evaluation
Run every compared configuration on test once (`final=1`), with `compare.py` statistics; build `extensions_results.md`
with a script (like `freeze_results.py`). Commit on `stage2` with a full explanatory message (as for Stage 1).

## 3. Deliverables
- `extensions_results.md` — one table per extension: mean ± std, paired t-test p, bootstrap 95% CI; controls; negative
  results stated plainly.
- Figures: `reject_curve.png`, `reliability_before_after.png`, `mlr_weights.png`, `rejection_by_class.png`
  (+ Grad-CAM / SHAP if done).
- Scripts that regenerate all of the above. Updated `tuning_log.md`, `context.md`, `implementation_plan.md`.
- Commit on branch `stage2` with a full message. (Report text is Part D — not part of Stage 2.)

## 4. Rough timeline (after sign-off)
| Day | Work |
|---|---|
| 1 | Step 0 (seeds, script changes, B-mode baseline); Step 1 TTA; start Step 2 grid |
| 2 | Step 2: Stack-3, controls, statistics |
| 3 | Step 3 calibration + reject option; Step 4 weight bars |
| 4–5 | Step 4 optional Grad-CAM/SHAP; Steps 5–6 freeze, test evaluation, write-up, commit |
| buffer | 2 days |

## 5. Known limitations to carry into the report
- CNN is a single seed in both stages; the Stage 1 CNN tuning was stopped at the run budget, not formally saturated.
- Near-full has 9 test wafers (30 in B): one wafer ≈ 0.01 macro-F1. Hence the bootstrap CIs.
- Our split is one fixed stratified split; the paper averages 10 random splits.

---

## Decision record (grilling session, 2026-10-07 — all answered "go with the recommendation")

| # | Question | Decision |
|---|---|---|
| Q1 | Which contributions? | C1 + C2 core, C4 add-on, C3 if time; **skip C5** (semi-supervised on unlabeled wafers: many GPU-days, high risk) |
| Q2 | Budget | ~1 week, CPU-mostly, GPU inference only, no new CNN training |
| Q3 | Baseline to beat | Both stacks; **MLR headline**, FNN alongside |
| Q4 | "Really better?" test | Paired t-test across seeds + paired bootstrap CI over wafers; test touched once |
| Q5 | Honest XGB inputs | XGB fitted on A (early-stop B_fit); **all Stage 2 stackers fitted on B**; Stack-2 re-run in B mode |
| Q6 | XGB tuning | ~20-config grid incl. class weights on/off; select by stack score |
| Q7 | C1 controls | Control A (XGB replaces MFE-FNN) + Control B (2nd MFE-FNN seed as third learner) |
| Q8 | Calibration data | Stacker's 2-fold cross-fitted predictions on B (not test halves) |
| Q9 | Calibration method | Temperature scaling (1 parameter); ECE + reliability diagrams |
| Q10 | Reject rule | Max probability, one global threshold; risk–coverage, fixed-coverage table, coverage@99/99.5% accuracy, per-class rejection breakdown |
| Q11 | TTA scope | TTA on train and test (no train/test input mismatch for the stacker) |
| Q12 | Inclusion rule | Improve B by > seed std; CI reported; failures reported as negative results |
| Q13 | Order | C4 → C1 → C2 → C3; branch `stage2` from `stage1-frozen` |
| Q14 | Explainability | MLR weight bars = core; Grad-CAM, SHAP = if time |
| Q15 | Seeds | MFE-FNN 3 → 5 seeds; CNN stays 1 seed (limitation) |
| Q16 | Deliverables | as §3 |
