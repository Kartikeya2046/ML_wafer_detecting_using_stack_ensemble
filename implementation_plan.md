# Implementation Plan — Reproduction + Original Contribution

Paper: Kang & Kang 2021, *Computers in Industry* 129:103450 (stacking ensemble of handcrafted-feature FNN + VGG-style CNN for WM-811K wafer map classification).
Companion files: `context.md` (background), `plan.md` (original reproduction plan, phases 0–7).
Last updated: 2026-10-08.

> **STATUS (2026-10-07): Part B (Stage 1) is DONE and FROZEN** — git tag `stage1-frozen`, branch `stage1-tuning`.
> Results: `results_comparison.md`; every trial and decision: `tuning_log.md`.
> **STATUS (2026-10-08): Part C (Stage 2) is DONE** on branch `stage2` (user signed off `stage2_plan.md`). Results:
> `extensions_results.md`; log: `tuning_log.md` "Stage 2". **Next: Part D** (plan the report structure with the user first).

---

## Part A — What has been done

### A1. Reproduction (original plan.md Phases 1–6)

| Phase | Status | Where |
|---|---|---|
| 1. Read paper + both reference repos | Done | `context.md` §3, §5 |
| 2. Data pipeline (WM-811K, 9 classes, 59 MFE features, 64×64 maps) | Done | `phase2_data.py`, `data/X_MFE.pkl`, `data/X_CNN.pkl`, `data/y.pkl` |
| 3. MFE + FNN base learner (59→128→128→9, tanh) | Done | `training_pipeline.ipynb` cells 3–5, `models/mfe_fnn_model.keras` |
| 4. VGG16-style CNN (64×64, 9-class softmax) | Done | notebook cells 7–11, `models/cnn_model.keras` |
| 5. Stacking meta-learner (18→10→9 FNN on concatenated softmax outputs) | Done, **had a bug** (see A2) | notebook cells 14–16 |
| 6. Three-way comparison | Done (Stage 1) | `results_comparison.md` — ours vs paper Table 7/8, built by `freeze_results.py` |
| 7. Report, slides, viva one-pager | Not started | — |

Data facts (from saved pickles): train = 162,946 wafers, test = 10,000. Test class counts: Center 248, Donut 32, Edge-Loc 300, Edge-Ring 560, Loc 208, Random 50, Scratch 69, Near-full **9**, none 8524.

Reported results before the fix (macro-F1): MFE+FNN 0.8147, CNN 0.8496, Stacking 0.7614.

### A2. Findings made in this session

1. **Stacking bug found and fixed.** The meta-learner was trained without class weights (the base learners use them). It early-stopped at ~23 epochs and ignored rare classes, so Near-full F1 = 0.0000 even though both base learners predicted it correctly. With `class_weight='balanced'` one test run gave macro-F1 0.850 and Near-full F1 0.737.
   - Fix is applied in notebook cell 14 (`class_weight=class_weight_dict`). *(Resolved in Part B: the fixed baseline was re-run as B1 by `stack_tune.py`, and `results_comparison.md` is now built by `freeze_results.py`. `data/stacking_outputs.pkl` is the old notebook output, kept for history.)*
2. **Seed noise is large.** 5 seeds of the fixed 2-learner stack: macro-F1 **0.8424 ± 0.0405**. Near-full has 9 test samples, so one wafer moves its F1 by roughly 0.1. Any comparison must use multi-seed mean ± std.
3. **XGBoost on the 59 features** reaches macro-F1 **0.8285** standalone (5-fold out-of-fold train predictions cached in `data/xgb_outputs.pkl`). This is above the MFE+FNN (0.8147), which is why it's the candidate third learner.
4. **Meta-learner trains on in-sample base predictions** (`train_prob` comes from models trained on the same data). This matches the reference repo, so we keep it for the baseline, but it is a known weakness and a candidate for tuning (see B2).

### A3. Prototype files (now committed)
- `phase7_extensions.py` — early prototype of the 3-learner stack + calibration/reject option. **Superseded by `stage2_plan.md`** (its 5-fold OOF XGB design breaks Stage 2 rule 2); kept for history, not to be run.
- `data/xgb_outputs.pkl` (not in git, not used by Stage 2), `ext.log` — by-products of that prototype run.
- `training_pipeline.ipynb`, `results_comparison.md`, `results_comparisons.ipynb` — committed with Stage 1 (`4bfadf7`).

### A4. Resolved: N=5000 vs full data
- Settled in B0 (2026-10-06): headline numbers are full data, N=162,946 train / 10,000 test (`context.md` §6). The `models/*_5k.keras` files are the superseded N=5000 workaround.

---

## Part B — Stage 1: Tune the existing approach until saturated

Goal: get the paper's 2-learner pipeline as strong as honestly possible before adding anything new. Every change is judged against the same yardstick.

**Rules**
- Tune on a validation split carved from training data. The 10,000 test wafers are used only for the final reported numbers.
- Metric: macro-F1 as mean ± std over 5 seeds; also per-class F1.
- Saturation: stop when two consecutive rounds improve the 5-seed mean macro-F1 by less than the seed std.
- Log every trial (config → val macro-F1) in `tuning_log.md`.

### B0. Confirm baseline scope
- Settle the N=5000 vs full-data question (A4); write the answer into `context.md`.
- Keep a record of exactly which model files feed the headline table.

### B1. Re-run the fixed baseline
- Re-run notebook cells 14–16 with the class-weight fix, 5 seeds.
- Regenerate `stacking_outputs.pkl` and `results_comparison.md` with mean ± std.
- Output: the **reference baseline** every later step is compared to.

### B2. Tune the stacker (cheap, minutes per trial)
Trials, one factor at a time then a small combination:
- Hidden size (10 → 16/32/64), learning rate (1e-4 → 1e-3), batch size.
- Class-weight strength: full balanced vs square-root balanced vs none.
- Stratified validation split for early stopping (currently the last 20% unstratified-by-class, so rare classes can be missing).
- Early-stopping patience and monitored metric (val loss vs val macro-F1).
- Optional: dropout / L2 on the meta-learner.
- Optional, compare to paper-faithful: train the meta-learner on out-of-fold base predictions instead of in-sample ones (needs base-learner K-fold, so only if cheap for the MFE-FNN; the CNN is too slow to K-fold on CPU).

### B3. Tune the MFE + FNN base learner (cheap)
- Width/depth, activation, dropout, learning rate, class-weight strength, feature standardisation check.
- Retrain, save new `mfe_fnn_outputs.pkl`, re-run B2's best stacker on top.

### B4. CNN improvements (expensive — a few runs at most)
CNN runs cost hours on CPU, so no grid search. Candidates, in order of expected value:
1. Data augmentation (flips, 90° rotations — wafer defect classes are mostly rotation-invariant; Scratch/Edge-Loc are the classes to watch).
2. Label smoothing.
3. Learning-rate schedule (reduce-on-plateau) in place of fixed 1e-4.
- Only run (2) and (3) if (1) helps or is inconclusive. Retrain, re-save `cnn_outputs.pkl`, re-run best stacker.

### B5. Saturation check and freeze
- Apply the saturation rule; record the final tuned 2-learner configuration and its 5-seed numbers.
- Freeze models and outputs (tag in git). This frozen result is both our "improved reproduction" and the baseline for Stage 2.

**Deliverable of Part B:** `tuning_log.md`, updated `results_comparison.md` (before-fix / after-fix / tuned), frozen model files.

**Part B outcome (2026-10-07):** B0–B5 done. Frozen pipeline = MFE+FNN (paper arch, no class weights) + VGG16 CNN (ImageNet init, no class weights, flip/rotation aug, fp32) + MLR stacker (paper's proposed, α=0.1). Test macro-F1: Stacking-MLR 0.8967 ± 0.0060 (paper 0.8949 ± 0.0121), tuned Stacking-FNN 0.9041 ± 0.0028 (paper 0.8991 ± 0.0096), vs B1 reference 0.8402 ± 0.0393. Deviations from this plan, all logged in tuning_log.md: added the paper's MLR meta-learner and ImageNet init (missing from the original reproduction); B2's "out-of-fold" option done for free via split B (base learners never fitted on it); CNN not formally saturated (stopped at 7 GPU runs). Scripts: `common.py`, `stack_tune.py`, `mfe_tune.py`, `cnn_train.py`, `freeze_results.py`, `watch_training.py`, `run_gpu.sh`/`run_cpu.sh`.

---

## Part C — Stage 2: Original contribution (detailed plan is written first, then implemented)

Do not start until Part B is frozen. First step of Part C is to write `stage2_plan.md` and get approval; the outline below is the starting point.

### C1. Third base learner (XGBoost on the 59 MFE features)
- Hypothesis: a differently-built learner adds complementary information to the stack, beyond the paper's two.
- Build: 5-fold out-of-fold probabilities for train (avoids in-sample leakage), full-train fit for test. Already prototyped in `phase7_extensions.py`; re-tune XGBoost hyperparameters on validation (depth, trees, learning rate) before the final run.
- Evaluate: Stack-2 (tuned, frozen) vs Stack-3, same meta-learner config, 5 seeds, per-class F1, paired comparison.
- Control: also try XGBoost *replacing* the MFE-FNN, to show the gain is from the extra learner and not just a stronger feature model.
- Document the deliberate deviation: XGB uses out-of-fold train probabilities while the other two use in-sample ones.

### C2. Calibration + reject option
- Temperature-scale the stacker's probabilities; report ECE before/after.
- Risk–coverage curve: accuracy and macro-F1 on accepted wafers vs fraction auto-classified (e.g. 100/98/95/90/85/80%), with the rest routed to a human.
- Calibration set must be separate from the evaluation set (the prototype cross-fits on two halves of the test set; revisit whether to use a held-out slice of training data instead).

### C3. Optional (only if time allows)
- Explainability: Grad-CAM on the CNN, SHAP on the 59 features, to show why the stacker favours one learner for particular classes.

**Deliverable of Part C:** `stage2_plan.md` (approved), `extensions_results.md`, `reject_curve.png`, a clearly labelled "our extensions" section for the report.

**Part C outcome (2026-10-08):** all of C1–C4 done incl. the optional Grad-CAM and SHAP. Final pipelines (chosen on B): MLR = MFE-FNN + CNN-TTA (test 0.9051 ± 0.0069), FNN = MFE-FNN + CNN-TTA + XGB (test 0.9070 ± 0.0037). TTA passes the inclusion rule for both stackers; XGB as third learner only for FNN, and never significantly beats a second MFE-FNN seed (control B); XGB replacing the MFE-FNN (control A) is the best MLR stack (test 0.9160). Reject option: 95% coverage → accuracy 99.6%, macro-F1 0.979. Temperature scaling does not help the MLR. Figures in `figures/`.

---

## Part D — Closing the project

1. **Paper comparison (Phase 6 remainder):** extract the paper's own per-class and macro numbers from `journal.pdf` (never from memory), place them next to ours, and write the honest discussion paragraph.
2. **Report** (Word): Introduction, Related Work, Dataset, Methodology, Results (reproduction + tuning + extensions), Discussion/Limitations, Conclusion. State clearly what is the paper's method and what is our own contribution. Note that implementation was AI-assisted, as the professor approved.
3. **Slides** mirroring the report.
4. **Viva one-pager:** wafer maps and why classification matters, handcrafted vs CNN features, why stacking helps, why the class-weight bug mattered, what the extensions add.
5. **Housekeeping:** update `context.md` and `plan.md`, commit with sensible messages, remove stray scripts (`fix_*.py`, `test_geom*.py`) or move them to an `archive/` folder.

---

## Order of work

| # | Step | Cost | Depends on |
|---|---|---|---|
| 1 | B0 confirm scope | minutes | — |
| 2 | B1 fixed baseline, 5 seeds | ~30 min CPU | B0 |
| 3 | B2 stacker tuning | hours, cheap trials | B1 |
| 4 | B3 MFE-FNN tuning | hours | B2 |
| 5 | B4 CNN improvements | many hours per run | B3 |
| 6 | B5 saturation check, freeze | minutes | B2–B4 |
| 7 | Write and approve `stage2_plan.md` | — | B5 |
| 8 | C1, C2 (then C3 if time) | hours | stage2 plan |
| 9 | Part D: paper comparison, report, slides, viva | days | all of the above |
