# Tuning Log — Stage 1 (implementation_plan.md Part B)

Raw per-trial records: `logs/stack_trials.jsonl`, `logs/mfe_trials.jsonl`, `logs/cnn_*.log|csv`.

## Protocol (fixed for all of Part B)

- **Split of the 162,946 training wafers** (`common.py:splits`):
  - **A** = first 130,356 (80%). The original base learners were fitted on exactly this part (Keras `validation_split=0.2` keeps the first 80%).
  - **B** = last 32,590. The base learners never fitted on B (it was only their early-stopping set), so their predictions on B are honest. B is split once, stratified, into **B_fit** / **B_val** (16,295 each).
  - The 10,000 test wafers are used only for final numbers.
- **Stacker trials** (`stack_tune.py`): 2-fold cross-fit over B (B_fit↔B_val), predictions pooled over all of B (30 Near-full, 218 Scratch, 99 Donut). Score = macro-F1, mean ± std over 5 seeds (MLR is deterministic: 1 run).
- **MFE-FNN trials** (`mfe_tune.py`): fit A, early-stop on B_fit, standalone macro-F1 on B_val, 3 seeds.
- **CNN runs** (`cnn_train.py`): fit A, early-stop on B_fit, single seed (GPU hours), standalone macro-F1 on B_val. Mixed precision + batch 128 (LR scaled ×√4 → 2e-4). Throughput benchmark: batch 32 fp32 537 img/s, batch 32 fp16 953, batch 128 fp16 1413 (2.6×). Run `ctrl` = same recipe, no other change, so the speed settings are not confounded with the improvements.
- **Known small leak:** the *original* base learners early-stopped on all of B, so B carries epoch-selection information for them. It affects every stacker config equally.
- **Saturation rule:** stop when two consecutive rounds improve the 5-seed mean by less than the seed std.

## B0 — scope
Headline = full data (N=162,946). See context.md §6. Paper reference: Table 7, N=162,946.

## B1 — reference baseline on the test set (original base learners, notebook stacker FNN 18→10→9, 5 seeds)

| config | test macro-F1 | Near-full F1 (mean) | epochs |
|---|---|---|---|
| notebook as committed (no class weights in the meta-learner, = repo) | 0.6298 ± 0.0904 | 0.000 | ~23 |
| **+ class-weight fix (reference baseline)** | **0.8402 ± 0.0393** | 0.657 | ~24 |

The old single-run 0.7614 in results_comparison.md was one draw of the unweighted configuration. The large std of the fixed baseline is Near-full (9 test wafers) flipping between ~0 and ~0.9 by seed — the instability B2 removes.

## B2 — stacker, round 1 (one factor at a time; base = notebook cell 14 + class-weight fix: FNN 18→10→9, lr 1e-4, batch 32, balanced cw, mode A)
| trial | B-cv macro-F1 | seed std | Near-full F1 | Scratch F1 | epochs (fold1, fold2) |
|---|---|---|---|---|---|
| base | 0.8369 | 0.0401 | 0.661 | 0.702 | [25, 24] |
| mode_B | 0.8320 | 0.0020 | 0.896 | 0.535 | [123, 232] |
| mode_AB | 0.8644 | 0.0021 | 0.915 | 0.674 | [102, 67] |
| h16 | 0.8697 | 0.0024 | 0.875 | 0.720 | [24, 24] |
| h32 | 0.8718 | 0.0043 | 0.894 | 0.703 | [22, 22] |
| h64 | 0.8758 | 0.0037 | 0.917 | 0.721 | [22, 22] |
| lr1e-3 | 0.8592 | 0.0114 | 0.815 | 0.707 | [21, 21] |
| bs128 | 0.8602 | 0.0059 | 0.829 | 0.710 | [34, 33] |
| bs512 | 0.8662 | 0.0041 | 0.870 | 0.716 | [67, 65] |
| cw_sqrt | 0.8013 | 0.0468 | 0.342 | 0.701 | [24, 24] |
| cw_none | 0.6295 | 0.0880 | 0.000 | 0.541 | [23, 23] |
| mon_f1 | 0.8692 | 0.0041 | 0.898 | 0.696 | [42, 32] |
| pat50 | 0.8369 | 0.0401 | 0.661 | 0.702 | [55, 54] |
| drop02 | 0.8175 | 0.0493 | 0.495 | 0.702 | [25, 24] |
| l2_1e-4 | 0.8633 | 0.0039 | 0.873 | 0.699 | [24, 24] |

**MLR (paper's proposed meta-learner, `meta=mlr`, deterministic → 1 run; default α=0.1).**

| trial | B-cv macro-F1 | seed std | Near-full F1 | Scratch F1 | epochs (fold1, fold2) |
|---|---|---|---|---|---|
| mlr_A_cw0 | 0.8802 | — | 0.909 | 0.742 | — |
| mlr_A_cw1 | 0.8574 | — | 0.909 | 0.702 | — |
| mlr_B_cw0 | 0.8808 | — | 0.909 | 0.737 | — |
| mlr_B_cw1 | 0.8621 | — | 0.857 | 0.713 | — |
| mlr_B_cw05 | 0.8687 | — | 0.877 | 0.719 | — |
| mlr_A_a001 | 0.8802 | — | 0.909 | 0.742 | — |
| mlr_A_a1 | 0.8797 | — | 0.909 | 0.742 | — |
| mlr_A_a10 | 0.8826 | — | 0.929 | 0.744 | — |
| mlr_A_a100 | 0.8781 | — | 0.909 | 0.728 | — |
| mlr_B_a1 | 0.8810 | — | 0.909 | 0.739 | — |
| mlr_B_a10 | 0.8812 | — | 0.909 | 0.743 | — |
| mlr_AB_a01 | 0.8801 | — | 0.909 | 0.741 | — |

## B2 — stacker, round 2 (combinations of round-1 winners)

| trial | B-cv macro-F1 | seed std | Near-full F1 | Scratch F1 | epochs (fold1, fold2) |
|---|---|---|---|---|---|
| r2_h64_bs512 | 0.8754 | 0.0029 | 0.925 | 0.714 | [39, 38] |
| r2_h64_bs512_l2 | 0.8750 | 0.0024 | 0.925 | 0.717 | [40, 38] |
| r2_h128_bs512 | 0.8778 | 0.0021 | 0.921 | 0.717 | [35, 34] |
| r2_h64_bs512_f1 | 0.8746 | 0.0018 | 0.921 | 0.706 | [33, 61] |
| r2_h64_B | 0.8329 | 0.0010 | 0.897 | 0.540 | [63, 57] |
| r2_h64_B_cw05 | 0.8695 | 0.0011 | 0.893 | 0.707 | [63, 83] |
| r2_h64_AB_bs512 | 0.8641 | 0.0011 | 0.915 | 0.662 | [38, 121] |
| r2_h64_B_cw0 | 0.8790 | 0.0041 | 0.890 | 0.734 | [77, 200] |

## B2 — stacker, round 3

| trial | B-cv macro-F1 | seed std | Near-full F1 | Scratch F1 | epochs (fold1, fold2) |
|---|---|---|---|---|---|
| r3_h128_B_cw0_bs128 | 0.8782 | 0.0024 | 0.883 | 0.736 | [174, 306] |
| r3_h128_B_cw0 | 0.8789 | 0.0024 | 0.883 | 0.736 | [77, 279] |
| r3_h256_bs512 | 0.8744 | 0.0025 | 0.929 | 0.708 | [31, 31] |
| r3_h64_B_cw0_f1 | 0.8777 | 0.0036 | 0.889 | 0.729 | [57, 83] |

### B2 findings
- **Root cause of the original stacker failure is mode A, not only class weights.** On A the base learners' outputs are in-sample (MFE-FNN macro-F1 0.94 on A vs 0.81 on B), so the FNN meta-learner sees near-perfect, overconfident inputs; its validation loss rises almost from the start and it early-stops at ~23 epochs regardless of patience (pat50 = identical result). Without class weights it then never predicts rare classes (cw_none 0.630, Near-full 0). Class weights or a wider hidden layer mask this; training on honest outputs (mode B) without class weights removes it (0.879).
- **The paper's own MLR meta-learner is as good as the best tuned FNN** (0.880 vs 0.879 ± 0.004) and insensitive to α (0.878–0.883 over α=0.01…100; differences ≈ one Near-full wafer). Class weights hurt MLR (0.857).
- **Saturation:** round 1 best FNN 0.8758 → round 2 0.8790 (+0.003) → round 3 0.8789 (+0.000); both gains < seed std (~0.003). Stopped.
- **Chosen stacker: MLR, α=0.1, no class weights, fitted on A** (= paper eq. 5, paper-faithful, deterministic). Tuned FNN kept as the paper's "Stacking-FNN" comparison: hidden 64, mode B, no class weights, lr 1e-4, batch 32.
- First test-set look (single, for the record — not used for selection): MLR on the original base learners = **0.8688** test macro-F1 (MLR mode B 0.8666).

## B3 — MFE + FNN base learner (fit A, early-stop B_fit, 3 seeds)

Stack column = MLR stack (α=0.1, no cw) on [this MFE-FNN, original CNN], B-cv macro-F1; `stk3` rows average over the 3 MFE seeds.

| trial | arch | cw | lr | batch | dropout | standalone B_val macro-F1 | Scratch | Near-full | epochs | **stack B-cv** |
|---|---|---|---|---|---|---|---|---|---|---|
| cw0 | 128×2 tanh | 0.0 | 0.0001 | 32 | 0.0 | 0.8513 ± 0.0074 | 0.651 | 0.838 | [62, 47, 52] | 0.8914 (seed 0 only) |
| cw05 | 128×2 tanh | 0.5 | 0.0001 | 32 | 0.0 | 0.8333 ± 0.0031 | 0.598 | 0.847 | [67, 58, 76] | — |
| relu | 128×2 relu | 1.0 | 0.0001 | 32 | 0.0 | 0.7826 ± 0.0054 | 0.392 | 0.855 | [46, 103, 81] | — |
| base | 128×2 tanh | 1.0 | 0.0001 | 32 | 0.0 | 0.8032 ± 0.0186 | 0.469 | 0.870 | [187, 177, 84] | 0.8825 (seed 0 only) |
| d3 | 128×3 tanh | 1.0 | 0.0001 | 32 | 0.0 | 0.7958 ± 0.0234 | 0.459 | 0.886 | [67, 154, 76] | — |
| w256 | 256×2 tanh | 1.0 | 0.0001 | 32 | 0.0 | 0.8356 ± 0.0047 | 0.618 | 0.892 | [127, 120, 141] | 0.8689 (seed 0 only) |
| drop02 | 128×2 tanh | 1.0 | 0.0001 | 32 | 0.2 | 0.7633 ± 0.0016 | 0.289 | 0.884 | [75, 64, 99] | — |
| lr1e-3 | 128×2 tanh | 1.0 | 0.001 | 32 | 0.0 | 0.7449 ± 0.0093 | 0.375 | 0.816 | [26, 26, 68] | — |
| r2_cw0_w256 | 256×2 tanh | 0.0 | 0.0001 | 32 | 0.0 | 0.8518 ± 0.0014 | 0.658 | 0.860 | [41, 47, 43] | — |
| r2_cw0_pat50 | 128×2 tanh | 0.0 | 0.0001 | 32 | 0.0 | 0.8513 ± 0.0074 | 0.651 | 0.838 | [92, 77, 82] | — |
| r2_cw0_w256_pat50 | 256×2 tanh | 0.0 | 0.0001 | 32 | 0.0 | 0.8518 ± 0.0014 | 0.658 | 0.860 | [71, 77, 73] | — |
| r2_cw0_w512 | 512×2 tanh | 0.0 | 0.0001 | 32 | 0.0 | 0.8593 ± 0.0023 | 0.664 | 0.864 | [36, 37, 38] | 0.8852 (seed 0 only) |
| x3_cw0 | 128×2 tanh | 0.0 | 0.0001 | 32 | 0.0 | 0.8513 ± 0.0074 | 0.651 | 0.838 | [62, 47, 52] | 0.8886 ± 0.0034 |
| x3_cw0_w512 | 512×2 tanh | 0.0 | 0.0001 | 32 | 0.0 | 0.8593 ± 0.0023 | 0.664 | 0.864 | [36, 37, 38] | 0.8872 ± 0.0015 |
| r3_cw0_w512_bs128 | 512×2 tanh | 0.0 | 0.0001 | 128 | 0.0 | 0.8590 ± 0.0095 | 0.657 | 0.884 | [58, 46, 61] | 0.8859 ± 0.0015 |
| r3_cw0_w512_d3 | 512×3 tanh | 0.0 | 0.0001 | 32 | 0.0 | 0.8474 ± 0.0056 | 0.653 | 0.813 | [32, 31, 30] | 0.8841 ± 0.0022 |
| r3_cw0_w512_lr3e-4 | 512×2 tanh | 0.0 | 0.0003 | 32 | 0.0 | 0.8568 ± 0.0079 | 0.663 | 0.889 | [32, 31, 32] | 0.8872 ± 0.0032 |
| r3_cw0_w1024 | 1024×2 tanh | 0.0 | 0.0001 | 32 | 0.0 | 0.8506 ± 0.0047 | 0.659 | 0.827 | [36, 31, 32] | 0.8859 ± 0.0029 |

Round 1 = rows `base`…`lr1e-3` (one factor), round 2 = `r2_*`, round 3 = `x3_*`/`r3_*` (`x3_*` are re-runs of the round-2 leaders saving all seeds; identical scores, same seeds).
Feature check: no NaN/inf, no constant columns; a few Radon-std features are heavy-tailed (max |z|≈45, 0.07% of values beyond |z|>10) — left as is (tanh saturates them).

### B3 findings
- **Class weights hurt the MFE-FNN by ~5 pts** (0.803 → 0.851): with weights up to 129× (Near-full) the network over-predicts rare classes and loses precision (Scratch F1 0.47 → 0.65, Loc 0.64 → 0.74). The paper does not use class weights.
- Wider layers help standalone (w512 0.859) but **not inside the stack**: all cw0 variants tie at stack level (0.884–0.889, std ≈0.002–0.003). Depth 3, ReLU, dropout and lr 1e-3 are worse.
- **Saturation (judged on the stack score, the quantity we report):** round 1 best 0.8886 → round 2 0.8872 → round 3 0.8872; both rounds below seed std. Stopped.
- **Chosen MFE-FNN: paper architecture (59→128→128→9 tanh, lr 1e-4, batch 32), no class weights** (`mfe_x3_cw0[_s1,_s2]_outputs.pkl`; seeds 0–2 are all kept so final numbers can average over base-learner seeds).

## B4 — CNN runs (GPU)

| run | recipe | epochs (best) | train time | B_val macro-F1 | test macro-F1 (record only) |
|---|---|---|---|---|---|
| original (`cnn_outputs.pkl`) | from scratch, batch 32, fp32, lr 1e-4, balanced cw; early-stopped on all of B (so B_val slightly optimistic) | 63+ | ~6 h | 0.8611 | 0.8496 |
| ctrl | same but **batch 128, mixed precision, lr 2e-4** | 53 (33) | 1.86 h | 0.8174 | 0.8252 |

**Decision:** batch 128 cost ~4 pts macro-F1 (Donut 0.90→0.79, Scratch 0.71→0.61), so all later CNN runs use **batch 32 + mixed precision** (lr 1e-4 = paper). Mixed precision alone gives 1.8× over fp32 without changing the optimisation. The original CNN serves as the control for these runs.
| aug | batch 32, mixed precision, lr 1e-4, balanced cw, **flips + 90° rotations** | 55 (35) | 2.20 h | 0.8155 | 0.8006 |

**aug finding:** no gain — 0.816 vs the original 0.861 (per-class B_val: Center 0.86, Donut 0.86, Edge-Loc 0.78, Scratch 0.66, Near-full 0.75). Both new-recipe runs (ctrl, aug) sit ~4 pts under the original, also on test, which ES cannot touch. Common differences to the original: mixed precision, early stopping on B_fit (16k) instead of all of B (33k). Unresolved; single seeds (paper CNN std ≈ 0.013).
| imnet | **ImageNet-pretrained VGG16** (paper §4.3), batch 32, **fp32**, lr 1e-4, balanced cw | 44 (24) | 2.98 h | 0.8478 | 0.8515 |

Note: a mixed-precision `imnet` start (20:16) was stopped after the user chose fp32 for the remaining runs; partial logs kept as `logs/killed_mp_cnn_imnet.*`.

**CNN candidates scored inside the stack** (MLR α=0.1 no cw + tuned MFE-FNN `x3_cw0`, B-cv, mean ± std over the 3 MFE seeds):

| CNN | stack B-cv macro-F1 |
|---|---|
| original | 0.8886 ± 0.0034 |
| ctrl | 0.8841 ± 0.0012 |
| aug | 0.8796 ± 0.0056 |
| **imnet** | **0.8933 ± 0.0016** (Near-full 0.953) |

**imnet finding:** standalone it ties the original (B_val 0.848 vs 0.861, but the original early-stopped on data including B_val; test 0.852 vs 0.850), and **inside the stack it is the best CNN (+0.005, above seed std)**. It is also the paper-faithful choice. The two fp16 runs are the worst, consistent with mixed precision costing accuracy here; all remaining runs are fp32.
| imnet_cw0 | ImageNet init, fp32, batch 32, lr 1e-4, **no class weights** | 26 (6) | 1.88 h | **0.8698** | 0.8727 |

Note: a from-scratch `cw0` run was stopped minutes after starting and replaced by `imnet_cw0` (no-class-weights tested directly on top of the winning init — one factor vs `imnet`, saves a run). Partial logs: `logs/killed_scratch_cnn_cw0.*`.

| CNN in stack | stack B-cv macro-F1 |
|---|---|
| imnet | 0.8933 ± 0.0016 |
| **imnet_cw0** | **0.8961 ± 0.0015** |

**imnet_cw0 finding:** removing class weights helps the CNN too (standalone B_val 0.848 → 0.870; Scratch 0.66 → 0.78, Loc 0.70 → 0.78), same mechanism as the MFE-FNN. Stack +0.003 (~2× seed std). Both B4 rounds improved by more than the std, so one last round runs on top of imnet_cw0 (fp32): `imnet_cw0_aug` (augmentation, previously confounded by fp16) and `imnet_cw0_plat` (reduce-on-plateau LR). Label smoothing skipped: it distorts the probabilities the MLR stacker combines.

**B4 final round (on top of imnet_cw0, fp32):**

| run | change vs imnet_cw0 | epochs (best) | train time | B_val macro-F1 | test (record only) | stack B-cv |
|---|---|---|---|---|---|---|
| imnet_cw0_aug | + flips / 90° rotations | 31 (11) | 2.08 h | 0.8723 | 0.8777 | **0.9016 ± 0.0043** |

Augmentation helps once trained in fp32 (+0.0055 in the stack, > seed std; Edge-Loc 0.83→0.86, Loc 0.78→0.81). The earlier negative `aug` result was the fp16 recipe, not augmentation.
| imnet_cw0_plat | + ReduceLROnPlateau (×0.5, patience 5) | 24 (4) | 1.61 h | 0.8599 | 0.8882 | 0.8921 ± 0.0016 |

Plateau schedule does not help (stack 0.8921 < 0.8961 for imnet_cw0); its higher test score is not used for selection. No combined aug+plateau run.

## B5 — saturation check and freeze

**Frozen pipeline (all choices made on B, never on test):**

| Component | Frozen choice | Changed from the original |
|---|---|---|
| MFE+FNN | paper Table 3 (59→128→128→9 tanh, Adam 1e-4, batch 32), **no class weights**, 3 seeds | class weights removed |
| CNN | VGG16 **ImageNet-initialised** (paper §4.3), GAP + softmax, Adam 1e-4, batch 32, fp32, **no class weights**, **flips + 90° rotations** | init, class weights, augmentation |
| Stacker | **MLR** (ridge on one-hot targets, α = 0.1, no intercept, no class weights) on [MFE, CNN] probabilities fitted on A — the paper's proposed meta-learner | was an 18→10→9 FNN (paper's Stacking-FNN baseline) |
| Stacker (reported too) | Stacking-FNN tuned: hidden 64, fitted on B (honest base outputs), no class weights | hidden 10→64, B inputs, no cw |

**Saturation:** stacker (B2) and MFE-FNN (B3) met the rule (two consecutive rounds < seed std). The CNN (B4) did **not formally saturate** — each round so far still improved the stack by more than the seed std (original 0.8886 → imnet 0.8933 → imnet_cw0 0.8961 → +aug 0.9016; plateau LR did not help). B4 was stopped at 7 GPU runs under the plan's "a few runs at most" budget; further CNN gains (more seeds, longer augmentation schedules) are possible and are noted as a limitation.

**Final test results** (results_comparison.md, built by `freeze_results.py`): Stacking-MLR **0.8967 ± 0.0060** (paper 0.8949 ± 0.0121), Stacking-FNN **0.9041 ± 0.0028** (paper 0.8991 ± 0.0096), MFE+FNN 0.8572 ± 0.0062 (paper 0.8599), CNN 0.8777 (single run; paper 0.8679). Reference baseline B1 was 0.8402 ± 0.0393.

Frozen artefacts: `data/mfe_x3_cw0[_s1,_s2]_outputs.pkl`, `data/cnn_imnet_cw0_aug_outputs.pkl`, `models/cnn_imnet_cw0_aug.keras`, `data/stack_FINAL_stack_{mlr,fnn}_*.pkl`. Git tag `stage1-frozen`.

---

# Stage 2 (stage2_plan.md, implementation_plan.md Part C) — 2026-10-08, branch `stage2`

Machine: Linux workstation (RTX 4000 Ada 20 GB, 24 cores, 125 GB RAM), `.venv` TF 2.17.1 + tf_keras 2.17 (`setup_env.sh`).
Equivalence with the Stage 1 laptop checked first: `freeze_results.py` reproduces `results_comparison.md` with zero diff;
the frozen CNN (rebuilt + `load_weights`, since `load_model` cannot unmarshal its Lambda layer across Python versions)
re-predicts test within 1.2e-4 (macro-F1 0.8777); MFE-FNN seeds 0–2 reproduce `x3_cw0` exactly (0.8442 / 0.8481 / 0.8616).
Independent jobs ran side by side (GPU: TTA, Grad-CAM; CPU: MFE seeds, XGB grid ×4, up to 10 stacker trials at once).
All scores: `logs/stack_trials.jsonl`, `logs/xgb_trials.jsonl`, `logs/compare.jsonl`, `logs/calibrate.jsonl`.
Full tables: `extensions_results.md` (built by `extensions_results.py`).

## Step 0 — prerequisites
- `mfe_tune.py cw0_5s cw=0 seeds=5` (frozen config, 5 seeds): B_val 0.8525 ± 0.0074, seeds [0.8442, 0.8481, 0.8616, 0.8613, 0.8474], 5.3 min.
- `stack_tune.py`: saves pooled B predictions (`data/stack_cv_<name>_b*_s*.pkl`) and MLR coefficients; inputs may be written
  `<name>+k` (seed b+k in round b, used by control B). New `compare.py` (paired t-test + wafer bootstrap).
- **Deviation:** FNN stackers use `base_seeds=5 seeds=1` (n = 5, paired with MLR by MFE seed) instead of 5×5 = 25 runs per trial.
- Stack-2 B-mode baselines: MLR 0.9030 ± 0.0034, FNN 0.8971 ± 0.0034.

## Step 1 — C4 TTA
`cnn_train.py --tag imnet_cw0_aug --tta` (8 symmetries, all train + test, 4.4 min on GPU). CNN alone B_val 0.8723 → 0.8797.
Stack: MLR +0.0044 (p 0.011, > std 0.0034) **pass**; FNN +0.0061 (p 0.009, CI [+0.001, +0.012]) **pass** → TTA CNN used from here on.

## Step 2 — C1 XGB third learner
`xgb_tune.py` 12-config grid (depth {4,6,8} × lr {0.05,0.1} × cw {0,1}), 4 at a time, ~4 min total. **Deviation:** the plan said
~20 configs; its listed factors give 12. Selected by Stack-3 **MLR** B score (deterministic, instant): `d6_lr05_cw1`
(Stack-3 0.9105; also the best standalone, 0.8704). Class weights help XGB, unlike the neural nets.
- Stack-3 vs Stack-2+TTA: MLR +0.0031 (p 0.20, < std 0.0045) **fail**; FNN +0.0050 (p 0.019, > std 0.0041) **pass**.
- Control B (2nd MFE seed as 3rd learner): Stack-3 − ctlB = +0.0011 (MLR), +0.0023 (FNN), both n.s. → "diversity helps" not supported.
- Control A (XGB replaces MFE-FNN): MLR 0.9123 (best MLR on B), FNN 0.9017 ± 0.0007.
- A first `ctlA_fnn` run got `seeds=1` by mistake (shared argument string); rerun with 5 seeds, the 1-seed record is superseded.

## Step 5 — freeze on B (inclusion rule applied per stacker)
MLR (headline): MFE-FNN + CNN-TTA. FNN: MFE-FNN + CNN-TTA + XGB(d6_lr05_cw1). Both B mode, cw 0, MLR α 0.1, FNN h64.

## Step 6 — single test evaluation (all compared configs, `final=1`)
MLR: S2 0.9005, **S2tta 0.9051 ± 0.0069 (final)**, S3 0.9149, ctlA 0.9160, ctlB 0.9059.
FNN: S2 0.9004, S2tta 0.9067, **S3 0.9070 ± 0.0037 (final)**, ctlA 0.9130, ctlB 0.9029.
The B rule rejected Stack-3 for MLR, which scores higher on test (+0.0098, p 0.057, CI includes 0; Near-full 9 wafers
carry most of it). Reported, not switched after seeing test.

## Step 3 — C2 calibration + reject (`calibrate.py`, thresholds chosen on B)
- T ≈ 0.136 (MLR), ≈ 1.03 (FNN). Test ECE: MLR 0.0044 (clipped ridge) → 0.0074 (worse: negative result); FNN 0.0056 → 0.0047.
- Fixed during the run: macro-F1 on accepted wafers averaged absent classes as 0; "before" for MLR was a softmax of ridge
  scores (ECE 0.73, meaningless) → now ridge outputs clipped to [0, 1] and renormalised.
- MLR final: 95% coverage → acc 0.9961, macro-F1 0.9791; 90% → 0.9993 / 0.9951. FNN B thresholds do not transfer below
  ~86% test coverage (confidence scale shifts between the 2-fold and final stacker fits).

## Step 4 — C3 explainability
`mlr_weights.py` (handcrafted learner carries Random/Near-full; CNN the spatial classes; XGB takes Near-full in Stack-3),
`gradcam.py` (block4_conv3), `shap_xgb.py` (exact TreeSHAP; refit matches saved XGB probs exactly).

## Environment fixes made along the way
- tf_keras 2.17 + Python 3.12: `randint(1, 1e9)` TypeError after `set_random_seed` → patched in the venv (`setup_env.sh`).

---

# Stage 2b — best-results pipeline + missing paper baselines (2026-10-08, branch `stage2`)

User request: keep the best models, take the approach that gives the best results, document everything for the write-up
(`writeup_notes.md`). Selection stays on B only; test is evaluated once for the final candidates.

## Saved models (so the final pipeline can be re-run, not only its outputs)
- `mfe_tune.py ... save=1` → `models/mfe_cw0_5s[_s1..4].keras` + `mfe_cw0_5s_scale.npz`; re-run outputs equal the
  committed Stage 2 outputs within 2.4e-7 (B_val seeds identical). `xgb_tune.py` now always saves `models/xgb_<name>.json`;
  the `d6_lr05_cw1` refit is bit-identical (0.0 diff).

## Paper baselines missing from Stage 1
- **Stacking-DT** (`stack_tune.py meta=dt`, sklearn default `DecisionTreeClassifier`), paper protocol (fit on A, as the
  Stage 1 MLR), 3 MFE seeds: **test 0.8782 ± 0.0065 vs paper 0.8789 ± 0.0094** — matches. Fitted on B instead (Stage 2
  protocol): B-cv 0.8429 ± 0.0063, test 0.8621 ± 0.0119 (a tree needs more data than B's 26k wafers).
- **MultiNN** (`cnn_train.py --multinn`, architecture from `DMkelllog/wafermap_MultiNN`: VGG16 GAP 512 ⊕ 59 standardised
  features → dropout 0.2 → softmax; our CNN recipe otherwise): training on GPU (`logs/cnn_multinn.log`).

## CNN seed ensemble (fixes the single-CNN-seed limitation)
- Seeds 1–4 of the frozen recipe (`--imagenet --cw 0 --fp32 --aug --seed s`), 2 at a time + MultiNN on the GPU
  (~150 s/epoch with 2 jobs, ~240 s with 3). Detached queue `logs/cnn_queue_stage2b.sh` also runs TTA for each.

## XGB bagging — negative result
5 seeds with subsample = colsample = 0.8 (`xgb_d6_lr05_cw1_bag_s0..4`): standalone B_val 0.8726–0.8766, averaged
(`average_outputs.py xgb_bag5`) **0.8785** vs single XGB 0.8704. But inside the stacks (MLR, B-cv) it is slightly worse:
CNN-TTA + XGB 0.9123 → 0.9105 (−0.0018, CI [−0.007, +0.005]); Stack-3 0.9105 → 0.9091 (−0.0014, p 0.009).
→ keep the single XGB. (Same lesson as Stage 1 B3: a better standalone model is not automatically a better stack input.)
`average_outputs.py` prints test scores only with `test=1`, so selection runs cannot look at test.

## Training-size sweep (paper Table 7 other blocks, Fig. 3) — closing a reproduction gap
`nsweep.py` + `cnn_train.py --subset N --rep R`: per replicate a random N training wafers (`common.subset`), 80% fit /
20% early stop (paper 4.2), our frozen base-learner recipes, MLR / DT / FNN(18→10→9) stackers fitted on the fit part
(paper protocol), all scored on the 10,000 test wafers. Replicates: 10 for N = 500 and 5,000 (as the paper), 5 for
N = 50,000 (GPU budget). Paper's full Table 7 (all 6 models × 4 N) copied into `nsweep.py` from `journal.pdf` p. 7.
- MFE-FNN: all 25 replicates done on CPU (10 in parallel). CNN: detached queue `logs/cnn_queue_nsweep.sh`.
- Fix: `common.class_weights` crashed when a small subset misses a class (even with cw = 0) → cw = 0 now returns
  all-ones; balanced weights on full data verified unchanged (Stage 1/2 unaffected).
- **N = 500 / 5,000 results** (10 replicates each, test macro-F1, ours vs paper; `nsweep_results.md`):
  N=500: MFE 0.4621 ± 0.0599 (0.5558), CNN 0.4899 ± 0.0875 (0.4937), DT 0.2680 ± 0.1399 (0.5179), FNN 0.4352 ± 0.0318 (0.5085),
  **MLR 0.5140 ± 0.0779 (0.5872)**. N=5,000: MFE 0.7286 ± 0.0233 (0.6983), CNN 0.7880 ± 0.0411 (0.6954),
  DT 0.7394 ± 0.0414 (0.7217), FNN 0.7977 ± 0.0328 (0.7350), **MLR 0.8037 ± 0.0341 (0.7599)**.
  → paper's claim (MLR stack > both base learners) holds at both sizes. At 5,000 we beat the paper everywhere (our CNN
  recipe — augmentation, no class weights — matters most when data is scarce: +0.09). At 500 we are below the paper on
  MFE and the stackers, with large stds on both sides; 400 fit wafers often miss rare classes entirely (rep 0: no Donut,
  no Near-full). Stacking-DT collapses at 500: the base learners are near-perfect in-sample (99.8% / 98.8% acc), so the
  tree learns thresholds on near-one-hot probabilities that do not transfer to test (same in-sample overconfidence
  mechanism as Stage 1 finding 1, amplified at small N).

## CNN seed ensemble — first results
Seeds 1–2 (TF 2.17, Linux): B_val plain 0.8850 / 0.8974, TTA 0.9053 / 0.9143 vs seed 0 (TF 2.10, laptop) 0.8723 / 0.8797.
Best val loss is practically identical (0.0540 / 0.0538 / 0.0530; best epochs 11 / 12 / 21), so seed 0 is not broken:
macro-F1 on rare classes swings strongly between seeds at equal loss → the single-seed CNN was a real limitation.
3-seed TTA ensemble `cnn_ens3_tta` (B_val 0.9145) in MLR stacks, B-cv: MFE+CNN 0.9074 → **0.9165**; MFE+CNN+XGB
0.9105 → **0.9185**; CNN+XGB 0.9123 → **0.9194**. Final selection waits for seeds 3–4 (5-seed ensemble).

## CNN seeds 3–4, 5-seed ensemble, final selection (B) and single test evaluation
- Seeds 3 / 4: B_val plain 0.8858 / 0.8802, TTA 0.8987 / 0.8933. All 5 seeds TTA: B_val 0.8983 ± 0.0116.
  `cnn_ens5_tta` (mean of 5): **B_val 0.9221**.
- Candidates on B (mode B, cw 0): MLR — MFE+CNN×5 0.9165 ± 0.0057, MFE+CNN×5+XGB 0.9182 ± 0.0039, **CNN×5+XGB 0.9223**;
  FNN — 0.9089, 0.9145, 0.9097. CNN×5+XGB vs CNN×1+XGB (MLR): +0.0100, CI [+0.003, +0.018].
  With the strong CNN the MFE-FNN no longer helps (MFE+CNN×5+XGB is 0.0062 below CNN×5 alone on B_val).
  → **Final (chosen on B): CNN×5 (TTA) + XGB(d6_lr05_cw1) → MLR (α 0.1, fitted on B).**
- Test (once): final **0.9134** (acc 0.9821); MFE+CNN×5+XGB MLR 0.9125 ± 0.0015; vs Stage 1 headline +0.0168
  (CI [+0.0004, +0.0399]); vs Stage 2 MLR final +0.0083 (CI incl. 0); vs CNN×1+XGB −0.0025 (CI [−0.014, +0.009]: noise).
  CNN seeds alone on test: 0.8924 ± 0.0116; ensemble 0.9126. MultiNN: B_val 0.8994, test 0.8880 (paper 0.8455).
- Calibration / reject (`calibrate.py e5_ctlA_mlr`): T 0.133; ECE 0.0041 (clipped ridge) → 0.0070; 95% target →
  95.2% auto-classified at 99.72% accuracy, macro-F1 0.9861; 90% → 99.96%.
- `predict.py` (+ `models/final_pipeline.npz` via `predict.py export=1`) rebuilds the final test predictions from the saved
  models only: max |score diff| 7.4e-5, identical classes. GPU OOM at inference batch 512 while 3 sweep runs held the GPU →
  batch 128.
