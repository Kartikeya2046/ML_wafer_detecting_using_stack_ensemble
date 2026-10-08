# Stage 2 Results — Our Extensions (stage2_plan.md)

Built by `python extensions_results.py` from saved outputs and logs. Frozen Stage 1 starting point: `results_comparison.md`.
All choices were made on split B (2-fold cross-fitted stacker predictions pooled over B = 32,590 wafers); the 10,000 test wafers were evaluated once, after the B decisions (step 5) were fixed.
Statistics: mean ± std over runs (5 MFE-FNN seeds; FNN-stacker seed 0), paired t-test across runs, paired bootstrap 95% CI over wafers (2,000 resamples) of the macro-F1 difference. **Inclusion rule:** an extension joins the pipeline if it improves the B score by more than the baseline seed std.

## 1. Summary

| Configuration | Base learners | stacker | B-cv macro-F1 | n | **Test macro-F1** | Test F1 micro (acc.) |
|---|---|---|---|---|---|---|
| Stack-2 (Stage 2 baseline, B mode) | MFE-FNN + CNN | MLR | 0.9030 ± 0.0034 | 5 | **0.9005 ± 0.0076** | 0.9802 |
| Stack-2 + TTA (C4) ← **final (B rule)** | MFE-FNN + CNN-TTA | MLR | 0.9074 ± 0.0045 | 5 | **0.9051 ± 0.0069** | 0.9812 |
| Stack-3 (C1) | MFE-FNN + CNN-TTA + XGB | MLR | 0.9105 ± 0.0030 | 5 | **0.9149 ± 0.0025** | 0.9818 |
| Control A: XGB replaces MFE-FNN | CNN-TTA + XGB | MLR | 0.9123 ± 0.0000 | 1 | **0.9160 ± 0.0000** | 0.9816 |
| Control B: 2nd MFE-FNN seed as 3rd learner | MFE-FNN ×2 + CNN-TTA | MLR | 0.9094 ± 0.0040 | 5 | **0.9059 ± 0.0051** | 0.9813 |
| Stack-2 (Stage 2 baseline, B mode) | MFE-FNN + CNN | FNN | 0.8971 ± 0.0034 | 5 | **0.9004 ± 0.0048** | 0.9805 |
| Stack-2 + TTA (C4) | MFE-FNN + CNN-TTA | FNN | 0.9032 ± 0.0041 | 5 | **0.9067 ± 0.0062** | 0.9814 |
| Stack-3 (C1) ← **final (B rule)** | MFE-FNN + CNN-TTA + XGB | FNN | 0.9082 ± 0.0048 | 5 | **0.9070 ± 0.0037** | 0.9812 |
| Control A: XGB replaces MFE-FNN | CNN-TTA + XGB | FNN | 0.9017 ± 0.0007 | 5 | **0.9130 ± 0.0022** | 0.9812 |
| Control B: 2nd MFE-FNN seed as 3rd learner | MFE-FNN ×2 + CNN-TTA | FNN | 0.9059 ± 0.0034 | 5 | **0.9029 ± 0.0057** | 0.9812 |
| Stage 1 frozen headline (A mode) | MFE-FNN + CNN | MLR | — | — | 0.8967 ± 0.0060 | 0.9805 |
| Stage 1 tuned Stacking-FNN | MFE-FNN + CNN | FNN | — | — | 0.9041 ± 0.0028 | 0.9811 |
| Paper (Table 7, N = 162,946) | | MLR / FNN | — | — | 0.8949 ± 0.0121 / 0.8991 ± 0.0096 | 0.9801 / 0.9808 |

**Final Stage 2 pipelines, chosen on B before the test set was opened:** MLR (headline) = MFE-FNN + CNN-TTA (test 0.9051 ± 0.0069); FNN = MFE-FNN + CNN-TTA + XGB (test 0.9070 ± 0.0037).

**Findings in one line each:**
1. **C4 TTA helps** both stackers on B (> seed std, CI excludes 0 for FNN) and on test (+0.005 MLR, +0.006 FNN). Cheapest win: 8 forward passes per wafer, no retraining. The CNN alone barely moves on test (0.8777 → 0.8781); the gain shows up in the stack, where smoother CNN probabilities combine better.
2. **C1 XGB as a third learner: mixed.** On B it passes the rule for the FNN stacker but not for MLR. On test the MLR Stack-3 scores 0.9149 (+0.0098 vs the chosen MLR pipeline, p = 0.057, CI includes 0) - the rule, applied on B, rejected it; we report that rather than switch after seeing test. Most of the test gap is the 9 Near-full wafers.
3. **"Diversity helps" is not supported**: Stack-3 never beats Control B (a second MFE-FNN seed in the third slot) by a significant margin on B (on test, MLR: +0.009, p = 0.028, but the bootstrap CI includes 0). **XGB is simply a better handcrafted-feature model** (standalone B_val 0.870 vs 0.853 for the MFE-FNN); Control A (CNN-TTA + XGB, no FNN at all) is the best MLR stack on both B and test - a negative result for the paper's MFE-FNN, not part of the pre-registered pipeline.
4. **C2 reject option works**: routing the 5% least confident wafers to a human lifts accuracy 98.1% → 99.6% and macro-F1 0.905 → 0.979 (MLR). **Temperature scaling does not improve calibration for MLR** (its clipped ridge outputs are already well calibrated) and barely for FNN - a negative result.
5. **C3 weights** reproduce the paper's story: the handcrafted learner carries Random and Near-full, the CNN carries the spatial patterns (Edge-Ring, Scratch, Edge-Loc).

## 2. C4 — Test-time augmentation (8 symmetries: 4 rotations × flip)

| CNN alone | B_val macro-F1 | Test macro-F1 | Test acc. |
|---|---|---|---|
| plain (Stage 1 frozen) | 0.8723 | 0.8777 | 0.9790 |
| TTA | 0.8797 | 0.8781 | 0.9796 |

CNN alone, TTA − plain on B_val: +0.0075, bootstrap CI [-0.0130, +0.0274] (single CNN seed, no t-test).

| Stack, TTA − plain | set | diff | paired t-test p | bootstrap 95% CI | baseline seed std | rule |
|---|---|---|---|---|---|---|
| MLR | B | +0.0044 | 0.011 | [-0.0004, +0.0095] | 0.0034 | pass |
| MLR | test | +0.0045 | 0.122 | [-0.0053, +0.0146] | 0.0076 | (test: report only) |
| FNN | B | +0.0061 | 0.009 | [+0.0011, +0.0121] | 0.0034 | pass |
| FNN | test | +0.0063 | 0.023 | [-0.0048, +0.0187] | 0.0048 | (test: report only) |

## 3. C1 — XGBoost on the 59 handcrafted features as a third base learner

XGBoost fitted on A, early-stopped (number of trees) on B_fit, `tree_method=hist`. Grid: depth {4, 6, 8} × learning rate {0.05, 0.1} × class weights {none, balanced}; **selected by Stack-3 MLR score on B** (plan: by stack score, not standalone; the MLR stacker is deterministic and instant, so the grid was scored with it).

| XGB config | trees | standalone B_val macro-F1 | Stack-3 MLR B-cv |
|---|---|---|---|
| d6_lr05_cw1 ← selected | 1260 | 0.8704 | 0.9105 ± 0.0030 |
| d6_lr1_cw1 | 589 | 0.8667 | 0.9097 ± 0.0027 |
| d8_lr1_cw1 | 314 | 0.8645 | 0.9095 ± 0.0024 |
| d4_lr05_cw0 | 992 | 0.8665 | 0.9093 ± 0.0028 |
| d4_lr1_cw1 | 1564 | 0.8695 | 0.9091 ± 0.0021 |
| d4_lr1_cw0 | 473 | 0.8607 | 0.9088 ± 0.0036 |
| d8_lr05_cw1 | 688 | 0.8674 | 0.9088 ± 0.0019 |
| d6_lr1_cw0 | 209 | 0.8596 | 0.9084 ± 0.0034 |
| d6_lr05_cw0 | 442 | 0.8602 | 0.9081 ± 0.0035 |
| d4_lr05_cw1 | 3187 | 0.8675 | 0.9074 ± 0.0024 |
| d8_lr1_cw0 | 125 | 0.8536 | 0.9073 ± 0.0016 |
| d8_lr05_cw0 | 259 | 0.8532 | 0.9072 ± 0.0013 |
| d6_lr05_cw1_bag_s0 | 1155 | 0.8745 | — |
| d6_lr05_cw1_bag_s3 | 1163 | 0.8766 | — |
| d6_lr05_cw1_bag_s1 | 1218 | 0.8726 | — |
| d6_lr05_cw1_bag_s2 | 1149 | 0.8751 | — |
| d6_lr05_cw1_bag_s4 | 1151 | 0.8752 | — |

Selecting the best of 12 configs on the same B score inflates its B number slightly (winner's curse); the test evaluation is unaffected.

| Comparison | stacker | set | diff | paired t-test p | bootstrap 95% CI |
|---|---|---|---|---|---|
| Stack-3 − Stack-2 (both with TTA) | MLR | B | +0.0031 | 0.199 | [-0.0037, +0.0108] |
| Stack-3 − Stack-2 (both with TTA) | MLR | test | +0.0098 | 0.057 | [-0.0040, +0.0303] |
| Stack-3 − Stack-2 (both with TTA) | FNN | B | +0.0050 | 0.019 | [-0.0007, +0.0117] |
| Stack-3 − Stack-2 (both with TTA) | FNN | test | +0.0004 | 0.923 | [-0.0064, +0.0080] |
| Stack-3 − Control B (is it diversity?) | MLR | B | +0.0011 | 0.713 | [-0.0056, +0.0087] |
| Stack-3 − Control B (is it diversity?) | MLR | test | +0.0090 | 0.028 | [-0.0051, +0.0322] |
| Stack-3 − Control B (is it diversity?) | FNN | B | +0.0023 | 0.354 | [-0.0031, +0.0089] |
| Stack-3 − Control B (is it diversity?) | FNN | test | +0.0041 | 0.326 | [-0.0053, +0.0163] |
| Control B − Stack-2 | MLR | B | +0.0020 | 0.426 | [-0.0005, +0.0045] |
| Control B − Stack-2 | MLR | test | +0.0008 | 0.868 | [-0.0051, +0.0069] |
| Control B − Stack-2 | FNN | B | +0.0026 | 0.085 | [-0.0003, +0.0063] |
| Control B − Stack-2 | FNN | test | -0.0038 | 0.465 | [-0.0099, +0.0005] |
| Control A − Stack-2 | MLR | B | +0.0049 | — | [-0.0052, +0.0160] |
| Control A − Stack-2 | MLR | test | +0.0109 | — | [-0.0027, +0.0307] |
| Control A − Stack-2 | FNN | B | -0.0015 | — | [-0.0086, +0.0073] |
| Control A − Stack-2 | FNN | test | +0.0064 | — | [-0.0041, +0.0191] |
| Stack-3 − Control A | MLR | B | -0.0018 | — | [-0.0087, +0.0057] |

Control A has a single run for MLR (XGB and the CNN are single deterministic models), so no t-test there.

## 4. C2 — Calibration and reject option

Temperature T fitted per run by log-loss on the stacker's pooled 2-fold B predictions, applied to test unchanged. ECE: 15 equal-width bins. "Before" = uncalibrated probabilities (FNN softmax; MLR ridge outputs clipped to [0, 1], renormalised). Reject thresholds are chosen on B for each target coverage and applied to test, so the realised test coverage is close to (not exactly) the target. Macro-F1 on accepted wafers averages over the classes present.

| Stack | T | test ECE before | test ECE after | B ECE before | B ECE after |
|---|---|---|---|---|---|
| S2tta_fnn | 1.035 | 0.0056 ± 0.0002 | 0.0047 ± 0.0002 | 0.0044 ± 0.0002 | 0.0036 ± 0.0003 |
| S2tta_mlr | 0.136 | 0.0044 ± 0.0004 | 0.0074 ± 0.0003 | 0.0055 ± 0.0003 | 0.0052 ± 0.0001 |
| S3_fnn | 1.029 | 0.0062 ± 0.0003 | 0.0054 ± 0.0003 | 0.0046 ± 0.0002 | 0.0039 ± 0.0003 |
| S3_mlr | 0.136 | 0.0039 ± 0.0003 | 0.0074 ± 0.0001 | 0.0061 ± 0.0001 | 0.0052 ± 0.0002 |
| ctlA_mlr | 0.136 | 0.0032 ± 0.0000 | 0.0076 ± 0.0000 | 0.0052 ± 0.0000 | 0.0051 ± 0.0000 |
| e5_ctlA_mlr | 0.133 | 0.0041 ± 0.0000 | 0.0070 ± 0.0000 | 0.0059 ± 0.0000 | 0.0048 ± 0.0000 |

**Reject option — S2tta_mlr (final MLR pipeline), test set:**

| target coverage | test coverage | accuracy on accepted | macro-F1 on accepted |
|---|---|---|---|
| 100% | 1.000 | 0.9812 | 0.9051 |
| 98% | 0.982 | 0.9893 | 0.9425 |
| 95% | 0.955 | 0.9961 | 0.9791 |
| 90% | 0.901 | 0.9993 | 0.9951 |
| 85% | 0.851 | 0.9998 | 0.9977 |
| 80% | 0.804 | 0.9998 | 0.9984 |
| B accuracy ≥ 99.0% | 0.985 | 0.9884 | 0.9370 |
| B accuracy ≥ 99.5% | 0.969 | 0.9934 | 0.9612 |

**Reject option — S3_fnn (final FNN pipeline), test set:**

| target coverage | test coverage | accuracy on accepted | macro-F1 on accepted |
|---|---|---|---|
| 100% | 1.000 | 0.9812 | 0.9070 |
| 98% | 0.979 | 0.9908 | 0.9503 |
| 95% | 0.941 | 0.9973 | 0.9819 |
| 90% | 0.896 | 0.9992 | 0.8268 |
| 85% | 0.863 | 0.9995 | 0.7180 |
| 80% | 0.862 | 0.9995 | 0.7180 |
| B accuracy ≥ 99.0% | 0.982 | 0.9898 | 0.9454 |
| B accuracy ≥ 99.5% | 0.956 | 0.9954 | 0.9725 |

FNN test coverage stalls near 86% for targets ≤ 85%: the thresholds come from the 2-fold B stackers, but test is predicted by the final stacker fitted on B, and the FNN's confidence scale shifts between fits (most None wafers sit on a plateau near 0.999; the B threshold for 80% coverage, 0.9957, admits 86% of test). The MLR thresholds transfer cleanly (test coverage within 0.5 pt of target) - one more reason the MLR is the better basis for the reject rule.

**Which wafers go to the human** (% of each true class rejected, MLR final pipeline):

| target coverage | Center | Donut | Edge-Loc | Edge-Ring | Loc | Random | Scratch | Near-full | None |
|---|---|---|---|---|---|---|---|---|---|
| 95% | 10% | 29% | 32% | 4% | 42% | 22% | 32% | 27% | 2% |
| 90% | 20% | 41% | 57% | 7% | 58% | 38% | 45% | 40% | 6% |

The rejected wafers are mostly the hard local patterns (Loc, Edge-Loc, Scratch) and rare Donut/Near-full; the "None" class (85% of wafers) is almost never rejected, so the human workload stays small. No class is rejected wholesale, so per-class thresholds (plan follow-up) were not needed.

## 5. C3 — Explainability

**MLR weights** (diagonal coefficient: the weight each base learner's vote for class c gets in the score for class c; mean ± std over runs; the paper's Fig. 5 for our stacks). Figures: `figures/mlr_weights_*.png`.

| S2tta_mlr | mfe_cw0_5s | cnn_imnet_cw0_aug_tta |
|---|---|---|
| Center | 0.26 ± 0.02 | 0.76 ± 0.02 |
| Donut | 0.45 ± 0.07 | 0.60 ± 0.08 |
| Edge-Loc | 0.21 ± 0.01 | 0.82 ± 0.01 |
| Edge-Ring | 0.06 ± 0.01 | 0.95 ± 0.01 |
| Loc | 0.30 ± 0.01 | 0.75 ± 0.01 |
| Random | 0.65 ± 0.03 | 0.42 ± 0.03 |
| Scratch | 0.19 ± 0.01 | 0.93 ± 0.01 |
| Near-full | 0.90 ± 0.21 | 0.11 ± 0.21 |
| None | 0.17 ± 0.01 | 0.83 ± 0.01 |

| S3_mlr | mfe_cw0_5s | cnn_imnet_cw0_aug_tta | xgb_d6_lr05_cw1 |
|---|---|---|---|
| Center | 0.14 ± 0.02 | 0.71 ± 0.01 | 0.16 ± 0.01 |
| Donut | 0.30 ± 0.11 | 0.46 ± 0.04 | 0.29 ± 0.07 |
| Edge-Loc | 0.14 ± 0.02 | 0.79 ± 0.01 | 0.10 ± 0.01 |
| Edge-Ring | 0.04 ± 0.02 | 0.93 ± 0.01 | 0.03 ± 0.01 |
| Loc | 0.23 ± 0.02 | 0.71 ± 0.01 | 0.11 ± 0.01 |
| Random | 0.43 ± 0.07 | 0.34 ± 0.04 | 0.30 ± 0.04 |
| Scratch | 0.09 ± 0.02 | 0.89 ± 0.00 | 0.15 ± 0.01 |
| Near-full | 0.28 ± 0.29 | 0.00 ± 0.18 | 0.71 ± 0.12 |
| None | 0.06 ± 0.02 | 0.79 ± 0.01 | 0.15 ± 0.01 |

Handcrafted features dominate **Random** and **Near-full** (global density patterns); the CNN dominates the spatial patterns. In Stack-3, XGB takes over Near-full from the MFE-FNN (0.71 vs 0.28), which is where its test gain comes from.

**SHAP (XGBoost, exact TreeSHAP)** — top 3 handcrafted features per class by mean |SHAP| on test (`figures/shap_xgb_*.png`):

| Class | top features |
|---|---|
| Center | density_9, radon_mean_11, radon_mean_10 |
| Donut | density_9, radon_mean_7, radon_mean_14 |
| Edge-Loc | radon_std_2, geom_solidity, radon_std_20 |
| Edge-Ring | radon_mean_20, geom_solidity, radon_mean_2 |
| Loc | radon_mean_2, radon_std_14, geom_eccentricity |
| Random | geom_solidity, radon_mean_5, density_12 |
| Scratch | geom_eccentricity, geom_solidity, radon_std_11 |
| Near-full | geom_area, geom_minor_axis, radon_mean_7 |
| None | geom_perimeter, geom_area, density_3 |

**Grad-CAM** (`figures/gradcam_imnet_cw0_aug.png`, layer block4_conv3, 8×8): on correctly classified Center, Edge-Loc, Random and Near-full wafers the heat sits on the defect; several misclassified wafers light up on the wafer edge or a corner instead. Confident correct Edge-Ring/Scratch maps are nearly blank because the softmax is saturated (p = 1.00 gives vanishing gradients) - a known Grad-CAM limitation.

## 6. Per-class test F1 — final pipelines vs Stage 1

| Class | test n | Stage 1 MLR (A) | Stage 2 MLR: Stack-2 + TTA | Stage 2 MLR: Stack-3 | Stage 1 FNN | Stage 2 FNN: Stack-3 |
|---|---|---|---|---|---|---|
| Center | 248 | 0.9553 | 0.9570 | 0.9556 | 0.9444 | 0.9529 |
| Donut | 32 | 0.8915 | 0.9453 | 0.9098 | 0.8861 | 0.8849 |
| Edge-Loc | 300 | 0.8434 | 0.8463 | 0.8557 | 0.8540 | 0.8535 |
| Edge-Ring | 560 | 0.9866 | 0.9932 | 0.9930 | 0.9874 | 0.9904 |
| Loc | 208 | 0.7742 | 0.7640 | 0.7689 | 0.7791 | 0.7809 |
| Random | 50 | 0.8821 | 0.8830 | 0.9144 | 0.8800 | 0.8960 |
| Scratch | 69 | 0.8254 | 0.8379 | 0.8439 | 0.8488 | 0.8238 |
| Near-full | 9 | 0.9191 | 0.9265 | 1.0000 | 0.9647 | 0.9882 |
| None | 8524 | 0.9922 | 0.9924 | 0.9927 | 0.9927 | 0.9926 |
| **Macro** | | **0.8967** | **0.9051** | **0.9149** | **0.9041** | **0.9070** |

## 7. Stage 2b — best-results pipeline (CNN seed ensemble) and the missing paper baselines

After Stage 2 the test set had been seen once; Stage 2b still selects on B only, but this is a second look at test.

**CNN seeds** (same recipe, seeds 0–4; seed 0 = the Stage 1 frozen CNN, trained on the laptop / TF 2.10):

| CNN | B_val plain | B_val TTA | Test TTA |
|---|---|---|---|
| seed 0 | 0.8723 | 0.8797 | 0.8781 |
| seed 1 | 0.8850 | 0.9053 | 0.8879 |
| seed 2 | 0.8974 | 0.9143 | 0.8948 |
| seed 3 | 0.8858 | 0.8987 | 0.8883 |
| seed 4 | 0.8802 | 0.8933 | 0.9129 |
| mean ± std | | 0.8983 ± 0.0116 | 0.8924 ± 0.0116 |
| **5-seed ensemble** | | **0.9221** | **0.9126** |

Best validation losses are nearly equal across seeds (0.053–0.054), yet macro-F1 varies by ±0.012: the rare classes swing between seeds. Averaging the seeds removes most of that variance.

| Stack (CNN = 5-seed TTA ensemble) | stacker | B-cv macro-F1 | n | Test macro-F1 |
|---|---|---|---|---|
| MFE-FNN + CNN×5 | MLR | 0.9165 ± 0.0057 | 5 | 0.9075 ± 0.0054 |
| MFE-FNN + CNN×5 | FNN | 0.9089 ± 0.0036 | 5 | 0.9095 ± 0.0029 |
| MFE-FNN + CNN×5 + XGB | MLR | 0.9182 ± 0.0039 | 5 | 0.9125 ± 0.0015 |
| MFE-FNN + CNN×5 + XGB | FNN | 0.9145 ± 0.0041 | 5 | 0.9065 ± 0.0034 |
| CNN×5 + XGB ← **final (best on B)** | MLR | 0.9223 ± 0.0000 | 1 | 0.9134 ± 0.0000 |
| CNN×5 + XGB | FNN | 0.9097 ± 0.0022 | 5 | 0.9092 ± 0.0023 |

| Comparison | set | diff | paired t-test p | bootstrap 95% CI |
|---|---|---|---|---|
| CNN×5 + XGB − CNN×1 + XGB (MLR) | B | +0.0100 | — | [+0.0030, +0.0178] |
| MFE + CNN×5 − MFE + CNN×1 (MLR) | B | +0.0090 | 0.050 | [+0.0023, +0.0157] |
| final stack − CNN×5 alone (B_val) | B | +0.0027 | — | [-0.0120, +0.0178] |
| MFE + CNN×5 + XGB − CNN×5 alone (B_val) | B | -0.0062 | — | [-0.0175, +0.0036] |
| without MFE-FNN − with MFE-FNN (MLR) | B | +0.0041 | — | [-0.0026, +0.0117] |
| final − Stage 1 headline | test | +0.0168 | — | [+0.0004, +0.0399] |
| final − Stage 2 MLR final | test | +0.0083 | — | [-0.0079, +0.0292] |
| final − CNN×1 + XGB | test | -0.0025 | — | [-0.0143, +0.0091] |

**Final pipeline: CNN×5 (TTA) + XGB → MLR — test macro-F1 0.9134, accuracy 0.9821.** Reject option at 95% target: 95.2% auto-classified at accuracy 0.9972, macro-F1 0.9861. Runnable from saved models: `predict.py` (reproduces the test scores within 7e-5, identical classes).

Findings: (1) the CNN seed ensemble is the largest single gain of the project on B (+0.009 to +0.010, CI excludes 0); (2) once the CNN is this strong, **the paper's MFE-FNN no longer helps** - adding it is below the CNN ensemble alone on B_val; the handcrafted features still help through XGB, slightly; (3) on test the single-CNN version of the final stack (0.9160) and the ensemble (0.9134) are within noise (CI [−0.014, +0.009]); the ensemble was chosen on B.

**Paper baselines added:** Stacking-DT (paper protocol, fitted on A): test 0.8782 ± 0.0065 (paper 0.8789 ± 0.0094). MultiNN (single run, our CNN recipe + the 59 features): B_val 0.8994, test 0.8880 (paper 0.8455 ± 0.0170). Training-size sweep: `nsweep_results.md`.

## 8. Limitations

- Stages 1–2: one CNN seed (Stage 2b adds 4 more: seed std ±0.012 macro-F1). The wafer bootstrap covers test-sampling noise.
- Near-full has 9 test wafers (30 in B): one wafer ≈ 0.01 macro-F1, and several test differences above are mostly Near-full.
- One fixed stratified split (the paper averages 10 random splits).
- Stage 2 stackers are fitted on B only (honest base outputs, 26k wafers) while the Stage 1 headline MLR was fitted on A; the like-for-like Stage 2 baseline is Stack-2 in B mode.
- FNN stacker runs use one stacker seed per MFE seed (n = 5), so stacker-seed variance is only partly included.
- Environment: Stage 2 ran on Linux, TF 2.17 + tf_keras (Stage 1: Windows, TF 2.10). Verified equivalent: the frozen CNN re-predicts test within 1.2e-4, and MFE-FNN seeds 0–2 reproduce Stage 1 exactly (B_val 0.8442 / 0.8481 / 0.8616).
