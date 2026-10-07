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
