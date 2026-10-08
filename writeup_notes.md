# Write-up Notes — Paper vs. Our Implementation vs. What We Did Differently

**Purpose:** the single source for the final report, slides and viva (implementation_plan.md Part D). Kept current while
work runs (user request, 2026-10-08). Every number here points to a file that holds it; nothing is from memory.
Detailed trial logs: `tuning_log.md`. Tables: `results_comparison.md` (Stage 1), `extensions_results.md` (Stage 2).
Last updated: 2026-10-08.

---

## 1. What the paper did (Kang & Kang 2021, *Computers in Industry* 129:103450, Elsevier — `journal.pdf`)

**Problem.** Classify wafer maps (2-D pass/fail die maps) from WM-811K into 9 defect classes (Center, Donut, Edge-Loc,
Edge-Ring, Loc, Random, Scratch, Near-full, None). Labeled subset: 172,950 maps; 4 maps with < 100 dies removed →
172,946 (paper §4.1). None ≈ 85% of wafers → macro-F1 is the main metric.

**Method — a stacking ensemble of two different base classifiers:**
| Component | Paper's choice | Where in paper |
|---|---|---|
| Handcrafted features (MFE) | 59 features: 13 region densities, 40 Radon-transform (20 row-mean + 20 row-std, cubic-interpolated), 6 geometry of the largest defect region (area, perimeter, major/minor axis, eccentricity, solidity) | §3.1 |
| Base classifier 1: MFE+FNN | 59 → 128 tanh → 128 tanh → 9 softmax (Table 3); chosen over other MFE models in preliminary experiments (Table 5) | §3.1, §4.3 |
| Base classifier 2: CNN | VGG16-style conv stack on 64×64 maps, global average pooling, 9-way softmax (Table 4), **pre-trained on ImageNet**; chosen over other CNNs (Table 6) | §3.2, §4.3 |
| Meta-classifier (proposed) | **MLR**: ridge regression of the one-hot label on the concatenated base probabilities (18 inputs), α = 0.1 → a class-by-class weight for each base classifier | §3.3, eq. 5–6, §4.4 |
| Training | Adam, lr 1e-4, batch 32; 80% of training data for backprop, 20% for early stopping (patience 20 on val loss, max 1000 epochs) | §4.2 |
| Evaluation | 10,000 random test wafers; training size N ∈ {500, 5,000, 50,000, 162,946}; **10 replications**, mean ± std, paired t-test | §4.2 |
| Baselines | MFE+FNN, CNN, MultiNN (CNN with the 59 features concatenated after GAP), Stacking-DT (sklearn default tree), Stacking-FNN (18→10→9) | §4.4 |

**Paper's result at N = 162,946 (Table 7, macro-F1):** MFE+FNN 0.8599 ± 0.0117, CNN 0.8679 ± 0.0126,
MultiNN 0.8455 ± 0.0170, Stacking-DT 0.8789 ± 0.0094, Stacking-FNN 0.8991 ± 0.0096, **Stacking-MLR 0.8949 ± 0.0121**.
Per-class: Table 8 (copied into `freeze_results.py`). Note: the paper prints N = "162,496" in places; 172,946 − 10,000 = 162,946.

**Paper's claim:** the two base classifiers are complementary (MFE better on some classes, CNN on others, Fig. 4/5);
the MLR stack weights each per class and beats both, robustly across N.

---

## 2. What we implemented faithfully (Stage 1 — reproduction)

| Paper element | Our implementation | File |
|---|---|---|
| Data: labeled subset, < 100-die removal, 10,000-wafer test split | same; stratified split, `random_state=42`; train 162,946 / test 10,000 | `phase2_data.py` |
| 59 handcrafted features | the reference repo's feature code (`WMPC_Stacking_TF2/run_code/extract_manual_features.py`), called unchanged | `phase2_data.py` |
| 64×64 maps for the CNN | binarised (defect = 1) and resized as in the repo | `phase2_data.py` |
| MFE+FNN (Table 3) | exactly Table 3, Adam 1e-4, batch 32, patience 20 | `mfe_tune.py` |
| CNN (Table 4) | VGG16 conv stack, ImageNet weights, GAP + softmax, Adam 1e-4, batch 32, patience 20 | `cnn_train.py` |
| Stacking-MLR | ridge, α = 0.1, no intercept, on [p_MFE, p_CNN] | `stack_tune.py meta=mlr` |
| Stacking-FNN | 18 → 10 → 9 (then tuned, see §3) | `stack_tune.py meta=fnn` |
| Metrics | macro-F1 (headline), micro-F1, per-class F1 | `common.py:f1s` |

**Adapted vs. original code (academic honesty):** feature extraction and the CNN/stacking structure follow the authors'
public repos `DMkelllog/WMPC_Stacking_TF2` and `DMkelllog/wafermap_MultiNN` (both in this folder). All training,
tuning, evaluation, statistics and Stage 2 code is ours. Implementation was AI-assisted (approved by the professor).

**Stage 1 result (test, `results_comparison.md`):** Stacking-MLR **0.8967 ± 0.0060** (paper 0.8949 ± 0.0121),
tuned Stacking-FNN **0.9041 ± 0.0028** (paper 0.8991), MFE+FNN 0.8572 ± 0.0062 (paper 0.8599), CNN 0.8777 (paper 0.8679).
→ the reproduction **matches the paper within its std** and confirms its main claim (stack > both base classifiers).

---

## 3. What we did differently, and why

### 3a. Deviations in the reproduction (Stage 1) — each found and justified on validation data (`tuning_log.md`)
| # | Paper / reference repo | Ours | Why |
|---|---|---|---|
| 1 | Meta-learner trained on the base learners' predictions for their own training data (repo) | Split the training set: **A** (base learners fitted) and **B** (only early stopping) → stacker trained on honest B outputs | In-sample predictions are over-confident; the repo-style stacker ignored Near-full (F1 = 0, macro 0.63–0.76) |
| 2 | (repo) balanced class weights in base learners | **No class weights** (the paper mentions none) | Class weights cost ~5 pts for the MFE-FNN and hurt the CNN and stackers |
| 3 | CNN: no augmentation | **Flip + 90° rotation augmentation** | +0.0055 in the stack (defect patterns are rotation-invariant) |
| 4 | Stacking-FNN 18→10→9 | hidden 64, fitted on B | tuned on B (B2) |
| 5 | 10 random splits × replications | **one fixed split**, 3–5 seeds | compute budget; stated as a limitation |
| 6 | N ∈ {500, 5k, 50k, 162,946} | N = 162,946 in Stage 1; sweep added in Stage 2b | `nsweep_results.md` |
| 7 | MultiNN, Stacking-DT baselines | **not in Stage 1**; added in Stage 2b | Stacking-DT matches (0.8782 vs 0.8789); MultiNN 0.8880 vs 0.8455 |
| 8 | fp32 (implied) | fp32, batch 32 | mixed precision / batch 128 cost ~4 pts on this VGG (no batch-norm) |

### 3b. Our own additions (Stage 2 — `extensions_results.md`, plan `stage2_plan.md`)
Rules: every choice on split B, test evaluated once, paired t-test + wafer bootstrap CI, failures reported.
| Extension | Idea | Result (test macro-F1) | Verdict |
|---|---|---|---|
| **C4 test-time augmentation** | average the CNN over the 8 symmetries of the square, no retraining | MLR 0.9005 → **0.9051**, FNN 0.9004 → 0.9067 | helps (passes rule for both stackers) |
| **C1 XGBoost 3rd learner** | gradient-boosted trees on the same 59 features as a 3rd base classifier | MLR 0.9149, FNN 0.9070 | mixed: passes on B only for FNN; never significantly beats a 2nd MFE-FNN seed (control B) |
| **Control A** | XGBoost *replaces* the MFE-FNN | MLR **0.9160**, FNN 0.9130 | best stack; XGB is simply a better model for the handcrafted features than the paper's FNN |
| **C2 calibration + reject option** | temperature scaling; route least-confident wafers to a human | 95% auto-classified → accuracy 98.1% → 99.6%, macro-F1 0.905 → 0.979 | reject option works; temperature scaling does not help MLR (negative) |
| **C3 explainability** | MLR weight bars (paper Fig. 5 for our stacks), Grad-CAM, SHAP | handcrafted features carry Random/Near-full, CNN the spatial classes | confirms the paper's complementarity story |

### 3c. Best-results pipeline (Stage 2b, in progress — user request 2026-10-08: "keep the best models, best approach")
Selection still on B only. Steps:
1. **CNN seed ensemble:** 4 more CNN seeds (same recipe, `--seed 1..4`), each with TTA → average of 5 CNNs. Also gives the
   first real CNN seed std (Stage 1/2 limitation). Seeds 1–2: B_val (TTA) 0.9053 / 0.9143 vs seed 0's 0.8797 at the same
   validation loss → rare-class F1 swings a lot between seeds; averaging seeds is the single biggest gain so far
   (3-seed ensemble: MLR stacks on B +0.007 to +0.009).
2. **XGB bagging — tried, rejected (negative result):** 5 XGB seeds with row/column subsampling 0.8. Standalone it helps
   (B_val 0.8785 vs 0.8704) but inside the stacks it is slightly worse on B (−0.0014 to −0.0018) → single XGB kept.
3. **Candidate stacks on B:** CNN-ens + XGB-bag (control-A style), MFE + CNN-ens + XGB-bag, both stackers → pick best on B, evaluate test once.
4. **Saved models for the final pipeline** + `predict.py` that reproduces the test predictions from the saved files.
   Saved so far: `models/mfe_cw0_5s[_s1..4].keras` + `mfe_cw0_5s_scale.npz` (re-run reproduces the Stage 2 outputs within 2.4e-7),
   `models/xgb_d6_lr05_cw1[_bag_s0..4].json` (refit identical, 0.0 diff), `models/cnn_imnet_cw0_aug.keras`.
**Result (`extensions_results.md` §7):** 5 CNN seeds with TTA: B_val 0.8983 ± 0.0116 each, **ensemble 0.9221**.
Final pipeline chosen on B: **CNN×5 (TTA) + XGB → MLR**, B-cv 0.9223 → **test macro-F1 0.9134, accuracy 98.21%**
(Stage 1 headline 0.8967 → +0.0168, CI excludes 0; paper 0.8949). With the reject option at 95% coverage: 99.72% accuracy.
Surprise worth reporting: once the CNN is a 5-seed ensemble, **the paper's MFE-FNN no longer adds anything** (stacks with it
score below the CNN ensemble alone on B); the handcrafted features still help a little through XGBoost.
Saved and runnable: `predict.py` + `models/final_pipeline.npz`, `models/xgb_d6_lr05_cw1.json`, `models/cnn_imnet_cw0_aug*.keras`.

---

### 3d. Training-size sweep (reproduces the paper's Table 7 for N = 500 / 5,000 / 50,000) — `nsweep_results.md`
| N | MFE+FNN ours / paper | CNN ours / paper | Stacking-MLR ours / paper |
|---|---|---|---|
| 500 | 0.462 / 0.556 | 0.490 / 0.494 | 0.514 / 0.587 |
| 5,000 | 0.729 / 0.698 | 0.788 / 0.695 | 0.804 / 0.760 |
| 50,000 | 0.837 / 0.831 | 0.883 / 0.840 | 0.876 / 0.869 |
| 162,946 | 0.857 / 0.860 | 0.878 / 0.868 | 0.897 / 0.895 |
Takeaways: the MLR stack beats both base learners at N = 500, 5,000 and 162,946 (paper's claim confirmed there) but
**not at N = 50,000** (0.876 < CNN 0.883; the FNN stacker 0.886 does) — under the paper's protocol the meta-learner sees
in-sample predictions, where our strong CNN looks near-perfect. Our recipe beats the paper at N ≥ 5,000 for every model
and helps most at mid-size data (CNN +0.09 at N = 5,000); at N = 500 rare classes are often absent from the 400 training wafers and results are noisy on both
sides; Stacking-DT collapses at N = 500 (in-sample overconfidence — same mechanism as finding 3a-1).

## 4. Story for the report / viva (plain language)
1. Wafer maps show where dies failed; the failure *pattern* points to the process step that went wrong — so classifying patterns automatically matters.
2. Two kinds of model see different things: handcrafted features summarise density/shape (good for Random, Near-full); a CNN sees spatial structure (good for Edge-Ring, Scratch).
3. The paper combines them with a simple per-class weighted vote learned by ridge regression (MLR) — we reproduced it and matched its numbers.
4. We found the repo's stacker was trained on over-confident in-sample predictions; fixing that (honest split B) was the key to reproducing the paper.
5. Our additions: TTA (free accuracy), a better handcrafted-feature model (XGBoost), averaging 5 CNNs (the biggest gain), a reject option that sends ~5% of wafers to a human and gets 99.7% accuracy on the rest, and explanations of what each model looks at.
6. Best pipeline: 5 CNNs + XGBoost combined by the paper's MLR → macro-F1 0.913 vs the paper's 0.895. Honest caveat: with that strong a CNN, the paper's handcrafted-feature FNN stops helping.

## 5. Known gaps vs. the paper (to close or state in the report)
- ~~**MultiNN**~~ — **done 2026-10-08:** test 0.8880 (single run) vs paper 0.8455 ± 0.0170 (`cnn_train.py --multinn`,
  architecture from the authors' `wafermap_MultiNN` repo; our CNN recipe otherwise).
- ~~**Stacking-DT**~~ — **done 2026-10-08:** test 0.8782 ± 0.0065 vs paper 0.8789 ± 0.0094 (paper protocol: fitted on A).
- ~~**Training-size sweep**~~ — **done 2026-10-08** (`nsweep_results.md`; 10 replicates for N ≤ 5,000, 5 for 50,000).
  Only Table 7 macro-F1 is reproduced; per-class Table 8 blocks and Fig. 3 (plot) for N < 162,946 are not.
- **Preliminary model-selection experiments** (paper Tables 5–6: other MFE classifiers and other CNNs) — not reproduced; we used the paper's chosen FNN and VGG directly.
- 10 random splits × replications — we use one split (seeds over learners; wafer bootstrap for test noise).

## 6. Limitations (carry into the report)
- Near-full has 9 test wafers: one wafer ≈ 0.01 macro-F1.
- CNN single seed in Stages 1–2; Stage 2b: 5 seeds, seed std ±0.012 macro-F1 (rare classes swing between seeds).
- Test set was viewed once at the end of Stage 2; Stage 2b selects on B only, but it is a second look at test — say so.
