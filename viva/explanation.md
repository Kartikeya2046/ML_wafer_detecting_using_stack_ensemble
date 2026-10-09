# The Paper Explained, Section by Section

A plain-language companion to `paper/main.pdf`. Read this before presenting; it follows the paper's order.

---

## Title and abstract
**"Stacking Handcrafted and Convolutional Features for Wafer Map Pattern Classification, Revisited: Reproduction,
Meta-Training Pitfalls, and Improvements."**
"Revisited" signals that this is a *reproduction study*: we rebuilt a published method, checked its claims, found where it
breaks, and improved it. The abstract states the three headline numbers: our reproduction (0.8967 vs 0.8949 reported),
the best pipeline (0.9134), and the reject option (95.2% of wafers at 99.72% accuracy).

## I. Introduction
- **Why it matters:** wafer maps show failed dies; the *pattern* of failures diagnoses the process problem.
- **Two families:** handcrafted features + a classifier, or a CNN on the image.
- **Kang & Kang's idea:** combine both with *stacking* (a model that learns how to weigh the two models per class).
- **Our four contributions:** (1) full reproduction, (2) the meta-training pitfall, (3) controlled extensions with
  statistics, (4) selective prediction.

## II. Related Work
Three paragraphs: handcrafted-feature methods, CNN methods, and stacking. The key line is about Ting & Witten (1999):
the stacker must be trained on predictions for data the base models did *not* see. That is exactly the rule the
public reference code breaks (Section V-B).

## III. Data and Original Method
- **Data:** 172,946 labeled wafers; test = 10,000 wafers; the class counts show how rare some classes are (Near-full: 9).
- **59 handcrafted features:** densities in 13 regions, 40 Radon-transform values, 6 shape measures.
- **CNN:** VGG16 pretrained on ImageNet; the 64×64 binary map is copied to 3 channels so it fits.
- **MLR stacker:** a ridge regression from the 18 probabilities to the 9 classes. Its weights are interpretable: one
  weight per (model, class).
- **Baselines:** Stacking-FNN, Stacking-DT, MultiNN (CNN + features fed into the same final layer).

## IV. Experimental Protocol (the "fairness" section)
- **Split A / B:** base models learn on A; B is never used to fit them, so their predictions on B look like predictions
  on new wafers.
- **A-mode vs B-mode:** the original protocol fits the stacker on A (where the models have seen the answers); we also fit
  it on B.
- **Selection discipline:** every decision is made on B; the test set only reports.
- **Statistics:** paired t-test (does the improvement hold across seeds?) and a bootstrap over wafers (would it survive
  a different sample of test wafers?).
- **Inclusion rule:** an extension stays only if it beats the baseline by more than the normal seed-to-seed wobble.

## V. Reproduction
- **Table I:** all six models at full data, ours vs reported. Every model is within or above the reported std.
- **Three details that mattered:** no class weights (the paper does not use them; they hurt), ImageNet initialization
  and rotation/flip augmentation for the CNN, and 32-bit training (mixed precision cost ~4 points).
- **V-B, the pitfall:** trained in-sample, the stacker sees base models that are almost always right, so it learns to
  copy them and never predicts the rarest class. Fixed by training on B. MLR, being simple, suffers least.
- **V-C, training-size sweep (Table II, Fig. 2):** at N ≥ 5,000 we beat the paper on every model; at N = 500 results are
  noisy and below the paper; at N = 50,000 the in-sample MLR stack falls below our CNN, the same pitfall again.

## VI. Extensions
- **Table III** walks through every configuration; **Table IV** gives the statistics.
- **A. TTA:** average the CNN over the 8 rotations/flips. Free improvement.
- **B. XGBoost:** a stronger model on the same 59 features. Two controls show *why* it helps: it's simply better, not
  "diversity".
- **C. CNN seed ensembles:** 5 CNNs differ a lot on rare classes even with equal loss; averaging them is the biggest gain.
- **D. Final pipeline:** 5 CNNs (TTA) + XGBoost → MLR = **0.9134**. The single-CNN version is 0.9160 on test, but
  that difference is noise and the choice was made on B, so we report it transparently.

## VII. Selective Prediction
Turn MLR scores into probabilities (temperature softmax), pick confidence thresholds on B, apply them to test. At 95.2%
coverage: 99.72% accuracy. The wafers sent to an engineer are mostly the genuinely hard local patterns.

## VIII. What the Stack Learns
**Fig. 4 (MLR weights):** handcrafted features get the weight for Random and Near-full; the CNN for Edge-Ring and Scratch.
This matches Kang & Kang's own figure, and SHAP / Grad-CAM agree.

## IX. Negative Results and Limitations
Say these confidently; they show the work is careful:
1. The paper's handcrafted FNN stops helping once the CNN is an ensemble.
2. A "different" third model is not better than a second seed of an existing one.
3. Bagging XGBoost helped alone but hurt the stack.
4. Temperature scaling did not improve MLR calibration.
5. Class weights, mixed precision and a plateau learning-rate schedule did not help.
6. The in-sample MLR stack does not beat the CNN at N = 50,000.
Limitations: one split; 9 Near-full test wafers; test consulted twice; 5 replicates at 50k; MultiNN single run.

## X. Conclusion
The paper reproduces. Two lessons: train stackers on held-out predictions, and the value of handcrafted features
depends on how strong the CNN is. Best result: 0.9134 macro-F1.

## Appendix
Per-class F1 (Table VI), CNN seeds (Table VII), XGBoost grid (Table VIII), Grad-CAM maps (Fig. 5).

---

### Where every number comes from
`paper/make_assets.py` builds all tables, figures and in-text numbers from the saved results in `data/` and `logs/`.
Re-run it and recompile (`cd paper && ../tools/tectonic main.tex`) to regenerate the PDF.
