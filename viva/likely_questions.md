# Likely Viva Questions — with Short Answers

Answers use the numbers in `paper/main.pdf`. Keep answers to 2–4 sentences; offer to go deeper if asked.

### About the problem
**1. What is a wafer map and why classify its patterns?**
A grid showing which dies on a wafer failed electrical tests. The *shape* of the failures tells engineers which process
step went wrong (edge ring → wafer-edge process, scratch → handling), so automatic classification speeds up yield analysis.

**2. Why macro-F1 instead of accuracy?**
85% of wafers are "None". A model that always says "None" gets 85% accuracy but is useless. Macro-F1 averages the F1 of all
nine classes equally, so rare classes like Near-full (9 test wafers) count as much as None.

**3. What dataset and split?**
WM-811K: 172,946 labeled wafers after removing maps with fewer than 100 dies. 10,000 random (stratified) test wafers; the
rest are training data, split into A (fit base models) and B (validation / meta-training).

### About the original method
**4. What are the 59 handcrafted features?**
13 region densities (how many failed dies in each zone), 40 Radon-transform values (projections that capture lines and
rings), 6 geometry values of the largest defect blob (area, perimeter, axes, eccentricity, solidity).

**5. What is stacking and what is MLR?**
Stacking trains a second-level model on the outputs (class probabilities) of first-level models. MLR is a ridge regression
from the 18 probabilities to the 9 classes, so it learns a per-class weight for each model, e.g. "trust handcrafted
features for Random, trust the CNN for Edge-Ring".

**6. Why do the two models complement each other?**
Handcrafted features summarize global density and shape (good for Random, Near-full); the CNN sees spatial structure
(good for Edge-Ring, Scratch). Our MLR weights (paper Fig. 3) show exactly this split.

### About our reproduction
**7. Did you reproduce the paper?**
Yes. At full data our Stacking-MLR scores 0.8967 vs 0.8949 reported, and all six models (MFE+FNN, CNN, MultiNN,
Stacking-DT, Stacking-FNN, Stacking-MLR) match or beat the reported values. We also reproduced the training-size sweep.

**8. What was the "meta-training pitfall"?**
The public code trains the stacker on the base models' predictions for their *own training data*. There the models are
almost always right and very confident, so the stacker learns "just follow the models" and never learns when to override
them; the rarest class (Near-full) got F1 = 0. Training the stacker on held-out data B fixed it.

**9. Why didn't you use class weights?**
The paper does not mention them, and in our experiments they hurt every model (about −5 points for the MFE+FNN).

**10. Where do your results differ from the paper?**
At N = 500 our handcrafted model and stackers are below the paper (very noisy; 400 training wafers often miss rare
classes entirely). At N = 50,000, under the paper's protocol, the MLR stack is slightly below our CNN; this is the same
pitfall as in Q8.

### About our improvements
**11. What is test-time augmentation?**
Wafer patterns keep their class when rotated or flipped, so we ask the CNN about all 8 rotations/flips of each wafer and
average the answers. No retraining needed; significant gain on validation.

**12. Why XGBoost?**
It is a strong tree model for tabular features. On the same 59 features it beats the paper's FNN (0.8704 vs 0.8525 on
validation). We also ran two controls to see *why* it helps: it helps because it is a better model, not because a
"different kind" of model adds diversity.

**13. What is the CNN ensemble and why does it help so much?**
Five CNNs with the same recipe but different random seeds. They have the same validation loss but differ by ±0.012
macro-F1, because rare classes swing from seed to seed. Averaging them removes that randomness: the largest single gain.

**14. What is your final result?**
5 CNNs (with TTA) + XGBoost, combined by the paper's MLR: macro-F1 0.9134, accuracy 98.21% (paper: 0.8949). The
improvement over our reproduction is statistically clear (bootstrap interval excludes zero).

**15. What is the reject option?**
If the model's confidence is below a threshold, the wafer is sent to an engineer instead. Classifying 95.2% of wafers
automatically gives 99.72% accuracy on them; the deferred ones are mostly the hard local patterns (Loc, Scratch, Edge-Loc).

### About rigor and limitations
**16. How do you know the improvements are real and not luck?**
All decisions were made on validation split B; the test set was only used to report. We used paired t-tests and a
bootstrap over wafers, and a fixed rule: keep an extension only if it beats the baseline by more than its run-to-run
spread.

**17. Why is the single-CNN stack slightly better on the test set than your final ensemble (0.9160 vs 0.9134)?**
The difference is inside the bootstrap interval, so it is noise (mostly the 9 Near-full wafers: one wafer ≈ 0.01
macro-F1). We chose the ensemble on validation data before looking at the test set, which is the correct procedure.

**18. Limitations?**
One fixed train/test split (the paper averages ten); only 9 Near-full test wafers; the test set was looked at twice
(each time after selection on B); five instead of ten replicates at N = 50,000; MultiNN is a single run.

**19. What surprised you?**
That the paper's handcrafted-feature network stops helping once the CNN is strong (an ensemble). The handcrafted
features still help a little through XGBoost.

**20. What would you do next?**
Repeat over several random splits like the paper; try rotation-invariant CNNs; use unlabeled wafers (80% of WM-811K)
with self-supervised learning.
