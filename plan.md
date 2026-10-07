> **SUPERSEDED (2026-10-07):** this was the original phase-by-phase reproduction plan. Phases 1–6 are done.
> Current plans: `implementation_plan.md` (master plan) and `stage2_plan.md`. Status: `context.md` (START HERE block).

# Implementation Plan — Wafer Map Defect Classification (Stacking Ensemble)

Companion to `context.md` — read that first for full background, project history, taken
topics, dataset details, reference code, and the user's working style (low-bandwidth,
agent-autonomous — see context.md §0). This file is the phase-by-phase execution plan.

**Ground rule for every phase below: the agent executes it fully and autonomously, making
all technical decisions itself, and only surfaces something to the user if it's a data
upload need or a genuine scope/feasibility blocker (context.md §0, §6). Do not pause
mid-phase to ask the user technical questions.**

---

## Phase 0 — Setup (do this first, every fresh session should check this is done)

1. Confirm environment has: `pandas`, `numpy`, `scikit-learn`, `tensorflow` (or `keras`),
   `scikit-image` or `opencv-python` (likely needed for wafer map resizing/Radon transform
   features), `matplotlib`. Install with `pip install --break-system-packages <pkg>` as
   needed.
2. Check whether `LSWMD.pkl` has already been uploaded by the user (check
   `/mnt/user-data/uploads/`). If not present, this is the ONE thing to ask the user for —
   see context.md §4 for exact instructions to relay (download from Kaggle, e.g.
   kaggle.com/qingyi/wm811k-wafer-map, upload the `.pkl` file). Do not proceed to Phase 2+
   without this file, but Phase 1 (reading paper/repos) can and should happen regardless of
   whether the data has arrived yet.
3. Working directory: `/home/claude/wafer_pipeline/` (or current session equivalent) —
   rebuild from this plan if a fresh session has no access to prior files.

---

## Phase 1 — Read the actual paper and both reference repos in full (MANDATORY FIRST STEP)

**This phase is COMPLETED.** The journal paper and reference repos were read and details extracted.

1. Fetch the full paper (Kang & Kang 2021, Computers in Industry 129:103450) — try
   ScienceDirect directly; if paywalled, search for an accessible preprint/author copy, or
   extract as much methodology detail as possible from citing papers' descriptions (several
   were already found during earlier research — see search history if available, otherwise
   re-search "Kang Kang 2021 stacking ensemble wafer map handcrafted features methodology").
   Extract and record in this file (update this section):
   - The exact 59 handcrafted features (names/definitions, or at least the categories:
     density-based, geometry-based, Radon-transform-based, etc.)
   - The exact CNN/VGG16 architecture details (layers, input size, output layer)
   - The exact stacking mechanism — how are the two base classifiers' outputs combined? What
     is the meta-learner? How is the "assign larger weight to the superior classifier per
     class" implemented mathematically?
   - The paper's own reported results (per-class precision/recall/F1 or accuracy for
     handcrafted-only, CNN-only, and stacking ensemble) — this is essential for Phase 6's
     comparison and must not be invented.
   - Train/test split methodology used in the paper.
2. Fetch and read the full code of both reference repos:
   - `github.com/DMkelllog/WMPC_Stacking_TF2`
   - `github.com/DMkelllog/wafermap_MultiNN`
   Read every notebook/script, not just the README. Note exact preprocessing steps, feature
   extraction code, model architectures, and training procedures used. These repos are the
   PRIMARY implementation reference — prefer matching their code over inferring from the
   paper text alone, since code is unambiguous where paper text may be vague.
3. **Update context.md §3 and §5 with what was actually found** (replace the "confirmed from
   abstract/citations" placeholders with real extracted detail) before proceeding to Phase 2.
4. If, after genuine effort, some methodology detail truly cannot be pinned down (e.g. the
   paper is vague and the repos also don't clarify it), pick the most standard/defensible
   choice from the wider wafer-map-classification literature, document the assumption
   clearly, and proceed — don't stall waiting for perfect clarity.

**Deliverable:** an updated context.md with real extracted methodology detail, replacing
guesses with facts (or clearly-labeled documented assumptions where facts aren't available).

---

## Phase 2 — Data pipeline

1. Load `LSWMD.pkl`. Inspect its actual schema (column names, label encoding, whether a
   train/test field exists — context.md §4 flags this as something to check, not assume).
2. Filter to the labeled subset (~172,950 wafers across 9 classes).
3. Decide and document the train/test split:
   - If the file has a pre-existing Training/Test field, use it (closer fidelity to
     literature convention).
   - Otherwise, use a stratified split (e.g. 80/20) preserving class proportions, with a
     fixed random seed for reproducibility.
4. Handle class imbalance explicitly — e.g. `class_weight='balanced'` for the handcrafted
   classifier, and class-weighted loss or oversampling for the CNN (pick whichever the
   reference repos use if discoverable in Phase 1; otherwise use `class_weight='balanced'`-
   equivalent approaches for both, documented).
5. Prepare two parallel data representations:
   - Raw/resized wafer map images (64×64, per repo convention) for the CNN branch.
   - Extracted 59-dim handcrafted feature vectors (per Phase 1's findings) for the
     handcrafted-feature branch.
6. Sanity-check: class distribution counts, a few visualized wafer maps per class (to
   confirm parsing/orientation is sane), feature vector shape/range checks.

**Deliverable:** clean, split, dual-representation dataset ready for both base classifiers.

---

## Phase 3 — Handcrafted-feature base classifier

1. Implement the 59-feature extraction pipeline exactly as determined in Phase 1 (from
   paper text and/or reference repo code).
2. Train a conventional classifier on these features — use whatever the paper/repos specify
   (likely Random Forest or a similar ensemble/tree method, standard in this literature per
   related work like `iamxichen/Semiconductor-Wafer-Defect-Classification` which used Random
   Forest on this same dataset). If genuinely unclear from Phase 1, default to
   `RandomForestClassifier(class_weight='balanced', random_state=42)` and document this
   choice.
3. Evaluate: per-class precision/recall/F1 on the test split.

**Deliverable:** trained handcrafted-feature classifier + its standalone evaluation results.

---

## Phase 4 — CNN base classifier

1. Implement the VGG16-style CNN architecture per Phase 1's findings (input: 64×64 resized
   wafer map; output: 9-class softmax).
2. Train with appropriate class-imbalance handling (class weights and/or data augmentation —
   match reference repo approach if found in Phase 1).
3. Evaluate: per-class precision/recall/F1 on the same test split as Phase 3, for direct
   comparability.

**Deliverable:** trained CNN classifier + its standalone evaluation results.

---

## Phase 5 — Stacking ensemble

**This is the actual novel contribution of the paper — do not skip or trivialize this into a
simple average of Phase 3 + Phase 4 outputs.**

1. Implement the stacking meta-learner exactly as determined in Phase 1 — the paper's
   framing ("assigning a larger weight to the output of the superior [base classifier]")
   suggests a class-dependent weighting scheme, not a single global blend. Follow whatever
   the reference repo (`WMPC_Stacking_TF2`) actually implements here — this repo's whole
   purpose is this exact mechanism, so it should be the primary source of truth for this
   phase specifically.
2. Train/fit the meta-learner on top of the two base classifiers' outputs (typically: hold
   out a validation split, or use the base classifiers' predictions on the training set,
   depending on the stacking convention used in the reference repo).
3. Evaluate: per-class precision/recall/F1 on the test split, same as Phases 3 and 4.

**Deliverable:** trained stacking ensemble + its evaluation results, directly comparable to
the two standalone base classifiers.

---

## Phase 6 — Compare all three models + compare to the paper's own numbers

1. Build a single results table: handcrafted-only vs. CNN-only vs. stacking ensemble, across
   all 9 classes, with precision/recall/F1 (and overall accuracy as a secondary summary
   stat, not the headline metric given the class imbalance).
2. Confirm the stacking ensemble actually improves over both base classifiers (this is the
   paper's core claim — if it doesn't in our reproduction, investigate why before writing
   this up as a negative result; check for bugs in the meta-learner implementation first).
3. Place these results side-by-side with the paper's own reported numbers (extracted in
   Phase 1). Write a short, honest discussion: do the numbers roughly match? If not, note
   plausible reasons (different random seed / train-test split methodology, class imbalance
   handling differences, hyperparameter differences vs. an underspecified paper, dataset
   version differences if the Kaggle mirror differs slightly from the original MIR Lab
   release, etc.).

**Deliverable:** comparison table + written discussion paragraph — this is what "faithful
reproduction" projects are typically graded on most heavily.

---

## Phase 7 — Report and viva prep

Write the full report autonomously (don't co-write section by section with the user — draft
completely, then let the user review at the end, per context.md §0):

1. **Introduction** — what wafer map defect patterns are, why automated classification
   matters in semiconductor manufacturing (yield analysis, root-cause tracing), why a hybrid
   ML approach (handcrafted + CNN) is used.
2. **Related Work** — brief summary of the Kang & Kang 2021 paper and 2-3 nearby papers in
   the same space (several were surfaced during topic research — e.g. Nakazawa & Kulkarni
   2018 IEEE TSM CNN approach, Wu et al. 2014 handcrafted feature approach — use these as
   comparison points showing the lineage of handcrafted vs. CNN approaches this paper
   unifies).
3. **Dataset** — WM-811K, scale, class distribution, imbalance.
4. **Methodology** — data pipeline, the 59 handcrafted features, the CNN architecture, the
   stacking mechanism. Be transparent that implementation was AI-assisted per the
   professor's own instruction (context.md §0) — not something to hide.
5. **Results** — Phase 6's comparison table.
6. **Discussion/Limitations** — Phase 6's paper-comparison discussion, plus any documented
   assumptions from Phase 1 where the paper/repos were ambiguous.
7. **Conclusion.**

Also prepare:
- Slide deck mirroring the same structure.
- A short (~1 page) plain-English viva prep summary covering: what a wafer map is and why
  defect pattern classification matters, what handcrafted vs. CNN features are (in simple
  terms — no need for exact feature formulas), why a stacking ensemble helps (different
  approaches are better at different defect types), the reproduced results vs. the paper's,
  and an honest note on which parts were AI-assisted vs. the user's own direction/review.

**Deliverable:** final report (Word doc), slide deck, viva prep one-pager — all as
downloadable files.

---

## Suggested timeline (3-4 months)

| Weeks | Phase |
|---|---|
| 1 | Phase 0 (setup) + Phase 1 (read paper + both repos in full, update context.md) |
| 2-3 | Phase 2 (data pipeline) — waits on user's dataset upload; can start once received |
| 4-5 | Phase 3 (handcrafted classifier) + Phase 4 (CNN classifier) — can be done in parallel |
| 6-7 | Phase 5 (stacking ensemble) — the trickiest/most paper-specific phase |
| 8-9 | Phase 6 (full comparison + paper benchmark discussion) |
| 10-12 | Phase 7 (report, slides, viva prep) |
| 13-14 | Buffer — re-runs, polishing, addressing any issues found in Phase 6 |

Since the user checks in weekly/biweekly (context.md §0), the agent should aim to complete
1-2 phases between check-ins and report a concise progress summary each time, rather than
waiting for user prompts to advance.

---

## Explicit "Do Not Deviate" checklist

- [x] Actually read the full paper and both reference repos (Phase 1) before writing any
      modeling code — do not implement from the summary-level guesses in context.md alone.
- [x] Target full (100%) implementation — all three models (handcrafted, CNN, stacking) plus
      the full evaluation/comparison, not a partial subset, per context.md §6.
- [x] Use per-class precision/recall/F1 as the primary metric, not plain accuracy, given the
      severe class imbalance (context.md §4).
- [x] Implement the stacking ensemble as an actual class-dependent meta-learner matching the
      reference repo, not a naive average of the two base classifiers' outputs.
- [x] Make all technical/modeling decisions autonomously — do not pause to ask the user to
      choose between options (context.md §0).
- [x] Only contact the user for: the dataset upload, genuine scope/feasibility blockers, and
      final review — batch everything else into periodic progress updates.
- [x] Keep context.md and this file updated as each phase completes, so a fresh agent session
      can resume with zero prior conversation history.
- [x] Write the full report/slides/viva-prep autonomously once results exist — don't wait for
      user prompts to advance. Keep context.md and plan.md updated.
      the user to co-author it.
