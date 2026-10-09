# Project Context — Wafer Map Defect Pattern Classification (VLSI Semester Project)

> ## ▶ START HERE (status as of 2026-10-08)
> - **Stage 1 (reproduction + tuning) is DONE and FROZEN** — git tag `stage1-frozen`. Our tuned pipeline matches or
>   beats the paper: Stacking-MLR **0.8967 ± 0.0060** macro-F1 (paper 0.8949 ± 0.0121), tuned Stacking-FNN
>   **0.9041 ± 0.0028** (paper 0.8991 ± 0.0096). Numbers: `results_comparison.md`. Every trial/decision: `tuning_log.md`.
> - **Stage 2 (our own contribution) is DONE** (user signed off 2026-10-08), branch `stage2`. Results: `extensions_results.md`
>   (built by `extensions_results.py`). Final pipelines chosen on B: MLR = MFE-FNN + CNN-TTA, test **0.9051 ± 0.0069**;
>   FNN = MFE-FNN + CNN-TTA + XGB, test **0.9070 ± 0.0037**. TTA helps; XGB as 3rd learner is mixed (MLR Stack-3 0.9149 on
>   test but rejected on B); XGB replacing the MFE-FNN is best (0.9160); reject option: 95% coverage → 99.6% accuracy.
> - **Stage 2b (best results + missing paper baselines) DONE 2026-10-08** (user: "keep the best models, best approach").
>   **Final pipeline: 5 CNN seeds (TTA) + XGBoost → MLR, test macro-F1 0.9134** (accuracy 98.21%; 99.72% on the 95%
>   auto-classified with the reject option). Runnable: `predict.py`. Paper baselines added: Stacking-DT 0.8782 (paper
>   0.8789), MultiNN 0.8880 (paper 0.8455), training-size sweep N = 500 / 5,000 / 50,000 (`nsweep_results.md`).
> - **For the write-up use `writeup_notes.md`** (paper did / we implemented / we did differently — kept current).
> - **Part D DONE (2026-10-09)** — decisions in `partD_plan.md`. Paper: `paper/main.pdf` (IEEE two-column, 8 pages,
>   source `paper/main.tex`, every table/figure/number from `paper/make_assets.py`). Viva material (separate from the
>   LaTeX): `viva/one_pager.md`, `viva/likely_questions.md`, `viva/explanation.md`. Slides: `slides/slides.pdf`.
>   Compile: `cd paper && ../tools/tectonic main.tex` (Tectonic in `tools/`, not in git). **Never mention AI in the
>   paper/slides/viva** (user). Open: push to GitHub (needs the user's yes); instructor co-authorship (user's call).
> - **Machine:** since 2026-10-08 the project runs on a Linux workstation (§8, "Linux workstation"). Old Windows notes kept.
> - **Reading order:** this file → `implementation_plan.md` (master plan, Parts A–D) → `stage2_plan.md` →
>   `extensions_results.md` / `tuning_log.md` / `results_comparison.md` as needed. `plan.md` is superseded (history only).
> - **How the user wants to work now:** §0 plus the updates in §9 (they override older text where they differ).

This file exists so any AI agent (Claude, ChatGPT, or otherwise) picking up this project
later has full grounded context. No re-guessing, no re-asking things already answered here.
Read this file AND the companion `plan.md` fully before doing any work.

---

## 0. WHO THE USER IS AND HOW THEY WORK — READ THIS FIRST

**Background:** The user is an **IT/CS background student, NOT an ECE/VLSI person**. Their
domain is **prompt engineering and ML modeling**. Their professor explicitly approved having
AI handle implementation work, with the user directing at a high level.

**Bandwidth constraint (important, stated explicitly by the user):** This project is
**compulsory but low-priority** for the user — they have other, more important work
competing for their time. They have explicitly asked that:

- **The agent make ALL decisions autonomously** — technical, modeling, structural — and just
  execute them. Do not present multiple options and wait for the user to choose. Pick the
  standard, defensible choice and proceed.
- **The agent do essentially all the work itself** — reading the paper in depth, writing all
  code, running all experiments, debugging, writing the full report and slides.
- **The user's role is limited to exactly four things:**
  1. **Data uploads** — anything requiring a login/manual download (e.g. Kaggle datasets)
     that the agent cannot fetch directly.
  2. **Approvals when genuinely needed** — e.g. confirming a major scope/direction change,
     or explicit yes/no on something that materially affects grading risk.
  3. **Periodic check-ins** (weekly/biweekly, NOT every turn) — the agent should batch
     progress updates rather than pinging the user constantly. Don't ask the user
     turn-by-turn questions; work through multiple phases autonomously between check-ins.
  4. **Final review before submission + viva prep** — the user will skim final outputs and
     needs to be able to explain the project verbally, but does not need to build/understand
     implementation details deeply.
- **Do NOT ask the user low-level implementation questions.** If a fresh agent session finds
  itself about to ask the user something technical (hyperparameters, architecture choices,
  which metric to prioritize, etc.), it should instead consult this file + plan.md, make the
  most standard/defensible choice itself, document the choice made, and proceed.
- **Keep VLSI/technical explanations light when the user is actually present** — a few plain
  English sentences, enough for a viva, not implementation-level detail.
- The user works across **multiple separate conversations with a token/context budget** —
  each new conversation may have ZERO memory of prior ones. **This file and plan.md are the
  single source of truth** and must be kept current as work progresses, so a brand-new agent
  session can resume seamlessly by reading these two files alone.

---

## 1. Project constraints (from professor)

- Must be based on a genuinely **IEEE or Elsevier published paper** (journal or reputable
  venue — not MDPI, not an unverified/predatory venue).
- Should ideally have a **public dataset AND public code** — full 100% implementation is
  the target for this project (see §6 — this is a firmer requirement than earlier attempts,
  because the user is worried about grading risk from partial implementation, and because
  usable reference code exists this time, unlike the previous two attempts).
- **Timeline: 3-4 months** (this is a longer window than the original 2-month Trojan-
  detection attempt — that project was abandoned; see §2).
- **Topics/papers ALREADY TAKEN by other students (do not suggest or pick these):**
  Application mapping onto manycore processors (x2 variants), Parameter optimization of VLSI
  placement, A graph placement methodology for fast chip design, Implementing CNN in
  standard cell libraries, Deep learning based routability optimization, Efficient SoC power
  estimation with ML (x2 variants), A unified machine learning framework, CircuitNet VLSI
  routing/congestion, Minimal machine learning processor, Layout hotspot detection using
  CNN, IR drop prediction, Pre/post synthesis power estimation for CMOS, Beyond Tokens:
  enhancing RTL quality estimation, Pre-route timing prediction and optimization, GNN-RE for
  gate-level netlists/agentic AI, STEB-UNet for routability & DRC prediction.

---

## 2. Project history — why we're on attempt #3

**Attempt 1 (ABANDONED):** FPGA routing congestion prediction, based on an MDPI Electronics
2021 paper + github.com/Rahulprakash77/FPGA-Congetion. Abandoned because MDPI is not
IEEE/Elsevier — failed the publisher requirement — and the natural IEEE upgrade path
(CircuitNet-based congestion) was already taken by another student.

**Attempt 2 (ABANDONED):** Hardware Trojan Detection using Random Forest, based on Hasegawa
et al., IEEE ISCAS 2017 ("Trojan-feature extraction at gate-level netlists..."). This was a
genuinely IEEE-published, non-taken topic with a public dataset (TrustHub), and a full
custom pipeline (Verilog netlist parser, SCOAP controllability/observability engine,
structural feature extraction, RandomForest classifier) was built and CONFIRMED WORKING on
one real circuit (RS232-T1000: 268 gates, 308 nets parsed successfully, 12 Trojan nets
correctly labeled out of 308, 56-feature CSV generated). However: (a) no official public
code existed for that paper, meaning everything had to be built from scratch with no
ground-truth implementation to validate against, and (b) it required real Verilog/VLSI
domain engineering that, while doable by AI, added complexity and reproduction risk. The
user decided to pivot to a project space requiring zero VLSI/Verilog knowledge and offering
existing reference code, given their limited bandwidth for this project. **The Trojan
detection code/approach is NOT being continued — this is a clean pivot away from it,
documented here only for history.**

**Attempt 3 (ACTIVE — this is the current project):** Wafer Map Defect Pattern Classification
using a stacking ensemble — see §3 below.

---

## 3. Chosen paper (ACTIVE PROJECT)

**Title:** "A stacking ensemble classifier with handcrafted and convolutional features for
wafer map pattern classification"
**Authors:** Hyungu Kang, Seokho Kang (corresponding author)
**Venue:** Computers in Industry, Volume 129, 2021, Article 103450
**Publisher:** **Elsevier** — genuinely published, confirmed via ScienceDirect.
**DOI:** 10.1016/j.compind.2021.103450

**What the paper does (confirmed from abstract/citations):** Wafer map defect pattern
classification via a hybrid ensemble of two approaches:
   1. A classifier built on **handcrafted features** (59-dimensional feature vector per wafer
   map — 13 density features from 13 partitioned regions, 6 geometric features including area/perimeter/axes/solidity/eccentricity, and 40 Radon transform features from row-wise mean and standard deviation).
2. A **CNN classifier** based on a VGG16-style architecture, operating directly on the
   resized (64×64) wafer map image.
   3. A **stacking ensemble meta-learner** that combines both base classifiers' outputs,
   assigning greater weight to whichever base classifier is stronger for each specific
   defect class (implemented via a Multi-Response Linear Regression (MLR) meta-classifier that assigns weight matrices to the probability outputs of the two base classifiers).

The paper's key point: handcrafted and CNN-based approaches have complementary strengths
across different defect pattern classes, so the stacking ensemble improves over either
base classifier alone.

**Why this paper is a strong choice (vs. attempts 1 and 2):**
- Genuinely Elsevier-published journal — satisfies the publisher requirement cleanly.
- Recent (2021) — within a 5-year window as of 2026.
- **Public code exists** (unlike attempt 2) — see §5.
- No VLSI/Verilog/hardware-description-language knowledge required at all — this is a pure
  image/tabular ML task (wafer maps are just 2D categorical arrays: pass/fail/edge-die per
  die position). This removes the single biggest source of friction from attempt 2.
- Not on the taken-topics list.
- Large, well-known, genuinely public dataset (§4).

---

## 4. Dataset

**Name:** WM-811K (also referred to as LSWMD — Lot-based Semiconductor Wafer Map Dataset)
**Source:** Originally released by National Taiwan University's MIR Lab
(http://mirlab.org/dataset/public/); also mirrored on **Kaggle**
(commonly found as "WM-811K wafer map" dataset, e.g. kaggle.com/qingyi/wm811k-wafer-map).
**File format:** Distributed as `LSWMD.pkl` — a pickled pandas DataFrame.

**Scale/structure (confirmed from multiple sources during research):**
- 811,457 total wafer maps, collected from ~46,393 production lots in a real fabrication
  environment.
- Only **172,950 wafers are labeled** by domain experts with one of **9 classes**: Center,
  Donut, Edge-ring, Edge-local, Local, Random, Near-full, Scratch, and **None** (no defect
  pattern). The remaining ~80% are unlabeled — NOT usable for supervised training (ignore
  for this project's core classification task; the paper doesn't appear to use unlabeled
  data given its supervised stacking-ensemble framing — confirm this from full paper).
- Severe class imbalance: the "None" class dominates the labeled subset — MUST be handled
  explicitly (class weighting, stratified splits, and/or resampling) — do not report plain
  accuracy alone, use per-class metrics (F1, precision, recall) as primary reporting, matching
  the paper's own framing that different classes have different difficulty/behavior.
- The dataset provides a "Training"/"Test" field for the labeled subset in at least some
  distributions — CHECK whether the specific `LSWMD.pkl` file obtained has this field, and
  use it for the train/test split if present (for closer fidelity to how the original
  dataset is typically split in this literature), otherwise construct a stratified split
  ourselves (document this choice either way).

**USER ACTION REQUIRED — the only mandatory upload for this project:** the user must
download `LSWMD.pkl` from Kaggle (requires a Kaggle login the agent doesn't have) and upload
it to the conversation. **No other user action should be needed to start real work** — once
this file is uploaded, the agent should proceed through all phases autonomously per plan.md.

---

## 5. Reference code (confirmed to exist — a major advantage over attempt 2)

Two public GitHub repositories from the same author group, directly implementing this
paper's methodology:

1. **`DMkelllog/WMPC_Stacking_TF2`** — "Wafer map pattern classification using stacking
   ensemble with TF2 Keras." This is the primary reference — implements the actual stacking
   ensemble described in the paper, using TensorFlow 2 / Keras.
2. **`DMkelllog/wafermap_MultiNN`** — "Wafer map defect pattern classification with
   Multi-Input Neural Network using Convolutional and Handcrafted Features." A related
   variant by the same author, useful as a secondary reference — confirmed details: input is
   the wafer map resized to 64×64, handcrafted features are 59-dimensional, CNN features are
   512-dimensional (concatenated before final classification in this multi-input variant),
   model is VGG16-based, dataset file expected at `/data/LSWMD.pkl`, sourced from
   `kaggle.com/qingyi/wm811k-wafer-map`.

**Agent instructions regarding these repos:** fetch and read both repos' code in full before
writing any implementation. Use them as the primary structural reference for: data
preprocessing steps, exact handcrafted feature definitions if present in code (preferred
over inferring from the paper text alone), CNN architecture specifics, and the stacking
mechanism. Adapt/rewrite (don't just blindly copy) into a clean, well-documented pipeline —
and note clearly in the report which parts are direct adaptations of these repos vs.
original engineering, for academic honesty.

**Phase 1 Completed:** The agent has fetched and inspected the full contents of both repos in detail, as well as the Kang & Kang 2021 journal paper. The precise feature extraction methods and CNN architectures have been recorded, replacing earlier assumptions.

---

## 6. Scope target: 100% implementation (not 50%)

Unlike the abandoned attempt 2 (where 50% was pre-approved due to a tight 2-month window and
no reference code), **this project targets full (100%) reproduction**, because:
- The timeline is longer (3-4 months vs. 2 months).
- Reference code exists, substantially de-risking full reproduction.
- The user is worried about grading risk from a partial submission and would rather the
  agent handle the larger scope autonomously than submit a deliberately partial version.

**100% implementation requires ALL of the following (see plan.md for phase breakdown):**
1. Full data pipeline: load, filter to labeled subset, handle class imbalance, train/test
   split, resize wafer maps for CNN input.
2. Handcrafted-feature base classifier: extract the 59-dim feature vector per wafer map,
   train a conventional ML classifier on it.
3. CNN base classifier: VGG16-style architecture, trained directly on wafer map images.
4. The stacking ensemble meta-learner combining both base classifiers, with class-dependent
   weighting as described in the paper's abstract.
5. Full evaluation: per-class (all 9 classes) precision/recall/F1, comparing handcrafted-only
   vs. CNN-only vs. stacking ensemble, replicating the paper's own ablation structure.
6. Comparison of reproduced results against the paper's own reported numbers (extract these
   precisely from the full paper text — not yet done, see plan.md Phase 1).
7. Full report + slides + a viva prep one-pager for the user.

**If, partway through, full 100% reproduction turns out to be infeasible within reasonable
effort (e.g. a component of the paper is genuinely underspecified and irreproducible), the
agent should:** clearly document the gap in this file and plan.md, implement the closest
faithful approximation, and flag it as a genuine limitation rather than silently doing less
than 100% without saying so. This is one of the "approval needed" cases from §0 — surface it
to the user briefly rather than just quietly settling for less.

**Status of the 7 items above (2026-10-07):** 1–6 are done (Stage 1, frozen). Item 7 (report, slides, viva
one-pager) is implementation_plan.md Part D, after Stage 2.

**Scope (settled 2026-10-06):** headline numbers are the **full-data N=162,946** setting (train 162,946 / test 10,000,
stratified, `random_state=42`, from `phase2_data.py`). The old N=5000 workaround (`models/*_5k.keras`) is superseded:
the GPU works through the TF 2.10 conda env (see §8). Paper comparison point: Table 7 / Table 8 at N=162,946.

**Facts verified from journal.pdf:** 10 replicates (random splits), mean ± std, paired t-test. CNN = VGG16
**pre-trained on ImageNet** with global-average-pooling head (§4.3; both reference repos load pretrained VGG16).
Proposed meta-learner = **MLR** (ridge regression of one-hot y on both probability vectors, λ=0.1, eq. 5–6); an
18→10→9 FNN meta-learner is the paper's *Stacking-FNN* baseline. No class weights are mentioned anywhere.
Table 7, N=162,946, F1macro: MFE+FNN 0.8599±0.0117, CNN 0.8679±0.0126, MultiNN 0.8455±0.0170,
Stacking-DT 0.8789±0.0094, Stacking-FNN 0.8991±0.0096, Stacking-MLR 0.8949±0.0121.
F1micro: 0.9741, 0.9775, 0.9728, 0.9757, 0.9808, 0.9801 (same order). Per-class: Table 8, last block
(printed there as "162,496"), copied into `freeze_results.py`.

**What Stage 1 found (the story for the report/viva):**
1. The first reproduction's stacker (Near-full F1 = 0, macro 0.7614) failed because the meta-learner was trained on
   the base learners' **in-sample** predictions, which are overconfident; it early-stopped after ~23 epochs and
   ignored rare classes. Class weights only masked it (0.8402 ± 0.0393, very seed-dependent).
2. **Class weights hurt every model here** (MFE-FNN −5 pts, CNN, both stackers). The paper uses none.
3. The paper's own **MLR** meta-learner is as good as the best tuned FNN and deterministic → chosen as headline.
4. The CNN needs **ImageNet initialisation** (paper §4.3) and benefits from **flip/rot90 augmentation**.
5. **Mixed precision (fp16) and batch 128 cost ~4 pts** on this VGG (no batch-norm) → all final CNN runs are fp32, batch 32.
6. Frozen pipeline: MFE+FNN (paper arch, no cw) + VGG16 (ImageNet, no cw, aug, fp32) + MLR (α=0.1).
   Test: Stacking-MLR 0.8967 ± 0.0060, Stacking-FNN 0.9041 ± 0.0028, MFE+FNN 0.8572 ± 0.0062, CNN 0.8777 (1 run).
7. Limitations: CNN is a single seed and its tuning stopped at the run budget (each round still improved);
   Near-full has only 9 test wafers (one wafer ≈ 0.01 macro-F1); one fixed split vs the paper's 10.

**Tuning protocol (reused in Stage 2):** A = train[:130356] (base learners fitted here); B = train[130356:]
(base learners only early-stopped here, so their outputs on B are honest); B split once, stratified, into
B_fit / B_val (`common.py:splits`). All selection on B; test touched only for final numbers.

---

## 7. Strict instructions for any agent continuing this work

1. **Never invent numbers.** If a metric/result isn't confirmed in this file, plan.md, the
   actual paper, or freshly computed in the current session, say so explicitly and go verify
   it (re-fetch the paper, re-inspect the reference repos) rather than guessing.
2. **Do essentially all the work yourself** — data pipeline, feature extraction, model
   training, evaluation, report writing. Per §0, the user should not be asked to make
   implementation decisions.
3. **Make and execute all technical/modeling decisions autonomously** — pick the standard,
   defensible choice (matching the paper and/or reference repos where possible) and proceed.
   Document the choice made in this file or plan.md rather than asking the user.
4. **Only interrupt the user for:** data uploads the agent cannot obtain itself (Kaggle
   login-gated downloads), genuine scope/feasibility blockers (§6's last paragraph), and
   final review/approval before submission.
5. **Batch progress into periodic updates**, not turn-by-turn questions — the user checks in
   weekly/biweekly, not continuously.
6. **Actually fetch and read the real paper and the two reference repos' code** before
   writing implementation — do not proceed on the summary-level descriptions in this file
   alone; they were built from search snippets, not the full source materials. This is
   explicitly Phase 1 of plan.md and must happen before any modeling code is written.
7. **Update this file and plan.md continuously** as work progresses (once the paper/repos
   are fully read, once each phase completes, once real results exist) — since the user
   works across multiple sessions with a token budget, these files are the persistent single
   source of truth, not conversation history. A stale context.md is worse than a short one —
   keep it accurate.
8. **Keep any direct interaction with the user light and low-friction** — short, clear
   asks only when truly necessary (per §0), batched progress summaries otherwise, and a
   plain-English viva-prep summary near the end (not deep technical explanation) since the
   user needs to be able to talk about this project, not re-derive it.

---

## 8. Resuming on a new machine / new session (READ THIS when the desktop changes)

**Git** — repo `https://github.com/Kartikeya2046/ML_wafer_detecting_using_stack_ensemble`, branch `main`
(also `stage1-tuning`; tag `stage1-frozen`). Stage 2 work goes on a new branch `stage2` from `stage1-frozen`.

**Large files that are NOT in git** (GitHub rejects files > 100 MB). Copy these from the old machine (USB/drive),
keeping the same relative paths inside the project folder:

| File | Size | Needed for | If lost |
|---|---|---|---|
| `models/cnn_imnet_cw0_aug.keras` | 169 MB | **Stage 2** (TTA re-predicts with it; Grad-CAM) | retrain: `sh run_gpu.sh cnn_train.py --tag imnet_cw0_aug --imagenet --cw 0 --fp32 --aug` (~2 h on an RTX 3050) |
| `data/X_CNN.pkl` | 1.35 GB | CNN training/inference (64×64 maps) | regenerate with `phase2_data.py` from LSWMD (~1 h, needs scikit-image) |
| `LSWMD.pkl.zip` or `LSWMD.pkl/LSWMD.pkl` | 150 MB / 2 GB | only to regenerate data | Kaggle `qingyi/wm811k-wafer-map` (needs the user's login) |
| other `models/cnn_*.keras` | 169 MB each | not needed (Stage 1 trial models) | — |
| trial outputs `data/*_outputs.pkl`, `data/stack_*.pkl` not tracked | ~6 MB each | not needed (scores are in `logs/*.jsonl`) | rerun the trial |

Everything else needed — the 59-feature matrix `data/X_MFE.pkl`, labels `data/y.pkl`, the frozen Stage 1 outputs
(`data/mfe_x3_cw0*_outputs.pkl`, `data/cnn_imnet_cw0_aug_outputs.pkl`, `data/stack_FINAL_*`, `data/stack_B1_*`),
all scripts, logs and docs — **is in git**. `python freeze_results.py` must reproduce `results_comparison.md` exactly
on the new machine; run it as the first sanity check.

**Linux workstation (current, since 2026-10-08)** — Ubuntu 24.04, RTX 4000 Ada 20 GB, 24 cores, 125 GB RAM, Python 3.12.
- Env: project venv `.venv` (not in git), created by `sh setup_env.sh`: TF 2.17.1 + `tf_keras` 2.17 (pip CUDA 12.3 /
  cuDNN 8.9). `run_gpu.sh` / `run_cpu.sh` pick `.venv` automatically and set `TF_USE_LEGACY_KERAS=1` (tf.keras = Keras 2).
  `setup_env.sh` also patches a tf_keras + Python 3.12 bug (`randint(1, 1e9)` TypeError after `set_random_seed`).
- Large files live at the paths in the table above (`data/X_CNN.pkl`, `models/cnn_imnet_cw0_aug.keras`, `LSWMD.pkl/LSWMD.pkl`).
- **Load the CNN with rebuild + `load_weights`, never `load_model`** — its Lambda layer was saved as Python 3.10 bytecode
  ("bad marshal data" under 3.12). `cnn_train.py --tta` and `gradcam.py` do this.
- Equivalence verified: `freeze_results.py` zero diff; CNN re-predicts test within 1.2e-4; MFE-FNN seeds reproduce exactly.
- Speed: CNN TTA over all 173k wafers × 8 in 4.4 min; MFE-FNN 5 seeds 5 min; XGB config ~1 min. `run_cpu.sh` defaults to
  cores/4 threads per job so ~4 CPU jobs run side by side; the memory guard / detached-launch notes below were laptop issues.

**Python environments (old Windows laptop)**
- **GPU env (all training):** conda env `btp_lstm_gpu` — Python 3.10, **tensorflow 2.10.0** (last TF with native-Windows
  GPU; do not upgrade), conda-forge `cudatoolkit 11.2.2` + `cudnn 8.1.0.77`, numpy 1.26.4, scikit-learn 1.7.2,
  xgboost 2.1.4, scipy 1.15.3, pandas 2.3.3, matplotlib 3.10. Recreate with:
  `conda create -n btp_lstm_gpu -c conda-forge python=3.10 cudatoolkit=11.2 cudnn=8.1.0` then
  `pip install tensorflow==2.10.0 "numpy<2" scikit-learn xgboost scipy pandas matplotlib scikit-image`.
  Check: `sh run_gpu.sh -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`.
- `run_gpu.sh` / `run_cpu.sh` hard-code the env path `E=/d/Anaconda3/envs/btp_lstm_gpu` — **edit that one line** if
  conda lives elsewhere. They put `$E/Library/bin` (the CUDA/cuDNN DLLs) on PATH, which TF 2.10 needs.
- Base Anaconda Python (TF 2.21, CPU only) is fine for `watch_training.py`, `freeze_results.py` and quick analysis,
  but run all training through `run_gpu.sh` / `run_cpu.sh` so every result comes from the same TF.
- Old hard-coded paths: `training_pipeline.ipynb` and `phase2_data.py` use `d:\assignment college\VLSI_PROJECT\...`;
  `logs/cnn_queue*.sh` use `/d/assignment college/VLSI_PROJECT`. The new scripts (`common.py` & co.) use paths
  relative to the project folder and need no edits.

**Stage 2 scripts** (on top of the Stage 1 toolkit below): `xgb_tune.py` (XGB base learner, fit A / early-stop B_fit),
`compare.py X Y [test=1]` (paired t-test + wafer bootstrap), `calibrate.py NAME` (temperature + reject option),
`mlr_weights.py`, `gradcam.py`, `shap_xgb.py`, `extensions_results.py` (rebuilds `extensions_results.md`), `cnn_train.py --tta`.
Figures in `figures/`.

**Scripts (Stage 1 toolkit)**

| Script | What it does |
|---|---|
| `common.py` | splits A / B_fit / B_val, class weights, per-class F1, GPU setup, load/dump helpers |
| `stack_tune.py NAME key=value…` | stacker trials (FNN or `meta=mlr`), modes A/B/AB, `base_seeds`, `final=1` for test; appends `logs/stack_trials.jsonl` |
| `mfe_tune.py NAME key=value…` | MFE+FNN trials (fit A, early-stop B_fit, score B_val), saves every seed's outputs; `logs/mfe_trials.jsonl` |
| `cnn_train.py --tag T [--imagenet] [--cw 0] [--aug] [--plateau] [--fp32] [--batch 32]` | VGG16 CNN on the GPU; index-gather input pipeline (~2 GB RAM); `logs/cnn_T.log/csv` |
| `freeze_results.py` | rebuilds `results_comparison.md` from saved outputs (no training) |
| `watch_training.py` | live terminal view of CNN epochs + trial summaries (`python watch_training.py`) |

**Operational lessons (hardware: laptop, RTX 3050 6 GB, 15 GB RAM, 16 threads)**
- **CNN: always fp32, batch 32** (fp16/batch 128 cost ~4 pts). ~240–260 s/epoch, ~2–3 h per run with patience 20.
- **Claude Code's memory guard kills background shells** when RAM gets low (it happened 3×). Long GPU jobs must be
  launched **detached** (PowerShell `Start-Process` on Git `bash.exe` running a queue script — see
  `logs/cnn_queue5.sh`); the user approved this. Only one CPU tuning job at a time next to a CNN run.
- The small models (stacker, MFE-FNN) train on CPU (`run_cpu.sh`): they are faster there and leave the GPU to the CNN.
- Early stopping everywhere = lowest validation loss, patience 20, restore best weights (as in the paper's repo).

---

## 9. Working preferences learned in the 2026-10-06/07 sessions (override older text where they differ)

- **Technical tuning stays autonomous** (§0, §7 still apply): make the defensible choice, log it, keep going.
- **Plan-level decisions go to the user:** the Stage 2 plan was built with the user via the **grilling skill**
  (`mattpocock-skills:grilling`, from the `mattpocock-skills` plugin, enabled in `.claude/settings.json`). Do the same
  for any future plan (e.g. the Part D report structure): ask in numbered rounds with a recommended answer each;
  the user typically answers "go with your recommendations" but wants to be asked.
- **Wait for the user's sign-off on `stage2_plan.md` before implementing.**
- **GPU at full load** for training; the user likes to watch progress live (`watch_training.py`).
- **Commits:** the user asks for commits with a full explanatory message of everything done; commit at phase
  boundaries (Stage 1 = commit `4bfadf7`, tag `stage1-frozen`). Push to the GitHub repo above.
- Give short progress updates while long jobs run; explain choices in plain language when asked
  (e.g. "why CPU for the small models", "how is the best epoch chosen" — both answered in §8).
