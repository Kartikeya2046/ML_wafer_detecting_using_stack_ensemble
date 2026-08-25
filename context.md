# Project Context — Wafer Map Defect Pattern Classification (VLSI Semester Project)

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

**Phase 4/5 Limitation (Hardware Constraint):** Due to native Windows hardware acceleration constraints (TensorFlow GPU not supported >= 2.11), training the VGG16 CNN on the full 162,946 dataset would take multiple days on CPU. With explicit user approval, we faithfully reproduced the N=5000 benchmark directly studied in the paper instead (Table 7 in Kang & Kang 2021).

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
