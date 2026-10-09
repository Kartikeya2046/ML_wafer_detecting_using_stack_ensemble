# Files Kept Only on the Local Machine (not pushed to GitHub)

Generated 2026-10-09 from `git ls-files --others --ignored --exclude-standard`. These files exist in the working
copy on the Linux workstation but are excluded by `.gitignore`. Most are larger than GitHub's 100 MB file limit or can
be regenerated. **Back up group 3 (trained CNN weights) separately** (USB / cloud drive): it is the only group that
costs GPU hours to recreate and is needed to run `predict.py`.

Total local-only size: about 10.6 GB (5.7 GB of it is the Python environment).

## 1. Raw dataset — download, do not regenerate
| File | Size | Why not in git | How to get it back |
|---|---|---|---|
| `LSWMD.pkl/LSWMD.pkl` | 2.0 GB | > 100 MB; third-party dataset (WM-811K) | Kaggle `qingyi/wm811k-wafer-map` (login needed) |
| `LSWMD.pkl.zip` | 150 MB | > 100 MB; same dataset, compressed | same |

## 2. Preprocessed data
| File | Size | Why not in git | How to get it back |
|---|---|---|---|
| `data/X_CNN.pkl` | 1.4 GB | > 100 MB | `sh run_cpu.sh phase2_data.py` (~1 h, needs group 1) |

(`data/X_MFE.pkl` and `data/y.pkl` are small and **are** in git.)

## 3. Trained neural-network weights — needed by `predict.py`, `cnn_train.py --tta`, `gradcam.py`
| File | Size | Used for | How to get it back (≈2 h each on one GPU) |
|---|---|---|---|
| `models/cnn_imnet_cw0_aug.keras` | 169 MB | CNN seed 0 (Stage 1 frozen; final pipeline) | `sh run_gpu.sh cnn_train.py --tag imnet_cw0_aug --imagenet --cw 0 --fp32 --aug --seed 0` |
| `models/cnn_imnet_cw0_aug_s1.keras` | 169 MB | CNN seed 1 (final pipeline) | same with `--tag imnet_cw0_aug_s1 --seed 1` |
| `models/cnn_imnet_cw0_aug_s2.keras` | 169 MB | CNN seed 2 (final pipeline) | `--tag imnet_cw0_aug_s2 --seed 2` |
| `models/cnn_imnet_cw0_aug_s3.keras` | 169 MB | CNN seed 3 (final pipeline) | `--tag imnet_cw0_aug_s3 --seed 3` |
| `models/cnn_imnet_cw0_aug_s4.keras` | 169 MB | CNN seed 4 (final pipeline) | `--tag imnet_cw0_aug_s4 --seed 4` |
| `models/cnn_multinn.keras` | 169 MB | MultiNN baseline | `--tag multinn --multinn --imagenet --cw 0 --fp32 --aug` |

A retrained CNN will not be bit-identical (GPU nondeterminism), so results would move slightly. Their saved predictions
(`data/cnn_*_outputs.pkl`, `data/cnn_*_tta_outputs.pkl`) **are** in git, so every table in the paper can be rebuilt
without these weights. The final XGBoost model (`models/xgb_d6_lr05_cw1.json`, 44 MB) and the MLR / calibration file
(`models/final_pipeline.npz`) **are** in git.

## 4. Rejected or intermediate experiment files — not needed (scores are in `logs/`)
| Files | Count | Size | What they are |
|---|---|---|---|
| `models/xgb_d6_lr05_cw1_bag_s0..4.json` | 5 | 210 MB | XGBoost bagging models (rejected, `tuning_log.md` Stage 2b) |
| `data/xgb_d6_lr05_cw1_bag_s0..4_outputs.pkl`, `data/xgb_bag5_outputs.pkl` | 6 | 36 MB | their predictions |
| `data/xgb_d{4,6,8}_lr{05,1}_cw{0,1}_outputs.pkl` (all but the selected `d6_lr05_cw1`) | 11 | 66 MB | XGBoost grid predictions; scores in `logs/xgb_trials.jsonl` |
| `data/stack_cv_S3sel_*_b*_s0.pkl` | 60 | 72 MB | stacks used to select the XGBoost config; scores in `logs/stack_trials.jsonl` |
| `data/stack_cv_e3_*_b*_s0.pkl`, `data/cnn_ens3_tta_outputs.pkl` | 12 | 19 MB | preliminary 3-CNN ensemble check (superseded by the 5-CNN ensemble) |
| `data/final_predictions.pkl` | 1 | 0.8 MB | output of `predict.py` (re-created on every run) |

All regenerate with the commands logged in `tuning_log.md`.

## 5. Environment and build tools
| Path | Size | How to get it back |
|---|---|---|
| `.venv/` | 5.7 GB | `sh setup_env.sh` |
| `tools/` (Tectonic 0.15.0 LaTeX engine) | 36 MB | release archive `tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz` from github.com/tectonic-typesetting/tectonic |
| `__pycache__/`, `paper/*.blg` and other LaTeX by-products | < 1 MB | created automatically |
