# Results Comparison — Stage 1 frozen (implementation_plan.md B5)

Test set: 10,000 wafers, untouched by any tuning decision (all selection on the training split B, see tuning_log.md).
Paper: Kang & Kang 2021, Table 7 / Table 8, N = 162,946, mean ± std over 10 replicates (different splits and seeds).
Ours: mean ± std over the runs listed in the n column (seeds of the stacker, or of the MFE-FNN for MLR, which is deterministic).

## Macro- and micro-F1

| Model | Ours: stage | n | **F1 macro** | F1 micro (= accuracy) | Paper F1 macro | Paper F1 micro |
|---|---|---|---|---|---|---|
| MFE+FNN | original (balanced cw) | 1 | 0.8147 | 0.9560 | 0.8599 ± 0.0117 | 0.9741 ± 0.0010 |
| MFE+FNN | tuned (no cw) | 3 | 0.8572 ± 0.0062 | 0.9735 ± 0.0003 | 0.8599 ± 0.0117 | 0.9741 ± 0.0010 |
| CNN | original (scratch, balanced cw) | 1 | 0.8496 | 0.9670 | 0.8679 ± 0.0126 | 0.9775 ± 0.0014 |
| CNN | tuned (ImageNet init, no cw, aug) | 1 | 0.8777 | 0.9790 | 0.8679 ± 0.0126 | 0.9775 ± 0.0014 |
| Stacking-FNN | before fix (no cw, = notebook/repo) | 5 | 0.6298 ± 0.0904 | 0.9688 ± 0.0034 | 0.8991 ± 0.0096 | 0.9808 ± 0.0010 |
| Stacking-FNN | after fix (balanced cw) — B1 baseline | 5 | 0.8402 ± 0.0393 | 0.9736 ± 0.0006 | 0.8991 ± 0.0096 | 0.9808 ± 0.0010 |
| Stacking-FNN | tuned (h64, honest B inputs, no cw) | 5 | 0.9041 ± 0.0028 | 0.9811 ± 0.0002 | 0.8991 ± 0.0096 | 0.9808 ± 0.0010 |
| Stacking-MLR | original base learners | 1 | 0.8688 | 0.9723 | 0.8949 ± 0.0121 | 0.9801 ± 0.0010 |
| Stacking-MLR | **tuned (frozen pipeline)** | 3 | 0.8967 ± 0.0060 | 0.9805 ± 0.0004 | 0.8949 ± 0.0121 | 0.9801 ± 0.0010 |

## Per-class F1 — tuned pipeline vs paper (Table 8, N = 162,946)

| Class | test n | MFE+FNN ours | paper | CNN ours | paper | Stack-FNN ours | paper | **Stack-MLR ours** | paper |
|---|---|---|---|---|---|---|---|---|---|
| Center | 248 | 0.9354 | 0.9323 | 0.9516 | 0.9326 | 0.9444 | 0.9455 | 0.9553 | 0.9424 |
| Donut | 32 | 0.8601 | 0.8615 | 0.8966 | 0.8760 | 0.8861 | 0.8843 | 0.8915 | 0.8969 |
| Edge-Loc | 300 | 0.7989 | 0.8051 | 0.8294 | 0.8301 | 0.8540 | 0.8562 | 0.8434 | 0.8492 |
| Edge-Ring | 560 | 0.9713 | 0.9667 | 0.9857 | 0.9794 | 0.9874 | 0.9824 | 0.9866 | 0.9823 |
| Loc | 208 | 0.7236 | 0.7406 | 0.7588 | 0.7715 | 0.7791 | 0.8059 | 0.7742 | 0.7950 |
| Random | 50 | 0.8599 | 0.8928 | 0.8602 | 0.8653 | 0.8800 | 0.9033 | 0.8821 | 0.8979 |
| Scratch | 69 | 0.6571 | 0.6451 | 0.8254 | 0.7737 | 0.8488 | 0.8083 | 0.8254 | 0.7928 |
| Near-full | 9 | 0.9191 | 0.9053 | 0.8000 | 0.7920 | 0.9647 | 0.9142 | 0.9191 | 0.9053 |
| None | 8524 | 0.9895 | 0.9899 | 0.9918 | 0.9909 | 0.9927 | 0.9923 | 0.9922 | 0.9918 |
| **Macro** | | **0.8572** | 0.8599 | **0.8777** | 0.8679 | **0.9041** | 0.8992 | **0.8967** | 0.8948 |

Near-full has only 9 test wafers: one wafer moves its F1 by ~0.1 and the macro-F1 by ~0.01.
Frozen files: `data/mfe_x3_cw0[_s1,_s2]_outputs.pkl`, `data/cnn_imnet_cw0_aug_outputs.pkl` (+ `models/cnn_imnet_cw0_aug.keras`), stacker outputs `data/stack_FINAL_stack_{mlr,fnn}_*.pkl`. Regenerate this file with `python freeze_results.py`.
