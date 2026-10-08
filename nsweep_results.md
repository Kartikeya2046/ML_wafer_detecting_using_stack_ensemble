# Training-size sweep — ours vs paper Table 7 (macro-F1 on the 10,000 test wafers)

Built by `nsweep.py stack`. Ours: mean ± std over replicates (random training subsets, paper protocol); paper: 10 replications. N = 162,946 row of ours = Stage 1 frozen (`results_comparison.md`; stackers there use split A/B, see tuning_log.md).

| N | reps | MFE+FNN ours | paper | CNN ours | paper | Stacking-DT ours | paper | Stacking-FNN ours | paper | Stacking-MLR ours | paper |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 500 | 10 | 0.4621 ± 0.0599 | 0.5558 ± 0.0376 | 0.4899 ± 0.0875 | 0.4937 ± 0.0642 | 0.2680 ± 0.1399 | 0.5179 ± 0.0368 | 0.4352 ± 0.0318 | 0.5085 ± 0.0662 | 0.5140 ± 0.0779 | 0.5872 ± 0.0542 |
| 5,000 | 10 | 0.7286 ± 0.0233 | 0.6983 ± 0.0371 | 0.7880 ± 0.0411 | 0.6954 ± 0.0420 | 0.7394 ± 0.0414 | 0.7217 ± 0.0220 | 0.7977 ± 0.0328 | 0.7350 ± 0.0556 | 0.8037 ± 0.0341 | 0.7599 ± 0.0202 |
| 50,000 | 5 | 0.8366 ± 0.0053 | 0.8310 ± 0.0109 | 0.8829 ± 0.0054 | 0.8397 ± 0.0213 | 0.8533 ± 0.0099 | 0.8415 ± 0.0116 | 0.8858 ± 0.0031 | 0.8746 ± 0.0088 | 0.8758 ± 0.0079 | 0.8686 ± 0.0078 |
| 162,946 | 1–5 | 0.8572 ± 0.0062 | 0.8599 ± 0.0117 | 0.8777 | 0.8679 ± 0.0126 | 0.8782 ± 0.0065 | 0.8789 ± 0.0094 | 0.9041 ± 0.0028 (tuned) | 0.8991 ± 0.0096 | 0.8967 ± 0.0060 | 0.8949 ± 0.0121 |
