python phase3_mfe_fnn.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python phase4_cnn.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python phase5_stacking.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python phase6_compare.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
