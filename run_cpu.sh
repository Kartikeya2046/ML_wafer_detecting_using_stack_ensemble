#!/bin/sh
# Same env as run_gpu.sh, but CPU only (small nets, leaves the GPU to the CNN).
# THREADS limits TF threads per job (default: a quarter of the cores, so ~4 CPU jobs can run side by side).
cd "$(dirname "$0")"
E=${WAFER_ENV:-.venv}
[ -x "$E/bin/python" ] || E=${WAFER_ENV:-/d/Anaconda3/envs/btp_lstm_gpu}
N=$(nproc 2>/dev/null || echo 8)
export CUDA_VISIBLE_DEVICES=-1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 TF_USE_LEGACY_KERAS=1 \
  TF_NUM_INTRAOP_THREADS=${THREADS:-$(( N / 4 > 1 ? N / 4 : 2 ))} TF_NUM_INTEROP_THREADS=1
if [ -x "$E/bin/python" ]; then exec "$E/bin/python" "$@"; fi
PATH="$E:$E/Library/bin:$E/Scripts:$PATH" exec "$E/python.exe" "$@"
