#!/bin/sh
# Run a script in the project's TF env on the GPU.
# Linux: the project venv (.venv, TF 2.17 + tf_keras, so tf.keras is Keras 2 as in the TF 2.10 runs; CUDA via pip).
# Windows (old laptop): the TF 2.10 conda env (last TF with native-Windows GPU); its CUDA/cuDNN DLLs live in Library/bin.
# Override the env location with WAFER_ENV=/path/to/env.
cd "$(dirname "$0")"
E=${WAFER_ENV:-.venv}
[ -x "$E/bin/python" ] || E=${WAFER_ENV:-/d/Anaconda3/envs/btp_lstm_gpu}
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 TF_USE_LEGACY_KERAS=1
if [ -x "$E/bin/python" ]; then exec "$E/bin/python" "$@"; fi
PATH="$E:$E/Library/bin:$E/Scripts:$PATH" exec "$E/python.exe" "$@"
