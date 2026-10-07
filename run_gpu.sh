#!/bin/sh
# Run a script in the TF 2.10 env (last TF with native-Windows GPU); its CUDA/cuDNN DLLs live in Library/bin.
E=/d/Anaconda3/envs/btp_lstm_gpu
cd "$(dirname "$0")"
PATH="$E:$E/Library/bin:$E/Scripts:$PATH" PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 exec "$E/python.exe" "$@"
