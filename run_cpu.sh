#!/bin/sh
# Same TF 2.10 env as run_gpu.sh, but CPU only (small nets, leaves the GPU to the CNN). THREADS limits TF threads.
E=/d/Anaconda3/envs/btp_lstm_gpu
cd "$(dirname "$0")"
PATH="$E:$E/Library/bin:$E/Scripts:$PATH" CUDA_VISIBLE_DEVICES=-1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 \
  TF_NUM_INTRAOP_THREADS=${THREADS:-2} TF_NUM_INTEROP_THREADS=1 exec "$E/python.exe" "$@"
