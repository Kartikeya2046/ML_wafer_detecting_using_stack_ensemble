# Training-size sweep GPU queue (detached): small-N CNN runs now (2 at a time), N=50000 after the Stage 2b queue.
cd "$(dirname "$0")/.."
T="--imagenet --cw 0 --fp32 --aug"
for n in 500 5000; do for r in 0 1 2 3 4 5 6 7 8 9; do echo "$n $r"; done; done | \
  xargs -P 2 -L 1 sh -c 'T="--imagenet --cw 0 --fp32 --aug"; sh run_gpu.sh cnn_train.py --tag ns$0_r$1 --subset $0 --rep $1 --seed $1 $T > logs/cnn_ns$0_r$1.log 2>&1'
while [ ! -f logs/cnn_queue_stage2b.done ]; do sleep 60; done
for r in 0 1 2 3 4; do echo "50000 $r"; done | \
  xargs -P 3 -L 1 sh -c 'T="--imagenet --cw 0 --fp32 --aug"; sh run_gpu.sh cnn_train.py --tag ns$0_r$1 --subset $0 --rep $1 --seed $1 $T > logs/cnn_ns$0_r$1.log 2>&1'
echo "queue finished $(date)" >> logs/cnn_queue_nsweep.done
