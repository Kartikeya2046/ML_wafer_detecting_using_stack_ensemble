# Stage 2b GPU queue (detached): CNN seeds 3-4 after seeds 1-2 finish, then TTA predictions for seeds 1-4.
cd "$(dirname "$0")/.."
T="--imagenet --cw 0 --fp32 --aug"
wait_for() { while pgrep -f "cnn_train.py --tag $1 " >/dev/null; do sleep 30; done; }
wait_for imnet_cw0_aug_s1
sh run_gpu.sh cnn_train.py --tag imnet_cw0_aug_s3 $T --seed 3 > logs/cnn_imnet_cw0_aug_s3.log 2>&1 &
wait_for imnet_cw0_aug_s2
sh run_gpu.sh cnn_train.py --tag imnet_cw0_aug_s4 $T --seed 4 > logs/cnn_imnet_cw0_aug_s4.log 2>&1 &
for s in 1 2; do sh run_gpu.sh cnn_train.py --tag imnet_cw0_aug_s$s --tta --cw 0 > logs/cnn_imnet_cw0_aug_s${s}_tta.log 2>&1; done
wait
for s in 3 4; do sh run_gpu.sh cnn_train.py --tag imnet_cw0_aug_s$s --tta --cw 0 > logs/cnn_imnet_cw0_aug_s${s}_tta.log 2>&1; done
echo "queue finished $(date)" >> logs/cnn_queue_stage2b.done
