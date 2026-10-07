# GPU queue (detached from Claude Code, user-approved 2026-10-06): paper-exact fp32, batch 32, lr 1e-4
cd "/d/assignment college/VLSI_PROJECT"
sh run_gpu.sh cnn_train.py --tag imnet --imagenet --fp32 > logs/cnn_imnet.log 2>&1
sh run_gpu.sh cnn_train.py --tag cw0 --cw 0 --fp32 > logs/cnn_cw0.log 2>&1
echo "queue finished $(date)" >> logs/cnn_queue3.done
