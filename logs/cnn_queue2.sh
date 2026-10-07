# GPU queue after the memory-pressure kill: wait for aug, then imnet, then cw0 (batch 32 + mixed precision)
cd "/d/assignment college/VLSI_PROJECT"
until [ -f data/cnn_aug_outputs.pkl ] || grep -q Traceback logs/cnn_aug.log; do sleep 60; done
sleep 90
sh run_gpu.sh cnn_train.py --tag imnet --imagenet > logs/cnn_imnet.log 2>&1
sh run_gpu.sh cnn_train.py --tag cw0 --cw 0 > logs/cnn_cw0.log 2>&1
grep -a "B_val\|Traceback" logs/cnn_aug.log logs/cnn_imnet.log logs/cnn_cw0.log
