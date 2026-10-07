cd "/d/assignment college/VLSI_PROJECT"
sh run_gpu.sh cnn_train.py --tag ctrl > logs/cnn_ctrl.log 2>&1
sh run_gpu.sh cnn_train.py --tag aug --aug > logs/cnn_aug.log 2>&1
