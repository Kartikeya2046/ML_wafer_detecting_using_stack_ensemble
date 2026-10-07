# B4 final round on top of imnet_cw0 (fp32, batch 32). Detached from Claude Code.
cd "/d/assignment college/VLSI_PROJECT"
sh run_gpu.sh cnn_train.py --tag imnet_cw0_aug --imagenet --cw 0 --fp32 --aug > logs/cnn_imnet_cw0_aug.log 2>&1
sh run_gpu.sh cnn_train.py --tag imnet_cw0_plat --imagenet --cw 0 --fp32 --plateau > logs/cnn_imnet_cw0_plat.log 2>&1
echo "queue finished $(date)" >> logs/cnn_queue5.done
