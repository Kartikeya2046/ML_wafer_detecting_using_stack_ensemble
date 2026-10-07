# B4 last run: ImageNet init + no class weights, fp32 batch 32 (one factor vs imnet). Detached from Claude Code.
cd "/d/assignment college/VLSI_PROJECT"
sh run_gpu.sh cnn_train.py --tag imnet_cw0 --imagenet --cw 0 --fp32 > logs/cnn_imnet_cw0.log 2>&1
echo "queue finished $(date)" >> logs/cnn_queue4.done
