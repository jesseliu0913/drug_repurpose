CUDA_VISIBLE_DEVICES=0 nohup python train.py --model "Qwen/Qwen2.5-7B-Instruct" --task 'qwen25' --training_data 'kpath' > ./log/qwen25_kpath.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --model "Qwen/Qwen2.5-7B-Instruct" --task 'qwen25' --training_data 'pagerank' > ./log/qwen25_pagerank.log 2>&1 &
