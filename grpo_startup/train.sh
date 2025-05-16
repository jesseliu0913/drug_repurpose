CUDA_VISIBLE_DEVICES=0 nohup python train.py --model "Qwen/Qwen2.5-7B-Instruct" --task 'qwen25' --training_data 'kpath' > ./log/qwen25_kpath.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --model "Qwen/Qwen2.5-7B-Instruct" --task 'qwen25' --training_data 'pagerank' > ./log/qwen25_pagerank.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py --model "meta-llama/Llama-3.2-3B-Instruct" --task 'llama32-3b' --training_data 'kpath' --train_setting 'partial' > ./log/llama323b_kpath.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --model "meta-llama/Llama-3.2-3B-Instruct" --task 'llama32-3b' --training_data 'pagerank' --train_setting 'partial' > ./log/llama323b_pagerank.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --model "meta-llama/Llama-3.2-1B-Instruct" --task 'llama32-1b' --training_data 'kpath' --train_setting 'partial' > ./log/llama321b_kpath.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --model "meta-llama/Llama-3.2-1B-Instruct" --task 'llama32-1b' --training_data 'pagerank' --train_setting 'partial' > ./log/llama321b_pagerank.log 2>&1 &
