CUDA_VISIBLE_DEVICES=0 nohup python sft_train.py --model "meta-llama/Llama-3.2-1B-Instruct" --task 'llama32-1b-instruct' --data_type 'ddinter' > ./logs/llama321b_ddinter.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python sft_train.py --model "meta-llama/Llama-3.2-1B-Instruct" --task 'llama32-1b-instruct' --data_type 'drugbank' > ./logs/llama321b_drugbank.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python sft_train.py --model "meta-llama/Llama-3.2-1B-Instruct" --task 'llama32-1b-instruct' --data_type 'pharmaDB' > ./logs/llama321b_pharmaDB.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python sft_train.py --model "meta-llama/Llama-3.2-3B-Instruct" --task 'llama32-3b-instruct' --data_type 'ddinter' > ./logs/llama323b_ddinter.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python sft_train.py --model "meta-llama/Llama-3.2-3B-Instruct" --task 'llama32-3b-instruct' --data_type 'drugbank' > ./logs/llama323b_drugbank.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python sft_train.py --model "meta-llama/Llama-3.2-3B-Instruct" --task 'llama32-3b-instruct' --data_type 'pharmaDB' > ./logs/llama323b_pharmaDB.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python sft_train.py --model "Qwen/Qwen2.5-3B-Instruct" --task 'qwen25-3b-instruct' --data_type 'ddinter' > ./logs/qwen253b_ddinter.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python sft_train.py --model "Qwen/Qwen2.5-3B-Instruct" --task 'qwen25-3b-instruct' --data_type 'drugbank' > ./logs/qwen253b_drugbank.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python sft_train.py --model "Qwen/Qwen2.5-3B-Instruct" --task 'qwen25-3b-instruct' --data_type 'pharmaDB' > ./logs/qwen253b_pharmaDB.log 2>&1 &
