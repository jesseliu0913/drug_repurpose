from huggingface_hub import create_repo, upload_folder

create_repo("JesseLiu/llama32-3b-cold", exist_ok=True)
upload_folder(folder_path="/playpen/jesse/drug_repurpose/grpo_startup/model_weights/llama32-3b-baseline-model/checkpoint-450", path_in_repo="", repo_id="JesseLiu/llama32-3b-cold")
