from huggingface_hub import create_repo, upload_folder

create_repo("JesseLiu/llama32-3b-kpath-baseline", exist_ok=True)
upload_folder(folder_path="/data1/tlc/pengjie/drug_repurpose/grpo_startup/model_weights/llama32-3b-kpath-finalbaseline", path_in_repo="", repo_id="JesseLiu/llama32-3b-kpath-baseline")