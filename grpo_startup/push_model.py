from huggingface_hub import create_repo, upload_folder

create_repo("JesseLiu/llama32-1b-pagerank-partial-fullname", exist_ok=True)
upload_folder(folder_path="/playpen/jesse/drug_repurpose/grpo_startup/model_weights/llama32-1b-pagerank-partial-fullname-final", path_in_repo="", repo_id="JesseLiu/llama32-1b-pagerank-partial-fullname")