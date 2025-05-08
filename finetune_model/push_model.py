from huggingface_hub import create_repo, upload_folder

create_repo("JesseLiu/llama32-1b-ddbaseline", exist_ok=True)
upload_folder(folder_path="./model_weight/llama32-1b-baseline-final", path_in_repo="", repo_id="JesseLiu/llama32-1b-ddbaseline")
