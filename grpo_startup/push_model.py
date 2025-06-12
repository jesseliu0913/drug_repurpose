from huggingface_hub import create_repo, upload_folder

create_repo("JesseLiu/qwen25-3b-pagerank-naive", exist_ok=True)
upload_folder(folder_path="/playpen/jesse/drug_repurpose/grpo_startup/model_weights/qwen25-3b-pagerank-finalnaive", path_in_repo="", repo_id="JesseLiu/qwen25-3b-pagerank-naive")