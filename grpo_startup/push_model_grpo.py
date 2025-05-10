import argparse
from huggingface_hub import create_repo, upload_folder

# Parse command line arguments
parser = argparse.ArgumentParser(description='Push model to HuggingFace Hub')
parser.add_argument('--repo_name', type=str, required=True, 
                    help='Repository name on HuggingFace Hub')
parser.add_argument('--model_path', type=str, required=True, 
                    help='Path to the model folder to push')
args = parser.parse_args()

create_repo(args.repo_name, exist_ok=True)
upload_folder(folder_path=args.model_path, path_in_repo="", repo_id=args.repo_name)
