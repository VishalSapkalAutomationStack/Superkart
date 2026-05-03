from huggingface_hub import HfApi
import os

api = HfApi()

repo_id = "Vishal1986/Superkart"

api.upload_file(
    path_or_fileobj="final_model.pkl",
    path_in_repo="final_model.pkl",
    repo_id=repo_id,
    repo_type="space"
)

api.upload_file("app.py", "app.py", repo_id, repo_type="space")
api.upload_file("requirements.txt", "requirements.txt", repo_id, repo_type="space")
api.upload_file("Dockerfile", "Dockerfile", repo_id, repo_type="space")

print("Deployment completed!")
