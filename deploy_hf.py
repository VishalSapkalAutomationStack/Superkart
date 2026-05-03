from huggingface_hub import HfApi
import os

api = HfApi()
repo_id = "Vishal1986/Superkart"
token = os.environ["HF_TOKEN"]

files = ["final_model.pkl", "app.py", "requirements.txt", "Dockerfile"]

for file in files:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id=repo_id,
        repo_type="space",
        token=token
    )

print("Deployment completed successfully!")
