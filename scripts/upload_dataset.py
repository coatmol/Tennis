import os
import shutil
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

HF_PRIVATE_TOKEN = os.environ.get("HF_PRIVATE_TOKEN")
REPO = "Coatmol/Tennis"

if not HF_PRIVATE_TOKEN:
    raise ValueError("HF_PRIVATE_TOKEN not found in environment or .env file")

dataset_dir = "input/dataset"
zip_base = "input/dataset_archive"
zip_path = f"{zip_base}.zip"

print(f"Zipping {dataset_dir} into {zip_path}...")
# shutil.make_archive automatically appends the format extension
shutil.make_archive(zip_base, "zip", dataset_dir)

print(f"Uploading {zip_path} to Hugging Face repository {REPO}...")
api = HfApi()
api.upload_file(
    path_or_fileobj=zip_path,
    path_in_repo="dataset.zip",
    repo_id=REPO,
    repo_type="dataset",
    token=HF_PRIVATE_TOKEN,
)

print("Upload complete! Cleaning up local zip file...")
os.remove(zip_path)
print("Done.")
