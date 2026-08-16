from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os

load_dotenv()

HF_PUBLIC_TOKEN = os.environ.get("HF_PUBLIC_TOKEN")
REPO = "Coatmol/Tennis"
HEADERS = {"Authorization": f"Bearer {HF_PUBLIC_TOKEN}"}

local_dataset_path = snapshot_download(
    repo_id=REPO,
    repo_type="dataset",
    token=HF_PUBLIC_TOKEN,
    local_dir="./input/dataset",
)

print(f"Dataset downloaded to: {local_dataset_path}")
