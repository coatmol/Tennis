import os
import zipfile
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

HF_PUBLIC_TOKEN = os.environ.get("HF_PUBLIC_TOKEN")
REPO = "Coatmol/Tennis"

print("Downloading dataset.zip from Hugging Face...")
local_zip_path = hf_hub_download(
    repo_id=REPO,
    filename="dataset.zip",
    repo_type="dataset",
    token=HF_PUBLIC_TOKEN,
    local_dir="./input",
)

extract_path = "./input/dataset"
print(f"Extracting dataset to {extract_path}...")
os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Cleaning up local zip file...")
os.remove(local_zip_path)

print("Done.")
