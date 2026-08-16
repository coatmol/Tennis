from huggingface_hub import snapshot_download

HF_PUBLIC_TOKEN = "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"
REPO = "Coatmol/Tennis"
HEADERS = {"Authorization": f"Bearer {HF_PUBLIC_TOKEN}"}

local_dataset_path = snapshot_download(
    repo_id=REPO,
    repo_type="dataset",
    token=HF_PUBLIC_TOKEN,
    local_dir="./input/dataset",
)

print(f"Dataset downloaded to: {local_dataset_path}")
