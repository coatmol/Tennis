import os
import requests

HF_PUBLIC_TOKEN = "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"
REPO = "Coatmol/Tennis"
HEADERS = {"Authorization": f"Bearer {HF_PUBLIC_TOKEN}"}

tree_url = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true"
response = requests.get(tree_url, headers=HEADERS)
files = response.json()

for item in files:
    if item["type"] == "file":
        file_path = item["path"]
        download_url = (
            f"https://huggingface.co/datasets/{REPO}/resolve/main/{file_path}"
        )

        os.makedirs(
            os.path.dirname(os.path.join("./input/dataset", file_path)), exist_ok=True
        )

        # Stream download
        r = requests.get(download_url, headers=HEADERS, stream=True)
        with open(os.path.join("./data", file_path), "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
