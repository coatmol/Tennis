import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

HF_PRIVATE_TOKEN = os.environ.get("HF_PRIVATE_TOKEN")
REPO = "Coatmol/Tennis"

if not HF_PRIVATE_TOKEN:
    raise ValueError("HF_PRIVATE_TOKEN not found in environment or .env file")

api = HfApi()

checkpoints = [
    ("output/tennis_ball_detector.pth", "tennis_ball_detector.pth"),
    ("output/tennis_player_detector.pth", "tennis_player_detector.pth"),
    ("output/yolo_balls/weights/best.pt", "yolo_ball_detector.pt")
]

print(f"Uploading checkpoints to Hugging Face repository {REPO}...")

for local_path, repo_path in checkpoints:
    if os.path.exists(local_path):
        print(f"Uploading {local_path}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=REPO,
            repo_type="dataset",
            token=HF_PRIVATE_TOKEN,
        )
        print(f"Successfully uploaded {repo_path}!")
    else:
        print(f"Warning: {local_path} does not exist. Skipping.")

print("Done.")
