from training import *
from model import *
import torch
import random
import config

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load ball dataset
    full_ball_data = parse_balls("input/dataset/balls")

    # Split keys into Train (80%) and Validation (20%)
    all_keys = list(full_ball_data.keys())
    random.seed(42)  # Seed for reproducible splits
    random.shuffle(all_keys)

    split_idx = int(len(all_keys) * 0.8)
    train_keys = all_keys[:split_idx]
    val_keys = all_keys[split_idx:]

    train_data = {k: full_ball_data[k] for k in train_keys}
    val_data = {k: full_ball_data[k] for k in val_keys}

    train_dataset = BallDataset(train_data, is_train=True)
    val_dataset = BallDataset(val_data, is_train=False)

    # Create PyTorch DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    model = BallDetectionModel(pretrained=True)

    train_model(
        model,
        train_loader,
        val_loader,
        epochs=config.TRAIN_EPOCHS,
        device=device,
        is_ball_model=True,
    )
