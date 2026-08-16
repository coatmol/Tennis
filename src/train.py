from training import *
from model import *
import torch
import random

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = parse_players("input/dataset/players")

    # Split keys into Train (80%) and Validation (20%)
    all_keys = list(data.keys())
    random.seed(42)  # Seed for reproducible splits
    random.shuffle(all_keys)

    split_idx = int(len(all_keys) * 0.8)
    train_keys = all_keys[:split_idx]
    val_keys = all_keys[split_idx:]

    train_data = {k: data[k] for k in train_keys}
    val_data = {k: data[k] for k in val_keys}

    # Wrap in your PyTorch Dataset class
    train_dataset = PlayerDataset(train_data)
    val_dataset = PlayerDataset(val_data)

    # Create PyTorch DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,  # Adjust based on GPU VRAM
        shuffle=True,  # Shuffle every epoch for training
        collate_fn=collate_fn,  # Handles variable number of player boxes per image
        num_workers=2,  # Parallel data loading
        pin_memory=True,  # Faster CPU-to-GPU tensor transfers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,  # No need to shuffle validation data
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PlayerDetectionModel(pretrained=True)

    train_model(model, train_loader, val_loader, 20, device=device)
