import os
import time
import torch
from torch.utils.data import DataLoader
from model import *
import config
import torchvision.transforms.v2 as v2


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=20,
    lr=None,
    device="cuda",
    is_ball_model=False,
):
    if lr is None:
        lr = config.LEARNING_RATE

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if is_ball_model:
        criterion = FocalDetectionLoss(
            alpha=0.25, gamma=2.0, lambda_box=config.LAMBDA_BOX
        )
        # GPU-accelerated augmentations
        gpu_transforms = v2.Compose([
            v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)], p=0.8),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            v2.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
        ]).to(device)
    else:
        criterion = DetectionLoss(lambda_box=config.LAMBDA_BOX)
        gpu_transforms = None

    print(f"Training on device: {device}")

    torch.backends.cudnn.benchmark = True
    scaler = torch.GradScaler("cuda") if str(device).startswith("cuda") else None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0.0

        for images, batch_boxes in train_loader:
            images = images.to(device)
            
            if gpu_transforms is not None:
                images = gpu_transforms(images)

            # Build Targets on the fly
            if is_ball_model:
                target_conf, target_boxes = build_ball_targets(
                    batch_boxes,
                    grid_shape=config.BALL_GRID_SHAPE,
                    img_size=config.FINAL_IMAGE_SIZE,
                )
            else:
                target_conf, target_boxes = build_player_targets(
                    batch_boxes,
                    grid_shape=config.PLAYER_GRID_SHAPE,
                    img_size=config.FINAL_IMAGE_SIZE,
                )

            # Forward Pass & Compute Loss with AMP
            with torch.autocast(device_type="cuda", enabled=scaler is not None):
                pred_conf, pred_boxes = model(images)
                loss, loss_conf, loss_box = criterion(
                    pred_conf, pred_boxes, target_conf, target_boxes
                )

            # Backward Pass & Optimization
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_train_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, batch_boxes in val_loader:
                images = images.to(device)
                pred_conf, pred_boxes = model(images)

                if is_ball_model:
                    target_conf, target_boxes = build_ball_targets(
                        batch_boxes,
                        grid_shape=config.BALL_GRID_SHAPE,
                        img_size=config.FINAL_IMAGE_SIZE,
                    )
                else:
                    target_conf, target_boxes = build_player_targets(
                        batch_boxes,
                        grid_shape=config.PLAYER_GRID_SHAPE,
                        img_size=config.FINAL_IMAGE_SIZE,
                    )

                loss, _, _ = criterion(pred_conf, pred_boxes, target_conf, target_boxes)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time Taken: {epoch_time:.2f}s"
        )

    # Save trained model weights
    save_path = (
        "output/tennis_ball_detector.pth"
        if is_ball_model
        else "output/tennis_player_detector.pth"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model weights successfully saved to {save_path}!")
