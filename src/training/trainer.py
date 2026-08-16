import time
import torch
from torch.utils.data import DataLoader
from model import *


def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device="cuda"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = DetectionLoss(lambda_box=5.0)

    print(f"Training on device: {device}")

    torch.backends.cudnn.benchmark = True
    scaler = torch.GradScaler("cuda") if str(device).startswith("cuda") else None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0.0

        for images, batch_boxes in train_loader:
            images = images.to(device)

            # Build Targets on the fly
            target_conf, target_boxes = build_targets(
                batch_boxes, grid_shape=(12, 20), img_size=(640, 384)
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
                target_conf, target_boxes = build_targets(
                    batch_boxes, grid_shape=(12, 20), img_size=(640, 384)
                )
                loss, _, _ = criterion(pred_conf, pred_boxes, target_conf, target_boxes)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time Taken: {epoch_time:.2f}s"
        )

    # Save trained model weights
    torch.save(model.state_dict(), "output/tennis_player_detector.pth")
    print("Model weights successfully saved to tennis_player_detector.pth!")
