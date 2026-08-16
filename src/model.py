import torch
import math
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class PlayerDetectionModel(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Feature Extractor Backbone (ResNet18)
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)

        # Remove Average Pooling and FC layer (Keep up to layer4)
        # Input: [B, 3, 384, 640] -> Output Feature Map: [B, 512, 12, 20]
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Convolutional Detection Head
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Output 5 channels per grid cell: 1 confidence + 4 box parameters
            nn.Conv2d(128, 5, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        # Forward pass through backbone
        features = self.backbone(x)  # [Batch, 512, 12, 20]

        # Forward pass through head
        out = self.head(features)  # [Batch, 5, 12, 20]

        # Reshape output to [Batch, Grid_H, Grid_W, 5]
        out = out.permute(0, 2, 3, 1)

        # Separate confidence from bounding box offsets
        confidence = out[..., 0]
        if not self.training:
            confidence = torch.sigmoid(confidence)

        bbox_preds = out[..., 1:]

        return confidence, bbox_preds


class BallDetectionModel(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)

        # Keep up to layer2 (Indices 0 to 5)
        # Input: [B, 3, 384, 640] -> Output Feature Map: [B, 128, 48, 80]
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])

        # Convolutional Detection Head
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 5, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        # Forward pass through backbone
        features = self.backbone(x)  # [Batch, 128, 48, 80]

        # Forward pass through head
        out = self.head(features)  # [Batch, 5, 48, 80]

        # Reshape output to [Batch, Grid_H, Grid_W, 5]
        out = out.permute(0, 2, 3, 1)

        # Separate confidence from bounding box offsets
        confidence = out[..., 0]
        if not self.training:
            confidence = torch.sigmoid(confidence)

        bbox_preds = out[..., 1:]

        return confidence, bbox_preds


class DetectionLoss(nn.Module):
    def __init__(self, lambda_box=5.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.SmoothL1Loss(reduction="sum")
        self.lambda_box = lambda_box

    def forward(self, pred_conf, pred_boxes, target_conf, target_boxes):
        # Move target tensors to same device as predictions (CPU or CUDA)
        target_conf = target_conf.to(pred_conf.device)
        target_boxes = target_boxes.to(pred_boxes.device)

        # Confidence Loss across ALL 240 grid cells
        loss_conf = self.bce(pred_conf, target_conf)

        # Bounding Box Loss ONLY on cells containing a player (target_conf == 1.0)
        mask = target_conf == 1.0  # Boolean mask of shape [B, 12, 20]

        if mask.sum() > 0:
            loss_box = self.l1(pred_boxes[mask], target_boxes[mask]) / mask.sum()
        else:
            loss_box = torch.tensor(0.0, device=pred_conf.device)

        # Combined Total Loss
        total_loss = loss_conf + (self.lambda_box * loss_box)
        return total_loss, loss_conf, loss_box


class FocalDetectionLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, lambda_box=5.0):
        super().__init__()
        # Initialize BCE without reduction so we can apply the focal modulating factor
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.l1 = nn.SmoothL1Loss(reduction="sum")
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_box = lambda_box

    def forward(self, pred_conf, pred_boxes, target_conf, target_boxes):
        target_conf = target_conf.to(pred_conf.device)
        target_boxes = target_boxes.to(pred_boxes.device)

        # 1. Focal Loss for Confidence
        bce_loss = self.bce(pred_conf, target_conf)

        # Calculate p_t (probability of the correct class)
        # Using the property: pt = exp(-BCE)
        pt = torch.exp(-bce_loss)

        # Apply the modulating factor: alpha * (1 - pt)^gamma
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # Calculate final focal loss and mean across all cells
        focal_loss = focal_weight * bce_loss
        loss_conf = focal_loss.sum() / (
            pred_conf.size(0) * pred_conf.size(1) * pred_conf.size(2)
        )

        # 2. Bounding Box Loss (same as player model)
        mask = target_conf == 1.0

        if mask.sum() > 0:
            loss_box = self.l1(pred_boxes[mask], target_boxes[mask]) / mask.sum()
        else:
            loss_box = torch.tensor(0.0, device=pred_conf.device)

        # Combined Total Loss
        total_loss = loss_conf + (self.lambda_box * loss_box)
        return total_loss, loss_conf, loss_box


def assign_gaussian_target(target_conf, b, gy, gx, box_w, box_h, grid_h=12, grid_w=20):
    """
    Spreads positive confidence across neighboring cells for large boxes.
    """
    # Calculate radius based on box size (larger near-player gets wider Gaussian)
    sigma = max(1.0, min(box_w, box_h) / 64.0)
    radius = int(math.ceil(3 * sigma))

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = gy + dy, gx + dx
            if 0 <= ny < grid_h and 0 <= nx < grid_w:
                # Gaussian heat value based on distance from center
                value = math.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                target_conf[b, ny, nx] = max(target_conf[b, ny, nx].item(), value)


def build_player_targets(boxes_batch, grid_shape=(12, 20), img_size=(640, 384)):
    grid_h, grid_w = grid_shape
    img_w, img_h = img_size
    stride_w, stride_h = img_w / grid_w, img_h / grid_h

    batch_size = len(boxes_batch)
    target_conf = torch.zeros((batch_size, grid_h, grid_w), dtype=torch.float32)
    target_boxes = torch.zeros((batch_size, grid_h, grid_w, 4), dtype=torch.float32)

    for b in range(batch_size):
        for box in boxes_batch[b]:
            p_id, x1, y1, x2, y2 = (
                box.tolist() if isinstance(box, torch.Tensor) else box
            )

            xc, yc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            bw, bh = x2 - x1, y2 - y1

            gx = min(int(xc / stride_w), grid_w - 1)
            gy = min(int(yc / stride_h), grid_h - 1)

            # Spread target confidence using Gaussian heatmap
            assign_gaussian_target(target_conf, b, gy, gx, bw, bh, grid_h, grid_w)

            # Store box coordinate offsets at center cell
            dx = (xc / stride_w) - gx
            dy = (yc / stride_h) - gy
            target_boxes[b, gy, gx] = torch.tensor([dx, dy, bw / img_w, bh / img_h])

    return target_conf, target_boxes


def build_ball_targets(boxes_batch, grid_shape=(48, 80), img_size=(640, 384)):
    grid_h, grid_w = grid_shape
    img_w, img_h = img_size
    stride_w, stride_h = img_w / grid_w, img_h / grid_h

    batch_size = len(boxes_batch)
    target_conf = torch.zeros((batch_size, grid_h, grid_w), dtype=torch.float32)
    target_boxes = torch.zeros((batch_size, grid_h, grid_w, 4), dtype=torch.float32)

    for b in range(batch_size):
        for box in boxes_batch[b]:
            # BallData tuple has exactly 4 elements (no class_id)
            x1, y1, x2, y2 = box.tolist() if isinstance(box, torch.Tensor) else box

            xc, yc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            bw, bh = x2 - x1, y2 - y1

            gx = min(int(xc / stride_w), grid_w - 1)
            gy = min(int(yc / stride_h), grid_h - 1)

            # For the tiny ball, we use a much tighter Gaussian (sigma=0.5, radius=1)
            # This prevents the target from smearing across too many cells
            sigma = 0.5
            radius = 1

            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < grid_h and 0 <= nx < grid_w:
                        value = math.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                        target_conf[b, ny, nx] = max(
                            target_conf[b, ny, nx].item(), value
                        )

            # Store box coordinate offsets at center cell
            dx = (xc / stride_w) - gx
            dy = (yc / stride_h) - gy
            target_boxes[b, gy, gx] = torch.tensor([dx, dy, bw / img_w, bh / img_h])

    return target_conf, target_boxes
