import torch
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
        confidence = torch.sigmoid(out[..., 0])
        bbox_preds = out[..., 1:]

        return confidence, bbox_preds


class DetectionLoss(nn.Module):
    def __init__(self, lambda_box=5.0):
        super().__init__()
        self.bce = nn.BCELoss()
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


def build_targets(boxes_batch, grid_shape=(12, 20), img_size=(640, 384)):
    """
    Converts list of variable ground-truth boxes per image into
    dense target tensors matching model output shapes.
    """
    grid_h, grid_w = grid_shape
    img_w, img_h = img_size
    stride_w = img_w / grid_w  # 32 pixels
    stride_h = img_h / grid_h  # 32 pixels

    batch_size = len(boxes_batch)
    target_conf = torch.zeros((batch_size, grid_h, grid_w), dtype=torch.float32)
    target_boxes = torch.zeros((batch_size, grid_h, grid_w, 4), dtype=torch.float32)

    for b in range(batch_size):
        boxes = boxes_batch[b]  # Shape: [N_players, 4] -> [x1, y1, x2, y2]
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()

            # Compute center point and box dimensions
            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0
            bw = x2 - x1
            bh = y2 - y1

            # Find grid cell index
            gx = min(int(xc / stride_w), grid_w - 1)
            gy = min(int(yc / stride_h), grid_h - 1)

            # Calculate cell offsets (0.0 to 1.0) and normalized width/height
            dx = (xc / stride_w) - gx
            dy = (yc / stride_h) - gy
            w_norm = bw / img_w
            h_norm = bh / img_h

            # Assign to targets
            target_conf[b, gy, gx] = 1.0
            target_boxes[b, gy, gx] = torch.tensor([dx, dy, w_norm, h_norm])

    return target_conf, target_boxes
