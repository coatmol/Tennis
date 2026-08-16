import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2


class PlayerDataset(Dataset):
    def __init__(self, parsed_data: dict) -> None:
        super().__init__()
        self.keys = list(parsed_data.keys())
        self.data = parsed_data

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        items = self.data[key]

        # Format Image: [H, W, C] -> [C, H, W], normalized [0.0, 1.0]
        raw_img = items[0]
        img_tensor = torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255.0

        # Format Boxes: List of [id, x1, y1, x2, y2]
        boxes = []
        for player in items[1:]:
            boxes.append(
                [player.id, player.min_x, player.min_y, player.max_x, player.max_y]
            )

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 5), dtype=torch.float32)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)

        return img_tensor, boxes_tensor


def collate_fn(batch):
    """
    Combines individual items into a batch.
    Images stack into shape: [Batch_Size, 3, Height, Width]
    Boxes return as a tuple of variable-length Tensors.
    """
    images, boxes = zip(*batch)
    images_stacked = torch.stack(images, dim=0)
    return images_stacked, boxes


class BallDataset(Dataset):
    def __init__(self, parsed_data: dict, is_train: bool = False) -> None:
        super().__init__()
        self.keys = list(parsed_data.keys())
        self.data = parsed_data
        self.is_train = is_train

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        items = self.data[key]

        # Format Image: [H, W, C] -> [C, H, W], normalized [0.0, 1.0]
        raw_img = items[0]
        img_tensor = torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255.0

        # Format Boxes: List of [x1, y1, x2, y2]
        boxes = []
        for ball in items[1:]:
            boxes.append([ball.min_x, ball.min_y, ball.max_x, ball.max_y])
            
        # Hard Negative Mining: 20% of the time, physically erase the ball with a gray box
        # and delete its bounding box so the model is forced to look at empty courts!
        if self.is_train and len(boxes) > 0 and torch.rand(1).item() < 0.2:
            x1, y1, x2, y2 = boxes[0]
            # Add a small padding to completely wipe the ball
            pad = 10
            ex1 = max(0, int(x1) - pad)
            ey1 = max(0, int(y1) - pad)
            ex2 = min(640, int(x2) + pad)
            ey2 = min(384, int(y2) + pad)
            
            # Fill with gray
            img_tensor[:, ey1:ey2, ex1:ex2] = 0.5
            
            # Delete all boxes for this image
            boxes = []

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)

        return img_tensor, boxes_tensor
