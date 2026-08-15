import torch
from torch.utils.data import Dataset, DataLoader


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

        # Format Boxes: List of [x1, y1, x2, y2]
        boxes = []
        for player in items[1:]:
            boxes.append([player.min_x, player.min_y, player.max_x, player.max_y])

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
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
