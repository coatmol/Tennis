from training import *
import torch

if __name__ == "__main__":
    # data = parse_players("input/dataset/players")
    # preview_sample(data)

    dummy_input = torch.randn(2, 3, 384, 640)  # Batch size 2
    model = PlayerDetectionModel(pretrained=True)
    conf, boxes = model(dummy_input)

    print(f"Confidence map shape: {conf.shape}")  # Expect: [2, 12, 20]
    print(f"BBox predictions shape: {boxes.shape}")  # Expect: [2, 12, 20, 4]
