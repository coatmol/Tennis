import torch
import config


def decode_predictions(pred_conf, pred_boxes):
    """
    Extracts the top 2 highest-confidence boxes and returns them along with their scores.
    """
    img_w, img_h = config.FINAL_IMAGE_SIZE
    grid_h, grid_w = pred_conf.shape[0], pred_conf.shape[1]
    stride_w, stride_h = img_w / grid_w, img_h / grid_h

    boxes = []
    scores = []

    # Flatten the confidence map to find top 2 scores
    flat_conf = pred_conf.reshape(-1)
    top2_vals, top2_idx = torch.topk(flat_conf, 2)

    for i in range(2):
        idx = top2_idx[i].item()
        val = top2_vals[i].item()
        
        gy = idx // grid_w
        gx = idx % grid_w

        dx, dy, bw_norm, bh_norm = pred_boxes[gy, gx].tolist()

        # Convert back to pixel coordinates
        xc = (gx + dx) * stride_w
        yc = (gy + dy) * stride_h
        bw = bw_norm * img_w
        bh = bh_norm * img_h

        x1 = max(0, xc - bw / 2.0)
        y1 = max(0, yc - bh / 2.0)
        x2 = min(img_w, xc + bw / 2.0)
        y2 = min(img_h, yc + bh / 2.0)

        boxes.append(torch.tensor([x1, y1, x2, y2]))
        scores.append(val)

    return boxes, scores
