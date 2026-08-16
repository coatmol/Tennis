import torch
import torchvision
import config


def decode_predictions(pred_conf, pred_boxes, conf_thresh=0.3, iou_thresh=0.5):
    """
    Extracts high-confidence boxes using Non-Maximum Suppression (NMS)
    and returns the top 2.
    """
    img_w, img_h = config.FINAL_IMAGE_SIZE
    grid_h, grid_w = pred_conf.shape[0], pred_conf.shape[1]
    stride_w, stride_h = img_w / grid_w, img_h / grid_h

    # Filter out low confidence predictions early
    mask = pred_conf > conf_thresh
    
    if mask.sum() == 0:
        return [], []

    # Get the indices of valid cells
    indices = torch.nonzero(mask)
    
    boxes = []
    scores = []
    
    for idx in indices:
        gy, gx = idx[0].item(), idx[1].item()
        val = pred_conf[gy, gx].item()
        
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

        boxes.append([x1, y1, x2, y2])
        scores.append(val)
        
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    
    # Apply Non-Maximum Suppression to remove overlapping boxes
    keep_idx = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_thresh)
    
    # Limit to top 2 players
    keep_idx = keep_idx[:2]
    
    final_boxes = boxes_tensor[keep_idx]
    final_scores = scores_tensor[keep_idx]
    
    return [b for b in final_boxes], final_scores.tolist()
