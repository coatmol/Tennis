from utils import read_video, write_video
from model import PlayerDetectionModel, BallDetectionModel
from decoder import decode_predictions
import config
import torch
import cv2
import os
import time
import argparse


def run_inference(player_model, ball_model, device, frame_bgr, last_ball_pos):
    # Preprocess Frame
    h_orig, w_orig = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (640, 384))
    tensor_img = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor_img = tensor_img.to(device)

    # Predict
    with torch.no_grad():
        pred_conf_player, pred_boxes_player = player_model(tensor_img)
        pred_conf_ball, pred_boxes_ball = ball_model(tensor_img)
        
        # Squeeze batch dimension and decode
        p_boxes, p_scores = decode_predictions(
            pred_conf_player[0], pred_boxes_player[0], 
            conf_thresh=config.PLAYER_CONF_THRESH, max_detections=2
        )
        # Decode top 5 balls so we don't accidentally throw away the real ball if a shoe scores slightly higher!
        b_boxes_all, b_scores_all = decode_predictions(
            pred_conf_ball[0], pred_boxes_ball[0], 
            conf_thresh=config.BALL_CONF_THRESH, max_detections=5
        )

    # Scale boxes back to original video resolution
    scale_x = w_orig / config.FINAL_IMAGE_SIZE[0]
    scale_y = h_orig / config.FINAL_IMAGE_SIZE[1]

    valid_p_boxes = []
    
    # Draw Players (Green)
    for box, score in zip(p_boxes, p_scores):
        if score < config.PLAYER_CONF_THRESH:
            continue

        x1, y1, x2, y2 = box.cpu().numpy()

        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        norm_xc = max(0.0, min(1.0, xc / config.FINAL_IMAGE_SIZE[0]))
        norm_yc = max(0.0, min(1.0, yc / config.FINAL_IMAGE_SIZE[1]))
        norm_w = max(0.0, min(1.0, w / config.FINAL_IMAGE_SIZE[0]))
        norm_h = max(0.0, min(1.0, h / config.FINAL_IMAGE_SIZE[1]))

        valid_p_boxes.append((norm_xc, norm_yc, norm_w, norm_h))

        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_bgr,
            f"Player {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Filter and Track the Ball
    valid_b_boxes = []
    best_ball_idx = -1
    best_ball_dist = float('inf')

    # Find the best valid ball
    for i, (box, score) in enumerate(zip(b_boxes_all, b_scores_all)):
        if score < config.BALL_CONF_THRESH:
            continue

        x1, y1, x2, y2 = box.cpu().numpy()
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        
        # 1. SHOE FILTER
        is_shoe = False
        for px1, py1, px2, py2 in p_boxes:
            px1, py1, px2, py2 = px1.item(), py1.item(), px2.item(), py2.item()
            player_height = py2 - py1
            bottom_30_y = py1 + (player_height * 0.70)
            
            # Allow some margin around the player horizontally
            margin = (px2 - px1) * 0.2
            if (px1 - margin) <= xc <= (px2 + margin) and bottom_30_y <= yc <= (py2 + margin):
                is_shoe = True
                break
                
        if is_shoe:
            continue
            
        # 2. TRACKER: Distance Heuristic
        # If we have a last known position, prefer the ball that is physically closest to it
        if last_ball_pos is not None:
            dist = ((xc - last_ball_pos[0]) ** 2 + (yc - last_ball_pos[1]) ** 2) ** 0.5
            # A tennis ball won't travel more than ~150 normalized pixels in 1 frame
            if dist > 150:
                continue # Reject massive teleportations (like white court lines across the pitch)
            
            if dist < best_ball_dist:
                best_ball_dist = dist
                best_ball_idx = i
        else:
            # If no history, just take the highest confidence one that isn't a shoe
            best_ball_idx = i
            break
            
    current_ball_pos = last_ball_pos

    # If we found a valid ball, draw it
    if best_ball_idx != -1:
        box = b_boxes_all[best_ball_idx]
        score = b_scores_all[best_ball_idx]
        x1, y1, x2, y2 = box.cpu().numpy()
        
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        
        current_ball_pos = (xc, yc) # Update history

        norm_xc = max(0.0, min(1.0, xc / config.FINAL_IMAGE_SIZE[0]))
        norm_yc = max(0.0, min(1.0, yc / config.FINAL_IMAGE_SIZE[1]))
        norm_w = max(0.0, min(1.0, w / config.FINAL_IMAGE_SIZE[0]))
        norm_h = max(0.0, min(1.0, h / config.FINAL_IMAGE_SIZE[1]))

        valid_b_boxes.append((norm_xc, norm_yc, norm_w, norm_h))

        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame_bgr,
            f"Ball {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

    return frame_bgr, valid_p_boxes, valid_b_boxes, current_ball_pos


def main():
    parser = argparse.ArgumentParser(
        description="Run tennis player & ball detection inference."
    )
    parser.add_argument(
        "--save-dataset",
        action="store_true",
        help="Save frames and labels when players or ball are detected",
    )
    parser.add_argument(
        "--rm", action="store_true", help="Remove previous saved output dataset"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Player Model
    player_model = PlayerDetectionModel(pretrained=False)
    player_model.load_state_dict(
        torch.load("output/tennis_player_detector.pth", map_location=device, weights_only=True)
    )
    player_model.to(device).eval()

    # Load Ball Model
    ball_model = BallDetectionModel(pretrained=False)
    if os.path.exists("output/tennis_ball_detector.pth"):
        ball_model.load_state_dict(
            torch.load("output/tennis_ball_detector.pth", map_location=device, weights_only=True)
        )
    ball_model.to(device).eval()

    input_video_path = "input/input_video.mp4"
    frames = read_video(input_video_path)

    if args.rm:
        import shutil

        if os.path.exists("output/dataset"):
            shutil.rmtree("output/dataset")

    if args.save_dataset:
        os.makedirs("output/dataset/players/images", exist_ok=True)
        os.makedirs("output/dataset/players/data", exist_ok=True)
        os.makedirs("output/dataset/balls/images", exist_ok=True)
        os.makedirs("output/dataset/balls/data", exist_ok=True)

    run_timestamp = int(time.time())
    output_frames = []
    completed = 0
    last_ball_pos = None

    for frame_id, frame in enumerate(frames):
        # Keep a clean copy for the dataset so we don't save drawn boxes
        clean_frame = frame.copy()

        out_frame, valid_p_boxes, valid_b_boxes, last_ball_pos = run_inference(
            player_model, ball_model, device, frame, last_ball_pos
        )

        if args.save_dataset:
            # Save Players Data
            if len(valid_p_boxes) == 2:
                img_path = f"output/dataset/players/images/{run_timestamp}-{frame_id}.jpg"
                txt_path = f"output/dataset/players/data/{run_timestamp}-{frame_id}.txt"
                cv2.imwrite(img_path, clean_frame)
                with open(txt_path, "w") as f:
                    for b in valid_p_boxes:
                        f.write(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
            
            # Save Ball Data
            if len(valid_b_boxes) == 1:
                img_path = f"output/dataset/balls/images/{run_timestamp}-{frame_id}.jpg"
                txt_path = f"output/dataset/balls/data/{run_timestamp}-{frame_id}.txt"
                # If we didn't already save the image for the players, save it now
                if not os.path.exists(img_path):
                    cv2.imwrite(img_path, clean_frame)
                with open(txt_path, "w") as f:
                    for b in valid_b_boxes:
                        f.write(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")

        output_frames.append(out_frame)
        completed += 1

        if completed % 10 == 0:
            print(f"Completed Inference on [{completed}/{len(frames)}] frames")

    write_video(output_frames, "output/output_video.mp4")


if __name__ == "__main__":
    main()
