from utils import read_video, write_video
from model import PlayerDetectionModel
from decoder import decode_predictions
import config
import torch
import cv2


def run_inference(model_path: str, frame_bgr):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PlayerDetectionModel(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # Preprocess Frame
    h_orig, w_orig = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (640, 384))
    tensor_img = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor_img = tensor_img.to(device)

    # Predict
    with torch.no_grad():
        pred_conf, pred_boxes = model(tensor_img)
        # Squeeze batch dimension
        boxes, scores = decode_predictions(pred_conf[0], pred_boxes[0])

    # Scale boxes back to original video resolution
    scale_x = w_orig / config.FINAL_IMAGE_SIZE[0]
    scale_y = h_orig / config.FINAL_IMAGE_SIZE[1]

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.cpu().numpy()
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

    return frame_bgr


def main():
    input_video_path = "input/input_video.mp4"
    frames = read_video(input_video_path)

    output_frames = []
    completed = 0
    for frame in frames:
        out_frame = run_inference("output/tennis_player_detector.pth", frame)
        output_frames.append(out_frame)
        completed += 1

        print(f"Completed Inference on [{completed}/{len(frames)}] frames")

    write_video(output_frames, "output/output_video.mp4")


if __name__ == "__main__":
    main()
