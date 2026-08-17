import os
import shutil
import yaml
from ultralytics import YOLO

def main():
    base_dir = os.path.abspath("input/dataset/balls")
    
    yaml_path = "balls_dataset.yaml"
    yaml_data = {
        'path': base_dir,
        'train': 'images',
        'val': 'images', # Use same for validation since we didn't pre-split
        'names': {0: 'ball'}
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)
        
    print("Starting YOLOv8 Nano training...")
    model = YOLO("yolov8n.pt")
    
    # Train
    results = model.train(
        data=yaml_path,
        epochs=40,
        imgsz=640,
        project=os.path.abspath("output"),
        name="yolo_balls",
        exist_ok=True, # Prevent it from making yolo_balls-2, yolo_balls-3, etc.
        device="0", # Use GPU
        workers=0   # Disable multiprocessing to prevent Windows memory/paging file crashes
    )
    
    # Automatically copy the best weights to the root output folder
    best_weights_path = os.path.join(results.save_dir, "weights", "best.pt")
    final_weights_path = os.path.abspath(os.path.join("output", "yolo_ball_detector.pt"))
    
    if os.path.exists(best_weights_path):
        shutil.copy(best_weights_path, final_weights_path)
        print(f"Training complete! Best weights automatically copied to {final_weights_path} for inference.")
    else:
        print("Training complete, but could not find best weights to copy.")

if __name__ == "__main__":
    main()
