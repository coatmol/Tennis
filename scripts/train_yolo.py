import os
import shutil
import yaml
from ultralytics import YOLO

def main():
    base_dir = os.path.abspath("input/dataset/balls")
    data_dir = os.path.join(base_dir, "data")
    labels_dir = os.path.join(base_dir, "labels")
    
    # YOLO requires the annotation folder to be named 'labels', not 'data'.
    # We will simply symlink or copy it.
    if os.path.exists(data_dir) and not os.path.exists(labels_dir):
        print("Copying 'data' folder to 'labels' for YOLO compatibility...")
        shutil.copytree(data_dir, labels_dir)
        
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
        project="output",
        name="yolo_balls",
        device="0" # Use GPU
    )
    
    print("Training complete! Best weights saved to output/yolo_balls/weights/best.pt")

if __name__ == "__main__":
    main()
