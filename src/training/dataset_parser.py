from pathlib import Path
from collections import namedtuple
import cv2
import config

PlayerData = namedtuple("PlayerData", ["id", "min_x", "min_y", "max_x", "max_y"])


def parse_players(path: str) -> dict[str, list]:
    images_path = f"{path}/images"
    data_path = f"{path}/data"

    images_folder = Path(images_path)
    data_folder = Path(data_path)

    ret: dict[str, list] = {}

    print("Parsing player images...")

    for image_file in [f for f in images_folder.iterdir() if f.is_file()]:
        img = cv2.imread(image_file.absolute())

        if img is None:
            raise FileNotFoundError(
                f"Could not read image from path: {image_file.absolute()}"
            )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, config.FINAL_IMAGE_SIZE)
        ret[image_file.stem] = [img]

    print("Parsing player data...")

    for data_file in [f for f in data_folder.iterdir() if f.is_file()]:
        with open(data_file.absolute(), "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                class_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])

                # Convert normalized YOLO format to target pixel bounding box
                x1 = (xc - w / 2.0) * config.FINAL_IMAGE_SIZE[0]
                y1 = (yc - h / 2.0) * config.FINAL_IMAGE_SIZE[1]
                x2 = (xc + w / 2.0) * config.FINAL_IMAGE_SIZE[0]
                y2 = (yc + h / 2.0) * config.FINAL_IMAGE_SIZE[1]

                ret[data_file.stem].append(PlayerData(class_id, x1, y1, x2, y2))

    return ret


def preview_sample(parsed_data: dict, sample_key: str | None = None):
    # Grab the first item if no key provided
    if sample_key is None:
        sample_key = next(iter(parsed_data))

    items = parsed_data[sample_key]
    img = items[0].copy()  # Get resized RGB image
    players = items[1:]  # Get PlayerData objects

    # Convert RGB back to BGR for OpenCV display
    display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for player in players:
        pt1 = (int(player.min_x), int(player.min_y))
        pt2 = (int(player.max_x), int(player.max_y))
        cv2.rectangle(display_img, pt1, pt2, (0, 255, 0), 2)

    cv2.imshow(f"Sample: {sample_key}", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
