from pathlib import Path
from collections import namedtuple
import cv2
import os
import config

PlayerData = namedtuple("PlayerData", ["id", "min_x", "min_y", "max_x", "max_y"])


def parse(path: str) -> dict[str, list]:
    images_path = f"{path}/images"
    data_path = f"{path}/data"

    images_folder = Path(images_path)
    data_folder = Path(data_path)

    ret: dict[str, list] = {}

    for image_file in [f for f in images_folder.iterdir() if f.is_file()]:
        img = cv2.imread(image_file.absolute())

        if img is None:
            raise FileNotFoundError(
                f"Could not read image from path: {image_file.absolute()}"
            )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, config.FINAL_IMAGE_SIZE)
        ret[image_file.stem] = [img]

    for data_file in [f for f in data_folder.iterdir() if f.is_file()]:
        with open(data_file.absolute(), "r") as data:
            content = data.read()
            players = content.split("\n")
            p1 = players[0].split(" ")
            p2 = players[1].split(" ")

            p1xc = float(p1[1])
            p1yc = float(p1[2])
            p1w = float(p1[3])
            p1h = float(p1[4])

            p1x1 = (p1xc - p1w / 2.0) * config.FINAL_IMAGE_SIZE[0]
            p1y1 = (p1yc - p1h / 2.0) * config.FINAL_IMAGE_SIZE[1]
            p1x2 = (p1xc + p1w / 2.0) * config.FINAL_IMAGE_SIZE[0]
            p1y2 = (p1yc + p1h / 2.0) * config.FINAL_IMAGE_SIZE[1]

            p2xc = float(p2[1])
            p2yc = float(p2[2])
            p2w = float(p2[3])
            p2h = float(p2[4])

            p2x1 = (p2xc - p2w / 2.0) * config.FINAL_IMAGE_SIZE[0]
            p2y1 = (p2yc - p2h / 2.0) * config.FINAL_IMAGE_SIZE[1]
            p2x2 = (p2xc + p2w / 2.0) * config.FINAL_IMAGE_SIZE[0]
            p2y2 = (p2yc + p2h / 2.0) * config.FINAL_IMAGE_SIZE[1]

            ret[data_file.stem].append(PlayerData(1, p1x1, p1y1, p1x2, p1y2))
            ret[data_file.stem].append(PlayerData(2, p2x1, p2y1, p2x2, p2y2))

    return ret
