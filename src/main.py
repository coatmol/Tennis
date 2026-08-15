from utils import read_video, write_video


def main():
    input_video_path = "input/input_video.mp4"
    frames = read_video(input_video_path)

    write_video(frames, "output/output_video.mp4")


if __name__ == "__main__":
    main()
