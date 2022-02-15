import click
import cv2
import numpy as np
from tqdm import trange
from pathlib import Path
from utils import make_output_dir


@click.command()
@click.option("--input-video-path", "-i", default="./movie.mp4")
@click.option("--output-dir", "-o", default="images")
@click.option("--viewer-mode", "-v", is_flag=True)
@click.option("--resize-rate", "-r", default=1.0)
@click.option("--subsample-rate", "-sub", default=1.0)
def main(input_video_path, output_dir, viewer_mode, resize_rate, subsample_rate):
    output_dir_path = Path(output_dir)
    make_output_dir(output_dir_path, clean=True)
    cap = cv2.VideoCapture(input_video_path)

    n_flames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Number of (Raw) Frame: {n_flames}")

    index_list = np.arange(1, n_flames, int(1 / subsample_rate))
    n_flames_subsampled = len(index_list)
    print(f"Number of Frame: {n_flames_subsampled}")

    for i in trange(n_flames):
        ret, frame = cap.read()

        if i not in index_list:
            continue
        image_name = f"{i:0=3}.png"
        output_image_path = str(output_dir_path.joinpath(image_name))
        frame = frame if resize_rate == 1.0 else cv2.resize(frame, (int(frame.shape[1] * resize_rate), int(frame.shape[0] * resize_rate)))
        cv2.imwrite(output_image_path, frame)
        if not ret:
            break
        if viewer_mode:
            cv2.imshow("frame", frame)
        key = cv2.waitKey(1)

    cap.release()
    if viewer_mode:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()