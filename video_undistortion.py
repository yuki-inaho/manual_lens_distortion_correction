import cv2
import toml
import click
import numpy as np
from tqdm import trange
from pathlib import Path
from utils import make_output_dir
from functools import partial

SCRIPT_DIR = str(Path().parent)


def fisheye_undistort_rectify_map(k1, k2, fx, fy, cx, cy, width, height):
    camera_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    dist_coef = np.array([[k1, k2, 0.0, 0.0]])
    projection_camera_mat = cv2.getOptimalNewCameraMatrix(camera_mat, dist_coef, (width, height), 0)[0]
    return cv2.fisheye.initUndistortRectifyMap(camera_mat, dist_coef, np.eye(3), projection_camera_mat, (width, height), cv2.CV_16SC2)


@click.command()
@click.option("--input-video-path", "-i", default="{SCRIPT_DIR}/movie.mp4")
@click.option("--input-toml-path", "-t", default=f"{SCRIPT_DIR}/cfg/parameter.toml")
@click.option("--focal-length", "-f", default=900)
@click.option("--output-video-name", "-o", default="movie_undist.mp4")
@click.option("--input-image-width", "-iw", "image_width", default=1920)
@click.option("--input-image-height", "-ih",  "image_height", default=1080)
def main(input_video_path, input_toml_path, focal_length, output_video_name, image_width, image_height):
    reader = cv2.VideoCapture(input_video_path)

    n_flames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Number of Frame: {n_flames}")

    with open(input_toml_path, "r") as f:
        toml_dict = toml.load(f)
    k1 = toml_dict["k1"]
    k2 = toml_dict["k2"]

    map_u, map_v = fisheye_undistort_rectify_map(k1, k2, focal_length, focal_length, image_width // 2, image_height // 2, image_width, image_height)
    correction = partial(cv2.remap, map1=map_u, map2=map_v, interpolation=cv2.INTER_LINEAR)

    frame_rate = 30.0
    ret, frame = reader.read()
    size = (frame.shape[1], frame.shape[0])

    fmt = cv2.VideoWriter_fourcc("m", "p", "4", "v")

    output_video_path = str(Path(input_video_path).parent.joinpath(output_video_name))
    writer = cv2.VideoWriter(output_video_path, fmt, frame_rate, size)

    for i in trange(1, n_flames - 1):
        ret, frame = reader.read()
        if ret:
            writer.write(correction(frame))

    reader.release()
    writer.release()


if __name__ == "__main__":
    main()