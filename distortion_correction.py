import cv2
import toml
import click
import numpy as np
from pathlib import Path
from functools import partial
from tqdm import tqdm

SCRIPT_DIR = str(Path().parent)
WINDOW_NAME = "Distortion Correction"


def fisheye_undistort_rectify_map(k1, k2, fx, fy, cx, cy, width, height):
    camera_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    dist_coef = np.array([[k1, k2, 0.0, 0.0]])
    projection_camera_mat = cv2.getOptimalNewCameraMatrix(camera_mat, dist_coef, (width, height), 0)[0]
    return cv2.fisheye.initUndistortRectifyMap(camera_mat, dist_coef, np.eye(3), projection_camera_mat, (width, height), cv2.CV_16SC2)


@click.command()
@click.option("--input-image-dir", "-i", "input_image_dir_path", default=f"{SCRIPT_DIR}/image")
@click.option("--intput-toml", "-t", "input_toml_path", default=f"{SCRIPT_DIR}/parameter.toml")
@click.option("--output-image-dir", "-o", "output_image_dir_path", default=f"{SCRIPT_DIR}/output")
@click.option("--focal-length", "-f", default=900)
@click.option("--file-extention", "-ext", default=".jpg")
def main(input_image_dir_path, input_toml_path, output_image_dir_path, focal_length, file_extention):
    input_image_dir_pathlib = Path(input_image_dir_path)
    input_image_path_list = [str(path) for path in input_image_dir_pathlib.glob("*") if path.suffix in [".png", ".jpg"]]
    output_image_dir_pathlib = Path(output_image_dir_path)
    output_image_dir_pathlib.mkdir(exist_ok=True)

    with open(input_toml_path, "r") as f:
        toml_dict = toml.load(f)
    k1 = toml_dict["k1"]
    k2 = toml_dict["k2"]
    image_height, image_width, _ = cv2.imread(input_image_path_list[0]).shape
    map_u, map_v = fisheye_undistort_rectify_map(k1, k2, focal_length, focal_length, image_width // 2, image_height // 2, image_width, image_height)
    correction = partial(cv2.remap, map1=map_u, map2=map_v, interpolation=cv2.INTER_LINEAR)

    for input_image_path in tqdm(input_image_path_list):
        image = cv2.imread(input_image_path)
        image_undistorted = correction(image)

        output_image_name = Path(input_image_path).with_suffix(file_extention).name
        output_image_path = str(output_image_dir_pathlib.joinpath(output_image_name))

        cv2.imwrite(output_image_path, image_undistorted)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()