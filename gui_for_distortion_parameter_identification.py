import cv2
import cvui
import toml
import click
import numpy as np
from pathlib import Path
from functools import partial

SCRIPT_DIR = str(Path().parent)
WINDOW_NAME = "Distortion Correction"


def fisheye_undistort_rectify_map(k1, k2, fx, fy, cx, cy, width, height):
    camera_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    dist_coef = np.array([[k1, k2, 0.0, 0.0]])
    projection_camera_mat = cv2.getOptimalNewCameraMatrix(camera_mat, dist_coef, (width, height), 0)[0]
    return cv2.fisheye.initUndistortRectifyMap(camera_mat, dist_coef, np.eye(3), projection_camera_mat, (width, height), cv2.CV_16SC2)


@click.command()
@click.option("--input-image", "-i", "input_image_path", default=f"{SCRIPT_DIR}/image/rgb.jpg")
@click.option("--output-toml", "-o", "output_toml_path", default=f"{SCRIPT_DIR}/parameter.toml")
@click.option("--resize-rate", "-r", default=1.0)
def main(input_image_path, output_toml_path, resize_rate):
    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    image_height, image_width = image.shape[:-1]

    distortion_param_to_map = partial(
        fisheye_undistort_rectify_map, fx=900, fy=900, cx=image_width // 2, cy=image_height // 2, width=image_width, height=image_height
    )
    k1 = [0.0]
    k2 = [0.0]

    image_height_resized = int(image_height * resize_rate)
    image_width_resized = int(image_width * resize_rate)
    trackbar_width = int(image_width_resized * 0.5)

    frame = np.zeros((image_height_resized + 200, np.max([image_width_resized, trackbar_width + 410]), 3), np.uint8)
    frame[:] = 50
    trackbar_bottom = frame.shape[0] - 50

    cvui.init(WINDOW_NAME)
    while True:
        # Distortion Correction
        map_u, map_v = distortion_param_to_map(k1=k1[0], k2=k2[0])
        image_corrected = cv2.remap(image, map_u, map_v, interpolation=cv2.INTER_LINEAR)
        image_corrected_resized = cv2.resize(image_corrected, None, fx=resize_rate, fy=resize_rate)

        # Visualize information
        frame[:image_height_resized, :image_width_resized, :] = image_corrected_resized[:]
        frame[trackbar_bottom - 130 : trackbar_bottom + 50, 10 : trackbar_width + 100] = (50, 50, 50)
        cvui.printf(frame, 50, trackbar_bottom - 120, 0.8, 0x00FF00, "k1: %3.05lf", k1[0])
        cvui.trackbar(frame, 15, trackbar_bottom - 100, trackbar_width, k1, -0.5, 0.5)
        cvui.printf(frame, 50, trackbar_bottom - 30, 0.8, 0x00FF00, "k2: %3.05lf", k2[0])
        cvui.trackbar(frame, 15, trackbar_bottom, trackbar_width, k2, -0.5, 0.5)

        if cvui.button(frame, trackbar_width + 100, trackbar_bottom - 100, 300, 100, "Dump"):
            toml_dict = {"k1": k1[0], "k2": k2[0]}
            with open(output_toml_path, "w") as f:
                toml.dump(toml_dict, f)
            print("Parameters are dumped")

        cvui.update()
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(20) in [ord("q"), 27]:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()