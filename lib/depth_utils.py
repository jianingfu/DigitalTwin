from calendar import different_locale
import normalSpeed
from PIL import Image
import os
import numpy as np
import cv2


def norm2bgr(norm):
    norm = ((norm + 1.0) * 127).astype("uint8")
    return norm

#depth should be in millimeters
def compute_normals(depth, fx, fy, k_size=5, distance_threshold=2000, difference_threshold=20, point_into_surface=False):
    normals = normalSpeed.depth_normal(depth, fx, fy, k_size, distance_threshold, difference_threshold, point_into_surface)
    return normals

def main():
    ycb_root = "datasets/ycb/YCB_Video_Dataset/"
    test_img = np.array(Image.open(os.path.join(ycb_root, "data/0000/000001-depth.png")))

    cv2.imshow("xd", test_img)
    cv2.waitKey(0)
    print(test_img.shape, test_img.dtype)
    fx = 1066.778
    fy = 1067.487

    normals = compute_normals((test_img / 10).astype(np.uint16), fx, fy)
    print("normals_map_out z mean:",  normals[:, :, 2].mean())

    vis_normals = norm2bgr(normals)
    cv2.imshow("xd2", vis_normals)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()