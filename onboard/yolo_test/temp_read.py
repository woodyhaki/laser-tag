import numpy as np
import torch
import sys
from pathlib import Path
import os
import cv2
import onnx
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] # YOLOv5 root directory
ROOT_YOLO = os.path.join(str(FILE.parents[1]), './yolo_test')
if str(ROOT_YOLO) not in sys.path:
    sys.path.append(str(ROOT_YOLO))  # add ROOT to PATH
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import pdb
from yolo_test.models.experimental import attempt_load
#from yolo_test.utils.general import check_img_size
from yolo_test.utils.augmentations import letterbox


if __name__ == "__main__":
    raw_img111 = cv2.imread('yolo_test/43_img.png',cv2.IMREAD_COLOR)

    print(raw_img111.shape)
    cv2.imshow("raw_img",raw_img111)
    cv2.waitKey(0)
