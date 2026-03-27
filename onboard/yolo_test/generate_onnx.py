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
#from data_utils import *
import pdb
from yolo_test.models.experimental import attempt_load
from yolo_test.utils.general import check_img_size
from yolo_test.utils.augmentations import letterbox

def preprocess_input_image(raw_img,stride,img_size = 640,ch = 3):
    img = letterbox(raw_img, img_size, stride=stride, auto=True)[0]
    if ch == 3:
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    elif ch == 1:
        img = img.reshape(1, img.shape[0], img.shape[1])
    else:
        print("wrong image channel:",ch)
        exit()
    
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float().cuda()
    img_size = check_img_size(img_size)
    img = img[None]
    img /= 255
    return img

from PIL import Image

if __name__ == "__main__":
    img_size = 320
    imgsz = [img_size,img_size]
    stride = 64
    device = torch.device("cpu")
    weights_file = 'checkpoints/yolo/best.pt'
    model = attempt_load([weights_file], device=device)
    stride = int(model.stride.max())  # model stride
    yolo_model = model.eval()
    print(yolo_model)
    print("load yolo ok!!!")
    
    
    img = Image.open("yolo_test/43_img.png")
    raw_img = np.array(img)
    
    # raw_img = cv2.imread('yolo_test/43_img.png',cv2.IMREAD_COLORRE)

    # print(raw_img.shape)
    # cv2.imshow("raw_img",raw_img)
    # cv2.waitKey(0)

    # if len(raw_img.shape) == 2 or raw_img.shape[2] == 1:
    #     ch = 1
    # else:
    #     ch = 3
    ch = 3
    print("image channel:",ch)
    #
    stride = int(yolo_model.stride.max())  # model stride
    img = preprocess_input_image(raw_img,stride,img_size,ch)

    print(f'image shape {img.shape} stride {stride}')
    #pdb.set_trace()
    file_name = 'onnx/yolo_single.onnx'
    net = yolo_model.to(torch.float32)
    img = img.to(torch.float32).cpu()
    net.eval()
    
    # 配置导出ONNX时的动态轴参数
    # dynamic_axes = {
    #     'input': {0: 'batch_size', 1: 'channels', 2: 'height', 3: 'width'},
    #     'output': {0: 'batch_size'}
    # }

    torch.onnx.export(net.cpu(),img,file_name,verbose=False,opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None)
    model_onnx = onnx.load(file_name)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    onnx.save(model_onnx, file_name)
    print("save onnx ok!!!")

# if __name__ == "__main__":
#     img = Image.open("yolo_test/43_img.png")

#     # 显示图像
#     img.show()
#     img_array = np.array(img)
#     # # 打印图像信息
#     print(img.mode)   # 图像模式，例如 "RGB", "RGBA", "L"(灰度)
#     print(img_array.shape)   # (width, height)
#     # raw_img = cv2.imread(img_file,cv2.IMREAD_UNCHANGED)
#     # print(raw_img.shape)
    
#     # cv2.imshow("raw_img",raw_img)
#     # cv2.waitKey(0)