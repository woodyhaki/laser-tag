import rospy
import numpy as np
import torch
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import message_filters
import sys
from pathlib import Path
import time
import Jetson.GPIO as GPIO
import os
from std_msgs.msg import Bool 

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] # YOLOv5 root directory
ROOT_YOLO = os.path.join(str(FILE.parents[1]), './yolo_test')
if str(ROOT_YOLO) not in sys.path:
    sys.path.append(str(ROOT_YOLO))  # add ROOT to PATH
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# print("ROOT:",ROOT)
# print("ROOT_YOLO:",ROOT_YOLO)
# print(sys.path)

publish_yolo_debug = True
output_pin = 7

import copy
from yolo_test.utils.general import non_max_suppression,scale_boxes

if publish_yolo_debug:
    from yolo_test.utils.plots import Annotator,colors
# from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
# from std_msgs.msg import Header

import tritonclient.grpc as grpcclient
output_pin = 7

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class BoundingBox:
    def __init__(self, classID, confidence, x1, x2, y1, y2, image_width, image_height):
        self.classID = classID
        self.confidence = confidence
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.u1 = x1 / image_width
        self.u2 = x2 / image_width
        self.v1 = y1 / image_height
        self.v2 = y2 / image_height
    
    def box(self):
        return (self.x1, self.y1, self.x2, self.y2)
        
    def width(self):
        return self.x2 - self.x1
    
    def height(self):
        return self.y2 - self.y1

    def center_absolute(self):
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))
    
    def center_normalized(self):
        return (0.5 * (self.u1 + self.u2), 0.5 * (self.v1 + self.v2))
    
    def size_absolute(self):
        return (self.x2 - self.x1, self.y2 - self.y1)
    
    def size_normalized(self):
        return (self.u2 - self.u1, self.v2 - self.v1)
    
def xywh2xyxy(x, origin_h, origin_w, input_w, input_h):
    """
    description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    param:
        origin_h:   height of original image
        origin_w:   width of original image
        x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
    return:
        y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
    """
    y = np.zeros_like(x)
    r_w = input_w / origin_w
    r_h = input_h / origin_h
    if r_h > r_w:
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2 - (input_h - r_w * origin_h) / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2 - (input_h - r_w * origin_h) / 2
        y /= r_w
    else:
        y[:, 0] = x[:, 0] - x[:, 2] / 2 - (input_w - r_h * origin_w) / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2 - (input_w - r_h * origin_w) / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        y /= r_h

    return y

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    description: compute the IoU of two bounding boxes
    param:
        box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
        box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))            
        x1y1x2y2: select the coordinate format
    return:
        iou: computed iou
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                 np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression2(prediction, origin_h, origin_w, input_w, input_h, conf_thres=0.5, nms_thres=0.1):
    """
    description: Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    param:
        prediction: detections, (x1, y1, x2, y2, conf, cls_id)
        origin_h: original image height
        origin_w: original image width
        conf_thres: a confidence threshold to filter detections
        nms_thres: a iou threshold to filter detections
    return:
        boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
    """
    # Get the boxes that score > CONF_THRESH

    #conf_thres = 0.01
    boxes = prediction[prediction[:, 4] >= conf_thres]
    # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
    boxes[:, :4] = xywh2xyxy(boxes[:, :4], origin_h, origin_w, input_w, input_h )
    # clip the coordinates
    boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
    # Object confidence
    confs = boxes[:, 4]
    # Sort by the confs
    boxes = boxes[np.argsort(-confs)]
    # Perform non-maximum suppression
    keep_boxes = []
    while boxes.shape[0]:
        large_overlap = bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres

        label_match = np.abs(boxes[0, -1] - boxes[:, -1]) < 2e-1
        # Indices of boxes with lower confidence scores, large IOUs and matching labels
        invalid = large_overlap & label_match

        keep_boxes += [boxes[0]]
        boxes = boxes[~invalid]
    boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
    return boxes

class Param():
    def __init__(self,model='yolo', width=320, height=160, url='localhost:8001', 
                 confidence=0.9, nms=0.85, model_info=False, verbose=False, client_timeout=None):
        self.model = model
        self.width = width
        self.height = height
        self.url = url
        self.confidence = confidence
        self.nms = nms
        self.model_info = model_info
        self.verbose = verbose
        self.client_timeout = client_timeout


def infer(client, model_name, input_data):
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('input', input_data.shape, "FP32"))
    inputs[0].set_data_from_numpy(input_data)
    outputs.append(grpcclient.InferRequestedOutput('output'))

    results = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    return results.as_numpy('output')

model_name_list = ['va', 'yolo_car']

class ImageSub():
    def __init__(self):
        rospy.init_node('yolo_node',anonymous=True)

        self.yolo_debug_pub = rospy.Publisher('/robot/yolo_debug', Image, queue_size=10)
        self.heatmap_pub = rospy.Publisher("/heatmap/gaussian", Image, queue_size=10)
        self.heatmap_display_pub = rospy.Publisher("/heatmap/display", Image, queue_size=10)

        #self.bbxpub = rospy.Publisher('/bbx', bbx, queue_size=10)
        self.bridge = CvBridge()
        self.loop_rate = rospy.Rate(20)
        self.img = None
        self.img_msg_header = None
        self.yolo_params = Param()
        self.use_triton_server = True
        self.triton_client = grpcclient.InferenceServerClient(
            url=self.yolo_params.url,
            verbose=self.yolo_params.verbose,
            ssl=False,
            root_certificates=None,
            private_key=None,
            certificate_chain=None)
        
        message_list = []
        omni_img_sub = message_filters.Subscriber('/cam2/image_raw', Image)
        message_list.append(omni_img_sub)
        ts = message_filters.ApproximateTimeSynchronizer(message_list, 10, 1, allow_headerless=True)
        ts.registerCallback(self.callback)
        
        
        self.img_size = 320
        self.imgsz = [self.img_size,self.img_size]
        self.stride = 32
        self.pub_detection_debug = True
        ##-------------------------------------------------------------------------------
        self.fire_pub = rospy.Publisher(f'/robot/fire', Bool, queue_size=10)
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.LOW)
        ##-------------------------------------------------------------------------------
    def open_fire(self):
        GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH)
        self.fire_pub.publish(Bool(data=True))
        
    def cease_fire(self):
        GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.LOW)
        self.fire_pub.publish(Bool(data=False))
        
        
    def callback(self,data0):
        self.img = self.bridge.imgmsg_to_cv2(data0, "bgr8")[:-90,:,:]
        self.img_msg_header = data0.header
        #print(f"callback {self.img.shape}")

        
    def preprocess_input_image(self,raw_img):
        img = letterbox(raw_img, self.imgsz, stride=self.stride, auto=True)[0]
        #print(f"image after letterbox {img.shape}")
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img = img.reshape(3, img.shape[0], img.shape[1])
        img = np.ascontiguousarray(img)
        #self.img_size = check_img_size(self.img_size)
        img = np.expand_dims(img,0)
        img = img / 255
        #img = img.squeeze(0)
        #print("===========",img.shape)
        return img

    def preprocess_triton_input(self,raw_bgr_image, input_shape):
        """
        description: Preprocess an image before TRT YOLO inferencing.
                    Convert BGR image to RGB,
                    resize and pad it to target size, normalize to [0,1],
                    transform to NCHW format.          
        param:
            raw_bgr_image: int8 numpy array of shape (img_h, img_w, 3)
            input_shape: a tuple of (H, W)
        return:
            image:  the processed image float32 numpy array of shape (3, H, W)
        """
        input_w, input_h = input_shape
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = input_w / w
        r_h = input_h / h
        if r_h > r_w:
            tw = input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((input_h - th) / 2)
            ty2 = input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = input_h
            tx1 = int((input_w - tw) / 2)
            tx2 = input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        return image


    def postprocess(self,output, origin_w, origin_h, input_shape, conf_th=0.5, nms_threshold=0.5, letter_box=False):
        """Postprocess TensorRT outputs.
        # Args
            output: list of detections with schema 
            [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            conf_th: confidence threshold
            letter_box: boolean, referring to _preprocess_yolo()
        # Returns
            list of bounding boxes with all detections above threshold and after nms, see class BoundingBox
        """
        
        # Get the num of boxes detected
        # Here we use the first row of output in that batch_size = 1
        output = output[0]
        #num = int(output.shape[0])
        # Reshape to a two dimentional ndarray
        #pred = np.reshape(output[:,1:], (-1, 6))[:num, :]

        # Do nms
        # classes = [0,1]
        # #print(output.shape)
        # pred0 = non_max_suppression(pred, conf_thres, iou_thres, classes[0], agnostic_nms, max_det=max_det)
        # pred1 = non_max_suppression(pred, conf_thres, iou_thres, classes[1], agnostic_nms, max_det=max_det)
        #non_max_suppression()
        boxes = non_max_suppression2(output, origin_h, origin_w, input_shape[0], input_shape[1], conf_thres=0.6, nms_thres=0.1)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5].astype(int) if len(boxes) else np.array([])
            
        detected_objects = []
        for box, score, label in zip(result_boxes, result_scores, result_classid):
            #print(f"label {label}")
            detected_objects.append(BoundingBox(label, score, box[0], box[2], box[1], box[3], origin_h, origin_w))
        return detected_objects

    def annotate_image(self, annotator, im0, im0_flip, img, pred, names, hide_labels=False, hide_conf=False):
        xy_array = []
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                #rospy.loginfo(f"detected!!!!!!!!!!!!!!!!!!!!")
                #rospy.loginfo(f"11111111111111 img0 {im0.shape}")
                #rospy.loginfo(f"det[:, :4] {det[:, :4]}")
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    ########### flip xyxy ##########
                    xyxy_display = copy.deepcopy(xyxy)
                    xyxy_display[0] = im0.shape[1] - xyxy_display[0] -1
                    xyxy_display[2] = im0.shape[1] - xyxy_display[2] -1
                    xyxy_display[1] = im0.shape[0] - xyxy_display[1] -1
                    xyxy_display[3] = im0.shape[0] - xyxy_display[3] -1
                    ################################
                    annotator.box_label(xyxy_display, label, color=colors(c, True))
                    for i in xyxy:
                        i = i.cpu().detach().float().item()
                        xy_array.append(i)
        return xy_array

    def generate_gaussian_kernel(self,size=7, sigma=2):
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        return kernel / kernel.max()

    def paste_gaussian(self, heatmap, x_center, y_center, kernel):
        h, w, c = heatmap.shape
        k = kernel.shape[0]
        radius = k // 2

        x0 = x_center - radius
        y0 = y_center - radius
        x1 = x_center + radius + 1
        y1 = y_center + radius + 1

        x0_clip, x1_clip = max(0, x0), min(w, x1)
        y0_clip, y1_clip = max(0, y0), min(h, y1)

        kernel_x0 = x0_clip - x0
        kernel_x1 = k - (x1 - x1_clip)
        kernel_y0 = y0_clip - y0
        kernel_y1 = k - (y1 - y1_clip)

        heatmap_slice = heatmap[:, :, 0]  # shape (H, W)

        heatmap_slice[y0_clip:y1_clip, x0_clip:x1_clip] = np.maximum(
            heatmap_slice[y0_clip:y1_clip, x0_clip:x1_clip],
            kernel[kernel_y0:kernel_y1, kernel_x0:kernel_x1]
        )

        heatmap[:, :, 0] = heatmap_slice
        return heatmap
    
    def yolo_process(self):
        agnostic_nms = False
        hide_labels = False
        hide_conf = False
        use_triton_server = True
        names = {0: 'enemy', 1: 'ally'}
        while not rospy.is_shutdown():
            if self.img is not None:
                # Padded resize
                xy_array = []
                if True:
                    inputs = []
                    outputs = []
                    T1 = time.time()
                    input_image_buffer = self.preprocess_input_image(self.img).astype(np.float32)
                    inputs.append(grpcclient.InferInput('input', input_image_buffer.shape, "FP32"))
                    outputs.append(grpcclient.InferRequestedOutput('output'))
                    inputs[0].set_data_from_numpy(input_image_buffer)
                    T2 = time.time()

                    T1 = time.time()
                    result = self.triton_client.infer(model_name=self.yolo_params.model,
                                                        inputs=inputs,
                                                        outputs=outputs,
                                                        client_timeout=self.yolo_params.client_timeout)
                    T2 = time.time()
                    result = result.as_numpy('output')
                    
                    T1 = time.time()
                    #print(f"result {result.shape}")
                    classes = [0,1]
        #
                    pred = torch.FloatTensor(result)
                  #  print("input_image_buffer:",input_image_buffer.shape)

                    pred0 = non_max_suppression(pred, 0.85, 0.75, classes[0], agnostic_nms, max_det=1)
                    pred1 = non_max_suppression(pred, 0.85, 0.75, classes[1], agnostic_nms, max_det=1)


                    # detected_objects = self.postprocess(result, self.img.shape[1], self.img.shape[0], \
                    #                                [self.yolo_params.width, self.yolo_params.height],
                    #                                self.yolo_params.confidence,self.yolo_params.nms)
                    T2 = time.time()
                    print('postprocess:%s ms' % ((T2 - T1)*1000))
                    print(f'detected_objects class 0 {pred0}')
                    print(f'detected_objects class 1 {pred1}')
                    
                    # Process predictions
                    im0 = self.img.copy() 
                    #rospy.loginfo(f"img shape {img.shape}")
                    #rospy.loginfo(f"im0 shape {im0.shape} {img.shape}")
                    im0_flip = cv2.flip(im0,-1)

                    annotator = Annotator(im0_flip, line_width=3, example=str('enemy'))
                   # print(f"im0 {im0.shape} | {im0_flip.shape} | {input_image_buffer.shape}")
                    temp_tensor = torch.FloatTensor(input_image_buffer)
                    xy_array_0 = self.annotate_image(annotator,im0, im0_flip, temp_tensor, pred0, names, hide_labels, hide_conf)
                    xy_array_1 = self.annotate_image(annotator,im0, im0_flip, temp_tensor, pred1, names, hide_labels, hide_conf)
                    xy_array_0 = np.array(xy_array_0).reshape(-1, 4)
                    xy_array_1 = np.array(xy_array_1).reshape(-1, 4)

                   # print(xy_array_0)

                    h, w = self.img.shape[:2]

                    black_map0 = np.zeros((h, w,1), dtype=np.float32)
                    black_map1 = np.zeros((h, w,1), dtype=np.float32)

                    def draw_centers_from_xyxy_list(black_map,xy_array):
                        for bbox in xy_array:
                            if len(bbox) != 4:
                                continue

                            x_min, y_min, x_max, y_max = bbox
                            x_center = int((x_min + x_max) / 2)
                            y_center = int((y_min + y_max) / 2)

                            box_w = x_max - x_min
                            box_h = y_max - y_min
                            box_size = int(max(box_w, box_h))

                            size = int(np.ceil(box_size * 1.5))
                            if size % 2 == 0:
                                size += 1
                            size = max(size, 7)

                            sigma = size / 6.0

                            gaussian_kernel = self.generate_gaussian_kernel(size, sigma)
            
                            black_map = self.paste_gaussian(black_map, x_center, y_center, gaussian_kernel)

                    draw_centers_from_xyxy_list(black_map0, xy_array_0)
                    draw_centers_from_xyxy_list(black_map1, xy_array_1)

                    ##-------------------------Fire condition-----------------------------------------------
                    if xy_array_0.shape[0] > 0:
                        x1, y1, x2, y2 = xy_array_0[0]
                        cx = (x1 + x2) / 2
                        print(f"enemy bbx width {(x2 - x1)}")
                        
                        #cy = (y1 + y2) / 2
                        img_height, img_width = self.img.shape[:2]
                        cx_rot = img_width - cx
                        #cy_rot = img_height - cy
                    
                        if (abs(cx_rot - img_width / 2) < 20) and (x2 - x1) > 30:
                            self.open_fire()
                        else:
                            self.cease_fire()
                    else:
                        self.cease_fire()
                    
                    # disp_map0 = np.clip(black_map0 * 255, 0, 255).astype(np.uint8)
                    # disp_map1 = np.clip(black_map1 * 255, 0, 255).astype(np.uint8)

                    # cv2.imshow("Class Enemy Heatmaps", disp_map0)
                    # cv2.imshow("Class Ally Heatmaps", disp_map1)
                    # cv2.waitKey(1)

                    two_channel_heatmap = np.concatenate([black_map0, black_map1], axis=2)  # shape: (H, W, 2)

                    ##-------------------------Display heatmap-----------------------------------------------
                    h, w, _ = two_channel_heatmap.shape
                    heatmap_0 = two_channel_heatmap[:, :, 0]
                    heatmap_1 = two_channel_heatmap[:, :, 1]


                    heatmap_0_norm = np.clip(heatmap_0 * 255, 0, 255).astype(np.uint8)
                    heatmap_1_norm = np.clip(heatmap_1 * 255, 0, 255).astype(np.uint8)
                    color_map = np.zeros((h, w, 3), dtype=np.uint8)
                    color_map[:, :, 2] = heatmap_0_norm
                    color_map[:, :, 1] = heatmap_1_norm

                    ##--------------------------------------------------------------------------------------
                    ros_img = CvBridge().cv2_to_imgmsg(two_channel_heatmap, encoding="32FC2")
                    ros_img.header.stamp = rospy.Time.now()
                    ros_img.header.frame_id = ""
                    
                    heatmap_display = CvBridge().cv2_to_imgmsg(color_map, encoding="bgr8")
                    heatmap_display.header.stamp = rospy.Time.now()
                    heatmap_display.header.frame_id = ""
                    
                    self.heatmap_pub.publish(ros_img)
                    self.heatmap_display_pub.publish(heatmap_display)

                    im0_flip = annotator.result()
                    # cv2.imshow("img",im0_flip)
                    # cv2.waitKey(1)
                    if im0_flip is not None:
                        msg = self.bridge.cv2_to_imgmsg(im0_flip,"bgr8")
                        self.yolo_debug_pub.publish(msg)

                    print('\n')

                # array_msg = bbx()
                # header = Header()
                # header.stamp = self.img_msg_header.stamp

                # layout = MultiArrayLayout()
                # layout.dim.append(MultiArrayDimension())
                # layout.dim[0].label = "rows"
                # layout.dim[0].size = len(detected_objects)
                # layout.dim.append(MultiArrayDimension())
                # layout.dim[1].label = "cols"
                # layout.dim[1].size = 4
                # array_msg.layout = layout

                # array_msg.data = xy_array
                # array_msg.header = header
                # self.bbxpub.publish(array_msg)

            self.loop_rate.sleep()

if __name__ == '__main__':
    img_sub = ImageSub()
    img_sub.yolo_process()
