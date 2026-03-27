import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import torch
import cv2
import sys
import copy
from pathlib import Path
import os
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
from yolo_test.utils.general import check_img_size,non_max_suppression,scale_boxes
from yolo_test.utils.augmentations import letterbox
from yolo_test.utils.plots import Annotator,colors
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
from std_msgs.msg import Float32MultiArray,Header
from vas.msg import bbx
from std_msgs.msg import Bool 

class ImageSub():
    def __init__(self):
        rospy.init_node('yolo_node',anonymous=True)
        self.yolo_debug_pub = rospy.Publisher('/robot/yolo_debug', Image, queue_size=10)
        self.heatmap_pub = rospy.Publisher("/heatmap/gaussian", Image, queue_size=1)
        self.heatmap_display_pub = rospy.Publisher("/heatmap/display", Image, queue_size=1)
        
        self.bbxpub = rospy.Publisher('/bbx', bbx, queue_size=10)
        self.bridge = CvBridge()
        self.loop_rate = rospy.Rate(30)
        self.img = None
        self.img_msg_header = None
        message_list = []
      #  img_sub = message_filters.Subscriber('/vswarm301/cam2/image_raw', Image)
        img_sub = message_filters.Subscriber('/vswarm302/cam2/image_raw', Image)
        
       # img_sub = message_filters.Subscriber('/cam2/image_raw', Image)
        
        message_list.append(img_sub)
        ts = message_filters.ApproximateTimeSynchronizer(message_list, 10, 1, allow_headerless=True)
        ts.registerCallback(self.callback)
        
        self.fire_pub = rospy.Publisher(f'/vswarm301/robot/fire', Bool, queue_size=10)
        
        ###############################################
        self.img_size = 320
        self.imgsz = [self.img_size,self.img_size]
        self.stride = 64
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda else "cpu")
        
        weights_file = 'checkpoints/yolo/best.pt'
        
        model = attempt_load([weights_file], device=self.device)
        self.stride = int(model.stride.max())  # model stride
        self.yolo_model = model.eval()
        print(self.yolo_model)
        print("load yolo ok!!!")
        ##########################################

    def callback(self,data0):
        self.img = self.bridge.imgmsg_to_cv2(data0, "bgr8")
        self.img = self.img[90:,...]
        # self.img = self.bridge.imgmsg_to_cv2(data0, "mono8")
        self.img_msg_header = data0.header
        
    def preprocess_input_image(self,raw_img):
        img = letterbox(raw_img, self.img_size, stride=self.stride, auto=True)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float().cuda()
        self.img_size = check_img_size(self.img_size)
        img = img[None]
        img /= 255
        return img
    
    def annotate_image(self, annotator, im0, im0_flip, img, pred, names, hide_labels=False, hide_conf=False):
        xy_array = []
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
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
        """Generate a size x size Gaussian kernel with center value = 1"""
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        return kernel / kernel.max()

    def paste_gaussian(self, heatmap, x_center, y_center, kernel):
        """Paste a Gaussian kernel at a specified center location on heatmap[:, :, 0]"""
        h, w, c = heatmap.shape
        k = kernel.shape[0]
        radius = k // 2

        x0 = x_center - radius
        y0 = y_center - radius
        x1 = x_center + radius + 1
        y1 = y_center + radius + 1

        # Clip within image range
        x0_clip, x1_clip = max(0, x0), min(w, x1)
        y0_clip, y1_clip = max(0, y0), min(h, y1)

        # Clip kernel range
        kernel_x0 = x0_clip - x0
        kernel_x1 = k - (x1 - x1_clip)
        kernel_y0 = y0_clip - y0
        kernel_y1 = k - (y1 - y1_clip)

        heatmap_slice = heatmap[:, :, 0]  # shape (H, W)

        heatmap_slice[y0_clip:y1_clip, x0_clip:x1_clip] = np.maximum(
            heatmap_slice[y0_clip:y1_clip, x0_clip:x1_clip],
            kernel[kernel_y0:kernel_y1, kernel_x0:kernel_x1]
        )

        heatmap[:, :, 0] = heatmap_slice  # write back
        return heatmap
    
    def open_fire(self):
        print(f"[Onboard] Open fire!")
        self.fire_pub.publish(Bool(data=True))

    def cease_fire(self):
        print(f"[Onboard] Cease fire!")
        self.fire_pub.publish(Bool(data=False))

    def yolo_process(self):
        conf_thres = 0.85 # confidence threshold
        iou_thres = 0.75  # NMS IOU threshold
        max_det = 4  # maximum detections per image
        classes = [0,1]
        agnostic_nms = False
        hide_labels = False
        hide_conf = True
        names = self.yolo_model.names if hasattr(self.yolo_model, 'module') else self.yolo_model.names  # get class names
        img_cnt = 0
        self.stride = int(self.yolo_model.stride.max())  # model stride
        open_fire = False
        while not rospy.is_shutdown():
            if self.img is not None:
                # Padded resize
                rospy.loginfo(f"self.img {self.img.shape}")
                
                img = self.preprocess_input_image(self.img)
                rospy.loginfo(f"model input image shape {img.shape}")
                
                pred = self.yolo_model(img, False, False)[0]
                print(f"pred {pred.shape}")
                print(f"names {names}")
                pred0 = non_max_suppression(pred, conf_thres, iou_thres, classes[0], agnostic_nms, max_det=max_det)
                pred1 = non_max_suppression(pred, conf_thres, iou_thres, classes[1], agnostic_nms, max_det=max_det)
                print(f"pred0 {pred0} pred1 {pred1}")
                pred1 = [pred1[0]]
                # Process predictions
                im0 = self.img.copy() 
                im0_flip = cv2.flip(im0,-1)
                xy_array = []
                annotator = Annotator(im0_flip, line_width=3, example=str('enemy'))
                
                print(f"im0 shape {im0.shape} im0_flip shape {im0_flip.shape} img shape {img.shape}")
                xy_array_0 = self.annotate_image(annotator,im0, im0_flip, img, pred0, names, hide_labels, hide_conf)
                xy_array_1 = self.annotate_image(annotator,im0, im0_flip, img, pred1, names, hide_labels, hide_conf)
                xy_array_0 = np.array(xy_array_0).reshape(-1, 4)
                xy_array_1 = np.array(xy_array_1).reshape(-1, 4)

                num_detection = xy_array_0.shape[0]
                print(xy_array_0)
                im0_flip = annotator.result()

                # Get original image size
                h, w = self.img.shape[:2]

                # Create black single-channel images
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

                        # Gaussian kernel size = 1.5x bbox size, rounded up to odd number
                        size = int(np.ceil(box_size * 1.5))
                        if size % 2 == 0:
                            size += 1
                        size = max(size, 7)  # ensure at least 7x7

                        sigma = size / 6.0  # heuristic to ensure 3σ covers most energy

                        gaussian_kernel = self.generate_gaussian_kernel(size, sigma)
        
                        black_map = self.paste_gaussian(black_map, x_center, y_center, gaussian_kernel)

                draw_centers_from_xyxy_list(black_map0, xy_array_0)
                draw_centers_from_xyxy_list(black_map1, xy_array_1)

                ##-------------------------Fire condition-----------------------------------------------
                if xy_array_0.shape[0] > 0:
                    x1, y1, x2, y2 = xy_array_0[0]
                    cx = (x1 + x2) / 2
                    print(f"enemy bbx width {(x2 - x1)}")
                    
                    img_height, img_width = self.img.shape[:2]
                    cx_rot = img_width - cx
                
                    if (abs(cx_rot - img_width / 2) < 10) and (x2 - x1) > 15:
                        self.open_fire()
                        open_fire = True
                    else:
                        self.cease_fire()
                        open_fire = False
                else:
                    self.cease_fire()
                    open_fire = False
                    
                # Concatenate into a (H, W, 2) float32 image
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
                
                # Publish float32 two-channel image
                ros_img = CvBridge().cv2_to_imgmsg(two_channel_heatmap, encoding="32FC2")
                ros_img.header.stamp = rospy.Time.now()
                ros_img.header.frame_id = ""
                
                heatmap_display = CvBridge().cv2_to_imgmsg(color_map, encoding="bgr8")
                heatmap_display.header.stamp = rospy.Time.now()
                heatmap_display.header.frame_id = ""
                
                self.heatmap_pub.publish(ros_img)
                self.heatmap_display_pub.publish(heatmap_display)
                
                im0_flip = annotator.result()
                if im0_flip is not None:
                    # Reticle lines
                    self.reticle_size = 50
                    self.reticle_thickness = 1
                    self.reticle_color = (0, 0, 255)  # Red in BGR

                    height, width = im0_flip.shape[:2]
                    center_x, center_y = width // 2 , height // 2 + 50
                    
                    cv2.line(im0_flip, 
                            (center_x - self.reticle_size, center_y), 
                            (center_x + self.reticle_size, center_y), 
                            self.reticle_color, self.reticle_thickness)
                    cv2.line(im0_flip, 
                            (center_x, center_y - self.reticle_size), 
                            (center_x, center_y + self.reticle_size), 
                            self.reticle_color, self.reticle_thickness)

                    # Center box
                    half_box = self.reticle_size // 2
                    cv2.rectangle(im0_flip,
                                (center_x - half_box, center_y - half_box),
                                (center_x + half_box, center_y + half_box),
                                self.reticle_color, self.reticle_thickness)
        
                    if open_fire:
                        text = "FIRE"
                        color = (0, 0, 255)  # Red
                    else:
                        text = "CEASE FIRE"
                        color = (0, 255, 0)  # Green

                    cv2.putText(
                        im0_flip, 
                        text, 
                        (50, 50),  # top-left corner
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5,       # font scale
                        color, 
                        3          # thickness
                    )

                    msg = CvBridge().cv2_to_imgmsg(im0_flip, "bgr8")
                    msg.header.stamp = self.img_msg_header.stamp
                    self.yolo_debug_pub.publish(msg)

                array_msg = bbx()
                header = Header()
                header.stamp = self.img_msg_header.stamp

                layout = MultiArrayLayout()
                layout.dim.append(MultiArrayDimension())
                layout.dim[0].label = "rows"
                layout.dim[0].size = num_detection
                layout.dim.append(MultiArrayDimension())
                layout.dim[1].label = "cols"
                layout.dim[1].size = 4
                array_msg.layout = layout

                if xy_array_0.shape[0] == 0:
                    xy_array = []
                else:
                    xy_array = xy_array_0.astype(np.float32)[0].tolist()
                
                array_msg.data = xy_array
                array_msg.header = header
                self.bbxpub.publish(array_msg)
                

            self.loop_rate.sleep()

if __name__ == '__main__':
    img_sub = ImageSub()
    img_sub.yolo_process()
