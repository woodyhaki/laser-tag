import platform
ONBOARD = False
if platform.machine() == 'aarch64':
    print("Running on Jetson")
    ONBOARD = True
elif platform.machine() == 'x86_64':
    print("Running on x86")
else:
    raise Exception("Unknown platform")

import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import message_filters
import sys
sys.path.append("..")
sys.path.append(".")
from onboard.depth_utils import Resize,NormalizeImage,PrepareForNet
import time,threading
if ONBOARD:
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import InferenceServerException

def preprocess(img,scale):
    img = cv2.resize(img,dsize=(224,224),fx=scale,fy=scale,interpolation = cv2.INTER_NEAREST)
    return img

def convert_np(img_tensor):
    return img_tensor.detach().cpu().squeeze(0).squeeze(0).numpy()

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = ((image - min_val) / (max_val - min_val)) * 255
    normalized_image = normalized_image.astype(np.uint8)
    return normalized_image

def normalize_array(arr):
    if len(arr) == 0:
        return arr
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = ((np.array(arr) - min_val) / (max_val - min_val)) * 1
    return normalized_arr

def preprocess_normalize_image(
    img,
    input_size=224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
):
    """
    Image preprocessing function: Resize + Normalize + PrepareForNet
    
    Args:
        img_path (str): input image path
        input_size (int): size after resize (square)
        mean (tuple): channel mean (RGB)
        std (tuple): channel std (RGB)

    Returns:
        img (np.ndarray): processed image, shape = (1, C, H, W), float32
    """
    # 2. resize (keep aspect ratio, longest side = input_size)
    h, w = img.shape[:2]
    scale = input_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 3. padding to square
    top = (input_size - new_h) // 2
    bottom = input_size - new_h - top
    left = (input_size - new_w) // 2
    right = input_size - new_w - left
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(0,0,0))

    # 4. convert to float32, normalize to [0,1]
    img_norm = img_padded.astype(np.float32) / 255.0

    # 5. standardize (subtract mean, divide std)
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    img_norm = (img_norm - mean) / std

    # 6. HWC -> CHW, add batch dimension
    img_chw = np.transpose(img_norm, (2, 0, 1))
    img_chw = np.expand_dims(img_chw, 0)  # (1, C, H, W)

    return img_chw

class Param():
    def __init__(self,model='datv2', width=224, height=224, url='localhost:8001', model_info=False, verbose=False, client_timeout=None):
        self.model = model
        self.width = width
        self.height = height
        self.url = url
        self.model_info = model_info
        self.verbose = verbose
        self.client_timeout = client_timeout

MAX_N = 4
SAVE_FRAME = 4000

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
class ImageSub():
    def __init__(self):
        rospy.init_node('depth_node',anonymous=True)
        self.depth_pub = rospy.Publisher('/depth_image', Image, queue_size=10)
        self.bridge = CvBridge()
        self.loop_rate = rospy.Rate(30)

        self.omni_img = None
        self.bbx = None
        self.detected_tar_num = 0
        self.va_param = Param()
        self.lock = threading.Lock()

        self.current_time = None
        self.processed_frame = 0

        message_list = []
        omni_img_sub = message_filters.Subscriber('/cam2/image_raw', Image)
        message_list.append(omni_img_sub)
        ts = message_filters.ApproximateTimeSynchronizer(message_list, 10, 1, allow_headerless=True)
        ts.registerCallback(self.callback)
        
        if ONBOARD:
            self.triton_client = grpcclient.InferenceServerClient(
                url=self.va_param.url,
                verbose=self.va_param.verbose,
                ssl=False,
                root_certificates=None,
                private_key=None,
                certificate_chain=None)

    def callback(self,data0):
        self.lock.acquire()
        try:
            self.current_time = data0.header.stamp
            self.omni_img = self.bridge.imgmsg_to_cv2(data0, "bgr8")[90:,...]
            self.omni_img = cv2.resize(self.omni_img,(360,180))
            self.omni_img = cv2.flip(self.omni_img,-1)
        finally:
            self.lock.release()

    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)

    def dummy_process(self):
        while not rospy.is_shutdown():
            print(111)
            self.loop_rate.sleep()

    def depth_process(self):
        img_cnt = 0
        while not rospy.is_shutdown():
            if self.omni_img is not None:
                if ONBOARD:
                    T1 = time.time()
                    inputs = []
                    outputs = []
                    self.lock.acquire()
                    try:
                        image, (h,w) = self.image2tensor(self.omni_img,182)
                        input_image_buffer = image.detach().cpu().numpy().astype(np.float32)
                        print(f"input_image_buffer {input_image_buffer.shape} {image.shape}")
                    finally:
                        self.lock.release()

                    inputs.append(grpcclient.InferInput('images', input_image_buffer.shape, "FP32"))
                    outputs.append(grpcclient.InferRequestedOutput('depth'))
                    inputs[0].set_data_from_numpy(input_image_buffer)

                    T2 = time.time()
                    rospy.loginfo('pre process time:%s ms' % ((T2 - T1) * 1000))
                    T1 = time.time()
                    result = self.triton_client.infer(
                        model_name=self.va_param.model,
                        inputs=inputs,
                        outputs=outputs,
                        client_timeout=self.va_param.client_timeout
                    )
                    result = result.as_numpy('depth')   # numpy array


                    T2 = time.time()
                    rospy.loginfo('infer time:%s ms\n' % ((T2 - T1) * 1000))
                    rospy.loginfo(f'result {result.shape}')
                    depth_img = np.squeeze(result)   # [H,W] float32

                    depth_vis = depth_img.copy()
                    min_val = np.min(depth_vis)
                    max_val = np.max(depth_vis)

                    if max_val > min_val:
                        depth_vis = (255.0 * (depth_vis - min_val) / (max_val - min_val)).astype(np.uint8)
                    else:
                        depth_vis = np.zeros_like(depth_vis, dtype=np.uint8)

                    depth_msg = self.bridge.cv2_to_imgmsg(depth_vis, encoding="mono8")
                    depth_msg.header.stamp = rospy.Time.now()
                    self.depth_pub.publish(depth_msg)

            else:
                rospy.loginfo(f"No image!!")
            self.loop_rate.sleep()
            
if __name__ == '__main__':
    img_sub = ImageSub()
    img_sub.depth_process()
    #img_sub.dummy_process()