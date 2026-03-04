#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import sys
sys.path.append("./")
import threading
import time
import collections
from geometry_msgs.msg import Twist
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

class Param():
    def __init__(self,model='vi', width=224, height=224, url='localhost:8001', model_info=False, verbose=False, client_timeout=None):
        self.model = model
        self.width = width
        self.height = height
        self.url = url
        self.model_info = model_info
        self.verbose = verbose
        self.client_timeout = client_timeout

class OnboardHeatmapDepthSubscriber:
    def __init__(self,params):

        ##--------------------Parameters---------------------##
        self.show_images = False
        self.lock = threading.Lock()
        self.loop_rate = rospy.Rate(30)
        self.seq_len = params['seq_len']
        self.param = Param()
        ##---------------------------------------------------------
        self.depth_image = None
        self.heatmap_image = None
        ##---------------------------------------------------------
        self.heatmap_queue = collections.deque(maxlen=self.seq_len)
        self.depth_queue = collections.deque(maxlen=self.seq_len)
        ##---------------------------------------------------------
        
        self.bridge = CvBridge()
        heatmap_sub = message_filters.Subscriber("/heatmap/gaussian", Image)
        depth_sub = message_filters.Subscriber("/depth_image", Image)
        ts = message_filters.ApproximateTimeSynchronizer([heatmap_sub,depth_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.callback)
        rospy.loginfo("Subscribed to /heatmap/gaussian and /depth/image_raw (sync)")
        self.triton_client = grpcclient.InferenceServerClient(
            url=self.param.url,
            verbose=self.param.verbose,
            ssl=False,
            root_certificates=None,
            private_key=None,
            certificate_chain=None)
        
        self.vel_pub = rospy.Publisher('robot/velcmd', Twist, queue_size=1000)
        

    def callback(self, heatmap_msg, depth_msg):
        self.lock.acquire()
        try:
            # Process heatmap
            heatmap = self.bridge.imgmsg_to_cv2(heatmap_msg, desired_encoding="32FC2")
            h, w, _ = heatmap.shape
            heatmap_0 = heatmap[:, :, 0]
            heatmap_1 = heatmap[:, :, 1]
            # print(f"heatmap shape {heatmap.shape}")
            self.heatmap_image = heatmap
            
            if self.show_images:
                heatmap_0_norm = np.clip(heatmap_0 * 255, 0, 255).astype(np.uint8)
                heatmap_1_norm = np.clip(heatmap_1 * 255, 0, 255).astype(np.uint8)
                color_map = np.zeros((h, w, 3), dtype=np.uint8)
                color_map[:, :, 2] = heatmap_0_norm
                color_map[:, :, 1] = heatmap_1_norm
                cv2.imshow("Subscribed Heatmap", color_map)

            # Process depth image
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            self.depth_image = depth
            
            if self.show_images:
                depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                depth_uint8 = depth_norm.astype(np.uint8)
                cv2.imshow("Subscribed Depth", depth_uint8)
                cv2.waitKey(1)
                
            self.heatmap_queue.append(heatmap)
            self.depth_queue.append(depth)
        
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            
        finally:
           self.lock.release()

    def policy_process(self):
        motion_cmd = Twist()
        while not rospy.is_shutdown():
            if len(self.heatmap_queue) == self.seq_len and len(self.depth_queue) == self.seq_len:
                # Take 5 frames and concatenate them in temporal order
                heatmaps = list(self.heatmap_queue)
                depths = list(self.depth_queue)
                input_images = []

                T1 = time.time()
                inputs = []
                outputs = []
                self.lock.acquire()
                try:
                    for hmap, depth in zip(heatmaps, depths):
                        # Concatenate channels
                        hmap = cv2.resize(hmap,(364,182))
                        input_image = np.concatenate((hmap, depth[..., None]), axis=-1)  # (H, W, 3)
                        input_image_resize = cv2.resize(input_image, (224, 224))
                        # print(input_image_resize.shape)
                        input_image_resize = np.transpose(input_image_resize,(2,0,1))
                        input_images.append(input_image_resize)
                finally:
                    self.lock.release()

                input_images_np = np.stack(input_images, axis=0)  # (5, 224, 224, 3)
                input_images_np = np.expand_dims(input_images_np,axis=0)
                print(f"input_images_np shape {input_images_np.shape}")


                inputs.append(grpcclient.InferInput('images', input_images_np.shape, "FP32"))
                outputs.append(grpcclient.InferRequestedOutput('actions'))
                inputs[0].set_data_from_numpy(input_images_np)

                T2 = time.time()
                rospy.loginfo('pre process time:%s ms' % ((T2 - T1) * 1000))
                T1 = time.time()
                result = self.triton_client.infer(
                    model_name=self.param.model,
                    inputs=inputs,
                    outputs=outputs,
                    client_timeout=self.param.client_timeout
                )
                result = result.as_numpy('actions')   # numpy array


                T2 = time.time()
                rospy.loginfo('infer time:%s ms\n' % ((T2 - T1) * 1000))
                rospy.loginfo(f'result {result.shape}')
                action = result[0]
                motion_cmd.linear.x =  action[0]
                motion_cmd.linear.y =  action[1]
                motion_cmd.angular.z = action[2]
                motion_cmd.angular.x = 0
                motion_cmd.angular.y = 0

                # Example: publish command for a fixed duration
                # start_time = rospy.get_time()
                # while (rospy.get_time() - start_time) < 0.5:  # Check if duration reached
                #     self.vel_pub.publish(motion_cmd)  # Publish velocity command
                #     rospy.sleep(0.001)  # Short sleep to avoid CPU overuse
                
                ##-------------Offline version---------------------------------
                # input_tensor = ptu.from_numpy(input_images_np).unsqueeze(0).to(ptu.device)  # (1, 5, 224, 224, 3)
                # input_tensor = input_tensor.permute(0, 1, 4, 2, 3).to(ptu.device)  # (1, 3, 5, 224, 224)
                # # print(f"input_tensor shape {input_tensor.shape}")
                # sampled_action = self.policy(input_tensor)
                # print(f"sampled_action {sampled_action}")
                # print()
                ##-------------------------------------------------------------

                
            self.loop_rate.sleep()



if __name__ == "__main__":
    rospy.init_node("heatmap_depth_subscriber_node", anonymous=True)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--color_num_channel',       type=int,default=3)
    parser.add_argument('--seq_len',       type=int,default=5)

    args = parser.parse_args()
    params = vars(args)
    
    try:
        heatmap_depth_sub = OnboardHeatmapDepthSubscriber(params)
        heatmap_depth_sub.policy_process()
    except rospy.ROSInterruptException:
        pass