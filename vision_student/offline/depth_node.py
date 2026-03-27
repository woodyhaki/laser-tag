#!/usr/bin/env python
import sys
sys.path.append("./")
from data_utils import make_dav2_model
from torchinfo import summary
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class DepthEstimatorNode:
    def __init__(self):
        rospy.init_node('depth_estimator_node')
        
        self.bridge = CvBridge()
        self.depth_pub = rospy.Publisher("/depth/image_raw", Image, queue_size=1)
        self.depth_viz_pub = rospy.Publisher("/depth/image_mono8", Image, queue_size=1)
        
        self.dav2_model = make_dav2_model()
        self.image_sub = rospy.Subscriber("/vswarm302/cam2/image_raw", Image, self.image_callback, queue_size=1)
        #self.image_sub = rospy.Subscriber("/cam2/image_raw", Image, self.image_callback, queue_size=1)
        
        self.header = None
        self.loop_rate = rospy.Rate(30)
        self.img = None
        rospy.loginfo("DepthEstimatorNode initialized and waiting for images...")

    def image_callback(self, msg):
        try:
            self.img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.header = msg.header 
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {}".format(e))
            return

    def depth_process(self):
        while not rospy.is_shutdown():
            if self.img is not None:
                # Run depth inference, output shape: (H, W)
                depth = self.dav2_model.infer_image(self.img, input_size=244)

                # -------- 5–95 percentile filtering --------
                # Remove invalid values (NaN or Inf)
                valid_mask = np.isfinite(depth)

                if np.any(valid_mask):
                    # Compute 5th and 95th percentiles
                    d_min = np.percentile(depth[valid_mask], 5)
                    d_max = np.percentile(depth[valid_mask], 95)

                    # Clip extreme values to reduce outlier influence
                    depth_clipped = np.clip(depth, d_min, d_max)

                    # Normalize to [0, 1]
                    depth_norm = (depth_clipped - d_min) / (d_max - d_min + 1e-6)
                else:
                    # Fallback if no valid pixels
                    depth_norm = np.zeros_like(depth)

                # -------- Visualization --------
                # Convert normalized depth to 8-bit image for visualization
                depth_vis = (depth_norm * 255).astype(np.uint8)

                try:
                    # Publish original depth map as float32 (H, W)
                    depth_msg = self.bridge.cv2_to_imgmsg(
                        depth.astype(np.float32),
                        encoding="32FC1"
                    )

                    # Publish normalized visualization image
                    depth_viz_msg = self.bridge.cv2_to_imgmsg(
                        depth_vis,
                        encoding="mono8"
                    )

                    # Keep header consistent with input image
                    depth_msg.header = self.header
                    depth_viz_msg.header = self.header

                    self.depth_pub.publish(depth_msg)
                    self.depth_viz_pub.publish(depth_viz_msg)

                except CvBridgeError as e:
                    rospy.logerr(f"CvBridge Error when converting depth: {e}")

            self.loop_rate.sleep()

if __name__ == '__main__':
    try:
        img_sub = DepthEstimatorNode()
        img_sub.depth_process()
    except rospy.ROSInterruptException:
        pass