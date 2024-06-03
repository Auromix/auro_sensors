#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# Copyright © 2023-2024 Auromix.                                              #
#                                                                             #
# Licensed under the Apache License, Version 2.0 (the "License");             #
# You may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at                                     #
#                                                                             #
#     http://www.apache.org/licenses/LICENSE-2.0                              #
#                                                                             #
# Unless required by applicable law or agreed to in writing, software         #
# distributed under the License is distributed on an "AS IS" BASIS,           #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
#                                                                             #
# Description: Realsense ROS1 data receiver                                   #
# Author: Herman Ye                                                           #
###############################################################################


import sys
import numpy as np
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
from auro_utils.loggers.logger import Logger
from typing import Optional

LOG_MODULE_TAG = (
    "Realsense ROS1"  # Tag for identifying log messages related to this module
)


class RealsenseCameraROS1:
    """
    Class for interfacing with a RealSense camera in ROS1 environment.

    roslaunch realsense2_camera rs_camera.launch should be run first.
    
    If you are using ROS1 in conda env
    conda install -c conda-forge -c robostack ros-noetic-desktop

    """

    def __init__(self):
        """
        Initializes the RealsenseCameraROS1 instance.
        """
        # Init logger
        self.logger = Logger()
        # Init ROS node if not initialized
        if not rospy.core.is_initialized():
            rospy.init_node("realsense_camera_ros1_node", anonymous=True)
        # Init CV bridge
        self.ros_cv_bridge = CvBridge()
        self.ros_initialized = True
        # Log
        self.logger.log_info("Realsense Camera ROS1 initialized.", tag=LOG_MODULE_TAG)

    def get_color_data(self, topic: str = None) -> np.ndarray:
        """
        Retrieves color image data from the specified topic.

        Args:
            topic (str): ROS topic from which to receive the color image data. If None, defaults to "/camera/color/image_raw".

        Returns:
            np.ndarray: Color image data in OpenCV-compatible format.
        """
        if self.ros_cv_bridge is None:
            self.logger.log_error("ROS1 CV bridge not initialized.", tag=LOG_MODULE_TAG)
            return None

        if topic is None:
            topic = "/camera/color/image_raw"
        try:
            color_msg = rospy.wait_for_message(topic, Image, 3)

            color = self.ros_cv_bridge.imgmsg_to_cv2(
                color_msg, desired_encoding="passthrough"
            )
        except Exception as e:
            self.logger.log_error(
                f"Error occurred during getting color data:\n{e}", tag=LOG_MODULE_TAG
            )
            sys.exit(1)
        return color

    def get_depth_data(
        self, topic: str = None, clip: Optional[float] = None
    ) -> np.ndarray:
        """
        Retrieves depth image data from the specified topic.

        Args:
            topic (str): ROS topic from which to receive the depth image data. If None, defaults to "/camera/aligned_depth_to_color/image_raw".
            clip (float, optional): Value to clip depths to. If None, no clipping is performed.

        Returns:
            np.ndarray: Depth image data in meters.
        """
        if self.ros_cv_bridge is None:
            self.logger.log_error("ROS CV bridge not initialized.", tag=LOG_MODULE_TAG)
            return None
        if topic is None:
            topic = "/camera/aligned_depth_to_color/image_raw"

        try:
            depth_msg = rospy.wait_for_message(topic, Image, 2)
        except Exception as e:
            self.logger.log_error(
                f"Error occurred during getting depth data:\n{e}", tag=LOG_MODULE_TAG
            )
            # Depth align to color check
            if self.get_color_data() is not None:
                self.logger.log_warning(
                    "Detected color image, but no aligned depth image.\nPlease enable depth align to color in camera settings.",
                    tag=LOG_MODULE_TAG,
                )
            sys.exit(1)

        depth_image = self.ros_cv_bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding="passthrough"
        )

        # Millimeter to meter
        depth = np.float32(depth_image) / 1000.0

        # Clip depth
        if clip is not None:
            depth = np.clip(depth, 0.0, clip)
            depth[depth == clip] = 0
        return depth

    def get_camera_intrinsics(self, topic: str = None) -> np.ndarray:
        """
        Retrieves camera intrinsics matrix from the specified topic.

        Args:
            topic (str): ROS topic from which to receive the camera intrinsics data. If None, defaults to "/camera/color/camera_info".

        Returns:
            np.ndarray: Camera intrinsics matrix.
        """
        if topic is None:
            topic = "/camera/color/camera_info"

        try:
            info = rospy.wait_for_message(topic, CameraInfo, 3)
            camera_intrinsics_matrix = np.array(info.K).reshape(3, 3)
        except Exception as e:
            self.logger.log_error(
                f"Error occurred during getting camera intrinsics:\n{e}",
                tag=LOG_MODULE_TAG,
            )
            sys.exit(1)
        return camera_intrinsics_matrix

    def get_camera_info(self, topic: str = None) -> CameraInfo:
        """
        Retrieves camera info message from the specified topic.

        Args:
            topic (str): ROS topic from which to receive the camera info message. If None, defaults to "/camera/color/camera_info".

        Returns:
            CameraInfo: Camera information message.
        """
        if topic is None:
            topic = "/camera/color/camera_info"

        try:
            camera_info = rospy.wait_for_message(topic, CameraInfo, 3)

        except Exception as e:
            self.logger.log_error(
                f"Error occurred during getting camera info:\n{e}", tag=LOG_MODULE_TAG
            )
        return camera_info
