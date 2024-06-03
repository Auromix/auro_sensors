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
# Description: Realsense camera class                                         #
# Author: Herman Ye                                                           #
###############################################################################


import pyrealsense2 as rs
import numpy as np
import cv2
import os
import imageio
import time
from typing import List, Dict, Optional

from auro_utils.loggers.logger import Logger

from basic_camera import BasicCamera
from basic_camera import CameraIntrinsics


class RealsenseCamera(BasicCamera):
    def __init__(self, camera_config: dict, logger: Logger = None):
        # Init logger
        if logger is None:
            self.logger = Logger(log_level="debug")
        else:
            self.logger = logger

        # Initialize camera
        super().__init__(camera_config)

    def __del__(self):
        """Stop the RealSense pipeline."""

        self.stop()

    def init_camera(self, camera_config: Dict[str, any]) -> bool:
        """Initializes the camera with the provided configuration.

        Args:
            camera_config (Dict[str, any]): The configuration dictionary for the camera.

        Returns:
            bool: True if the initialization was successful, False otherwise.
        """
        self.camera_config = camera_config
        self.camera_started = False
        self.logger.log_debug(
            f"Initializing {camera_config['camera_type']}...")

        try:
            # Init realsense pipeline
            self.pipeline = rs.pipeline()
            # Init realsense config
            self.realsense_config = rs.config()
            # Enable specific camera device
            if camera_config.get('serial_number'):
                self.realsense_config.enable_device(
                    camera_config['serial_number'])
            # Config streams
            self.width = camera_config.get('width', 1280)
            self.height = camera_config.get('height', 720)
            self.fps = camera_config.get('fps', 30)

            self.color_format = camera_config.get(
                'color_format', 'bgr8')
            self.depth_format = camera_config.get(
                'depth_format', 'z16')
            self.ir_format = camera_config.get('ir_format', 'y8')
            if self.color_format not in [
                'bgr8', 'rgb8', 'bgra8', 'rgba8', 'yuyv', 'yuv422', 'yuv444', 'mono8', 'mono16'
            ]:
                raise ValueError(
                    f"Invalid color format: {self.color_format}")
            if self.depth_format not in [
                'z16', 'disparity16', 'xyz32f', 'y8'
            ]:
                raise ValueError(
                    f"Invalid depth format: {self.depth_format}")

            if self.ir_format not in [
                'y8', 'y16'
            ]:
                raise ValueError(
                    f"Invalid ir format: {self.ir_format}")

            # Remap to rs.format
            self.color_format = getattr(rs.format, self.color_format.lower())
            self.depth_format = getattr(rs.format, self.depth_format.lower())
            self.ir_format = getattr(rs.format, self.ir_format.lower())
            self.logger.log_debug(
                f"Configuring camera with width: {self.width}, height: {self.height}, fps: {self.fps}, color_format: {self.color_format}, depth_format: {self.depth_format}, ir_format: {self.ir_format}")
            # Enable streams
            self.realsense_config.enable_stream(
                rs.stream.depth, self.width, self.height, self.depth_format, self.fps
            )
            self.realsense_config.enable_stream(
                rs.stream.color, self.width, self.height, self.color_format, self.fps
            )
            self.realsense_config.enable_stream(
                rs.stream.infrared, 1, self.width, self.height, self.ir_format, self.fps)
            self.realsense_config.enable_stream(
                rs.stream.infrared, 2, self.width, self.height, self.ir_format, self.fps
            )

            # Align to
            align_to = camera_config.get('align_to', 'color')
            assert align_to in ['color', 'depth']
            align_to = getattr(rs.stream, align_to.lower())
            self.align = rs.align(align_to)
            self.logger.log_debug(f"Aligning to {align_to}")

            # Start camera pipeline
            self.start()

            # Get depth scale after starting the pipeline
            self.depth_scale = self.pipeline_profile.get_device(
            ).first_depth_sensor().get_depth_scale()
            self.logger.log_debug(f"Depth scale: {self.depth_scale}")

            self.logger.log_success(
                f"{camera_config['camera_type']} initialized.")
        except BaseException as e:
            self.logger.log_error(f"Failed to initialize camera: {e}")
            self.stop()
            raise

        return True

    def start(self) -> bool:
        """Starts the camera stream.

        Returns:
            bool: True if the camera stream was started successfully, False otherwise.
        """
        # Start pipeline
        try:
            self.pipeline_profile = self.pipeline.start(self.realsense_config)
            self.logger.log_info(
                f"{self.camera_config['camera_type']} started.")
            self.camera_started = True
            return True
        except BaseException as e:
            self.logger.log_error(
                f"Failed to start {self.camera_config['camera_type']}: {e}")
            raise

    def stop(self) -> bool:
        """Stops the camera stream.

        Returns:
            bool: True if the camera stream was stopped successfully, False otherwise.
        """
        try:
            if self.pipeline and self.camera_started:
                self.pipeline.stop()
                self.logger.log_info(
                    f"{self.camera_config['camera_type']} stopped."
                )
                self.camera_started = False
                return True
            else:
                self.logger.log_warning(
                    f"Received stop request, but {self.camera_config['camera_type']} was not running, skipping stop.")
                return False
        except BaseException as e:
            self.logger.log_error(
                f"Failed to stop {self.camera_config['camera_type']}: {e}")
            raise

    def get_current_frames(self) -> Dict[str, np.ndarray]:
        """Gets the current image frames from the camera.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the current frames captured by the camera. Keys could be 'color', 'depth', 'ir1', 'ir2', etc.
        """
        # Get realsense frames
        frames = self.pipeline.wait_for_frames()
        # Align the frames to color
        aligned_frames = self.align.process(frames)
        # Get frames data
        depth_data = np.asanyarray(aligned_frames.get_depth_frame().get_data())
        color_data = np.asanyarray(aligned_frames.get_color_frame().get_data())
        ir1_data = np.asanyarray(
            aligned_frames.get_infrared_frame(1).get_data())
        ir2_data = np.asanyarray(
            aligned_frames.get_infrared_frame(2).get_data())
        # Construct the dictionary of frames
        current_frames = {
            "color": color_data,
            "depth": depth_data,
            "ir1": ir1_data,
            "ir2": ir2_data,
        }

        return current_frames

    def get_depth(self, clip: Optional[float] = None, scale: Optional[float] = None) -> np.ndarray:
        """
        Gets the depth frame data from the camera.

        Args:
            clip (Optional[float]): Maximum depth value to clip to, in meters. Defaults to None.
            scale (Optional[float]): Scale factor for depth values. Defaults to None.

        Returns:
            np.ndarray: The depth frame data.
        """
        frames = self.get_current_frames()
        data = frames["depth"]

        # Clip
        if clip:
            clip = clip / self.depth_scale
            data = np.clip(data, 0, clip)
            data[data == clip] = 0
        # Scale
        if scale:
            data = data * scale

        return data

    def get_color(self, bgr2rgb: bool = False) -> np.ndarray:
        """
        Gets the color frame data from the camera.

        Args:
            bgr2rgb (bool): Whether to convert BGR to RGB. Defaults to False.

        Returns:
            np.ndarray: The color frame data.
        """
        frames = self.get_current_frames()
        data = frames["color"]
        if bgr2rgb:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        return data

    def get_ir(self, ir: int = 1) -> np.ndarray:
        """
        Gets the infrared frame data from the camera.

        Args:
            ir (int): Infrared channel to retrieve (1 or 2). Defaults to 1.

        Returns:
            np.ndarray: The infrared frame data.
        """
        assert ir in [1, 2], "ir should be 1 or 2"
        frames = self.get_current_frames()
        if ir == 1:
            data = frames["ir1"]
        else:
            data = frames["ir2"]
        return data

    def get_intrinsics(self, camera_type="color") -> CameraIntrinsics:
        """Gets the camera intrinsics.

        Args:
            camera_type (str, optional): The type of camera. Defaults to "color". Only used for multi-camera setups.
        Returns:
            CameraIntrinsics: An instance of the CameraIntrinsics class containing the camera intrinsics.
        """

        assert camera_type in ['color', 'depth', 'ir1',
                               'ir2'], "camera_type should be color, depth, ir1 or ir2"

        if camera_type == "color":
            intrinsics = self.pipeline_profile.get_stream(
                rs.stream.color).as_video_stream_profile().get_intrinsics()
        elif camera_type == "depth":
            intrinsics = self.pipeline_profile.get_stream(
                rs.stream.depth).as_video_stream_profile().get_intrinsics()

        elif camera_type == "ir1":
            intrinsics = self.pipeline_profile.get_stream(
                rs.stream.infrared1).as_video_stream_profile().get_intrinsics()

        elif camera_type == "ir2":
            intrinsics = self.pipeline_profile.get_stream(
                rs.stream.infrared2).as_video_stream_profile().get_intrinsics()
        else:
            raise ValueError("Invalid camera_type")
        # Construct a CameraIntrinsics object
        camera_intrinsics = CameraIntrinsics(width=intrinsics.width, height=intrinsics.height, fx=intrinsics.fx, fy=intrinsics.fy,
                                             ppx=intrinsics.ppx, ppy=intrinsics.ppy, distortion_coefficients=intrinsics.coeffs, distortion_model=intrinsics.model)
        return camera_intrinsics

    def save_data(self, data: np.ndarray, name: str, prefix: str = "", suffix: str = ".png") -> bool:
        """Saves the camera data to a file.

        Args:
            data (np.ndarray): The data to save.
            name (str): The name of the file.
            prefix (str, optional): The prefix to add to the filename. Defaults to "".
            suffix (str, optional): The suffix to add to the filename. Defaults to ".png".

        Returns:
            bool: True if the data was saved successfully, False otherwise.
        """
        try:
            # Init camera data save directory
            if not self.camera_config.get('camera_data_save_directory'):
                self.project_dir = os.path.dirname(
                    os.path.abspath(os.path.dirname(__file__)))
                self.camera_data_save_directory = os.path.join(
                    self.project_dir, 'camera_data')
            else:
                self.camera_data_save_directory = self.camera_config['camera_data_save_directory']

            # Check if directory exists
            if not os.path.exists(self.camera_data_save_directory):
                self.logger.log_warning(
                    f"Directory {self.camera_data_save_directory} does not exist.")
                self.logger.log_info(
                    f"Creating directory: {self.camera_data_save_directory}")
                os.makedirs(self.camera_data_save_directory)

            # self.logger.log_debug(
            #     f"Camera data save directory: {self.camera_data_save_directory}")

            full_path = os.path.join(
                self.camera_data_save_directory, f"{prefix}{name}{suffix}")

            imageio.imwrite(full_path, data)
            self.logger.log_debug(
                f"Saved {prefix}_{name}{suffix} to {full_path}.")
        except BaseException as e:
            self.logger.log_error(
                f"Failed to save {prefix}_{name}{suffix} to {full_path}.")
            self.logger.log_error(e)
            return False
        return True


def example_get_intrinsics():
    camera_config = {
        'serial_number': "",
        'camera_type': 'Realsense D415',
        'camera_data_save_directory': "",
        'width': 1280,
        'height': 720,
        'fps': 30,
        'color_format': 'bgr8',
        'depth_format': 'z16',
        'ir_format': 'y8',
    }

    my_realsense_camera = RealsenseCamera(camera_config)

    intrinsics = my_realsense_camera.get_intrinsics(camera_type='depth')
    print(intrinsics)
    fx = intrinsics.fx
    fy = intrinsics.fy
    ppx = intrinsics.ppx
    ppy = intrinsics.ppy
    print(fx, fy, ppx, ppy)
    distortion_coefficients = intrinsics.distortion_coefficients
    print(distortion_coefficients)
    matrix = intrinsics.get_intrinsics_matrix()
    print(matrix)


def example_display_video():
    camera_config = {
        'serial_number': "",
        'camera_type': 'Realsense D415',
        'camera_data_save_directory': "",
        'width': 1280,
        'height': 720,
        'fps': 30,
        'color_format': 'bgr8',
        'depth_format': 'z16',
        'ir_format': 'y8',
    }
    my_realsense_camera = RealsenseCamera(camera_config)

    try:
        while True:
            # # Get frames from each sensor at different time[method 1]
            # color = my_realsense_camera.get_color()
            # depth = my_realsense_camera.get_depth(clip=3.0)
            # ir1 = my_realsense_camera.get_ir(ir=1)
            # ir2 = my_realsense_camera.get_ir(ir=2)

            # Get frames from each sensor at same time [method 2]
            frames = my_realsense_camera.get_current_frames()
            color = frames['color']
            depth = frames['depth']
            ir1 = frames['ir1']
            ir2 = frames['ir2']

            # Apply color map to depth frame for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)

            # Resize frames if needed
            color_resized = cv2.resize(color, (640, 360))
            depth_colormap_resized = cv2.resize(depth_colormap, (640, 360))
            ir1_resized = cv2.resize(ir1, (640, 360))
            ir2_resized = cv2.resize(ir2, (640, 360))

            # Convert grayscale IR images to BGR for display
            ir1_resized_bgr = cv2.cvtColor(ir1_resized, cv2.COLOR_GRAY2BGR)
            ir2_resized_bgr = cv2.cvtColor(ir2_resized, cv2.COLOR_GRAY2BGR)

            # Combine frames into a 2x2 grid
            top_row = np.hstack((color_resized, depth_colormap_resized))
            bottom_row = np.hstack((ir1_resized_bgr, ir2_resized_bgr))
            combined_image = np.vstack((top_row, bottom_row))

            # Display the combined image
            cv2.imshow('Realsense D415 Frames', combined_image)

            # Check for key press to exit
            key = cv2.waitKey(1)
            if key & 0xFF in (27, ord('q')):  # 27 is the esc key
                break

    except Exception as e:
        my_realsense_camera.logger.log_error(f"An error occurred: {str(e)}")
    finally:
        # # Stop the camera
        # my_realsense_camera.stop()
        # Destroy all OpenCV windows
        cv2.destroyAllWindows()


def example_save_data():
    camera_config = {
        'serial_number': "",
        'camera_type': 'Realsense D415',
        'camera_data_save_directory': "",
        'width': 1280,
        'height': 720,
        'fps': 30,
        'color_format': 'bgr8',
        'depth_format': 'z16',
        'ir_format': 'y8',
    }

    my_realsense_camera = RealsenseCamera(camera_config)

    frames = my_realsense_camera.get_current_frames()
    color = my_realsense_camera.get_color(bgr2rgb=True)
    depth = my_realsense_camera.get_depth()
    ir1 = my_realsense_camera.get_ir(ir=1)
    ir2 = my_realsense_camera.get_ir(ir=2)
    # Save data
    my_realsense_camera.save_data(data=color, name='color')
    my_realsense_camera.save_data(data=depth, name='depth')
    my_realsense_camera.save_data(data=ir1, name='ir1')
    my_realsense_camera.save_data(data=ir2, name='ir2')


if __name__ == '__main__':
    # example_get_intrinsics()
    # example_save_data()
    example_display_video()
