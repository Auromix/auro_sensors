# !/usr/bin/env python3
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
# Description: BasicCamera class and CameraIntrinsics class.                  #
# Author: Herman Ye                                                           #
###############################################################################


from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict


class CameraIntrinsics:
    """Abstract base class for camera intrinsics."""

    def __init__(self, width: int, height: int, fx: float, fy: float, ppx: float, ppy: float, distortion_coefficients: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0], distortion_model: str = 'No Model'):
        """Initializes the CameraIntrinsics with the given parameters.

        Args:
            width (int): The width of the image.
            height (int): The height of the image.
            fx (float): The focal length in the x-axis.
            fy (float): The focal length in the y-axis.
            ppx (float): The x-coordinate of the principal point.
            ppy (float): The y-coordinate of the principal point.
            distortion_coefficients (List[float]): The distortion coefficients.
            distortion_model (str): The distortion model, for example 'Inverse Brown Conrady'.

        """
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy
        self.distortion_coefficients = distortion_coefficients
        self.distortion_model = distortion_model

    def get_intrinsics_matrix(self) -> np.ndarray:
        """Returns the camera intrinsics matrix.

        Returns:
            np.ndarray: The 3x3 camera intrinsics matrix.
        """
        return np.array([
            [self.fx, 0, self.ppx],
            [0, self.fy, self.ppy],
            [0, 0, 1]
        ])

    def __repr__(self) -> str:
        """Provides a string representation of the CameraIntrinsics object.

        Returns:
            str: String representation of the CameraIntrinsics object.
        """
        return (f"CameraIntrinsics(width={self.width}, height={self.height}, fx={self.fx}, fy={self.fy}, ppx={self.ppx}, "
                f"ppy={self.ppy}, distortion_coefficients={self.distortion_coefficients}, distortion_model={self.distortion_model})")


class BasicCamera(ABC):
    """Abstract base class for basic camera functionality."""

    def __init__(self, camera_config: Dict[str, any]):
        """Initializes the BasicCamera with a given configuration.

        Args:
            camera_config (Dict[str, any]): The configuration dictionary for the camera.
        """

        self.init_camera(camera_config=camera_config)

    @abstractmethod
    def init_camera(self, camera_config: Dict[str, any]) -> bool:
        """Initializes the camera with the provided configuration.

        Args:
            camera_config (Dict[str, any]): The configuration dictionary for the camera.

        Returns:
            bool: True if the initialization was successful, False otherwise.
        """
        self.camera_config = camera_config
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def start(self) -> bool:
        """Starts the camera stream.

        Returns:
            bool: True if the camera stream was started successfully, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def stop(self) -> bool:
        """Stops the camera stream.

        Returns:
            bool: True if the camera stream was stopped successfully, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_current_frames(self) -> Dict[str, np.ndarray]:
        """Gets the current image frames from the camera.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the current frames captured by the camera. Keys could be 'color', 'depth', 'ir1', 'ir2', etc.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_intrinsics(self, camera_type="color") -> CameraIntrinsics:
        """Gets the camera intrinsics.

        Args:
            camera_type (str, optional): The type of camera. Defaults to "color". Only used for multi-camera setups.
        Returns:
            CameraIntrinsics: An instance of the CameraIntrinsics class containing the camera intrinsics.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_params(self) -> Dict[str, any]:
        """Gets the camera parameters.

        Returns:
            Dict[str, any]: A dictionary containing the camera parameters.
        """
        raise NotImplementedError("Not implemented yet")

    def set_params(self, params: Dict[str, any]) -> bool:
        """Sets the camera parameters.

        Args:
            params (Dict[str, any]): A dictionary containing the camera parameters to set.

        Returns:
            bool: True if the parameters were set successfully, False otherwise.
        """
        raise NotImplementedError("Not implemented yet")

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
        raise NotImplementedError("Not implemented yet")
