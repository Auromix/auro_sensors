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
# Description: BasicCamera class and CameraIntrinsics class.                  #
# Author: Herman Ye                                                           #
###############################################################################

from abc import ABC, abstractmethod
import numpy as np
from typing import List


class CameraIntrinsics:
    def __init__(self, fx: float, fy: float, ppx: float, ppy: float, coeffs: List[float], width: int, height: int):
        """Initializes the CameraIntrinsics with the given parameters.

        Args:
            fx (float): The focal length in the x-axis.
            fy (float): The focal length in the y-axis.
            ppx (float): The x-coordinate of the principal point.
            ppy (float): The y-coordinate of the principal point.
            coeffs (List[float]): The distortion coefficients.
            width (int): The width of the image.
            height (int): The height of the image.
        """
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy
        self.coeffs = coeffs
        self.width = width
        self.height = height

    def get_intrinsics_matrix(self) -> List[List[float]]:
        """Returns the camera intrinsics matrix.

        Returns:
            List[List[float]]: The 3x3 camera intrinsics matrix.
        """
        return [
            [self.fx, 0, self.ppx],
            [0, self.fy, self.ppy],
            [0, 0, 1]
        ]

    def __repr__(self) -> str:
        """Provides a string representation of the CameraIntrinsics object.

        Returns:
            str: String representation of the CameraIntrinsics object.
        """
        return (f"CameraIntrinsics(fx={self.fx}, fy={self.fy}, ppx={self.ppx}, "
                f"ppy={self.ppy}, coeffs={self.coeffs}, width={self.width}, height={self.height})")


class BasicCamera(ABC):
    def __init__(self, camera_config: dict):
        """Initializes the BasicCamera with a given configuration.

        Args:
            camera_config (dict): The configuration dictionary for the camera.
        """
        self.config: dict = camera_config
        self.init_camera(self.config)

    @abstractmethod
    def init_camera(self, camera_config: dict) -> bool:
        """Initializes the camera with the provided configuration.

        Args:
            camera_config (dict): The configuration dictionary for the camera.

        Returns:
            bool: True if the initialization was successful, False otherwise.
        """
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
    def get_intrinsics(self) -> CameraIntrinsics:
        """Gets the camera intrinsics.

        Returns:
            CameraIntrinsics: An instance of the CameraIntrinsics class containing the camera intrinsics.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_current_frame(self) -> np.ndarray:
        """Gets the current image frame from the camera.

        Returns:
            np.ndarray: The current frame captured by the camera as a numpy array.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_params(self) -> dict:
        """Gets the camera parameters.

        Returns:
            dict: A dictionary containing the camera parameters.
        """
        raise NotImplementedError("Not implemented yet")

    def set_params(self, params: dict) -> bool:
        """Sets the camera parameters.

        Args:
            params (dict): A dictionary containing the camera parameters to set.

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
