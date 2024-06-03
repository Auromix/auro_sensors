import cv2
import numpy as np
import pyrealsense2 as rs
a=pyrealsense2.intrinsics

ARUCO_DICT = {
    'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
    'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
    'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
    'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
    'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
    'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
    'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
    'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
    'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
    'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
    'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
    'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000,
    'DICT_7X7_50': cv2.aruco.DICT_7X7_50,
    'DICT_7X7_100': cv2.aruco.DICT_7X7_100,
    'DICT_7X7_250': cv2.aruco.DICT_7X7_250,
    'DICT_7X7_1000': cv2.aruco.DICT_7X7_1000,
    'DICT_ARUCO_ORIGINAL': cv2.aruco.DICT_ARUCO_ORIGINAL,
    'DICT_APRILTAG_16h5': cv2.aruco.DICT_APRILTAG_16h5,
    'DICT_APRILTAG_25h9': cv2.aruco.DICT_APRILTAG_25h9,
    'DICT_APRILTAG_36h10': cv2.aruco.DICT_APRILTAG_36h10,
    'DICT_APRILTAG_36h11': cv2.aruco.DICT_APRILTAG_36h11
}


class Aruco:
    def __init__(self, aruco_config: dict, camera):
        # Get config
        self.config = aruco_config
        # Get camera
        self.camera = camera
        # Get intrinsics
        self.camera_intrinsics = self.camera.get_intrinsics(
            camera_type='color')
        self.camera_intrinsics_matrix = np.array([[self.camera_intrinsics.fx, 0, self.camera_intrinsics.ppx], [
                                                 0, self.camera_intrinsics.fy, self.camera_intrinsics.ppy], [0, 0, 1]])
        self.camera_distortion_coefficients = np.array(
            self.camera_intrinsics.coeffs)

        self.aruco_detectors = []

        for marker_config in self.config.values():
            # Get dictionary name
            dictionary_name = marker_config.get('dictionary_name')
            aruco_dict_key = ARUCO_DICT.get(dictionary_name)
            if aruco_dict_key is None:
                raise ValueError(f"Unknown dictionary name: {dictionary_name}")
            aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_key)
            # Get aruco parameters
            aruco_params = cv2.aruco.DetectorParameters()
            # Init aruco detector
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

            # Add detector to list
            self.aruco_detectors.append(
                (detector, marker_config))

    def get_aruco_poses(self, color):
        marker_poses = []
        for detector, config in self.aruco_detectors:
            corners, ids, _ = detector.detectMarkers(color)
            if ids is None:
                continue
            for idx, id in enumerate(ids):
                if id == config['marker_id']:
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, config['marker_size'], self.camera_intrinsics_matrix, self.camera_distortion_coefficients)
                    quaternion = self.rotation_vector_to_quaternion(
                        np.array(rvec[0][0]))
                    marker_pose = np.concatenate((tvec[0][0], quaternion))
                    marker_poses.append({
                        'name': config['marker_name'],
                        'pose': marker_pose
                    })
        return marker_poses

    def draw_aruco_poses(self, color):
        for detector, config in self.aruco_detectors:
            corners, ids, _ = detector.detectMarkers(color)
            # Check if aruco is detected
            if ids is None:
                continue
            # Estimate pose and draw axis for each marker
            for idx, id in enumerate(ids):
                if id == config['marker_id']:
                    # Draw detected marker
                    cv2.aruco.drawDetectedMarkers(color, [corners[idx]])

                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners[idx], config['marker_size'], self.camera_intrinsics_matrix, self.camera_distortion_coefficients)
                    translation = tvec[0][0]
                    quaternion = self.rotation_vector_to_quaternion(
                        np.array(rvec[0][0]))

                    # Draw axis
                    axis_length = 0.05
                    # Ensure rvec and tvec have the correct shape
                    rvec = rvec[0].reshape((3, 1))
                    tvec = tvec[0].reshape((3, 1))
                    color = cv2.drawFrameAxes(
                        color, self.camera_intrinsics_matrix, self.camera_distortion_coefficients, rvec, tvec, axis_length)
                    # Draw pose
                    pose_text1 = f"{config['marker_name']}({config['dictionary_name']}, {config['marker_id']}, {config['marker_size']}):"
                    pose_text2 = f"[{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}, {quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f}]"
                    # Get the position of the first corner to place the text
                    corner_pos = corners[idx][0][0]
                    # Adjust the position above the corner
                    text_pos1 = (int(corner_pos[0]), int(corner_pos[1]) - 30)
                    text_pos2 = (int(corner_pos[0]), int(corner_pos[1]) - 10)
                    # Put the pose text on the image
                    cv2.putText(color, pose_text1, text_pos1, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(color, pose_text2, text_pos2, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return color

    def rotation_vector_to_quaternion(self, rvec):
        norm = np.linalg.norm(rvec)
        if norm < 1e-5:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = rvec / norm
        half_angle = norm / 2.0
        sin_half = np.sin(half_angle)
        cos_half = np.cos(half_angle)
        qw = cos_half
        qx = axis[0] * sin_half
        qy = axis[1] * sin_half
        qz = axis[2] * sin_half
        return np.array([qx, qy, qz, qw])


if __name__ == '__main__':
    from auro_sensors.cameras.realsense_camera import RealsenseCamera
    camera_config = {
        'serial_number': "",
        'camera_type': 'Realsense D415',
        'camera_data_save_directory': "",
    }
    my_realsense_camera = RealsenseCamera(camera_config)

    aruco_config = {
        "my_marker1": {
            "dictionary_name": "DICT_ARUCO_ORIGINAL",
            "marker_name": "my_marker1",
            "marker_size": 0.1,
            "marker_id": 233
        },
        "my_marker2": {
            "dictionary_name": "DICT_ARUCO_ORIGINAL",
            "marker_name": "my_marker2",
            "marker_size": 0.1,
            "marker_id": 996
        },
        "my_marker3": {
            "dictionary_name": "DICT_ARUCO_ORIGINAL",
            "marker_name": "my_marker3",
            "marker_size": 0.1,
            "marker_id": 789
        }
    }

    my_aruco = Aruco(aruco_config=aruco_config, camera=my_realsense_camera)

    try:
        while True:
            color = my_realsense_camera.get_color()
            if color is None:
                continue

            marker_pose = my_aruco.get_aruco_poses(color)
            if marker_pose is not None:
                print(f'Marker Pose: {marker_pose}')

            output_color = my_aruco.draw_aruco_poses(color)

            cv2.imshow('Aruco Detection', output_color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        my_realsense_camera.stop()
        cv2.destroyAllWindows()
