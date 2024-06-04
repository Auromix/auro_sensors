import cv2
from auro_sensors.cameras.realsense_camera import RealsenseCamera
from auro_sensors.utils.aruco import Aruco
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
    cv2.destroyAllWindows()
