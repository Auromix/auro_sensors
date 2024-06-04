import cv2
import numpy as np
from auro_sensors.cameras.realsense_camera import RealsenseCamera


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

    color = my_realsense_camera.get_color(bgr2rgb=True)
    depth = my_realsense_camera.get_depth()
    ir1 = my_realsense_camera.get_ir(ir=1)
    ir2 = my_realsense_camera.get_ir(ir=2)
    # Save data
    my_realsense_camera.save_data(data=color, name='color')
    my_realsense_camera.save_data(data=depth, name='depth')
    my_realsense_camera.save_data(data=ir1, name='ir1')
    my_realsense_camera.save_data(data=ir2, name='ir2')


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
        # Destroy all OpenCV windows
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # example_get_intrinsics()
    # example_save_data()
    example_display_video()
