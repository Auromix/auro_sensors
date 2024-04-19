import matplotlib.pyplot as plt
from auro_sensors.camera.realsense.realsense_camera import RealsenseCamera

camera = RealsenseCamera()
color_data = camera.get_color_data()
depth_data = camera.get_depth_data()
depth_scale = camera.depth_scale
intrinsics = camera.get_camera_intrinsics()

# Print info
print("Color data shape:", color_data.shape)
print("Depth data shape:", depth_data.shape)
print("Depth scale:", depth_scale)
print("Camera Intrinsics:")
print(intrinsics)

# Show color&depth image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(color_data)
plt.title("Color Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(depth_data, cmap="gray")
plt.title("Depth Image")
plt.axis("off")

plt.show()
