# import open3d as o3d
# from open3d.visualization import draw_plotly
# import cv2
# color_path = "/home/ubuntu/cluster/nbl-users/Shreyas-Sushrut-Raghu/ria/3DFace_DB/iPhone12_filled/color/digital/aligned/train/S14_0.JPG"
# depth_path ="/home/ubuntu/cluster/nbl-users/Shreyas-Sushrut-Raghu/ria/3DFace_DB/iPhone12_filled/depth/digital/aligned/train/S14_0.JPG" 
# color = cv2.imread(color_path)
# depth = cv2.imread(depth_path)
# depth = cv2.resize(depth, (224,224))
# color = o3d.geometry.Image(color)
# depth = o3d.geometry.Image(depth)
# pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()

# rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
# pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

# # flip the orientation, so it looks upright, not upside-down
# pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

# # o3d.io.write_point_cloud("paper_exmp.ply", pcd)
# # draw_plotly([pcd])    # visualize the point clou

import open3d as o3d
import numpy as np
import cv2
import os
# Load images
color_path = "/home/ubuntu/cluster/nbl-users/Shreyas-Sushrut-Raghu/ria/3DFace_DB/iPhone12_filled/color/digital/aligned/train/S14_0.JPG"
depth_path ="/home/ubuntu/cluster/nbl-users/Shreyas-Sushrut-Raghu/ria/3DFace_DB/iPhone12_filled/depth/digital/aligned/train/S14_0.JPG" 

# Check file existence
if not os.path.exists(color_path) or not os.path.exists(depth_path):
    raise FileNotFoundError("Error: One or both image files not found!")

# Load images
color = cv2.imread(color_path)
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

if color is None or depth is None:
    raise ValueError("Error: Failed to load color or depth image.")

# Convert color to RGB
color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

# Resize depth to match color
depth = cv2.resize(depth, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)

# Convert to Open3D images
color_o3d = o3d.geometry.Image(color)
depth_o3d = o3d.geometry.Image(depth)

# iPhone 12 estimated intrinsics (Adjust if needed)
fx = 80
fy = 150.0  # Focal length (Change if distorted)
cx, cy = color.shape[1] / 2, color.shape[0] / 2  # Image center

pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=color.shape[1], height=color.shape[0],
    fx=fx, fy=fy, cx=cx, cy=cy
)

# Convert to RGBD Image with correct depth scale
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d, depth_o3d,
    depth_scale=1000.0,  # Try 500, 1000, 1500, 2000
    depth_trunc=3.0,
    convert_rgb_to_intensity=False
)

# Generate point cloud
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

# Flip the orientation, so it looks upright, not upside-down
# Use the correct transformation matrix
# pcd.transform([[-1, 1, 0, 0],    # 90 degrees clockwise in XY-plane
#                [-1, 0, 0, 0],
#                [0, 0, 1, 0],
#                [0, 0, 0, 1]])
pcd.transform([[-1, 0, 0, 0],  # Invert X
               [ 0, -1, 0, 0],  # Invert Y
               [ 0, 0, 1, 0],   # Keep Z
               [ 0, 0, 0, 1]])
# Save and visualize
o3d.io.write_point_cloud("output.ply", pcd)
