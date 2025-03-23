import open3d as o3d
import numpy as np
import cv2
import os

def get_pc(color, depth):
    if color is None or depth is None:
        raise ValueError("Error: Failed to load color or depth image.")
    color = color.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC, move to CPU if necessary
    depth = depth.cpu().numpy() 
    depth = depth[0, :, :] 
    color = np.asarray(color, dtype=np.uint8)
    depth = np.asarray(depth, dtype=np.float32)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    depth = cv2.resize(depth, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
    depth = depth.squeeze()
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
    return pcd
# Save and visualize
# o3d.io.write_point_cloud("output.ply", pcd)
