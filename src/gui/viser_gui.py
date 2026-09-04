import numpy as np
import viser
import viser.transforms as tf
import torch
import time
import threading

class ViserGUI:
    """
    A lightweight, high-performance Viser-based GUI for visualizing 3D Gaussian Splatting SLAM.
    This runs entirely in the background and streams to localhost:8080.
    """
    def __init__(self, port=8080):
        self.port = port
        print(f"[ViserGUI] Starting Viser server on http://localhost:{port}...")
        self.server = viser.ViserServer(port=self.port)
        
        # Keep track of added objects
        self.frustums = set()
        
        # Setup basic UI elements
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup basic UI toggles in the Viser web interface"""
        with self.server.gui.add_folder("Visibility"):
            self.show_trajectory = self.server.gui.add_checkbox("Show Trajectory", initial_value=True)
            self.show_points = self.server.gui.add_checkbox("Show Point Cloud", initial_value=True)
            self.show_gaussians = self.server.gui.add_checkbox("Show Gaussian Splats", initial_value=True)
            
            # Link toggles to scene node visibility
            self.show_trajectory.on_update(
                lambda _: self.server.scene.client_components[0].set_visibility("/trajectory", self.show_trajectory.value)
                if len(self.server.scene.client_components) > 0 else None
            )
            
    def update_camera_frustum(self, cam_id: int, pose_c2w: np.ndarray, fx: float, fy: float, cx: float, cy: float, img_w: int, img_h: int, image=None):
        """
        Adds or updates a camera frustum in the viewer.
        pose_c2w: 4x4 camera to world transformation matrix
        """
        if not self.show_trajectory.value:
            return

        # Calculate FOV from focal length
        fov_y = 2.0 * np.arctan(img_h / (2.0 * fy))
        aspect = img_w / img_h
        
        # Extract translation and rotation (wxyz)
        pos = pose_c2w[:3, 3]
        rot_mat = pose_c2w[:3, :3]
        quat = tf.SO3.from_matrix(rot_mat).wxyz
        
        self.server.scene.add_camera_frustum(
            f"/trajectory/cam_{cam_id}",
            fov=fov_y,
            aspect=aspect,
            scale=0.05,
            image=image,
            position=pos,
            wxyz=quat,
            color=(255, 0, 0) if cam_id == "current" else (0, 255, 0)
        )
        self.frustums.add(cam_id)

    def update_point_cloud(self, points: np.ndarray, colors: np.ndarray = None, point_size=0.01):
        """
        Updates the global point cloud (e.g. from the tracking frontend).
        """
        if not self.show_points.value or len(points) == 0:
            return
            
        if colors is None:
            colors = np.ones_like(points) * 150 # Default gray
            
        self.server.scene.add_point_cloud(
            "/map/points",
            points=points,
            colors=colors,
            point_size=point_size
        )

    def update_gaussian_splats(self, xyz, scales, rotations_wxyz, colors, opacities):
        """
        Natively renders 3D Gaussian Splats in the browser using Viser's built-in splat rasterizer.
        Note: Convert tensors to numpy before passing.
        """
        if not self.show_gaussians.value or len(xyz) == 0:
            return
            
        self.server.scene.add_gaussian_splats(
            "/map/gaussians",
            centers=xyz,
            scales=scales,
            quaternions=rotations_wxyz, # Expects wxyz
            colors=colors,              # RGB values [0, 1] or [0, 255]
            opacities=opacities
        )

    def clear_trajectory(self):
        """Removes all camera frustums"""
        for cam_id in self.frustums:
            self.server.scene.remove(f"/trajectory/cam_{cam_id}")
        self.frustums.clear()

if __name__ == "__main__":
    # Example standalone usage test
    gui = ViserGUI()
    print("Viser test running. Open http://localhost:8080 in your browser.")
    
    # Generate some dummy data
    import math
    for i in range(100):
        # Dummy camera going in a circle
        c2w = np.eye(4)
        c2w[0, 3] = math.cos(i * 0.1)
        c2w[1, 3] = math.sin(i * 0.1)
        gui.update_camera_frustum(i, c2w, 500, 500, 320, 240, 640, 480)
        
        # Dummy point cloud
        pts = np.random.randn(1000, 3) * 0.5
        colors = np.random.randint(0, 255, (1000, 3), dtype=np.uint8)
        gui.update_point_cloud(pts, colors)
        
        time.sleep(0.1)
