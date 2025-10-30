import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

class MapManager:
    def __init__(self, grid_downsample_res=0.2, filter_dis=0.04, merge_dis=0.4, bound_radius=20.0):
        self.grid_downsample_res = grid_downsample_res
        self.filter_dis = filter_dis
        self.merge_dis = merge_dis
        self.bound_radius = bound_radius
        self.Map = np.empty((0, 3))  # Initialize the global map as 3D points

    def update_point_map(self, global_map, scan, transformation):
        """Update the global point map with new 3D points"""
        # Draw global map
        #o3d.visualization.draw_geometries([global_map], window_name="Global Map", width=640, height=480, left=50, top=50, mesh_show_back_face=True)
        # Draw local scan
        #o3d.visualization.draw_geometries([scan], window_name="Scan", width=640, height=480, left=50, top=50, mesh_show_back_face=True)
        
        # Transform local scan to global frame
        transformed_scan = self.transform_scan(scan, transformation)
        
        # Remove points that are too close to existing ones
        # global_map, removed_points = self.filter_close_points_3d(global_map, transformed_scan, self.filter_dis)
        
        # Apply 3D grid downsampling
        # global_map = self.grid_downsample(global_map, self.grid_downsample_res)
        
        # Merge close points (optional, commented out as in original)
        # global_map = self.merge_close_points_3d(global_map, self.merge_dis)
        
        # Add the transformed scan points to the global map
        global_map += transformed_scan
        
        #Draw global map
        #o3d.visualization.draw_geometries([global_map], window_name="Global Map", width=640, height=480, left=50, top=50, mesh_show_back_face=True)
        
        global_map = self.clip_local_map_by_box_3d(global_map, transformation, x_half=self.bound_radius, y_half=self.bound_radius, z_half=self.bound_radius)

        
        # Optionally apply voxel downsampling to reduce the number of points
        global_map = global_map.voxel_down_sample(self.grid_downsample_res)

        
        
        # Visualize the point map
        # self.plot_point_map(global_map, transformed_scan, removed_points)
        
        return global_map

    def transform_scan(self, scan, transformation):
        if not isinstance(scan, o3d.geometry.PointCloud):
                raise TypeError("Input scan must be an Open3D PointCloud object.")

        points = np.asarray(scan.points)                                                # Convert Open3D PointCloud to NumPy array
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))         # Convert points to homogeneous coordinates (Nx4)
       
        transformed_points = (transformation @ homogeneous_points.T).T                  # Apply transformation matrix (4x4) to points (Nx4) 
        transformed_pcd = o3d.geometry.PointCloud()                                     # Create a new Open3D PointCloud object 
        transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points[:, :3])  # Keep only the 3D coordinates

        return transformed_pcd

    def filter_close_points_3d(self, global_map, new_points, min_dist=0.04):
        """Removes new points that are too close to existing ones and returns removed points"""
        if len(global_map) == 0:
            return new_points, np.array([])  # No previous points, keep all new ones

        tree = cKDTree(global_map)
        distances, _ = tree.query(new_points)
        
        # Keep points that are farther than min_dist
        mask = distances > min_dist
        filtered_points = new_points[mask]
        
        # Removed points are those that were too close
        removed_points = new_points[~mask]
        
        return np.vstack((global_map, filtered_points)), removed_points

    def grid_downsample(self, points, grid_size):
        """Reduces the number of points by keeping one per 3D grid cell"""
        grid_indices = np.floor(points / grid_size).astype(int)
        unique_indices, unique_idx = np.unique(grid_indices, axis=0, return_index=True)
        return points[unique_idx]

    def merge_close_points_3d(self, points, merge_dist=0.4):
        """Merges nearby points by averaging their positions"""
        tree = cKDTree(points)
        clusters = tree.query_ball_tree(tree, merge_dist)

        new_points = []
        visited = set()
        for i, cluster in enumerate(clusters):
            if i in visited:
                continue
            cluster_points = points[cluster]
            new_points.append(cluster_points.mean(axis=0))
            visited.update(cluster)

        return np.array(new_points)
    
    def clip_local_map_by_box_3d(self, point_cloud, transformation, x_half, y_half, z_half):
        """Clip point cloud to a box (AABB) with custom dimensions around the robot."""
        # Robot position in global frame
        center = transformation[:3, 3]

        # Define box half-dimensions in x, y, z
        min_bound = center - np.array([x_half, y_half, z_half])
        max_bound = center + np.array([x_half, y_half, z_half])

        # Create the bounding box
        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        # Crop the point cloud
        return point_cloud.crop(aabb)


    def plot_point_map(self, global_map, transformed_scan=None, removed_points=None):
        """Visualize the 3D point map using Open3D"""
        # Create Open3D point clouds
        global_pcd = o3d.geometry.PointCloud()
        global_pcd.points = o3d.utility.Vector3dVector(global_map)
        global_pcd.paint_uniform_color([0, 0, 0])  # Black for global map

        geometries = [global_pcd]
        
        if transformed_scan is not None:
            scan_pcd = o3d.geometry.PointCloud()
            scan_pcd.points = o3d.utility.Vector3dVector(transformed_scan)
            scan_pcd.paint_uniform_color([0, 0, 1])  # Blue for new scan
            geometries.append(scan_pcd)
        
        if removed_points is not None and len(removed_points) > 0:
            removed_pcd = o3d.geometry.PointCloud()
            removed_pcd.points = o3d.utility.Vector3dVector(removed_points)
            removed_pcd.paint_uniform_color([1, 0, 0])  # Red for removed points
            geometries.append(removed_pcd)

        # Visualize trajectories (assuming they are available as 3D points)
        try:
            for traj, color, name in [
                (self.odom_trajectory, [1, 0, 0], "Odometry"),
                (self.estimated_trajectory, [0, 0, 1], "Estimated"),
                (self.ground_truth_trajectory, [0, 1, 0], "Ground Truth")
            ]:
                traj = np.array(traj)
                if len(traj) > 0:
                    traj_pcd = o3d.geometry.PointCloud()
                    traj_pcd.points = o3d.utility.Vector3dVector(traj)
                    traj_pcd.paint_uniform_color(color)
                    geometries.append(traj_pcd)
        except AttributeError:
            pass  # Trajectories not available

        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="SLAM: 3D Point Cloud Map")
        for geom in geometries:
            vis.add_geometry(geom)
        
        # Set rendering options
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([1, 1, 1])  # White background
        
        # Run visualization
        vis.run()
        vis.destroy_window()

    def save_point_map(self, local_map, filename):
        """Save the 3D point map as a PLY file"""
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(local_map)
        
        # Save to PLY file
        o3d.io.write_point_cloud(f"{filename}.ply", pcd)
        print(f"Point map saved as {filename}.ply")