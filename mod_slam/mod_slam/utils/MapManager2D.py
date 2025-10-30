import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

class MapManager2D:
    def __init__(self, grid_downsample_res=0.1, filter_dis=0.04, merge_dis=0.4, exclusion_radius=0.2, max_map_points=3500):
        self.grid_downsample_res = grid_downsample_res
        self.filter_dis = filter_dis
        self.merge_dis = merge_dis
        self.exclusion_radius = exclusion_radius  # Exclusion zone radius around lidar sensor
        self.max_map_points = max_map_points  # Maximum points to keep in local map
        self.point_ages = []  # Track point insertion order/age for temporal management
        self.frame_counter = 0  # Counter for tracking frames

  # Point map functions and processes   
    def update_point_map(self, local_map, scan, transformation):
        """Update the global point map with new points"""
        
        # print(f"scan shape: {scan}")
        # filtered_scan = self.apply_exclusion_zone(scan)
        filtered_scan = scan
        # print(f"filtered scan shape: {filtered_scan}")
        
        # Transform local scan to global frame
        transformed_scan = self.transform_scan(filtered_scan, transformation)
        
        # Add points raw
        local_map =  np.vstack((local_map, transformed_scan))
        
        # Remove points that are too close to existing ones
        # local_map, removed_points = self.filter_close_points_2d(local_map, transformed_scan, self.filter_dis)
        
        # Apply 2D grid downsampling
        local_map = self.grid_downsample(local_map, self.grid_downsample_res)
        
        # Choose one of the following map management strategies:
        
        # Method 1: SAFE block clipping (prevents drift with safety checks)
        # local_map = self.safe_clip_local_map_by_block(local_map, transformation, x_distance=12.0, y_distance=12.0, min_points=2000)
        
        # Method 2: Conservative management (only clip when really necessary)
        # local_map = self.conservative_map_management(local_map, transformation, max_points=3000)
        
        # Method 3: Simple FIFO management (SAFEST - removes oldest points when threshold exceeded)
        # local_map = self.simple_fifo_management(local_map, max_points_threshold=2000)
        
        # Method 4: Original FIFO using class parameter (uses self.max_map_points)
        # local_map = self.remove_oldest_points(local_map)
        
        # Method 5: Original block clipping (can cause drift if not careful)
        # local_map = self.clip_local_map_by_block(local_map, transformation, x_distance=12.0, y_distance=12.0, min_points=1000)
        
        # Method 6: NO CLIPPING - for testing SLAM stability without map management
        # pass  # No clipping at all
        print(f"Map size: {len(local_map)} points")

        # self.plot_point_map(local_map, transformed_scan, removed_points=None, transformation=transformation, x_distance=12.0, y_distance=12.0)
        # self.plot_point_map(local_map, transformed_scan)
        
        return local_map
    
    def update_global_point_map(self, local_map, scan, transformation):
        """Update the global point map with new points"""
        
        # Apply exclusion zone filtering to scan before transformation
        filtered_scan = self.apply_exclusion_zone(scan)
        
        # Transform local scan to global frame
        transformed_scan = self.transform_scan(filtered_scan, transformation)
        
        # Remove points that are too close to existing ones
        local_map, removed_points = self.filter_close_points_2d(local_map, transformed_scan, self.filter_dis)
        
        # Apply 2D grid downsampling
        local_map = self.grid_downsample(local_map, self.grid_downsample_res)
        
        # # Merge close points
        local_map = self.merge_close_points_2d(local_map, self.merge_dis)
        
        # Clip local map to stay within a radius around the robot
        # local_map = self.clip_local_map_by_radius(local_map, transformation, radius=10.0)  # you can change radius

        # self.plot_point_map(local_map, transformed_scan, removed_points=None)
        # self.plot_point_map(local_map, transformed_scan)
        
        return local_map
        
    def transform_scan(self, scan, transformation):
        """Apply a 2D transformation to a set of points"""
        tx, ty = transformation[0, 2], transformation[1, 2]
        theta = np.arctan2(transformation[1, 0], transformation[0, 0])
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        return scan @ rotation_matrix.T + np.array([tx, ty])
    
    def filter_close_points_2d(self, local_map, new_points, min_dist=0.04):
        """Removes new points that are too close to existing ones and returns removed points."""
        if len(local_map) == 0:
            return new_points, np.array([])  # No previous points, keep all new ones

        tree = cKDTree(local_map)
        distances, _ = tree.query(new_points)
        
        # Keep points that are farther than min_dist
        mask = distances > min_dist
        filtered_points = new_points[mask]
        
        # Removed points are those that were too close
        removed_points = new_points[~mask]
        
        return np.vstack((local_map, filtered_points)), removed_points
       
    def grid_downsample(self, points, grid_size):
        """Reduces the number of points by keeping one per 2D grid cell."""
        grid_indices = np.floor(points / grid_size).astype(int)
        unique_indices, unique_idx = np.unique(grid_indices, axis=0, return_index=True)
        return points[unique_idx]
    
    def merge_close_points_2d(self, points, merge_dist=0.4):
        """Merges nearby points by averaging their positions."""
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
    
    def safe_clip_local_map_by_block(self, points, transformation, x_distance=12.0, y_distance=12.0, min_points=2000, transition_zone=2.0):
        """
        SAFER clipping that prevents SLAM drift by:
        1. Ensuring minimum point count
        2. Gradual transition zones
        3. Preserving dense areas for scan matching
        
        Args:
            points: Nx2 array of map points
            transformation: 3x3 transformation matrix
            x_distance: distance threshold in x direction
            y_distance: distance threshold in y direction 
            min_points: minimum number of points to maintain (CRITICAL for stability)
            transition_zone: gradual fade zone to prevent abrupt changes
        """
        if len(points) == 0:
            return points
            
        robot_position = transformation[:2, 2]
        
        # Step 1: Calculate distances for safety fallback
        distances = np.linalg.norm(points - robot_position, axis=1)
        
        # Step 2: Create core rectangular area (strict clipping)
        x_min = robot_position[0] - x_distance
        x_max = robot_position[0] + x_distance
        y_min = robot_position[1] - y_distance
        y_max = robot_position[1] + y_distance
        
        core_mask = ((points[:, 0] >= x_min) & (points[:, 0] <= x_max) & 
                     (points[:, 1] >= y_min) & (points[:, 1] <= y_max))
        
        # Step 3: CRITICAL SAFETY CHECK - ensure minimum points
        if np.sum(core_mask) < min_points:
            # Fallback: keep closest points to maintain scan matching quality
            closest_indices = np.argpartition(distances, min_points)[:min_points]
            safety_mask = np.zeros_like(core_mask, dtype=bool)
            safety_mask[closest_indices] = True
            print(f"WARNING: Block clipping would leave only {np.sum(core_mask)} points. Using safety fallback with {min_points} closest points.")
            return points[safety_mask]
        
        return points[core_mask]
    
    def conservative_map_management(self, points, transformation, max_points=3000):
        """
        Very conservative approach - only remove points when absolutely necessary.
        Prioritizes SLAM stability over memory efficiency.
        """
        if len(points) <= max_points:
            return points  # No clipping needed
            
        robot_position = transformation[:2, 2]
        distances = np.linalg.norm(points - robot_position, axis=1)
        
        # Only remove the farthest points, keep the closest max_points
        closest_indices = np.argpartition(distances, max_points)[:max_points]
        
        print(f"Conservative clipping: {len(points)} -> {max_points} points")
        return points[closest_indices]
    
    def clip_local_map_by_block(self, points, transformation, x_distance=12.0, y_distance=12.0, min_points=1000):
        """
        Clips the local map using a rectangular block around the robot.
        Much more stable than radius-based clipping as it doesn't shift the map's center of mass.
        
        Args:
            points: Nx2 array of map points
            transformation: 3x3 transformation matrix
            x_distance: distance threshold in x direction (front/back of robot)
            y_distance: distance threshold in y direction (left/right of robot)
            min_points: minimum number of points to maintain
        """
        if len(points) == 0:
            return points
            
        robot_position = transformation[:2, 2]  # Get [x, y] of robot from transformation matrix
        
        # Create rectangular boundary around robot
        x_min = robot_position[0] - x_distance
        x_max = robot_position[0] + x_distance
        y_min = robot_position[1] - y_distance
        y_max = robot_position[1] + y_distance
        
        # Keep points within the rectangular block
        mask = ((points[:, 0] >= x_min) & (points[:, 0] <= x_max) & 
                (points[:, 1] >= y_min) & (points[:, 1] <= y_max))
        
        # CRITICAL: Always keep minimum number of closest points for stability
        if np.sum(mask) < min_points and len(points) > min_points:
            # Calculate distances and keep closest points
            distances = np.linalg.norm(points - robot_position, axis=1)
            closest_indices = np.argpartition(distances, min_points)[:min_points]
            mask = np.zeros_like(mask, dtype=bool)
            mask[closest_indices] = True
            print(f"Block clipping safety: Using {min_points} closest points instead of {np.sum(mask)} within block")
            
        return points[mask]

    def apply_exclusion_zone(self, points, exclusion_radius=None):
        """
        Remove points that are within the exclusion zone around the sensor origin.
        
        Args:
            points: Nx2 numpy array of points [x, y]
            exclusion_radius: radius of exclusion zone (if None, uses self.exclusion_radius)
            
        Returns:
            Filtered points outside the exclusion zone
        """
        if exclusion_radius is None:
            exclusion_radius = self.exclusion_radius
            
        if len(points) == 0:
            return points
            
        # Remove NaN and infinite points first
        valid_points_mask = np.isfinite(points).all(axis=1)
        points = points[valid_points_mask]
        
        if len(points) == 0:
            return points
            
        # Calculate distances from sensor origin (0, 0)
        distances = np.linalg.norm(points, axis=1)
        
        # Keep only points outside the exclusion zone
        valid_mask = distances > exclusion_radius
        
        return points[valid_mask]

    def remove_oldest_points(self, points):
        """
        Remove oldest points if map exceeds maximum size.
        Uses FIFO (First In, First Out) strategy.
        """
        if len(points) <= self.max_map_points:
            return points
            
        # Keep only the most recent max_map_points
        return points[-self.max_map_points:]
    
    def simple_fifo_management(self, points, max_points_threshold=3000):
        """
        Simple FIFO point management - removes oldest points when threshold exceeded.
        Very safe for SLAM as it maintains temporal consistency and doesn't create
        spatial gaps that could confuse scan matching.
        
        Args:
            points: Nx2 array of map points (assumed to be in chronological order)
            max_points_threshold: maximum number of points before removal starts
            
        Returns:
            Filtered points with oldest ones removed if threshold exceeded
        """
        if len(points) <= max_points_threshold:
            return points
            
        # Remove oldest points (first in the array), keep newest ones
        points_to_remove = len(points) - max_points_threshold
        filtered_points = points[points_to_remove:]
        
        print(f"FIFO management: Removed {points_to_remove} oldest points. Map size: {len(points)} -> {len(filtered_points)}")
        return filtered_points
    
    def adaptive_clip_map(self, points, transformation, target_points=None, min_radius=8.0, max_radius=15.0):
        """
        Adaptive clipping based on point count and distance.
        Dynamically adjusts radius to maintain target point count.
        """
        if target_points is None:
            target_points = self.max_map_points
            
        if len(points) == 0:
            return points
            
        robot_position = transformation[:2, 2]
        distances = np.linalg.norm(points - robot_position, axis=1)
        
        # Start with minimum radius and increase until we have acceptable point count
        for radius in np.linspace(min_radius, max_radius, 20):
            mask = distances < radius
            if np.sum(mask) <= target_points:
                return points[mask]
                
        # If still too many points, keep closest target_points
        closest_indices = np.argpartition(distances, target_points)[:target_points]
        return points[closest_indices]
    
    def density_aware_clip_map(self, points, transformation, base_radius=10.0, density_threshold=50):
        """
        Clips map based on local point density.
        Reduces radius in dense areas, increases in sparse areas.
        """
        if len(points) == 0:
            return points
            
        robot_position = transformation[:2, 2]
        distances = np.linalg.norm(points - robot_position, axis=1)
        
        # Calculate local density around robot
        close_points = points[distances < 2.0]  # Points within 2m
        local_density = len(close_points)
        
        # Adjust radius based on density
        if local_density > density_threshold:
            # High density: use smaller radius
            adjusted_radius = base_radius * 0.7
        else:
            # Low density: use larger radius
            adjusted_radius = base_radius * 1.3
            
        mask = distances < adjusted_radius
        return points[mask]
    
    def hierarchical_map_management(self, points, transformation, levels=[5.0, 10.0, 20.0], densities=[0.05, 0.1, 0.2]):
        """
        Multi-level map management with different resolutions at different distances.
        Close points: high resolution, far points: lower resolution.
        """
        if len(points) == 0:
            return points
            
        robot_position = transformation[:2, 2]
        distances = np.linalg.norm(points - robot_position, axis=1)
        
        filtered_points = []
        
        for i, (radius, grid_res) in enumerate(zip(levels, densities)):
            if i == 0:
                # First level: points within first radius
                mask = distances < radius
            else:
                # Subsequent levels: points between previous and current radius
                mask = (distances >= levels[i-1]) & (distances < radius)
            
            level_points = points[mask]
            if len(level_points) > 0:
                # Apply different downsampling for this level
                downsampled = self.grid_downsample(level_points, grid_res)
                filtered_points.append(downsampled)
        
        if filtered_points:
            return np.vstack(filtered_points)
        else:
            return np.array([]).reshape(0, 2)

    def temporal_map_management(self, points, max_age_frames=100):
        """
        Remove points based on age (how long they've been in the map).
        Useful for dynamic environments.
        """
        if len(points) == 0 or len(self.point_ages) == 0:
            return points
            
        # Remove points older than max_age_frames
        current_frame = self.frame_counter
        valid_mask = np.array([(current_frame - age) <= max_age_frames for age in self.point_ages])
        
        if np.any(valid_mask):
            self.point_ages = [age for i, age in enumerate(self.point_ages) if valid_mask[i]]
            return points[valid_mask]
        else:
            self.point_ages = []
            return np.array([]).reshape(0, 2)
    
    def sliding_window_map(self, points, transformation, window_size=50.0):
        """
        Maintains a sliding window of the map based on robot trajectory.
        Only keeps points within a certain distance of recent robot positions.
        """
        if len(points) == 0:
            return points
            
        # Get current robot position
        robot_position = transformation[:2, 2]
        
        # For sliding window, we'd need to track robot trajectory
        # This is a simplified version that just uses current position
        distances = np.linalg.norm(points - robot_position, axis=1)
        mask = distances < window_size
        
        return points[mask]
    
    def update_point_map_with_age_tracking(self, local_map, scan, transformation):
        """
        Enhanced update method that tracks point ages for temporal management.
        """
        self.frame_counter += 1
        
        # Apply exclusion zone filtering (optional)
        filtered_scan = scan  # or self.apply_exclusion_zone(scan)
        
        # Transform local scan to global frame
        transformed_scan = self.transform_scan(filtered_scan, transformation)
        
        # Add new points and track their ages
        if len(local_map) == 0:
            local_map = transformed_scan
            self.point_ages = [self.frame_counter] * len(transformed_scan)
        else:
            local_map = np.vstack((local_map, transformed_scan))
            self.point_ages.extend([self.frame_counter] * len(transformed_scan))
        
        # Apply grid downsampling (this will change point count, so we need to update ages accordingly)
        old_count = len(local_map)
        local_map = self.grid_downsample(local_map, self.grid_downsample_res)
        new_count = len(local_map)
        
        # If downsampling reduced points, we need to adjust point_ages
        if new_count < old_count and len(self.point_ages) == old_count:
            # Simple approach: keep most recent ages
            self.point_ages = self.point_ages[-new_count:]
        
        # Apply temporal management (remove old points)
        local_map = self.temporal_map_management(local_map, max_age_frames=100)
        
        # Apply spatial management with stable block clipping
        local_map = self.clip_local_map_by_block(local_map, transformation, x_distance=12.0, y_distance=12.0, min_points=1000)
        
        return local_map

    def plot_point_map(self, local_map, transformed_scan=None, removed_points=None, transformation=None, x_distance=12.0, y_distance=12.0):
        """Plot the global map with clipping boundary visualization"""
        
        # self.get_logger().info(f"Map size: {local_map.shape}")
        
        # Setup real-time visualization
        plt.ion()  # Interactive mode ON
        plt.clf()
        
        # Initial plot
        target_scatter = plt.scatter(local_map[:, 0], local_map[:, 1], color='black', label="Map (Reference)", s=0.05)
        if transformed_scan is not None:
            source_scatter = plt.scatter(transformed_scan[:, 0], transformed_scan[:, 1], color='#1f77b4', label="New scan", alpha=0.5, s=5)
        if removed_points is not None:
            plt.scatter(removed_points[:, 0], removed_points[:, 1], color='red', label="Removed points", s=5, alpha=0.5)
        
        # Draw exclusion zone circle for current robot position (assuming origin for visualization)
        circle = plt.Circle((0, 0), self.exclusion_radius, fill=False, color='red', linestyle='--', alpha=0.7, label=f'Exclusion zone ({self.exclusion_radius*100:.0f}cm)')
        plt.gca().add_patch(circle)
        
        # Draw clipping bounding box if transformation is provided
        if transformation is not None:
            robot_position = transformation[:2, 2]  # Get robot position from transformation
            
            # Calculate bounding box corners
            x_min = robot_position[0] - x_distance
            x_max = robot_position[0] + x_distance
            y_min = robot_position[1] - y_distance
            y_max = robot_position[1] + y_distance
            
            # Draw bounding box rectangle
            from matplotlib.patches import Rectangle
            bbox = Rectangle((x_min, y_min), 2*x_distance, 2*y_distance, 
                           fill=False, color='green', linestyle='-', linewidth=2, 
                           alpha=0.8, label=f'Clipping box ({x_distance}×{y_distance}m)')
            plt.gca().add_patch(bbox)
            
            # Mark robot position
            plt.scatter(robot_position[0], robot_position[1], color='green', marker='x', 
                       s=100, linewidth=3, label='Robot position')
        
        #Plot trajectories
        # odom_trajectory = np.array(self.odom_trajectory)
        # estimated_trajectory = np.array(self.estimated_trajectory)
        # ground_truth_trajectory = np.array(self.ground_truth_trajectory)
        # plt.plot(odom_trajectory[:, 0], odom_trajectory[:, 1], color='red', label="Odometry Trajectory")
        # plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], color='blue', label="Estimated Trajectory")
        # plt.plot(ground_truth_trajectory[:, 0], ground_truth_trajectory[:, 1], color='green', label="Ground Truth Trajectory")
        
        plt.title("SLAM: Scan-to-Scan Matching")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        plt.grid(True)
        plt.draw()
        plt.axis('equal')   
        plt.pause(0.1)
        
        # plt.savefig("Optimised_point_map.png")

        
    def save_point_map(self, local_map, filename):
        """Save the global point as a plot"""
        Number_of_map_points = local_map.shape[0]
        
        # Configure LaTeX fonts
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 16,
            # Additional LaTeX-specific settings for better rendering
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            "mathtext.default": "regular",
            "axes.unicode_minus": False,  # Use ASCII minus for LaTeX compatibility
        })
        
        plt.ion()  # Interactive mode ON
        plt.clf()
        
        # Set figure size
        plt.figure(figsize=(8, 6))
        
        # Initial plot - increased marker size for solid dots
        target_scatter = plt.scatter(local_map[:, 0], local_map[:, 1], color='black', label="Map (Reference)", s=0.1)
        
        # # Plot trajectories
        # odom_trajectory = np.array(self.odom_trajectory)
        # estimated_trajectory = np.array(self.estimated_trajectory)
        # ground_truth_trajectory = np.array(self.ground_truth_trajectory)
        # plt.plot(odom_trajectory[:, 0], odom_trajectory[:, 1], color='red', label="Odometry Trajectory")
        # plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], color='blue', label="Estimated Trajectory")
        # plt.plot(ground_truth_trajectory[:, 0], ground_truth_trajectory[:, 1], color='green', label="Ground Truth Trajectory")
        
        # Add additional text or variables to the legend
        additional_text = f"Grid downsample resolution: {self.grid_downsample_res}"
        additional_text_2 = f"Filter res: {self.filter_dis}"
        additional_text_3 = f"Number of map points: {Number_of_map_points}"
        additional_text_4 = f"Lidar exclusion radius: {self.exclusion_radius} m"
        plt.plot([], [], ' ', label=additional_text)
        plt.plot([], [], ' ', label=additional_text_2)
        plt.plot([], [], ' ', label=additional_text_3)
        plt.plot([], [], ' ', label=additional_text_4)
        
        plt.title(f"Point cloud map with {Number_of_map_points} points")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        
        # plt.xlim(-25, 16.5)  # Adjusted limits based on Husky lab map
        # plt.ylim(-12.6, 7.6)
        plt.gca().set_aspect('equal', adjustable='box')
        
        # plt.legend(loc='upper left', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{filename}_pointmap.png", dpi=600, bbox_inches='tight')

        print(f"Point map saved as {filename}_pointmap.png")
