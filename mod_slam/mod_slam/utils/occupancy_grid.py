import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
import numpy as np
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
import tf_transformations
import open3d as o3d

class ProbabilisticOccupancyGrid:
    def __init__(self, resolution=0.05, initial_size=10, p_occ=0.7, p_free=0.3, uniform_expand=False):
        self.resolution = resolution
        self.p_occ = np.log(p_occ / (1 - p_occ))
        self.p_free = np.log(p_free / (1 - p_free))
        
        # Initial grid size
        self.grid_size_x = int(initial_size / resolution)
        self.grid_size_y = int(initial_size / resolution)
        self.log_odds = np.zeros((self.grid_size_x, self.grid_size_y))
        
        # Set the origin to the center of the grid
        self.origin = np.array([self.grid_size_x // 2, self.grid_size_y // 2])
        self.extents = np.array([self.grid_size_x, self.grid_size_y])  
        self.uniform_expand = uniform_expand  # Option to expand uniformly

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices."""
        x_idx = int(x / self.resolution + self.origin[0])
        y_idx = int(y / self.resolution + self.origin[1])
        return x_idx, y_idx

    
    def expand_grid(self, new_x, new_y):
        """Expand the grid only in the required direction (not symmetrically)."""
        x_idx, y_idx = self.world_to_grid(new_x, new_y)

        # Current grid limits
        min_x, max_x = 0, self.extents[0]
        min_y, max_y = 0, self.extents[1]

        expand_x = expand_y = False

        # Check if expansion is needed in x direction
        if x_idx < 0:
            expand_x = True
            min_x = x_idx  # Expand left
        elif x_idx >= self.extents[0]:
            expand_x = True
            max_x = x_idx + 1  # Expand right

        # Check if expansion is needed in y direction
        if y_idx < 0:
            expand_y = True
            min_y = y_idx  # Expand downward
        elif y_idx >= self.extents[1]:
            expand_y = True
            max_y = y_idx + 1  # Expand upward

        # If no expansion needed, return
        if not expand_x and not expand_y:
            return

        # Calculate new extents
        new_extents = np.array([self.extents[0], self.extents[1]])

        if expand_x:
            new_extents[0] = max_x - min_x
        if expand_y:
            new_extents[1] = max_y - min_y

        # Create new grid
        new_log_odds = np.zeros(new_extents)

        # Offsets to shift old grid into new grid
        x_offset = -min_x if x_idx < 0 else 0
        y_offset = -min_y if y_idx < 0 else 0

        # Copy old grid into new grid
        new_log_odds[x_offset:x_offset + self.extents[0], y_offset:y_offset + self.extents[1]] = self.log_odds

        # Update origin to account for shift
        self.origin += np.array([x_offset, y_offset])

        # Update class variables
        self.extents = new_extents
        self.log_odds = new_log_odds

    def update(self, robot_pose, lidar_ranges, lidar_angles, max_range):
        x_r, y_r, theta_r = robot_pose

        # Expand grid if needed
        for r, angle in zip(lidar_ranges, lidar_angles):
            if r >= max_range:  # Ensure r is a scalar
                continue
            
            end_x = x_r + r * np.cos(theta_r + angle)
            end_y = y_r + r * np.sin(theta_r + angle)
            
            self.expand_grid(end_x, end_y)

        # Perform occupancy updates
        for r, angle in zip(lidar_ranges, lidar_angles):
            if r >= max_range:  # Ensure r is a scalar
                continue
            
            end_x = x_r + r * np.cos(theta_r + angle)
            end_y = y_r + r * np.sin(theta_r + angle)
            
            x_idx, y_idx = self.world_to_grid(end_x, end_y)
            rx_idx, ry_idx = self.world_to_grid(x_r, y_r)
            
            # Get free cells along the ray
            free_cells = self.bresenham(rx_idx, ry_idx, x_idx, y_idx)
            
            for (fx, fy) in free_cells:
                if 0 <= fx < self.extents[0] and 0 <= fy < self.extents[1]:
                    self.log_odds[fx, fy] += self.p_free  # Mark free space
        
            if 0 <= x_idx < self.extents[0] and 0 <= y_idx < self.extents[1]:
                self.log_odds[x_idx, y_idx] += self.p_occ  # Mark occupied space
                
    def update_from_local_points(self, transform_matrix, scan_points_local):
        """
        Update occupancy grid using a transformation matrix and local scan points.
        
        Args:
            transform_matrix: 3x3 or 4x4 transformation matrix from local to world frame
            scan_points_local: List of (x, y) points in local frame
        """
        # Extract robot position from transformation matrix
        x_r, y_r = transform_matrix[0, 2], transform_matrix[1, 2]

        # Transform scan points from local frame to world frame using matrix multiplication
        scan_points_homogeneous = np.column_stack([
            [x for x, y in scan_points_local],
            [y for x, y in scan_points_local],
            np.ones(len(scan_points_local))
        ])
        
        world_points_homogeneous = (transform_matrix @ scan_points_homogeneous.T).T
        world_points = world_points_homogeneous[:, :2]

        # Expand the grid if needed
        for x_world, y_world in world_points:
            self.expand_grid(x_world, y_world)

        # Perform occupancy updates
        for x_world, y_world in world_points:
            x_idx, y_idx = self.world_to_grid(x_world, y_world)
            rx_idx, ry_idx = self.world_to_grid(x_r, y_r)

            # Get free cells along the ray
            free_cells = self.bresenham(rx_idx, ry_idx, x_idx, y_idx)

            for (fx, fy) in free_cells:
                if 0 <= fx < self.extents[0] and 0 <= fy < self.extents[1]:
                    self.log_odds[fx, fy] += self.p_free  # Mark free space

            if 0 <= x_idx < self.extents[0] and 0 <= y_idx < self.extents[1]:
                self.log_odds[x_idx, y_idx] += self.p_occ  # Mark occupied space

    def update_from_optimized_pose_and_pointcloud(self, optimized_pose, point_cloud):
        if point_cloud is None or len(point_cloud) == 0:
            return

        dim_pose = optimized_pose.shape[0] - 1  # 2 for SE(2), 3 for SE(3)

        # Extract robot position (only XY for mapping)
        x_r, y_r = optimized_pose[0, 2], optimized_pose[1, 2]

        # Determine point dimensionality
        dim_pc = point_cloud.shape[1]  # 2 or 3

        # Convert to homogeneous coordinates
        if dim_pc == 2:  # XY points
            pc_hom = np.hstack([point_cloud, np.ones((len(point_cloud), 1))])
        elif dim_pc == 3:  # XYZ points
            pc_hom = np.hstack([point_cloud, np.ones((len(point_cloud), 1))])
        else:
            raise ValueError("Point cloud must have shape (N, 2) or (N, 3)")

        # Apply transformation
        world_pc_hom = (optimized_pose @ pc_hom.T).T

        # Use only XY for mapping
        world_points = world_pc_hom[:, :2]

        # Expand the grid for all points
        for x_world, y_world in world_points:
            self.expand_grid(x_world, y_world)

        # Perform occupancy updates
        for x_world, y_world in world_points:
            x_idx, y_idx = self.world_to_grid(x_world, y_world)
            rx_idx, ry_idx = self.world_to_grid(x_r, y_r)

            # Free space along ray
            free_cells = self.bresenham(rx_idx, ry_idx, x_idx, y_idx)
            for (fx, fy) in free_cells:
                if 0 <= fx < self.extents[0] and 0 <= fy < self.extents[1]:
                    self.log_odds[fx, fy] += self.p_free

            # Occupied space at the hit point
            if 0 <= x_idx < self.extents[0] and 0 <= y_idx < self.extents[1]:
                self.log_odds[x_idx, y_idx] += self.p_occ

    def update_from_pointcloud(self, pose_stamped: PoseStamped, cloud_msg, z_threshold=0.2):
        """
        Updates occupancy grid from a local PointCloud2 scan.
        
        Args:
            pose_stamped: geometry_msgs/PoseStamped - Robot pose in map/world frame.
            cloud_msg: sensor_msgs/PointCloud2 - Local point cloud (relative to robot).
            z_threshold: float - Ignore points with |z| > z_threshold.
        """
        
        # Note that the pose is not the correct format for pose stamped
        
        # Extract robot pose
        x_r = pose_stamped.position.x
        y_r = pose_stamped.position.y

        # Convert quaternion to yaw
        q = pose_stamped.orientation
        quat = [q.x, q.y, q.z, q.w]
        _, _, theta_r = tf_transformations.euler_from_quaternion(quat)

        # Convert PointCloud2 to list of (x, y, z)
        points = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)

        # First pass: Expand grid if needed
        for px, py, pz in points:
            # if abs(pz) > z_threshold:
            #     continue
            
            # Transform from robot frame to world frame
            wx = x_r + (px * np.cos(theta_r) - py * np.sin(theta_r))
            wy = y_r + (px * np.sin(theta_r) + py * np.cos(theta_r))

            self.expand_grid(wx, wy)

        # Second pass: Perform occupancy updates
        points = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        for px, py, pz in points:
            # if abs(pz) > z_threshold:
            #     continue

            # Transform to world coordinates
            wx = x_r + (px * np.cos(theta_r) - py * np.sin(theta_r))
            wy = y_r + (px * np.sin(theta_r) + py * np.cos(theta_r))

            # Convert to grid indices
            gx, gy = self.world_to_grid(wx, wy)
            rx_idx, ry_idx = self.world_to_grid(x_r, y_r)

            # Get free cells along the ray
            free_cells = self.bresenham(rx_idx, ry_idx, gx, gy)
            for (fx, fy) in free_cells:
                if 0 <= fx < self.extents[0] and 0 <= fy < self.extents[1]:
                    self.log_odds[fx, fy] += self.p_free  # Free space

            # Mark endpoint as occupied
            if 0 <= gx < self.extents[0] and 0 <= gy < self.extents[1]:
                self.log_odds[gx, gy] += self.p_occ

    def update_from_open3d(self, pose_stamped: PoseStamped, cloud, z_threshold=0.2):
        """
        Updates occupancy grid from a local Open3D point cloud (or Nx3/Nx2 numpy array).

        Args:
            pose_stamped: geometry_msgs/PoseStamped - Robot pose in map/world frame.
            cloud: open3d.geometry.PointCloud or (N,3)/(N,2) numpy array of local points (robot frame).
            z_threshold: float - Ignore points with |z| > z_threshold (if provided).
        """

        # Extract robot pose (support PoseStamped or Pose)
        pose = getattr(pose_stamped, "pose", pose_stamped)
        x_r = pose.position.x
        y_r = pose.position.y

        # Convert quaternion to yaw
        q = pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        _, _, theta_r = tf_transformations.euler_from_quaternion(quat)

        # Convert Open3D / numpy cloud to numpy array of shape (N, 3)
        if isinstance(cloud, o3d.geometry.PointCloud):
            pts = np.asarray(cloud.points)  # (N,3)
        else:
            pts = np.asarray(cloud)

        if pts.ndim != 2 or pts.shape[1] not in (2, 3):
            raise ValueError("cloud must be an Open3D PointCloud or an (N,2)/(N,3) numpy array")

        if pts.shape[1] == 2:
            # pad z with zeros if only (x,y) provided
            pts = np.hstack([pts, np.zeros((pts.shape[0], 1), dtype=pts.dtype)])
            
        print(f"Updating with {len(pts)} points before z-thresholding")

        print(f"Z values range: {pts[:, 2].min():.3f} to {pts[:, 2].max():.3f}")
        # Optional z filtering
        if z_threshold is not None:
            pts = pts[np.abs(pts[:, 2]) <= z_threshold]
            
        print(f"Updating with {len(pts)} points after z-thresholding")

        # First pass: Expand grid if needed
        cos_t = np.cos(theta_r)
        sin_t = np.sin(theta_r)
        for px, py, pz in pts:
            # Transform from robot frame to world frame
            wx = x_r + (px * cos_t - py * sin_t)
            wy = y_r + (px * sin_t + py * cos_t)
            self.expand_grid(wx, wy)

        # Second pass: Perform occupancy updates
        rx_idx, ry_idx = self.world_to_grid(x_r, y_r)
        for px, py, pz in pts:
            # Transform to world coordinates
            wx = x_r + (px * cos_t - py * sin_t)
            wy = y_r + (px * sin_t + py * cos_t)

            # Convert to grid indices
            gx, gy = self.world_to_grid(wx, wy)

            # Get free cells along the ray
            free_cells = self.bresenham(rx_idx, ry_idx, gx, gy)
            for fx, fy in free_cells:
                if 0 <= fx < self.extents[0] and 0 <= fy < self.extents[1]:
                    self.log_odds[fx, fy] += self.p_free  # Free space

            # Mark endpoint as occupied
            if 0 <= gx < self.extents[0] and 0 <= gy < self.extents[1]:
                self.log_odds[gx, gy] += self.p_occ

    def bresenham(self, x0, y0, x1, y1):
        """Bresenham's line algorithm for ray tracing."""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while (x0, y0) != (x1, y1):
            cells.append((x0, y0))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return cells

    def get_probability_map(self):
        """Convert log odds to probability values."""
        log_odds_clamped = np.clip(self.log_odds, -100, 100)
        return 1 - 1 / (1 + np.exp(log_odds_clamped))

    def plot_map(self):
        """Plot the occupancy grid."""
        # plt.imshow(self.get_probability_map().T, cmap='gray_r', origin='lower',
        # extent=[-self.extents[0]//2, self.extents[0]//2, -self.extents[1]//2, self.extents[1]//2])
        plt.imshow(self.get_probability_map().T, cmap='gray_r', origin='lower')
        plt.colorbar(label='Occupancy Probability')
        plt.show()
        
    def save_map_only(self, filename, dpi=300):
        """Plot the occupancy grid with proper color correction."""
        # Convert log odds to probability values
        probability_map = self.get_probability_map().T
        
        # Create the output map following SLAM Toolbox conventions
        output_map = np.full_like(probability_map, 205, dtype=np.uint8)  # Unknown areas = 205 (light gray)
        observed_mask = np.abs(self.log_odds.T) > 1e-6  # Small threshold to account for floating point precision
        free_mask = (probability_map < 0.25) & observed_mask
        occupied_mask = (probability_map > 0.65) & observed_mask
        uncertain_mask = observed_mask & ~free_mask & ~occupied_mask
        output_map[free_mask] = 254  # Free space = white
        output_map[occupied_mask] = 0  # Occupied space = black
        output_map[uncertain_mask] = 205  # Uncertain = light gray
        
        plt.imshow(output_map, cmap='gray', origin='lower')
        # plt.colorbar(label='Occupancy Map')
        plt.savefig(filename, dpi=dpi)
        plt.show()
        
    def plot_map_realtime(self):
        """Plot the occupancy grid in real time."""
        plt.ion()  # Enable interactive mode
        plt.clf()  # Clear the figure
        
        # Plot the occupancy grid
        plt.imshow(self.get_probability_map().T, cmap='gray_r', origin='lower',
           extent=[-self.extents[0]//2, self.extents[0]//2, -self.extents[1]//2, self.extents[1]//2])
        plt.colorbar(label='Occupancy Probability')
        
        # # Plot trajectories
        # if hasattr(self, 'odom_trajectory') and self.odom_trajectory:
        #     odom_trajectory = np.array(self.odom_trajectory)
        #     plt.plot(odom_trajectory[:, 0], odom_trajectory[:, 1], color='red', label="Odometry Trajectory")
        
        # if hasattr(self, 'estimated_trajectory') and self.estimated_trajectory:
        #     estimated_trajectory = np.array(self.estimated_trajectory)
        #     plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], color='blue', label="Estimated Trajectory")
        
        # if hasattr(self, 'ground_truth_trajectory') and self.ground_truth_trajectory:
        #     ground_truth_trajectory = np.array(self.ground_truth_trajectory)
        #     plt.plot(ground_truth_trajectory[:, 0], ground_truth_trajectory[:, 1], color='green', label="Ground Truth Trajectory")
        
        # plt.legend()
        plt.axis('equal')
        plt.draw()
        plt.pause(0.1)  # Pause to update the figure

    def save_map(self, filename):
        """Save the occupancy grid map to a .png file and a .yaml file."""

        # Convert log odds to probability values
        probability_map = self.get_probability_map().T
        print("Probability map shape:", probability_map.shape)
        # probability_map = np.flipud(probability_map)

        # Normalize the probability map to 0-255 for saving as an image
        #normalized_map = ((1 - probability_map) * 255).astype(np.uint8)  # Invert colors
        
        # Create the output map following SLAM Toolbox conventions
        output_map = np.full_like(probability_map, 205, dtype=np.uint8)  # Unknown areas = 205 (light gray)
        observed_mask = np.abs(self.log_odds.T) > 1e-6  # Small threshold to account for floating point precision
        free_mask = (probability_map < 0.25) & observed_mask
        occupied_mask = (probability_map > 0.65) & observed_mask
        uncertain_mask = observed_mask & ~free_mask & ~occupied_mask
        output_map[free_mask] = 254  # Free space = white
        output_map[occupied_mask] = 0  # Occupied space = black
        output_map[uncertain_mask] = 205  # Uncertain = light gray

        # Save the map as a .png file
        image_filename = f"{filename}.png"
        # cv2.imwrite(image_filename, normalized_map) 
        cv2.imwrite(image_filename, output_map) # Treats the first index of an array as the row index (y) and assumes (0,0) is the top-left corner

        # Create the .yaml file with metadata
        yaml_data = {
            'image': f"{filename}.png",
            'resolution': self.resolution,
            'origin': [float(-self.origin[0] * self.resolution),
               float(-self.origin[1] * self.resolution), 0.0],  # Matches publisher
            'negate': 0,
            'occupied_thresh': 0.65,
            'free_thresh': 0.196
        }

        yaml_filename = f"{filename}.yaml"
        with open(yaml_filename, 'w') as yaml_file:
            yaml.dump(yaml_data, yaml_file, default_flow_style=False)
            
        print("Saving origin:", yaml_data['origin'])
            
        # print(f"Map saved as {image_filename} and {yaml_filename}")
        
# **Test Functions**
def test_moving_robot_map():
    # Initialize the grid with dimensions and resolution
    grid = ProbabilisticOccupancyGrid(resolution=0.1, initial_size=5, uniform_expand=True)

    # x+
    # First test: Robot at the center (0, 0) and lidar scan
    robot_pose_1 = (0, 0, 0)  # Robot at the center of the map
    angles = np.linspace(0, 2 * np.pi, 360)  # Full lidar scan around the robot
    ranges = np.full_like(angles, 3.0)  # Circular lidar readings at 3m
    grid.update(robot_pose_1, ranges, angles, max_range=3.5)  # Update grid with first scan
    grid.plot_map()  # Plot the map after the first update
    # Second test: Robot moves to a new position (e.g., (2, 2)) and scans again
    robot_pose_2 = (2, 0, np.pi / 4)  # Robot moves to (2, 2) and faces 45 degrees
    ranges_2 = np.full_like(angles, 3)  # Lidar readings at 3.5m
    grid.update(robot_pose_2, ranges_2, angles, max_range=4.0)  # Update grid with second scan
    grid.plot_map()  # Plot the map after the second update
    robot_pose_3 = (4, 0, np.pi / 4)  # Robot moves to (2, 2) and faces 45 degrees
    ranges_3 = np.full_like(angles, 3)  # Lidar readings at 3.5m
    grid.update(robot_pose_3, ranges_3, angles, max_range=4.0)  # Update grid with second scan
    grid.plot_map()  # Plot the map after the second update
    grid.save_map("/home/ruan/dev_ws/src/landmark_extract/Results/Maps/Test")  # Save the map to a .png and .yaml file
    
    # y+
    # First test: Robot at the center (0, 0) and lidar scan
    robot_pose_1 = (0, 0, 0)  # Robot at the center of the map
    angles = np.linspace(0, 2 * np.pi, 360)  # Full lidar scan around the robot
    ranges = np.full_like(angles, 3.0)  # Circular lidar readings at 3m
    grid.update(robot_pose_1, ranges, angles, max_range=3.5)  # Update grid with first scan
    grid.plot_map()  # Plot the map after the first update
    # Second test: Robot moves to a new position (e.g., (2, 2)) and scans again
    robot_pose_2 = (0, 2, np.pi / 4)  # Robot moves to (2, 2) and faces 45 degrees
    ranges_2 = np.full_like(angles, 3)  # Lidar readings at 3.5m
    grid.update(robot_pose_2, ranges_2, angles, max_range=4.0)  # Update grid with second scan
    grid.plot_map()  # Plot the map after the second update
    robot_pose_3 = (0, 4, np.pi / 4)  # Robot moves to (2, 2) and faces 45 degrees
    ranges_3 = np.full_like(angles, 3)  # Lidar readings at 3.5m
    grid.update(robot_pose_3, ranges_3, angles, max_range=4.0)  # Update grid with second scan
    grid.plot_map()  # Plot the map after the second update
    grid.save_map("/home/ruan/dev_ws/src/landmark_extract/Results/Maps/Test")  # Save the map to a .png and .yaml file
    
    # x-
    # First test: Robot at the center (0, 0) and lidar scan
    robot_pose_1 = (0, 0, 0)  # Robot at the center of the map
    angles = np.linspace(0, 2 * np.pi, 360)  # Full lidar scan around the robot
    ranges = np.full_like(angles, 3.0)  # Circular lidar readings at 3m
    grid.update(robot_pose_1, ranges, angles, max_range=3.5)  # Update grid with first scan
    grid.plot_map()  # Plot the map after the first update
    # Second test: Robot moves to a new position (e.g., (2, 2)) and scans again
    robot_pose_2 = (-2, 0, np.pi / 4)  # Robot moves to (2, 2) and faces 45 degrees
    ranges_2 = np.full_like(angles, 3)  # Lidar readings at 3.5m
    grid.update(robot_pose_2, ranges_2, angles, max_range=4.0)  # Update grid with second scan
    grid.plot_map()  # Plot the map after the second update
    robot_pose_3 = (-4, 0, np.pi / 4)  # Robot moves to (2, 2) and faces 45 degrees
    ranges_3 = np.full_like(angles, 3)  # Lidar readings at 3.5m
    grid.update(robot_pose_3, ranges_3, angles, max_range=4.0)  # Update grid with second scan
    grid.plot_map()  # Plot the map after the second update
    grid.save_map("/home/ruan/dev_ws/src/landmark_extract/Results/Maps/Test")  # Save the map to a .png and .yaml file
    
    # y-
    # First test: Robot at the center (0, 0) and lidar scan
    robot_pose_1 = (0, 0, 0)  # Robot at the center of the map
    angles = np.linspace(0, 2 * np.pi, 360)  # Full lidar scan around the robot
    ranges = np.full_like(angles, 3.0)  # Circular lidar readings at 3m
    grid.update(robot_pose_1, ranges, angles, max_range=3.5)  # Update grid with first scan
    grid.plot_map()  # Plot the map after the first update
    # Second test: Robot moves to a new position (e.g., (2, 2)) and scans again
    robot_pose_2 = (0, -2, np.pi / 4)  # Robot moves to (2, 2) and faces 45 degrees
    ranges_2 = np.full_like(angles, 3)  # Lidar readings at 3.5m
    grid.update(robot_pose_2, ranges_2, angles, max_range=4.0)  # Update grid with second scan
    grid.plot_map()  # Plot the map after the second update
    robot_pose_3 = (0, -4, np.pi / 4)  # Robot moves to (2, 2) and faces 45 degrees
    ranges_3 = np.full_like(angles, 3)  # Lidar readings at 3.5m
    grid.update(robot_pose_3, ranges_3, angles, max_range=4.0)  # Update grid with second scan
    grid.plot_map()  # Plot the map after the second update
    grid.save_map("/home/ruan/dev_ws/src/landmark_extract/Results/Maps/Test")  # Save the map to a .png and .yaml file

def test_dynamic_expansion():
    grid = ProbabilisticOccupancyGrid(resolution=0.1, initial_size=5, uniform_expand=False)

    # First scan: robot in a corridor
    robot_pose_1 = (1, 1, 0)
    angles = np.linspace(-np.pi / 4, np.pi / 4, 90)
    ranges = np.full_like(angles, 3.0)
    
    grid.update(robot_pose_1, ranges, angles, max_range=3.5)
    grid.plot_map()

    # Second scan: robot moves into an open room
    robot_pose_2 = (5, 8, np.pi / 2)
    ranges_2 = np.full_like(angles, 6.0)
    
    grid.update(robot_pose_2, ranges_2, angles, max_range=6.5)
    grid.plot_map()
    
    robot_pose_3 = (-2, -5, -np.pi / 2)
    ranges_3 = np.full_like(angles, 6.0)
    
    grid.update(robot_pose_3, ranges_3, angles, max_range=6.5)
    grid.plot_map()
    

if __name__ == "__main__":
    test_moving_robot_map()
    # test_dynamic_expansion()
