import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path
from std_msgs.msg import Header
from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates
from tf2_ros import TransformBroadcaster
from tf2_ros import Buffer, TransformListener
import tf2_ros
from geometry_msgs.msg import TransformStamped
import tf_transformations
from std_srvs.srv import Empty
import time

import sys
import os
from ament_index_python.packages import get_package_share_directory
import time
import numpy as np
from scipy.spatial import KDTree, cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import BSpline
from scipy.linalg import svd
import matplotlib.pyplot as plt
import select
import csv
import json
from pathlib import Path as dir_path
import yaml
from scipy.interpolate import splprep, splev
from scipy import interpolate
import copy
from concurrent.futures import ThreadPoolExecutor
import open3d as o3d

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)  # Add the script directory to the system path for importing custom modules

from utils.ICP_scanmatcher import ICP_Scanmatcher
from utils.occupancy_grid import ProbabilisticOccupancyGrid

import utils.ICP as ICP
from utils.ScanContextManager2D import ScanContextManager
from utils.PoseGraphManager2D import PoseGraphManager
from utils.UtilsMisc2D import ScanContextResultSaver
from utils.MapManager2D import MapManager2D
from utils.SegmentationManager2D import ScansegmentationManager2D
from utils.UtilsMisc2D import yawdeg2se2

import std_msgs.msg
import geometry_msgs.msg
from sensor_msgs.msg import PointCloud2, LaserScan
from sensor_msgs_py import point_cloud2
from slam_interfaces.msg import SubMapT, MapArrayT
from builtin_interfaces.msg import Time

import gtsam

class Feature_scan_SLAM(Node):
    def __init__(self):
        super().__init__('scan_to_scan_matching')

        # Declare parameters
        # Topic parameters
        self.declare_parameter('lidar_topic', "/scan") #/a200_1057/sensors/lidar2d_1/scan

        # File and directory parameters
        self.declare_parameter('save_dir', "/home/ruan/dev_ws/src/lidar_slam_2d/results")
        self.declare_parameter('test_name', "real_lab_euc")  # Name of the log file without extension
        self.declare_parameter('log_data', True)  # Flag to enable/disable data logging
        
        # Frame parameters
        self.declare_parameter('global_frame_id', 'map')
        self.declare_parameter('sensor_frame_id', "laser_frame")  #lidar2d_1_laser
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('robot_frame_id', 'base_link')   
        
        # Other parameters
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('use_odom', True)  # Flag to use odometry data
        self.declare_parameter('trans_for_mapupdate_', 1.0)  # Distance threshold for map update # 1
        
        # Scan-to-map matching parameters
        self.declare_parameter('scan_map_max_iterations', 300)
        self.declare_parameter('scan_map_tolerance', 1e-15)
        self.declare_parameter('scan_map_visualize', False)
        self.declare_parameter('scan_matching_error_threshold', 2.0) # lab 0,5 # 1.0 #0.15# Threshold for scan matching error
        self.declare_parameter('max_translation', 1.0) #lab 0,4 # Maximum allowable translation between scans (in meters)
        self.declare_parameter('max_rotation', 30)  # Maximum allowable rotation between scans (in degrees)
        
        # Noise filtering parameters
        self.declare_parameter('enable_noise_filtering', False)
        self.declare_parameter('noise_filter_method', 'statistical')  # 'statistical', 'median', 'gradient', 'combined'
        self.declare_parameter('statistical_k_neighbors', 10)
        self.declare_parameter('statistical_std_threshold', 2.0)
        self.declare_parameter('median_window_size', 5)
        self.declare_parameter('gradient_max_threshold', 0.5)
        
        # Map management parameters
        self.declare_parameter('filter_dis', 0.1)
        self.declare_parameter('grid_downsample_res', 0.1) #0.08 #lab 0.25
        self.declare_parameter('merge_dis', 0.2)
        self.declare_parameter('lidar_exclusion_radius', 0.1)  # 10 cm exclusion zone around lidar
        
        # Occupancy grid map parameters
        self.declare_parameter('occupancy_grid_publish', False)
        self.declare_parameter('map_resolution', 0.1)
        self.declare_parameter('map_initial_size', 10)
        self.declare_parameter('map_p_occ', 0.7)
        self.declare_parameter('map_p_free', 0.3)
        self.declare_parameter('map_uniform_expand', True)
        
        # Other parameters
        self.declare_parameter('model_state', "/model_states")
        
        
        # Old scan-to-scan matching parameters
        self.declare_parameter('scan_max_iterations', 20)
        self.declare_parameter('scan_tolerance', 1e-5)
        self.declare_parameter('scan_visualize', False)
        
        # Backend parameters
        self.declare_parameter('num_rings', 20)
        self.declare_parameter('num_sectors', 60)
        self.declare_parameter('num_candidates', 10)
        self.declare_parameter('try_gap_loop_detection', 30)
        self.declare_parameter('loop_threshold', 0.16)
        self.declare_parameter('save_gap', 50)
        self.declare_parameter("SCM_radius", 8)
        self.declare_parameter("SCM_recent_node_exclusion", 500)  # Number of recent nodes to exclude in loop detection

        self.declare_parameter('PGM_prior_covariance', [1e-6, 1e-6, 1e-6])
        self.declare_parameter('PGM_odom_covariance', [0.5, 0.5, 0.1])
        self.declare_parameter('PGM_loop_covariance', [0.5, 0.5, 0.1])

        #-------------------------------------------------
        # Topic parameters
        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.model_state = self.get_parameter('model_state').value
        
        # File and directory parameters
        self.global_frame_id = self.get_parameter('global_frame_id').get_parameter_value().string_value
        self.odom_frame_id = self.get_parameter('odom_frame_id').get_parameter_value().string_value
        self.sensor_frame_id = self.get_parameter('sensor_frame_id').value
        self.robot_frame_id = self.get_parameter('robot_frame_id').value
        
        # Initialize TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Scan context parameters
        self.num_rings = self.get_parameter('num_rings').get_parameter_value().integer_value
        self.num_sectors = self.get_parameter('num_sectors').get_parameter_value().integer_value
        self.num_candidates = self.get_parameter('num_candidates').get_parameter_value().integer_value
        self.try_gap_loop_detection = self.get_parameter('try_gap_loop_detection').get_parameter_value().integer_value
        self.loop_threshold = self.get_parameter('loop_threshold').get_parameter_value().double_value
        self.save_gap = self.get_parameter('save_gap').get_parameter_value().integer_value
        self.SCM_radius = self.get_parameter('SCM_radius').get_parameter_value().integer_value
        self.SCM_recent_node_exclusion = self.get_parameter('SCM_recent_node_exclusion').get_parameter_value().integer_value
        
        # Pose graph parameters
        self.PGM_prior_covariance = np.array(self.get_parameter('PGM_prior_covariance').get_parameter_value().double_array_value)
        self.PGM_odom_covariance = np.array(self.get_parameter('PGM_odom_covariance').get_parameter_value().double_array_value)
        self.PGM_loop_covariance = np.array(self.get_parameter('PGM_loop_covariance').get_parameter_value().double_array_value)

        # Other parameters
        self.log_data = self.get_parameter('log_data').get_parameter_value().bool_value
        self.use_odom = self.get_parameter('use_odom').get_parameter_value().bool_value
        self.timestamp_list = [] 
        self.publish_tf = self.get_parameter('publish_tf').get_parameter_value().bool_value
        self.get_logger().info(f"Publish TF: {self.publish_tf}")

        # Scan-to-scan matching parameters
        self.max_iterations = self.get_parameter('scan_max_iterations').get_parameter_value().integer_value
        self.tolerance = self.get_parameter('scan_tolerance').get_parameter_value().double_value
        self.visualize = self.get_parameter('scan_visualize').get_parameter_value().bool_value

        # Scan-to-map matching parameters
        self.max_iterations_map = self.get_parameter('scan_map_max_iterations').get_parameter_value().integer_value
        self.tolerance_map = self.get_parameter('scan_map_tolerance').get_parameter_value().double_value
        self.visualize_map = self.get_parameter('scan_map_visualize').get_parameter_value().bool_value
        self.scan_matching_error_threshold = self.get_parameter('scan_matching_error_threshold').get_parameter_value().double_value
        self.max_translation = self.get_parameter('max_translation').get_parameter_value().double_value
        self.max_rotation = self.get_parameter('max_rotation').get_parameter_value().integer_value

        # Noise filtering parameters
        self.enable_noise_filtering = self.get_parameter('enable_noise_filtering').get_parameter_value().bool_value
        self.noise_filter_method = self.get_parameter('noise_filter_method').get_parameter_value().string_value
        self.statistical_k_neighbors = self.get_parameter('statistical_k_neighbors').get_parameter_value().integer_value
        self.statistical_std_threshold = self.get_parameter('statistical_std_threshold').get_parameter_value().double_value
        self.median_window_size = self.get_parameter('median_window_size').get_parameter_value().integer_value
        self.gradient_max_threshold = self.get_parameter('gradient_max_threshold').get_parameter_value().double_value

        # Local map variables
        self.filter_dis = self.get_parameter('filter_dis').get_parameter_value().double_value
        self.grid_downsample_res = self.get_parameter('grid_downsample_res').get_parameter_value().double_value
        self.merge_dis = self.get_parameter('merge_dis').get_parameter_value().double_value
        self.lidar_exclusion_radius = self.get_parameter('lidar_exclusion_radius').get_parameter_value().double_value

        # Occupancy grid map variables
        self.map_resolution = self.get_parameter('map_resolution').get_parameter_value().double_value
        self.map_initial_size = self.get_parameter('map_initial_size').get_parameter_value().integer_value
        self.p_occ = self.get_parameter('map_p_occ').get_parameter_value().double_value
        self.p_free = self.get_parameter('map_p_free').get_parameter_value().double_value
        self.uniform_expand = self.get_parameter('map_uniform_expand').get_parameter_value().bool_value
        self.occupancy_grid_publish = self.get_parameter('occupancy_grid_publish').get_parameter_value().bool_value
        
        # map keyframe_list
        self.keyframe_list = []
        self.key_frame_maps = []
        self.optimised_map = None
        self.trans_for_mapupdate_ = self.get_parameter('trans_for_mapupdate_').get_parameter_value().double_value
        
        # Subscribers
        self.subscription = self.create_subscription(LaserScan, self.get_parameter('lidar_topic').value, self.laserscan_CB, 10)
        self.gt_subscriber = self.create_subscription(ModelStates, self.get_parameter('model_state').value, self.model_CB, 10)
        
        # Publishers
        self.publisher_ = self.create_publisher(OccupancyGrid, 'map', 10)
        self.gt_pub = self.create_publisher(Path, 'ground_truth_path', 10)
        self.odom_pub = self.create_publisher(Path, 'odom_path', 10)
        self.estimated_pub = self.create_publisher(Path, 'estimated_path', 10)
        self.optimized_pub = self.create_publisher(Path, 'optimized_path', 10)
        self.optimized_map_pub = self.create_publisher(OccupancyGrid, 'optimized_map', 10)
        self.submap_pub = self.create_publisher(SubMapT, '/submap', 10)
        self.map_array_pub = self.create_publisher(MapArrayT, '/map_array', 10)
        self.pose_pub_ = self.create_publisher(PoseStamped, '/current_pose', 10)
        self.path_est_pub_ = self.create_publisher(Path, '/path', 10)
        self.path_gt_pub_ = self.create_publisher(Path, '/path_gt', 10)
        
        # Service
        self.map_save_srv = self.create_service(Empty, 'map_save', self.map_save_callback)
        
        # Timers
        if self.occupancy_grid_publish:
            self.timer_occupancy_grid = self.create_timer(1.0, self.publish_occupancy_grid)
        self.timer = self.create_timer(0.1, self.publish_paths)  # Publish at 10 Hz
        
        # TransformBroadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Initialize managers and classes
        self.SegmentationManager = ScansegmentationManager2D()
        self.scan_scan_matching = ICP_Scanmatcher(self.max_iterations, self.tolerance, self.visualize)
        self.scan_map_matching = ICP_Scanmatcher(self.max_iterations_map, self.tolerance_map, self.visualize_map)
        self.MapManager = MapManager2D(self.grid_downsample_res, self.filter_dis, self.merge_dis, self.lidar_exclusion_radius)
        self.og_map = ProbabilisticOccupancyGrid(self.map_resolution, self.map_initial_size, self.p_occ, self.p_free, self.uniform_expand)
        self.og_optimised_map = ProbabilisticOccupancyGrid(self.map_resolution, self.map_initial_size, self.p_occ, self.p_free, self.uniform_expand)
        
        self.PGM = PoseGraphManager(self.PGM_prior_covariance,
                                    odom_covariance=self.PGM_odom_covariance,
                                    loop_covariance=self.PGM_loop_covariance)
        self.PGM.addPriorFactor()
        
        # self.ResultSaver = PoseGraphResultSaver2D(
        #     init_pose=self.PGM.curr_se2,
        #     save_gap=self.save_gap,
        #     num_frames=999999,  # Unknown in live mode
        #     seq_idx='ros2',
        #     save_dir="/home/ruan/dev_ws/src/lidar_slam_2d/results/Posegraph"
        # )
        self.SCM = ScanContextManager(
            shape=[self.num_rings, self.num_sectors],
            num_candidates = self.num_candidates,
            threshold = self.loop_threshold,
            pointcloud_radius = self.SCM_radius,
            recent_node_exclusion = self.SCM_recent_node_exclusion
        )
        self.sc_saver = ScanContextResultSaver(
            save_gap = 5, 
            save_dir = "/home/ruan/dev_ws/src/lidar_slam_2d/results/Posegraph"
        )
        
        # Point cloud dynamic variables, transforms and map (Core variables)
        self.frame_idx = 0
        self.current_pcd = None
        self.prev_pcd = None
        self.odom_transform = np.eye(3) # Relative tranformation between two scans
        self.icp_initial = np.eye(3) # Initial guess for the scan matching
        self.odom_estimate = np.eye(3)  #Accumalulation of relative scans
        self.intermediate_estimate = np.eye(3)
        self.estimate = np.eye(3)
        self.prev_estimate = np.eye(3)
        self.Map = None
        self.flag = False
        self.loop_closure = False
        self.initial_transformation = np.eye(3) 
        
        # State variables and trajectories
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.x_gt = 0.0
        self.y_gt = 0.0
        self.yaw_gt = 0.0
        self.x_est = 0.0
        self.y_est = 0.0
        self.yaw_est = 0.0
        self.ground_truth_trajectory = [np.array([0, 0, 0])]
        self.odom_trajectory = [np.array([0, 0, 0])]
        self.estimated_trajectory = [np.array([0, 0, 0])]
        self.T_odom = np.eye(3)  
        self.T_odom_prev = np.eye(3) 
        self.T_est = np.eye(3) 
        
        ## Ros messages
        self.current_pose_stamped_ = PoseStamped()
        self.current_pose_gt_stamped = PoseStamped()
        self.gt_pose_full = PoseStamped()
        self.path_est = Path()
        self.path_gt = Path()
        self.subMap = SubMapT()
        self.map_array_msg = MapArrayT()
        self.previous_position_ = np.zeros(3)
        self.displacement = 0.0
        self.sub_map_previous_position = np.zeros(3)
        self.latest_distance = 0.0
        self.sub_map_displacement = 0.0
        
        ## Threading variables
        self.loop_executor = ThreadPoolExecutor(max_workers=1)
        self.mapping_flag_= False

        # Construct relative paths and files for saving data
        self.save_dir = dir_path(self.get_parameter('save_dir').value)
        self.test_name = self.get_parameter('test_name').value
        results_data_dir = self.save_dir  / "Data" # Define directories
        results_map_dir = self.save_dir / "Maps"
        results_map_keyframe_dir = self.save_dir / "Maps" / "Keyframe_maps"
        results_data_dir.mkdir(parents=True, exist_ok=True) # Ensure directories exist
        results_map_dir.mkdir(parents=True, exist_ok=True)
        results_map_keyframe_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_log_file = results_data_dir / f"{self.test_name}_trajectories.csv"
        self.scanmatch_log_file = results_data_dir / f"{self.test_name}_scanmatch.csv"
        self.map_log_file = results_map_dir / f"{self.test_name}" # Map file without extension for YAML and PNG (Required for saving function)
        self.keyframe_log_file = results_map_keyframe_dir / f"{self.test_name}_keyframe_map" 
        
        # Initialize log files
        with open(self.trajectory_log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "X", "Y", "Yaw", "Odom_X", "Odom_Y", "Odom_Yaw", "Est_X", "Est_Y", "Est_Yaw"])      
        with open(self.scanmatch_log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Algo_iteration", "Algo_iter_time" "ss_final_error", "ss_num_iterations", "ss_time_taken",
                    "sm_final_error", "sm_num_iterations", "sm_time_taken"])

        self.save_dir_tum = self.save_dir / "Data"
        self.gt_tum_log_file = os.path.join(self.save_dir_tum, f'Sim_groundtruth_{self.test_name}.txt')
        self.est_tum_log_file = os.path.join(self.save_dir_tum, f'Sim_estimate_{self.test_name}.txt')
        self.odom_tum_log_file = os.path.join(self.save_dir_tum, f'Sim_odometry_{self.test_name}.txt')
        self.optimised_tum_log_file = os.path.join(self.save_dir_tum, f'Sim_optimised_{self.test_name}.txt')

        open(self.gt_tum_log_file, 'w').close()
        open(self.est_tum_log_file, 'w').close()
        open(self.odom_tum_log_file, 'w').close()
        open(self.optimised_tum_log_file, 'w').close()
        
        # Scan matching logging variables
        self.iteration_time = 0
        self.ss_final_error = 0
        self.ss_num_iterations = 0
        self.ss_time_taken = 0
        self.sm_final_error = 0
        self.sm_num_iterations = 0
        self.sm_time_taken = 0
            
        # Print initialization information
        self.get_logger().info(f"Trajectory log file: {self.trajectory_log_file}")
        self.get_logger().info(f"Scanmatch log file: {self.scanmatch_log_file}")
        self.get_logger().info(f"Map file: {self.map_log_file}")
        self.get_logger().info("SLAM node initialized")
    
    # Callback functions
    def model_CB(self, model_data):

        try:
            # Find the index of the robot in the model states
            robot_index = model_data.name.index("my_bot")
            robot_pose = PoseStamped()
            robot_pose.header.stamp = self.get_clock().now().to_msg()
            robot_pose.pose = model_data.pose[robot_index]
            
            # Extract position and orientation
            x = robot_pose.pose.position.x
            y = robot_pose.pose.position.y
            z = robot_pose.pose.position.z
            qx = robot_pose.pose.orientation.x
            qy = robot_pose.pose.orientation.y
            qz = robot_pose.pose.orientation.z
            qw = robot_pose.pose.orientation.w
            
            # Convert quaternion to roll, pitch, yaw
            roll, pitch, yaw = tf_transformations.euler_from_quaternion((qx, qy, qz, qw))

            # Update the ground truth pose
            self.x_gt = x
            self.y_gt = y
            self.yaw_gt = yaw
            
            # Update the PoseStamped message
            self.gt_pose_full.header.stamp = robot_pose.header.stamp
            self.gt_pose_full.header.frame_id = "map"
            self.gt_pose_full.pose.position.x = x
            self.gt_pose_full.pose.position.y = y
            self.gt_pose_full.pose.position.z = z
            self.gt_pose_full.pose.orientation.x = qx
            self.gt_pose_full.pose.orientation.y = qy
            self.gt_pose_full.pose.orientation.z = qz
            self.gt_pose_full.pose.orientation.w = qw

        except ValueError:
            self.get_logger().warn("Robot 'my_bot' not found in model states.")
        
    def laserscan_CB(self, msg):
        
        self.timestamp = msg.header.stamp
        self.current_pose_gt_stamped = copy.deepcopy(self.gt_pose_full)
        self.iteration_start_time = time.time()
        
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        
        # Extract every second beam
        # ranges = ranges[::2]
        # angles = angles[::2]
        
        # For F1tenh car config
        # ranges = ranges[:-1] 
        if len(ranges) != len(angles):
            self.get_logger().error("Scan and ranges are not the same length.")
            return
        
        # Using raw scans with noise filtering
        points = []
        valid_ranges = []
        valid_angles = []
        
        # First pass: basic range filtering
        for r, theta in zip(msg.ranges, angles):
            if msg.range_min < r < msg.range_max:
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                points.append([x, y, 0.0])  # Z = 0 for 2D lidar scans
                valid_ranges.append(r)
                valid_angles.append(theta)
        
        points = np.array(points)
        # points = points[::2]  # Downsample by taking every second point
        # print(f"Z values range: {points[:, 2].min():.3f} to {points[:, 2].max():.3f}")

        # Apply noise filtering if enabled and we have enough points
        if self.enable_noise_filtering and len(points) > 10:
            points = self.filter_noisy_points(points, valid_ranges, valid_angles, self.noise_filter_method)
        
        self.current_pcd = points
        
        # Transform the point cloud to robot base frame
        try:
            # Get the transform from sensor frame to robot base frame
            time_point = rclpy.time.Time.from_msg(msg.header.stamp)
            transform = self.tf_buffer.lookup_transform(
                self.robot_frame_id,    # target frame
                self.sensor_frame_id,   # source frame
                Time(),       # time
                timeout=rclpy.duration.Duration(seconds=2.0)
            )
            if self.frame_idx == 0:
                initial_transform = self.tf_buffer.lookup_transform(
                self.sensor_frame_id, 
                self.robot_frame_id,
                Time(),       # time
                timeout=rclpy.duration.Duration(seconds=2.0)
            )
                trans = np.array([
                    initial_transform.transform.translation.x,
                    initial_transform.transform.translation.y
                ])
                quat = initial_transform.transform.rotation
                r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
                yaw = r.as_euler('xyz')[2]  # Extract yaw angle
                self.initial_transformation = np.array([
                    [np.cos(yaw), -np.sin(yaw), trans[0]],
                    [np.sin(yaw), np.cos(yaw), trans[1]],
                    [0, 0, 1]
                ])
            self.current_pcd = self.transform_pointcloud_array(self.current_pcd, transform)

            # print(f"Z values range: {self.current_pcd[:, 2].min():.3f} to {self.current_pcd[:, 2].max():.3f}")

            # Run SLAM iteration
            self.run_iteration(self.current_pcd, ranges, angles, timestamp=msg.header.stamp)
        except Exception as e:
            self.get_logger().warn(f'Could not transform point cloud from {self.sensor_frame_id} to {self.robot_frame_id}: {str(e)}')
        
        # self.run_iteration(self.current_pcd, ranges, angles, timestamp=msg.header.stamp)
        
        # Create a PointCloud2 message from scan
        header = std_msgs.msg.Header()
        header.stamp = msg.header.stamp
        header.frame_id = msg.header.frame_id
        pointcloud_msg = point_cloud2.create_cloud_xyz32(header, self.current_pcd[:, :3]) 
        
        # # Using segmented scans
        # scan_segments, range_bearing_segments, _ = self.SegmentationManager.segment_scan(ranges, angles)
        # segment_points = []
        # seg_ranges = []
        # seg_bearings = []
        # for segment, rb_segment in zip(scan_segments, range_bearing_segments):
        #     for point, rb in zip(segment, rb_segment):
        #         segment_points.append(point)
        #         seg_ranges.append(rb[0])
        #         seg_bearings.append(rb[1])
        # self.current_pcd = np.array(segment_points)
        # self.get_logger().info(f'Segmented point cloud shape: {self.current_pcd.shape}')
        # self.run_iteration(self.current_pcd, seg_ranges, seg_bearings, timestamp=msg.header.stamp)
        
        # Publish the transform from map to base_link
        # self.publish_transform(msg.header.stamp)
        
        if self.log_data:
            
            # Save in TUM format for benchmarking
            self.ground_truth_trajectory.append(np.array([self.x_gt, self.y_gt,  self.yaw_gt]))
            self.save_pose_to_tum(self.timestamp, self.ground_truth_trajectory[-1], self.gt_tum_log_file)
            self.save_pose_to_tum(self.timestamp, self.odom_trajectory[-1], self.odom_tum_log_file)
            self.save_pose_to_tum(self.timestamp, self.estimated_trajectory[-1], self.est_tum_log_file)
            
            # Log the scan matching results
            with open(self.scanmatch_log_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([self.frame_idx, self.iteration_time, self.ss_final_error, self.ss_num_iterations, self.ss_time_taken,
                            self.sm_final_error, self.sm_num_iterations, self.sm_time_taken])
        
        self.iteration_end_time = time.time()
        self.iteration_time = (self.iteration_end_time - self.iteration_start_time) * 1000  # in milliseconds
        self.get_logger().info(f'Frame: {self.frame_idx}, Iteration time: {self.iteration_time:.2f} msec')

    # Main functions
    def run_iteration(self, current_pcd, ranges, angles, timestamp):

        # Extract pointcloud and prepare for processing
        point_set = current_pcd.copy()
        current_pcd = current_pcd[:, :2]  # Keep only x and y coordinates
        self.timestamp_list.append(timestamp)

        self.PGM.curr_node_idx = self.frame_idx
        self.SCM.addNode(node_idx=self.PGM.curr_node_idx, ptcloud=current_pcd)
        
        # First iteration
        if self.frame_idx == 0:
            self.prev_pcd = copy.deepcopy(current_pcd)
            self.Map = copy.deepcopy(current_pcd)
            #self.ResultSaver.saveUnoptimizedPoseGraphResult(self.PGM.curr_se2, self.frame_idx)
            self.prev_estimate = self.estimate
            self.frame_idx += 1
            # self.ResultSaver.filecount = self.frame_idx
            
            self.current_pose_stamped_.header.stamp = timestamp
            self.update_map_array(point_set,  self.current_pose_stamped_)
            return
        
        # Odom initial guess
        Initial_guess = np.eye(3)
        T_odom_relative = np.eye(3)
        if (self.use_odom):
            
            # Get odom to base_link transform from tf tree
            try:
                odom_trans = self.tf_buffer.lookup_transform(
                    self.odom_frame_id,
                    self.robot_frame_id,
                    #Time() # timestamp rclpy.time.Time() #stamp - When using stamp the lidar timestamp seems to be ahead of tf tree
                    rclpy.time.Time()         
            )
            except Exception as e:
                self.get_logger().error(str(e))
                return
            
            # Create 3x3 SE2 transformation matrix
            t = odom_trans.transform.translation
            q = odom_trans.transform.rotation
            roll, pitch, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            self.T_odom = np.array([
                [np.cos(yaw), -np.sin(yaw), t.x],
                [np.sin(yaw), np.cos(yaw), t.y],
                [0, 0, 1]
            ], dtype=np.float32)
            
            T_odom_relative = np.linalg.inv(self.T_odom_prev) @ self.T_odom
            Initial_guess = self.prev_estimate @ T_odom_relative
        else:
            # Use constant velocity model based on previous transformation
            if self.frame_idx > 1:
                # Calculate the relative transformation from the last two frames
                prev_prev_estimate = getattr(self, 'prev_prev_estimate', np.eye(3))
                velocity_transform = np.linalg.inv(prev_prev_estimate) @ self.prev_estimate
                Initial_guess = self.prev_estimate @ velocity_transform
                # Store previous estimate for next iteration
                self.prev_prev_estimate = self.prev_estimate.copy()
            else:
                # For the second frame, just use previous estimate
                Initial_guess = self.prev_estimate
                self.prev_prev_estimate = np.eye(3)
        self.T_odom_prev = self.T_odom.copy()
        
        #-------------------------------------------------
        # Scan matching
        start_time = time.time()
        transformation, self.sm_final_error, self.sm_num_iterations, self.sm_time_taken, transformed_source = self.scan_map_matching.icp_scanmatching_map(self.Map, current_pcd, Initial_guess)
        
        
        # Error threshold check
        # self.get_logger().info(f"Scan matching error: {self.sm_final_error}, iterations: {self.sm_num_iterations}, time: {self.sm_time_taken*1000:.2f} ms")
        if(self.sm_final_error > self.scan_matching_error_threshold):
            # self.estimate = transformation
            self.get_logger().warn(f"Scan matching error too high: {self.sm_final_error}. Using odom estimate.")
            self.estimate = Initial_guess
            self.Map = self.MapManager.update_point_map(self.Map, current_pcd, self.estimate)
        elif self.jump_detected(Initial_guess, transformation, self.max_translation, np.deg2rad(self.max_rotation)):
            # # Check for jumps
            self.get_logger().warn("Warning: Sudden jump detected! Keeping odom estimate.")
            self.estimate = Initial_guess
            # self.Map = self.MapManager.update_point_map(self.Map, current_pcd, self.estimate)
        else:
            self.estimate = transformation
            self.get_logger().info("Incremental map update performed.")
            self.Map = self.MapManager.update_point_map(self.Map, current_pcd, self.estimate)

        #-----------------------------------------------------------
        # # Two stage scan matching
        # # Odom scan matching (Assumes constant velocity model)
        # initial_guess = np.eye(3)
        # self.odom_transform, self.ss_final_error, self.ss_num_iterations, self.ss_time_taken, transformed_source = self.scan_scan_matching.icp_scanmatching(self.prev_pcd, current_pcd, initial_guess)
        # self.odom_estimate = self.odom_estimate @ self.odom_transform
        # self.intermediate_estimate = self.estimate @ self.odom_transform
    
        # # Extract x, y, and yaw from the transformation matrix, format and print
        # x_odom, y_odom = self.odom_estimate[0, 2], self.odom_estimate[1, 2]
        # yaw_odom = np.arctan2(self.odom_estimate[1, 0], self.odom_estimate[0, 0])  # Extract rotation in radians
        # self.odom_trajectory.append(np.array([x_odom, y_odom, yaw_odom]))
        
        # # Global scan to map matching
        # transformation, self.sm_final_error, self.sm_num_iterations, self.sm_time_taken, transformed_source = self.scan_map_matching.icp_scanmatching_map(self.Map, current_pcd,  self.intermediate_estimate)
        #Error 0.04 threshold
        # -------------------------------------------------------------------
        
        # Update pose graph
        # self.PGM.curr_se2 = self.PGM.curr_se2 @ self.odom_transform
        # self.PGM.addOdometryFactor(self.odom_transform)
        # self.PGM.prev_node_idx = self.PGM.curr_node_idx
        # self.icp_initial = self.odom_transform
        # self.prev_pcd = copy.deepcopy(current_pcd)
        # self.sc_saver.saveScanContextSVG(self.SCM.scancontexts[self.frame_idx], self.frame_idx)
        # self.sc_saver.saveScanContextHeatmapSVG(self.SCM.scancontexts[self.frame_idx], self.frame_idx)
        
        # # Check for jumps
        # if self.jump_detected(self.estimate, transformation):
        #     self.estimate = transformation
        #     self.T_est = self.estimate
        # else:
        #     self.get_logger().info("Warning: Sudden jump detected! Keeping previous estimate.")
        # Update the local point map
        
        # self.Map = self.MapManager.update_point_map(self.Map, current_pcd, self.estimate)
        # self.MapManager.plot_point_map(self.Map)
            
        # Extract pose and update the occupancy grid map
        self.og_map.update_from_local_points(self.estimate, current_pcd)
        # self.og_map.plot_map_realtime()
        
        # Convert SE2 to SE3 for publishing
        transformation_3d = np.eye(4)
        transformation_3d[:2, :2] = self.estimate[:2, :2]  # Copy rotation
        transformation_3d[:2, 3] = self.estimate[:2, 2]  # Copy translation
        transformation_3d[2, 2] = 1.0  # Set z-axis to 1 for 3D
        transformation_3d[3, 3] = 1.0  #
        self.publish_pose_and_pointcloud(transformation_3d, point_set, timestamp)

        # Update pose graph
        estimate_transform = np.linalg.inv(self.prev_estimate) @ self.estimate
        self.PGM.curr_se2 = self.estimate
        self.PGM.addOdometryFactor(estimate_transform)
        self.PGM.prev_node_idx = self.PGM.curr_node_idx
        self.icp_initial = self.odom_transform
        self.prev_pcd = copy.deepcopy(current_pcd)
        self.prev_estimate = self.estimate
        
        # # --------------------------------------------------------------
        # # Save keyframe of local map
        # if self.frame_idx % self.try_gap_loop_detection == 0:
        #     # self.get_logger().info(f"Saving keyframe at frame {self.frame_idx}")
        #     self.keyframe_list.append(self.frame_idx)
        #     # Transform points to the robot's reference frame
        #     # Points may already be in local reference frame, so commenting out transformation
        #     homogeneous_map = np.hstack((self.Map, np.ones((self.Map.shape[0], 1))))  # Convert to homogeneous coordinates
        #     map_in_local_frame = (np.linalg.inv(self.estimate) @ homogeneous_map.T).T[:, :2]  # Transform and convert back to 2D
        #     # map_in_local_frame = self.Map.copy()  # Use map points directly
        #     self.key_frame_maps.append(map_in_local_frame)
            
        # # --------------------------------------------------------------
        # # Loop detection
        # if self.frame_idx > 1 and self.frame_idx % self.try_gap_loop_detection == 0:
        #     self.get_logger().info(f"Checking for loop closure at frame {self.frame_idx}")
        #     loop_idx, loop_dist, yaw_diff_deg = self.SCM.detectLoop()
        #     if loop_idx is not None:
        #         self.get_logger().info(f"Loop closure detected at {self.frame_idx} <-> {loop_idx}")
        #         self.loop_closure = True
                
        #         self.ResultSaver.saveUnoptimizedPoseGraphResult_forced(self.PGM.curr_se2, self.frame_idx)
        #         self.get_logger().info(f"Unoptimised pose graph saved at frame {self.frame_idx}")
                
        #         loop_pts = self.SCM.getPtcloud(loop_idx)
        #         loop_transform, _, _ = ICP.icp(current_pcd, loop_pts,
        #                                        init_pose=yawdeg2se2(yaw_diff_deg),
        #                                        max_iterations=20)
        #         self.PGM.addLoopFactor(loop_transform, loop_idx)
                
        #         self.PGM.optimizePoseGraph()
                
        #         self.ResultSaver.saveOptimizedPoseGraphResult(self.frame_idx, self.PGM.graph_optimized)
        #         self.get_logger().info(f"Optimised pose graph saved at frame {self.frame_idx}")
        #         self.ResultSaver.logLoopClosure(loop_idx, self.frame_idx)
                
        #         # Publish optimized trajectory for visualization
        #         self.publish_optimized_trajectory(self.frame_idx, self.PGM.graph_optimized)
                
        #         # self.make_optimised_map_from_scm2()
        #         # self.publish_optimized_map()
        
        
        # self.ResultSaver.saveUnoptimizedPoseGraphResult(self.PGM.curr_se2, self.frame_idx)
        # --------------------------------------------------------------
        
        # Store the estimated trajectory
        self.x_est, self.y_est = self.estimate[0, 2], self.estimate[1, 2]
        self.yaw_est = np.arctan2(self.estimate[1, 0], self.estimate[0, 0]) 
        self.estimated_trajectory.append(np.array([self.x_est, self.y_est, self.yaw_est]))

        # Increment frame index and update file count
        self.frame_idx += 1
        # self.ResultSaver.filecount = self.frame_idx
        
    def transform_pointcloud_array(self, points, transform):
        """
        Transform a numpy array of points using a TF transform.
        
        Args:
            points: Nx3 numpy array of points
            transform: TransformStamped message
            
        Returns:
            Transformed Nx3 numpy array of points
        """
        
        if len(points) == 0:
            return points
            
        # Extract translation and rotation from transform
        translation = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ])
        
        # Convert quaternion to rotation matrix
        quat = transform.transform.rotation
        r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        rotation_matrix = r.as_matrix()
        
        # Apply transformation: R * p + t
        transformed_points = (rotation_matrix @ points.T).T + translation
        
        return transformed_points
    
    # Filters
    def filter_noisy_points(self, points, ranges, angles, method='combined'):
        """
        Filter noisy points from laser scan data using multiple techniques.
        
        Args:
            points: Nx3 numpy array of points [x, y, z]
            ranges: array of range values
            angles: array of angle values
            method: filtering method ('statistical', 'median', 'gradient', 'combined')
            
        Returns:
            Filtered Nx3 numpy array of points
        """
        if len(points) < 10:
            return points
            
        if method == 'statistical':
            return self._statistical_outlier_removal(points)
        elif method == 'median':
            return self._median_filter(points, ranges, angles)
        elif method == 'gradient':
            return self._gradient_filter(points, ranges, angles)
        elif method == 'combined':
            # Apply multiple filters in sequence
            filtered_points = self._median_filter(points, ranges, angles)
            if len(filtered_points) > 5:
                filtered_points = self._gradient_filter(filtered_points, 
                                                      ranges[:len(filtered_points)], 
                                                      angles[:len(filtered_points)])
            if len(filtered_points) > 5:
                filtered_points = self._statistical_outlier_removal(filtered_points)
            return filtered_points
        else:
            return points
    
    def _statistical_outlier_removal(self, points, k_neighbors=None, std_threshold=None):
        """
        Remove statistical outliers using k-nearest neighbors.
        """
        if k_neighbors is None:
            k_neighbors = self.statistical_k_neighbors
        if std_threshold is None:
            std_threshold = self.statistical_std_threshold
        if len(points) < k_neighbors + 1:
            return points
            
        try:
            # Build KDTree for efficient neighbor search
            tree = cKDTree(points[:, :2])  # Use only x, y coordinates
            
            # Find k nearest neighbors for each point
            distances, _ = tree.query(points[:, :2], k=k_neighbors+1)  # +1 because first is the point itself
            mean_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self-distance
            
            # Calculate statistics
            global_mean = np.mean(mean_distances)
            global_std = np.std(mean_distances)
            
            # Filter points based on statistical threshold
            threshold = global_mean + std_threshold * global_std
            valid_mask = mean_distances < threshold
            
            return points[valid_mask]
            
        except Exception as e:
            self.get_logger().warn(f"Statistical filtering failed: {e}")
            return points
    
    def _median_filter(self, points, ranges, angles, window_size=None):
        """
        Apply median filter to range data and reconstruct points.
        """
        if window_size is None:
            window_size = self.median_window_size
        if len(ranges) < window_size:
            return points
            
        try:
            # Apply median filter to ranges
            from scipy.signal import medfilt
            filtered_ranges = medfilt(ranges, kernel_size=window_size)
            
            # Reconstruct points from filtered ranges
            filtered_points = []
            for r, theta in zip(filtered_ranges, angles):
                if r > 0:  # Valid range
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    filtered_points.append([x, y, 0.0])
            
            return np.array(filtered_points) if filtered_points else points
            
        except Exception as e:
            self.get_logger().warn(f"Median filtering failed: {e}")
            return points
    
    def _gradient_filter(self, points, ranges, angles, max_gradient=None):
        """
        Filter points based on range gradient to remove sudden jumps.
        """
        if max_gradient is None:
            max_gradient = self.gradient_max_threshold
        if len(ranges) < 3:
            return points
            
        try:
            # Calculate gradient of range values
            range_gradient = np.gradient(ranges)
            
            # Filter based on gradient threshold
            valid_mask = np.abs(range_gradient) < max_gradient
            
            # Also remove isolated points (points with invalid neighbors)
            for i in range(1, len(valid_mask) - 1):
                if valid_mask[i] and not (valid_mask[i-1] or valid_mask[i+1]):
                    valid_mask[i] = False
            
            return points[valid_mask] if np.any(valid_mask) else points
            
        except Exception as e:
            self.get_logger().warn(f"Gradient filtering failed: {e}")
            return points
    
    # Utility functions
    def publish_occupancy_grid(self):
        prob_map = self.og_map.get_probability_map().T
        
        # Convert probability map to integer occupancy values (-1 unknown, 0 free, 100 occupied)
        occupancy_values = np.clip((prob_map * 100).astype(int), 0, 100).flatten()
        
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        msg.info.resolution = self.og_map.resolution
        msg.info.width = int(self.og_map.extents[0])
        msg.info.height = int(self.og_map.extents[1])
        origin_pose = Pose()
        origin_pose.position.x = -self.og_map.origin[0] * self.og_map.resolution
        origin_pose.position.y = -self.og_map.origin[1] * self.og_map.resolution
        msg.info.origin = origin_pose
        
        msg.data = occupancy_values.tolist()
        
        self.publisher_.publish(msg)
        # self.get_logger().info("Published probability map")
        
        # print("Publishing origin:", -self.og_map.origin[0] * self.og_map.map_resolution, -self.og_map.origin[1] * self.og_map.map_resolution)
        
    def publish_paths(self):
        time_msg = self.get_clock().now().to_msg()
        time_sec = time_msg.sec + time_msg.nanosec * 1e-9
        gt_traj = self.ground_truth_trajectory
        odom_traj = self.odom_trajectory
        est_traj = self.estimated_trajectory
        
        self.gt_pub.publish(self.convert_to_path(self.ground_truth_trajectory, "map"))
        self.odom_pub.publish(self.convert_to_path(self.odom_trajectory, "map"))
        self.estimated_pub.publish(self.convert_to_path(self.estimated_trajectory, "map"))
        
        with open(self.trajectory_log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([time_sec, gt_traj[-1][0], gt_traj[-1][1], gt_traj[-1][2],
                    odom_traj[-1][0], odom_traj[-1][1], odom_traj[-1][2],
                    est_traj[-1][0], est_traj[-1][1], est_traj[-1][2]])
            
    def convert_to_path(self, trajectory, frame_id="map"):
        """ Converts a list of (x, y, theta) into a Path message """
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = frame_id

        for pose in trajectory:
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = frame_id
            pose_stamped.pose.position.x = float(pose[0])
            pose_stamped.pose.position.y = float(pose[1])
            pose_stamped.pose.position.z = 0.0

            # Convert yaw (theta) to quaternion
            from tf_transformations import quaternion_from_euler
            q = quaternion_from_euler(0, 0, float(pose[2]))
            pose_stamped.pose.orientation.x = q[0]
            pose_stamped.pose.orientation.y = q[1]
            pose_stamped.pose.orientation.z = q[2]
            pose_stamped.pose.orientation.w = q[3]

            path_msg.poses.append(pose_stamped)

        return path_msg
    
    def publish_transform(self, timestamp):
        """
        Publish the transform from the map frame to the base_link frame.
        """
        t = TransformStamped()

        # Set the header
        # t.header.stamp = self.get_clock().now().to_msg()
        t.header.stamp = timestamp
        t.header.frame_id = "map"  # Parent frame
        t.child_frame_id = "base_link"  # Child frame

        # Set the translation (x, y, z)
        t.transform.translation.x = self.x_est
        t.transform.translation.y = self.y_est
        t.transform.translation.z = 0.0  # Assuming 2D SLAM

        # Set the rotation (convert yaw to quaternion)
        from tf_transformations import quaternion_from_euler
        q = quaternion_from_euler(0, 0, self.yaw_est)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(t)

    def jump_detected(self, prev_estimate, new_estimate, max_translation=0.3, max_rotation=np.deg2rad(60)):
        """
        Checks for sudden jumps in the estimated transformation.

        Args:
            prev_estimate (np.ndarray): Previous transformation matrix (3x3).
            new_estimate (np.ndarray): New transformation matrix (3x3).
            max_translation (float): Maximum allowed translation difference (meters).
            max_rotation (float): Maximum allowed rotation difference (radians).

        Returns:
            bool: True if the new estimate is valid, False if it is a jump.
        """

        # Extract translation (x, y)
        prev_x, prev_y = prev_estimate[0, 2], prev_estimate[1, 2]
        new_x, new_y = new_estimate[0, 2], new_estimate[1, 2]

        # Compute translation difference
        translation_diff = np.sqrt((new_x - prev_x)**2 + (new_y - prev_y)**2)
        self.get_logger().info(f"Translation diff: {translation_diff:.3f} m")

        # Extract rotation (yaw)
        prev_yaw = np.arctan2(prev_estimate[1, 0], prev_estimate[0, 0])
        new_yaw = np.arctan2(new_estimate[1, 0], new_estimate[0, 0])

        # Compute rotation difference
        rotation_diff = np.abs(new_yaw - prev_yaw)
        rotation_diff = (rotation_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
        self.get_logger().info(f"Rotation diff: {np.rad2deg(rotation_diff):.2f} degrees")

        # Check if jump exceeds threshold
        if translation_diff > max_translation or np.abs(rotation_diff) > max_rotation:
            return True 
        return False  

    def publish_pose_and_pointcloud(self, transformation: np.ndarray, pointcloud, stamp: Time):
        
        # Update and publish the current gt pose
        self.current_pose_gt_stamped.header.stamp = stamp
        self.current_pose_gt_stamped.header.frame_id = "map"
        
        # Append to the path gt and publish path (Local copy required otherwise all poses will be the same)
        pose_for_path = copy.deepcopy(self.current_pose_gt_stamped)
        self.path_gt.header.stamp = stamp
        self.path_gt.header.frame_id = "map"
        self.path_gt.poses.append(pose_for_path)
        self.path_gt_pub_.publish(self.path_gt)
        
        # Extract position and orientation from the transformation matrix
        position = transformation[0:3, 3].astype(np.float64)
        quat = tf_transformations.quaternion_from_matrix(transformation.astype(np.float64))
        quat_msg = geometry_msgs.msg.Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        
        # If we want to publish a TF from map → base_link (or robot frame)
        # NB - a frame can only be assinged one parent, otherwise the transform will not be published
        if self.publish_tf:
            base_to_map_msg = TransformStamped()
            base_to_map_msg.header.stamp = stamp
            # base_to_map_msg.header.stamp = self.get_clock().now().to_msg()
            base_to_map_msg.header.frame_id = self.global_frame_id
            base_to_map_msg.child_frame_id = self.robot_frame_id
            base_to_map_msg.transform.translation.x = float(position[0])
            base_to_map_msg.transform.translation.y = float(position[1])
            base_to_map_msg.transform.translation.z = float(position[2])
            base_to_map_msg.transform.rotation = quat_msg

            if self.use_odom:
                odom_to_map_msg = self.calculateMaptoOdomTransform(base_to_map_msg, stamp)
                self.tf_broadcaster.sendTransform(odom_to_map_msg)
            else:
                self.tf_broadcaster.sendTransform(base_to_map_msg)
        
        # Update and publish the current pose
        self.current_pose_stamped_.header.stamp = stamp
        self.current_pose_stamped_.header.frame_id = "map"
        self.current_pose_stamped_.pose.position.x = float(position[0])
        self.current_pose_stamped_.pose.position.y = float(position[1])
        self.current_pose_stamped_.pose.position.z = float(position[2])
        self.current_pose_stamped_.pose.orientation = quat_msg
        self.pose_pub_.publish(self.current_pose_stamped_)

        # Append to the path and publish path (Local copy required otherwise all poses will be the same)
        pose_for_path = copy.deepcopy(self.current_pose_stamped_)
        self.path_est.header.stamp = stamp
        self.path_est.header.frame_id = "map"
        self.path_est.poses.append(pose_for_path)
        self.path_est_pub_.publish(self.path_est)
        
        # Check if we've moved far enough to trigger a map update
        self.displacement = np.linalg.norm(position - self.previous_position_)
        self.previous_position_ = position.copy()
        self.sub_map_displacement = np.linalg.norm(position - self.sub_map_previous_position)
        if (self.sub_map_displacement >= self.trans_for_mapupdate_) and (not self.mapping_flag_):
            
            # Make a local copy of the pose stamped for passing into the thread
            current_pose_copy = PoseStamped()
            current_pose_copy.header = self.current_pose_stamped_.header
            current_pose_copy.pose = self.current_pose_stamped_.pose

            # Update previous_position_ immediately
            self.sub_map_previous_position = position.copy()
            self.get_logger().info(f"Map update triggered with displacement: {self.sub_map_displacement:.3f} m")
            
            # Launch the map-updating task on a new thread using the ThreadPoolExecutor
            def mapping_job():
                self.update_map_array(pointcloud,  current_pose_copy)

            # Submit the mapping job to the executor
            self.mapping_flag_ = True
            self.loop_executor.submit(mapping_job)
        
        # Create a PointCloud2 message from scan
        header = std_msgs.msg.Header()
        header.stamp = stamp
        header.frame_id = "map"
        pointcloud_msg = point_cloud2.create_cloud_xyz32(header, pointcloud) 
        
        # Create and populate the SubMapT message
        self.subMap.header.stamp = stamp
        self.subMap.header.frame_id = "map"
        self.subMap.distance += self.displacement
        self.subMap.pose = self.current_pose_stamped_.pose
        self.subMap.cloud = pointcloud_msg
        
        # Publish the SubMapT message
        self.submap_pub.publish(self.subMap)
        
    def update_map_array(self, downsampled_cloud, current_pose_stamped: PoseStamped):
        
        # Start the map update process
        self.get_logger().info("Updating map array with new point cloud data)")
        num_submaps = len(self.map_array_msg.submaps)
        self.get_logger().info(f"Number of submaps: {num_submaps}")

        # Create SubMap message and 
        current_pose_stamped_copy = copy.deepcopy(current_pose_stamped)
        points = np.asarray(downsampled_cloud)
        header = Header()
        header.stamp = current_pose_stamped_copy.header.stamp
        header.frame_id = self.global_frame_id  # or another appropriate frame
        cloud_msg = point_cloud2.create_cloud_xyz32(header, points)
        submap = SubMapT()
        submap.header.frame_id = self.global_frame_id
        submap.header.stamp = current_pose_stamped_copy.header.stamp
        self.latest_distance += self.sub_map_displacement
        submap.distance = self.latest_distance
        submap.pose = current_pose_stamped_copy.pose
        submap.cloud = cloud_msg
        submap.cloud.header.frame_id = self.global_frame_id

        # Add to Maparray msg
        self.map_array_msg.header.stamp = current_pose_stamped_copy.header.stamp
        self.map_array_msg.header.frame_id = self.global_frame_id
        self.map_array_msg.submaps.append(submap)
        
        # # Conditional map publishing
        # map_time = self.get_clock().now()
        # dt = (map_time - self.last_map_time_).nanoseconds * 1e-9
        # if dt > self.map_publish_period_:
        #     self.publish_map(self.map_array_msg, self.global_frame_id)
        #     self.last_map_time_ = map_time
        # pass

        # Publish the updated Maparray message
        self.map_array_pub.publish(self.map_array_msg)
        self.get_logger().info("Map updated successfully.")
        self.mapping_flag_ = False
    
    def calculateMaptoOdomTransform(self, base_to_map_msg: TransformStamped, stamp):

        odom_to_map_msg = TransformStamped()
        try:
            # Convert base_to_map_msg to a transformation matrix
            t = base_to_map_msg.transform.translation
            q = base_to_map_msg.transform.rotation
            base_to_map_mat = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
            base_to_map_mat[0, 3] = t.x
            base_to_map_mat[1, 3] = t.y
            base_to_map_mat[2, 3] = t.z

            # Lookup transform from odom to base_link
            tf_stamped = self.tf_buffer.lookup_transform(
                self.odom_frame_id,
                self.robot_frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Create the transformation matrix
            t_odom = tf_stamped.transform.translation
            q_odom = tf_stamped.transform.rotation
            odom_to_base_mat = tf_transformations.quaternion_matrix([q_odom.x, q_odom.y, q_odom.z, q_odom.w])
            odom_to_base_mat[0, 3] = t_odom.x
            odom_to_base_mat[1, 3] = t_odom.y
            odom_to_base_mat[2, 3] = t_odom.z

            # map->odom = map->base_link * base_link->odom
            map_to_odom_mat = base_to_map_mat @ np.linalg.inv(odom_to_base_mat)

            # Extract translation and quaternion
            translation = tf_transformations.translation_from_matrix(map_to_odom_mat)
            quat = tf_transformations.quaternion_from_matrix(map_to_odom_mat)

            # Populate the TransformStamped message
            odom_to_map_msg.header.stamp = stamp
            odom_to_map_msg.header.frame_id = self.global_frame_id
            odom_to_map_msg.child_frame_id = self.odom_frame_id
            odom_to_map_msg.transform.translation.x = translation[0]
            odom_to_map_msg.transform.translation.y = translation[1]
            odom_to_map_msg.transform.translation.z = translation[2]
            odom_to_map_msg.transform.rotation.x = quat[0]
            odom_to_map_msg.transform.rotation.y = quat[1]
            odom_to_map_msg.transform.rotation.z = quat[2]
            odom_to_map_msg.transform.rotation.w = quat[3]

        except Exception as e:
            self.get_logger().error(f"Transform from {self.robot_frame_id} to {self.odom_frame_id} failed: {str(e)}")

        return odom_to_map_msg
    
    def save_pose_to_tum(self, timestamp, pose, filename):
        """
        Save a pose to TUM format file.
        TUM format: timestamp tx ty tz qx qy qz qw
        
        Args:
            timestamp: ROS timestamp
            pose: numpy array [x, y, z, roll, pitch, yaw]
            filename: path to TUM file
        """
        # Convert timestamp to seconds
        time_sec = timestamp.sec + timestamp.nanosec * 1e-9
        
        # Extract position
        tx, ty, tz = pose[0], pose[1], 0.0
        
        # Convert roll, pitch, yaw to quaternion
        roll, pitch, yaw =  0.0,  0.0, pose[2]
        qx, qy, qz, qw = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
        
        # Write to file in TUM format
        with open(filename, 'a') as f:
            f.write(f"{time_sec:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
    
    def save_map_as_pcd(self, filename=None):
        """
        Save the current map (self.Map) as a PCD file.
        
        Args:
            filename: Optional filename. If None, uses default naming based on map_log_file
        """
        if self.Map is None or len(self.Map) == 0:
            self.get_logger().warn("No map data available to save as PCD")
            return False
            
        try:
            # Generate filename if not provided
            if filename is None:
                filename = f"{self.map_log_file}_pointmap.pcd"
            
            # Ensure filename has .pcd extension
            if not filename.endswith('.pcd'):
                filename += '.pcd'
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            
            # Convert 2D points to 3D by adding z=0
            if self.Map.shape[1] == 2:
                # Add z-coordinate as 0 for 2D points
                points_3d = np.column_stack([self.Map, np.zeros(self.Map.shape[0])])
            else:
                # Use points as-is if already 3D
                points_3d = self.Map
            
            # Set points
            pcd.points = o3d.utility.Vector3dVector(points_3d.astype(np.float64))
            
            # Save the PCD file
            success = o3d.io.write_point_cloud(filename, pcd)
            
            if success:
                self.get_logger().info(f"Map saved as PCD file: {filename}")
                self.get_logger().info(f"Number of points saved: {len(self.Map)}")
                return True
            else:
                self.get_logger().error(f"Failed to save PCD file: {filename}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Error saving map as PCD: {str(e)}")
            return False

    # Backend stuff
    def map_save_callback(self, request, response):
        self.get_logger().info("Received a request to save the map")
        
        self.MapManager.save_point_map(self.Map, self.map_log_file)
        self.og_map.save_map(self.map_log_file)
        
        # Save map as PCD file
        self.save_map_as_pcd(f"{self.map_log_file}_pointmap.pcd")
        
        if self.loop_closure:
            self.get_logger().info("Loop closure was detected. Saving optimised map.")
            # self.make_optimised_map()
            self.make_optimised_map_from_scm2()
            self.og_optimised_map.save_map(f'{self.map_log_file}_optimised')
            # Also save optimised map as PCD if available
            if hasattr(self, 'optimised_map') and self.optimised_map is not None:
                self.save_map_as_pcd(f"{self.map_log_file}_optimised_pointmap.pcd")
        else:
            self.get_logger().info("No loop closure was detected. No optimised map saved.")
            
    def convert_optimized_graph_to_path(self, cur_node_idx, graph_optimized, frame_id="map"):
        """ Converts optimized pose graph to a ROS2 Path message for visualization """
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = frame_id
        
        # Extract all optimized poses from the graph
        for node_idx in range(cur_node_idx + 1):  # Include current node
            try:
                # Get optimized pose from the graph
                pose_opt = graph_optimized.atPose2(gtsam.symbol('x', node_idx))
                x = pose_opt.x()
                y = pose_opt.y()
                theta = pose_opt.theta()
                
                # Create PoseStamped message
                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = self.get_clock().now().to_msg()
                pose_stamped.header.frame_id = frame_id
                pose_stamped.pose.position.x = float(x)
                pose_stamped.pose.position.y = float(y)
                pose_stamped.pose.position.z = 0.0
                
                # Convert yaw (theta) to quaternion
                from tf_transformations import quaternion_from_euler
                q = quaternion_from_euler(0, 0, float(theta))
                pose_stamped.pose.orientation.x = q[0]
                pose_stamped.pose.orientation.y = q[1]
                pose_stamped.pose.orientation.z = q[2]
                pose_stamped.pose.orientation.w = q[3]
                
                path_msg.poses.append(pose_stamped)
                
                pose = np.array([x, y, theta])
                self.save_pose_to_tum(self.timestamp_list[node_idx], pose, self.optimised_tum_log_file)

            except Exception as e:
                self.get_logger().warn(f"Failed to extract pose at node {node_idx}: {e}")
                continue
        
        return path_msg
    
    def publish_optimized_trajectory(self, cur_node_idx, graph_optimized):
        """ Publishes the optimized trajectory as a ROS2 Path message """
        try:
            optimized_path = self.convert_optimized_graph_to_path(cur_node_idx, graph_optimized)
            self.optimized_pub.publish(optimized_path)
            self.get_logger().info(f"Published optimized trajectory with {len(optimized_path.poses)} poses")
        except Exception as e:
            self.get_logger().error(f"Failed to publish optimized trajectory: {e}")
    
    def make_optimised_map(self):
        # Make optimised point map
        for i in range(len(self.key_frame_maps)):
            if i == 0:
                self.optimised_map = self.key_frame_maps[i]
                self.MapManager.save_point_map(self.optimised_map, f'{self.keyframe_log_file }_{self.keyframe_list[i]}')
                self.og_optimised_map.update_from_local_points(np.eye(3), self.key_frame_maps[i])
                self.og_optimised_map.save_map_only(f'{self.keyframe_log_file}_og_{self.keyframe_list[i]}')
            else:
                local_map_pcd = self.key_frame_maps[i]
                optimised_transform = self.PGM.get_optimized_pose_transform_at_node(self.keyframe_list[i])
                self.optimised_map = self.MapManager.update_global_point_map(self.optimised_map, local_map_pcd, optimised_transform)
                self.MapManager.save_point_map(self.optimised_map, f'{self.keyframe_log_file }_{self.keyframe_list[i]}')

                self.og_optimised_map.update_from_local_points(optimised_transform, self.key_frame_maps[i])
                self.og_optimised_map.save_map_only(f'{self.keyframe_log_file}_og_{self.keyframe_list[i]}')
                # Keyframe maps are too little for updating occupancy grid
                
    def make_optimised_map_from_scm2(self):
        total_frames = self.PGM.graph_optimized.size()
        print(f"Total frames for optimised map: {total_frames}")
        self.optimised_map = None
        for node_idx in range(total_frames-1):
   
            ptcloud = self.SCM.getPtcloud(node_idx)
            optimized_pose = self.PGM.get_optimized_pose_transform_at_node(node_idx)
            if node_idx == 0:
                self.optimised_map = self.transform_pointcloud_2d(ptcloud, optimized_pose)
            else:
                self.optimised_map = self.MapManager.update_global_point_map(self.optimised_map, ptcloud, optimized_pose)

            self.og_optimised_map.update_from_optimized_pose_and_pointcloud(optimized_pose, ptcloud)

            if node_idx % 30 == 0:  # Save every 30th point map
                temp_filename = f'{self.keyframe_log_file}_scm_pgm_{node_idx}'
                self.MapManager.save_point_map(self.optimised_map, temp_filename)
                self.og_optimised_map.save_map_only(temp_filename)
                # self.publish_optimized_map()
            
    def transform_pointcloud_2d(self, ptcloud, transformation):
        """
        Transform a 2D point cloud using a 2D transformation matrix.
        
        Args:
            ptcloud (np.ndarray): Point cloud as Nx2 array
            transformation (np.ndarray): 3x3 transformation matrix
            
        Returns:
            np.ndarray: Transformed point cloud
        """
        if ptcloud is None or len(ptcloud) == 0:
            return np.array([]).reshape(0, 2)
            
        # Ensure point cloud is 2D
        if ptcloud.shape[1] != 2:
            self.get_logger().warn(f"Point cloud has {ptcloud.shape[1]} dimensions, expected 2D")
            ptcloud = ptcloud[:, :2]  # Take only x, y coordinates
        
        # Convert to homogeneous coordinates
        homogeneous_pts = np.hstack([ptcloud, np.ones((ptcloud.shape[0], 1))])
        
        # Transform points
        transformed_pts = (transformation @ homogeneous_pts.T).T
        
        # Return only x, y coordinates
        return transformed_pts[:, :2]

    def publish_optimized_map(self):
        """
        Publish the optimized occupancy grid map for visualization in RViz.
        """
        try:
            prob_map = self.og_optimised_map.get_probability_map().T
            
            # Convert probability map to integer occupancy values (-1 unknown, 0 free, 100 occupied)
            occupancy_values = np.clip((prob_map * 100).astype(int), 0, 100).flatten()
            
            msg = OccupancyGrid()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'
            
            msg.info.resolution = self.og_optimised_map.resolution
            msg.info.width = int(self.og_optimised_map.extents[0])
            msg.info.height = int(self.og_optimised_map.extents[1])
            origin_pose = Pose()
            origin_pose.position.x = -self.og_optimised_map.origin[0] * self.og_optimised_map.resolution
            origin_pose.position.y = -self.og_optimised_map.origin[1] * self.og_optimised_map.resolution
            msg.info.origin = origin_pose
            
            msg.data = occupancy_values.tolist()
            
            self.optimized_map_pub.publish(msg)
            self.get_logger().info("Published optimized map")
            
        except Exception as e:
            self.get_logger().error(f"Error publishing optimized map: {e}")
    
def main(args=None):
    
    rclpy.init(args=args)
    node = Feature_scan_SLAM()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown detected: Ctrl+C pressed.")
    finally:
        node.MapManager.save_point_map(node.Map, node.map_log_file)
        node.og_map.save_map(node.map_log_file)
        if node.loop_closure:
            node.get_logger().info("Loop closure was detected. Saving optimised map.")
            node.make_optimised_map_from_scm2()
            node.og_map.save_map(f'{node.map_log_file}_unoptimised')
            node.og_optimised_map.save_map(f'{node.map_log_file}_optimised')
        else:
            node.get_logger().info("No loop closure was detected. No optimised map saved.")

        node.get_logger().info("Maps saved before exit.")
        node.get_logger().info(f"Map saved to {node.map_log_file} ")

        # Ensure shutdown is only called once
        if rclpy.ok():  
            node.destroy_node()
            
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()
