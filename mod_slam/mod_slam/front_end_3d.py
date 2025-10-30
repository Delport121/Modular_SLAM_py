#!/usr/bin/env python3

import os
import sys
import copy
import numpy as np
import open3d as o3d
import small_gicp
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Quaternion
import geometry_msgs.msg
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from ament_index_python.packages import get_package_share_directory
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured

import tf_transformations
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from sensor_msgs.msg import PointCloud2
import tf2_ros
import tf2_sensor_msgs 

from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import PoseStamped, Pose
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from nav_msgs.msg import Path
from sensor_msgs_py import point_cloud2
from builtin_interfaces.msg import Time
from concurrent.futures import ThreadPoolExecutor

from nav_msgs.msg import Odometry  # Add at the top if not already
from slam_interfaces.msg import SubMapT, MapArrayT

from pathlib import Path as dir_path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)  # Add the script directory to the system path for importing custom modules

from mod_slam.utils.ScanContextManager import ScanContextManager
from mod_slam.utils.PoseGraphManager import PoseGraphManager, PoseGraphResultSaver
from mod_slam.utils.UtilsMisc  import ScanContextResultSaver
from mod_slam.utils.UtilsMisc import yawdeg2se3
import mod_slam.utils.UtilsPointcloud as Ptutils
import mod_slam.utils.ICP as ICP
from mod_slam.utils.MapManager import MapManager
from mod_slam.utils.lidar_undistortion import LidarUndistortion

# from utils.ScanContextManager import ScanContextManager
# from utils.PoseGraphManager import PoseGraphManager, PoseGraphResultSaver
# from utils.UtilsMisc  import ScanContextResultSaver
# from utils.SegmentationManager import SegmentationManager
# from utils.UtilsMisc import yawdeg2se3
# import utils.UtilsPointcloud as Ptutils
# import utils.ICP as ICP
# from utils.MapManager import MapManager

import csv
import struct
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Imu
import time

class PointCloudOdometryNode(Node):
    def __init__(self):
        super().__init__('frontend_node_3D')

        # Declare parameters
        self.declare_parameter('lidar_topic','/ouster/points') # '/ouster/points' #/points_raw  #/velodyne_points  # #cloud_deskewed
        self.declare_parameter('imu_topic', '/imu/data')  # #/gpsimu_driver/imu_data #/imu/data #/imu_plugin/out
        self.declare_parameter('odom_topic', '/odometry/local') #/odometry/local #/odometry/filtered
        
        self.declare_parameter('save_dir', '/home/ruan/dev_ws/src/lidar_slam_3d/results') #/home/voyager/ros-workspace/high-level/src/ruan_dev/lidar_slam_3d/results/Data
        self.declare_parameter('test_name', 'real_lab_sc')
        self.declare_parameter('Log_data', True) 
        
        self.declare_parameter('global_frame_id', 'map')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('robot_frame_id', 'base_link')
        
        self.declare_parameter('publish_tf', False)
        self.declare_parameter('use_odom', True)
        self.declare_parameter("publish_point_map", False)
        self.declare_parameter('publish_map_array_msg', False) # Required for Euclidean
        
        self.declare_parameter('voxel_downsample_res', 0.2)
        self.declare_parameter('map_local_voxel_res', 0.2)
        self.declare_parameter('map_local_horizon', 800) #Lab 1300 # Any voxel not accessed after this number of frames can be deleted
        self.declare_parameter('map_local_clear_cycle', 100) # lab 500# Any voxel not accessed after this number of frames can be deleted
        self.declare_parameter('map_local_bound_radius', 20.0)
        self.declare_parameter('trans_for_mapupdate_', 1.0)  # Distance threshold for map update # 1
        self.declare_parameter('map_pub_downsample_res', 0.1)
        
        # Outlier filtering parameters
        self.declare_parameter('enable_outlier_filtering', False)
        self.declare_parameter('statistical_nb_neighbors', 20)  # Number of neighbors for statistical outlier removal
        self.declare_parameter('statistical_std_ratio', 2.0)   # Standard deviation ratio threshold
        self.declare_parameter('radius_outlier_nb_points', 16) # Minimum number of neighbors in radius
        self.declare_parameter('radius_outlier_radius', 0.05)  # Search radius for outlier removal
        self.declare_parameter('range_filter_min', 0.3)        # Minimum range filter (meters)
        self.declare_parameter('range_filter_max', 50.0)       # Maximum range filter (meters)
        
        # Other parameters
        self.declare_parameter('lio_odom_topic', '/preintegrated_odom')  # LIO-SAM odometry topic
        self.declare_parameter('model_state_topic', '/model_states') 
        self.declare_parameter('use_lio_odometry', False)  # Whether to use LIO odometry data
        self.declare_parameter("Get_gt", False)  # Whether to get ground truth data
        self.declare_parameter("use_imu", False)
        
        # Get parameters
        self.lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value
        self.imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.lio_odom_topic = self.get_parameter('lio_odom_topic').get_parameter_value().string_value
        self.model_state_topic = self.get_parameter('model_state_topic').get_parameter_value().string_value
        self.save_dir = self.get_parameter('save_dir').get_parameter_value().string_value
        self.save_dir_trajectory = f"{self.save_dir}/Data"
        self.test_name = self.get_parameter('test_name').get_parameter_value().string_value
        self.global_frame_id = self.get_parameter('global_frame_id').get_parameter_value().string_value
        self.odom_frame_id = self.get_parameter('odom_frame_id').get_parameter_value().string_value
        self.robot_frame_id = self.get_parameter('robot_frame_id').get_parameter_value().string_value
        self.publish_tf = self.get_parameter('publish_tf').get_parameter_value().bool_value # Whether to publish the TF from map to base_link
        self.use_imu = self.get_parameter('use_imu').get_parameter_value().bool_value # Whether to use IMU data for undistortion
        self.use_odom = self.get_parameter('use_odom').get_parameter_value().bool_value # Whether to use odometry data
        self.use_lio_odometry = self.get_parameter('use_lio_odometry').get_parameter_value().bool_value # Whether to use LIO odometry data
        self.voxel_downsample_res = self.get_parameter('voxel_downsample_res').get_parameter_value().double_value
        self.map_local_voxel_res = self.get_parameter('map_local_voxel_res').get_parameter_value().double_value
        self.map_local_horizon = self.get_parameter('map_local_horizon').get_parameter_value().integer_value
        self.map_local_clear_cycle = self.get_parameter('map_local_clear_cycle').get_parameter_value().integer_value
        self.map_local_bound_radius = self.get_parameter('map_local_bound_radius').get_parameter_value().double_value
        self.trans_for_mapupdate_ = self.get_parameter('trans_for_mapupdate_').get_parameter_value().double_value
        self.map_global_downsample_res = self.get_parameter('map_pub_downsample_res').get_parameter_value().double_value
        self.gt_on = self.get_parameter('Get_gt').get_parameter_value().bool_value
        self.publish_unmodified_map = self.get_parameter("publish_map_array_msg").get_parameter_value().bool_value
        self.publish_map_array_msg = self.get_parameter('publish_map_array_msg').get_parameter_value().bool_value
        self.log_data = self.get_parameter('Log_data').get_parameter_value().bool_value
        
        # Outlier filtering parameters
        self.enable_outlier_filtering = self.get_parameter('enable_outlier_filtering').get_parameter_value().bool_value
        self.statistical_nb_neighbors = self.get_parameter('statistical_nb_neighbors').get_parameter_value().integer_value
        self.statistical_std_ratio = self.get_parameter('statistical_std_ratio').get_parameter_value().double_value
        self.radius_outlier_nb_points = self.get_parameter('radius_outlier_nb_points').get_parameter_value().integer_value
        self.radius_outlier_radius = self.get_parameter('radius_outlier_radius').get_parameter_value().double_value
        self.range_filter_min = self.get_parameter('range_filter_min').get_parameter_value().double_value
        self.range_filter_max = self.get_parameter('range_filter_max').get_parameter_value().double_value
        
        # Initialize managers
        # self.segmentation_manager = SegmentationManager()
        self.MapManager = MapManager(grid_downsample_res=self.map_local_voxel_res, bound_radius=self.map_local_bound_radius)
        self.lidar_undistortion = LidarUndistortion()

        # Define a QoS profile
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        
        # Tf listeners
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # TransformBroadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribers
        self.lidar_sub = self.create_subscription(PointCloud2, self.lidar_topic, self.pointcloud_callback, qos_profile) 
        self.gt_sub = self.create_subscription(ModelStates, self.model_state_topic, self.model_CB, 10)
        self.imu_sub = self.create_subscription(Imu, self.imu_topic, self.imu_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.unmodified_map_pub = self.create_publisher(PointCloud2, '/unmodified_map', 10)
        # self.lio_odom = self.create_subscription(Odometry, self.lio_odom_topic, self.odom_callback, 10)
        
        # Publishers
        self.pose_pub_ = self.create_publisher(PoseStamped, '/current_pose', 10)
        self.map_publisher = self.create_publisher(PointCloud2, "/point_map", 10)
        self.path_est_pub_ = self.create_publisher(Path, '/path', 10)
        self.path_gt_pub_ = self.create_publisher(Path, '/path_gt', 10)
        self.submap_pub = self.create_publisher(SubMapT, '/submap', 10)
        self.map_array_pub = self.create_publisher(MapArrayT, '/map_array', 10)
        self.odom_pub_ = self.create_publisher(Odometry, '/odometry', 10)  # For lio-sam
        
        # Variables
        ## Logistic
        self.frame_idx = 0
        self.timestamp = None
        self.Map = o3d.geometry.PointCloud()
        
        ## Ros messages
        self.current_pose_stamped_ = PoseStamped()
        self.current_pose_gt_stamped = PoseStamped()
        self.gt_pose_full = PoseStamped()
        self.current_pose_stamped_.pose.position.x = 0.0
        self.current_pose_stamped_.pose.position.y = 0.0
        self.current_pose_stamped_.pose.position.z = 0.0
        self.current_pose_stamped_.pose.orientation.x = 0.0
        self.current_pose_stamped_.pose.orientation.y = 0.0
        self.current_pose_stamped_.pose.orientation.z = 0.0
        self.current_pose_stamped_.pose.orientation.w = 1.0
        self.path_est = Path()
        self.path_gt = Path()
        self.subMap = SubMapT()
        self.map_array_msg = MapArrayT()
        self.previous_position_ = np.zeros(3)
        self.sub_map_previous_position = np.zeros(3)
        
        ## Open3d scanmatching
        self.prev_scan_pts = None
        self.initial_odom_estimate = np.eye(4)
        self.intermediate_estimate = np.eye(4)
        self.initial_transform = np.eye(4)
        self.estimate = np.eye(4)
        self.icp_initial = np.eye(4) # Unused currently
    
        ## GICP scanmatching
        self.target = None
        self.T_last_current = np.identity(4)
        self.T_world_lidar = np.identity(4)
        self.target = small_gicp.GaussianVoxelMap(self.map_local_voxel_res)
        self.target.set_lru(horizon=self.map_local_horizon, clear_cycle=self.map_local_clear_cycle)

        ## Other Variables
        self.gt_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # x, y, z, roll, pitch, yaw
        self.odom_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # x, y, z, roll, pitch, yaw
        self.estimated_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # x, y, z, roll, pitch, yaw
        self.displacement = 0.0
        self.sub_map_displacement = 0.0
        self.latest_distance = 0.0
        self.previous_odom_mat = np.identity(4, dtype=np.float32)
        self.sim_trans = np.identity(4, dtype=np.float32)
        
        ## Tightly coupled  variables
        self.use_imu_thight = True
        self.odom_que_length_ = 200  # equivalent to static const int odom_que_length_ {200};
        self.odom_que_ = [Odometry() for _ in range(self.odom_que_length_)]  # like std::array<nav_msgs::msg::Odometry, odom_que_length_>
        self.odom_ptr_front_ = 0  # int odom_ptr_front_ {0};
        self.odom_ptr_last_ = -1  # int odom_ptr_last_ {-1};
        
        ## Threading variables
        self.loop_executor = ThreadPoolExecutor(max_workers=1)
        self.mapping_flag_= False

        # Data logging
        if not os.path.exists(self.save_dir_trajectory):
            os.makedirs(self.save_dir_trajectory)
        self.trajectory_log_file = os.path.join(self.save_dir_trajectory, f'Sim_trajectories_{self.test_name}.csv')
        ## Trajectories
        with open(self.trajectory_log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Time",
                "GT_X", "GT_Y", "GT_Z", "GT_Roll", "GT_Pitch", "GT_Yaw",
                "Odom_X", "Odom_Y", "Odom_Z", "Odom_Roll", "Odom_Pitch", "Odom_Yaw",
                "Est_X", "Est_Y", "Est_Z", "Est_Roll", "Est_Pitch", "Est_Yaw"
            ])
        
        ## TUM format trajectory files for benchmarking
        self.gt_tum_log_file = os.path.join(self.save_dir_trajectory, f'Sim_groundtruth_{self.test_name}.txt')
        self.est_tum_log_file = os.path.join(self.save_dir_trajectory, f'Sim_estimate_{self.test_name}.txt')
        self.odom_tum_log_file = os.path.join(self.save_dir_trajectory, f'Sim_odometry_{self.test_name}.txt')
        
        # Initialize TUM format files (no headers needed for TUM format)
        open(self.gt_tum_log_file, 'w').close()
        open(self.est_tum_log_file, 'w').close()
        open(self.odom_tum_log_file, 'w').close()
        
        # Log file paths for user information
        self.get_logger().info(f"CSV trajectory log: {self.trajectory_log_file}")
        self.get_logger().info(f"TUM ground truth log: {self.gt_tum_log_file}")
        self.get_logger().info(f"TUM estimate log: {self.est_tum_log_file}")
        self.get_logger().info(f"TUM odometry log: {self.odom_tum_log_file}")
        
        ## Scan match data
        self.scanscan_converged = False
        self.scanscan_iterations = 0
        self.scanscan_RMSE = 0.0
        self.scanscan_error = 0.0
        self.scanscan_num_inliers = 0
        self.scanmap_converged = False
        self.scanmap_iterations = 0
        self.scanmap_RMSE = 0.0
        self.scanmap_error = 0.0
        self.scanmap_num_inliers = 0
        
        self.scanmatch_log_file = os.path.join(self.save_dir_trajectory, f'Sim_scan_match_data_{self.test_name}.csv')
        with open(self.scanmatch_log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Time",
                "ScanScan_converged", "ScanScan_iterations", "ScanScan_RMSE", "ScanScan_error", "ScanScan_num_inliers",
                "ScanMap_converged", "ScanMap_iterations", "ScanMap_RMSE", "ScanMap_error", "ScanMap_num_inliers"
            ])
            
        ## Pipeline times
        self.time_log_file = os.path.join(self.save_dir_trajectory, f'Time_data_{self.test_name}.csv')
        self.pipeline_time = 0.0
        self.frame_transform_time = 0.0
        self.imu_undistortion_time = 0.0
        self.downsampling_time = 0.0
        self.scan_matching_time = 0.0
        self.map_update_time = 0.0
        with open(self.time_log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Total_pipeline_time_ms", "frame_transform_time_ms", "imu_undistortion_time_ms", "downsampling_time_ms", "scan_matching_time_ms", "map_update_time_ms"
            ])
        
        self.get_logger().info(f"Data log mode: {self.log_data}")
        self.get_logger().info('Lidar SLAM 3D node initialized')

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
            self.gt_pose = np.array([x, y, z, roll, pitch, yaw])
            
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
    
    def odom_callback(self, msg: Odometry):
        
        self.odom_pose[0] = msg.pose.pose.position.x
        self.odom_pose[1] = msg.pose.pose.position.y
        self.odom_pose[2] = msg.pose.pose.position.z
        self.odom_pose[3], self.odom_pose[4], self.odom_pose[5] = tf_transformations.euler_from_quaternion(
            [msg.pose.pose.orientation.x,
             msg.pose.pose.orientation.y,
             msg.pose.pose.orientation.z,
             msg.pose.pose.orientation.w]
        )   
        
        self.receive_odom(msg)
    
    def imu_callback(self, msg: Imu):
        
        self.receiveIMU(msg)
        
    def pointcloud_callback(self, msg: PointCloud2):
        
        self.timestamp = msg.header.stamp
        gt_pose_copy = copy.deepcopy(self.gt_pose) # Models update at a higher rate than the lidar, so we need to copy the gt pose here
        self.current_pose_gt_stamped = copy.deepcopy(self.gt_pose_full)
        pipeline_start_time = time.time()
        
        # Print field names of the incoming point cloud message
        # field_names = [field.name for field in msg.fields]
        # self.get_logger().info(f"Point cloud field names: {field_names}")
        
        # Had to rebuilt cloud because of point field issues (Incoming msg pointcloud is different from python pointcloud2)
        # starttime = time.time()  
        msg_py = self.rebuild_xyz_msg_fast(msg)
        # rebuild_time = (time.time() - starttime) * 1000 
        # self.get_logger().info(f"Rebuild XYZI time: {rebuild_time:.2f} ms")
        
        # Transform the point cloud to the robot frame 
        transform = None
        try:
            time_point = rclpy.time.Time.from_msg(msg.header.stamp)
            transform = self.tf_buffer.lookup_transform(
                self.robot_frame_id,  # target frame
                msg.header.frame_id,  # source frame
                time_point
            )
            
            # starttime = time.time()  
            transformed_msg = self.transform_pointcloud_msg_optimized(msg_py, transform)
            # frame_transform_time = (time.time() - starttime) * 1000 
            # self.frame_transform_time = frame_transform_time  # Store for logging
            # self.get_logger().info(f"Optimized frame transform time: {frame_transform_time:.2f} ms")
            
            # starttime = time.time()
            transformed_pointcloud_open3d = self.ros2_to_open3d(transformed_msg)
            # rostoopen3d_time = (time.time() - starttime) * 1000
            # self.get_logger().info(f"ROS2 to Open3D conversion time: {rostoopen3d_time:.2f} ms")
            
            # starttime = time.time()
            self.pipeline(transformed_pointcloud_open3d, msg.header.stamp)
            # self.pipeline_time = (time.time() - starttime) * 1000
            # self.get_logger().info(f"Pipeline execution time: {self.pipeline_time:.2f} ms")

           
        except Exception as e:
            self.get_logger().warn(f"Transform failed: {e}")
            return
        

        
        # starttime = time.time()
        if self.log_data:
            
            
            # Log the pipeline times
            with open(self.time_log_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [self.pipeline_time, 
                    self.frame_transform_time, 
                    self.imu_undistortion_time, 
                    self.downsampling_time, 
                    self.scan_matching_time, 
                    self.map_update_time
                ])
                
            # Log the scan matching results
            with open(self.scanmatch_log_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    self.timestamp.sec + self.timestamp.nanosec * 1e-9,
                    self.scanscan_converged,
                    self.scanscan_iterations,
                    self.scanscan_RMSE,
                    self.scanscan_error,
                    self.scanscan_num_inliers,
                    self.scanmap_converged,
                    self.scanmap_iterations,    
                    self.scanmap_RMSE,
                    self.scanmap_error,
                    self.scanmap_num_inliers
                ])
            
            # Save in CSV format (existing functionality)
            with open(self.trajectory_log_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([self.timestamp.sec + self.timestamp.nanosec * 1e-9, 
                                self.gt_pose[0], self.gt_pose[1], self.gt_pose[2], self.gt_pose[3], self.gt_pose[4], self.gt_pose[5],
                                self.odom_pose[0], self.odom_pose[1], self.odom_pose[2], self.odom_pose[3], self.odom_pose[4], self.odom_pose[5],
                                self.estimated_pose[0], self.estimated_pose[1], self.estimated_pose[2], self.estimated_pose[3], self.estimated_pose[4], self.estimated_pose[5]])
            
            # Save in TUM format for benchmarking
            self.save_pose_to_tum(self.timestamp, gt_pose_copy, self.gt_tum_log_file)
            self.save_pose_to_tum(self.timestamp, self.estimated_pose, self.est_tum_log_file)
            self.save_pose_to_tum(self.timestamp, self.odom_pose, self.odom_tum_log_file)
        
        # log_time = (time.time() - starttime) * 1000
        # self.get_logger().info(f"Data logging time: {log_time:.2f} ms")
        
        pipeline_end_time = time.time()  # End timer
        self.pipeline_time = (pipeline_end_time - pipeline_start_time) * 1000
        self.get_logger().info(f"Full Pipeline execution time: {self.pipeline_time:.2f} ms")

    # Main functions
    def pipeline(self, current_pcd, stamp):
        
        # IMU undistortion
        if (self.use_imu):
            time_point = rclpy.time.Time.from_msg(stamp)
            scan_time = time_point.nanoseconds * 1e-9  # Convert nanoseconds to seconds
            distortion_start_time = time.time()
            self.lidar_undistortion.adjust_distortion(current_pcd, scan_time)
            distortion_end_time = time.time()
            self.imu_undistortion_time = (distortion_end_time - distortion_start_time) * 1000 
            self.get_logger().info(f"IMU undistortion time: {self.imu_undistortion_time:.2f} ms")

        # Outlier filtering (early in pipeline, after undistortion)
        if self.enable_outlier_filtering:
            filtering_start_time = time.time()
            current_pcd = self.filter_outliers(current_pcd)
            filtering_end_time = time.time()
            filtering_time = (filtering_end_time - filtering_start_time) * 1000
            self.get_logger().info(f"Outlier filtering time: {filtering_time:.2f} ms")

        # # Downsampling
        # downsampling_start = time.time()
        # curr_scan_down_pts = current_pcd.voxel_down_sample(self.voxel_downsample_res)
        # downsampling_end = time.time()
        # self.downsampling_time = (downsampling_end - downsampling_start) * 1000
        
        # Get current pose
        self.sim_trans = self.get_transformation_matrix_from_pose(self.current_pose_stamped_.pose)
        
        # Integrate current odom data for initial guess
        if (self.use_odom):
            try:
                odom_trans = self.tf_buffer.lookup_transform(
                    self.odom_frame_id,
                    self.robot_frame_id,
                    rclpy.time.Time() #stamp - When using stamp the lidar timestamp seems to be ahead of tf tree
            )
            except Exception as e:
                self.get_logger().error(str(e))
                return
            
            # Convert TransformStamped to 4x4 matrix
            t = odom_trans.transform.translation
            q = odom_trans.transform.rotation
            odom_mat = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
            odom_mat[0, 3] = t.x
            odom_mat[1, 3] = t.y
            odom_mat[2, 3] = t.z
            odom_mat = odom_mat.astype(np.float32)

            # Update sim_trans using relative odometry
            if not np.allclose(self.previous_odom_mat, np.identity(4, dtype=np.float32)):
                rel_transform = np.linalg.inv(self.previous_odom_mat) @ odom_mat
                self.sim_trans = self.sim_trans @ rel_transform

            self.previous_odom_mat = odom_mat
            
        # IMU pre_integration
        if self.use_imu_thight and self.odom_ptr_last_ != -1:
            odom_ptr = self.odom_ptr_front_
            while odom_ptr != self.odom_ptr_last_:
                odom_stamp = rclpy.time.Time.from_msg(self.odom_que_[odom_ptr].header.stamp)
                if odom_stamp.nanoseconds > rclpy.time.Time.from_msg(stamp).nanoseconds:
                    break
                odom_ptr = (odom_ptr + 1) % self.odom_que_length_

            odom_position = self.get_transformation_matrix_from_pose(self.odom_que_[odom_ptr].pose.pose)
            sim_trans = odom_position
            self.odom_ptr_front_ = odom_ptr

                #---------------------------------------
        
        #---------------------------------------------------
        # # Open3D based SLAM pipeline
        # if self.frame_idx == 0:
        #     self.prev_scan_pts = curr_scan_down_pts
        #     self.Map = self.MapManager.update_point_map(self.Map, curr_scan_down_pts, np.eye(4))
        #     self.Map = copy.deepcopy(curr_scan_down_pts)
        #     self.publish_map()
        #     self.frame_idx += 1
        #     return
        
        # # Scan-to-scan matching
        # initial_transform, self.scanscan_error, self.scanscan_RMSE, self.scanscan_num_inliers = self.match_scans(curr_scan_down_pts, self.prev_scan_pts)
        # self.initial_odom_estimate = self.initial_odom_estimate @ initial_transform
        # self.intermediate_estimate = self.estimate @ initial_transform
        
        # self.odom_pose[0] = self.initial_odom_estimate[0, 3]
        # self.odom_pose[1] = self.initial_odom_estimate[1, 3]
        # self.odom_pose[2] = self.initial_odom_estimate[2, 3]
        # self.odom_pose[3], self.odom_pose[4], self.odom_pose[5] = tf_transformations.euler_from_matrix(self.initial_odom_estimate)
        
        # # Scan-to-map matching
        # transformation, self.scanmap_error, self.scanmap_RMSE, self.scanmap_num_inliers = self.match_scans(curr_scan_down_pts, self.Map,  self.intermediate_estimate)
        # if self.scanmap_error < 0.01:
        #     self.get_logger().warn(f"Low fitness score: {self.scanmap_error:.3f} at frame {self.frame_idx}. Skipping frame.")
        # else:
        #     self.estimate = transformation 
        #     self.Map = self.MapManager.update_point_map(self.Map, current_pcd, self.estimate)
        
        #---------------------------------------------------
        # # Open3D scan-to-model based SLAM pipeline
        # if self.frame_idx == 0:
        #     self.prev_scan_pts = curr_scan_down_pts
        #     self.Map = self.MapManager.update_point_map(self.Map, curr_scan_down_pts, np.eye(4))
        #     self.Map = copy.deepcopy(curr_scan_down_pts)
        #     self.publish_map()
        #     self.frame_idx += 1
        #     return
        
        # # Scan-to-map matching
        # with_odom = self.sim_trans
        # transformation, self.scanmap_error, self.scanmap_RMSE, self.scanmap_num_inliers = self.match_scans(curr_scan_down_pts, self.Map,  with_odom)
        # if self.scanmap_error < 0.5:
        #     self.get_logger().warn(f"Low fitness score: {self.scanmap_error:.3f} at frame {self.frame_idx}. Skipping frame.")
        # else:
        #     self.estimate = transformation 
        #     self.Map = self.MapManager.update_point_map(self.Map, current_pcd, self.estimate)
        
        #---------------------------------------
        # # # # Small GICP based SLAM pipeline
        # num_threads = os.cpu_count() or 1
        # downsampled, tree = small_gicp.preprocess_points(np.asarray(current_pcd.points), 0.25, num_threads=num_threads)
        
        # # First frame initialization
        # if self.frame_idx == 0:
        #     self.target = (downsampled, tree)
        #     # self.target.insert(downsampled)
        #     self.Map = self.MapManager.update_point_map(self.Map, curr_scan_down_pts, np.eye(4))
        #     self.Map = copy.deepcopy(curr_scan_down_pts)

        #     self.update_map(curr_scan_down_pts, np.eye(4), self.current_pose_stamped_)
        #     self.frame_idx += 1
        #     returncurr_scan_down_pts
        
        # downsampled_map, tree_map = small_gicp.preprocess_points(np.asarray(self.Map.points), 0.25, num_threads=num_threads)
        
        # # Scan-to-scan matching
        # result = small_gicp.align(self.target[0], downsampled, self.target[1], self.T_last_current, num_threads=num_threads)
        # self.T_last_current = result.T_target_source
        # self.scanscan_converged = result.converged
        # self.scanscan_iterations = result.iterations
        # self.scanscan_error = result.error
        # self.scanscan_num_inliers = result.num_inliers
        
        # # Scan-to-map matching
        # result = small_gicp.align(downsampled_map, downsampled, tree_map, self.T_world_lidar @ self.T_last_current, num_threads=num_threads)
        # self.scanmap_converged = result.converged
        # self.scanmap_iterations = result.iterations
        # self.scanmap_error = result.error
        # self.scanmap_num_inliers = result.num_inliers

        # # self.T_last_current = np.linalg.inv(self.T_world_lidar) @ result.T_target_source
        # self.T_world_lidar = result.T_target_source
        # # self.target.insert(downsampled, self.T_world_lidar)
        
        # transformation = self.T_world_lidar
        # self.Map = self.MapManager.update_point_map(self.Map, current_pcd, transformation)
        
        # self.target = (downsampled, tree)
        
        
        
        #---------------------------------------
        
        
        # Small GICP based SLAM pipeline scan-to-model matching
        num_threads = os.cpu_count() or 1
    
        # small_gicp prefers contiguous float32 Nx3
        # starttime = time.time() 
        points_np = np.asarray(current_pcd.points)
        downsampled, tree = small_gicp.preprocess_points(points_np, self.voxel_downsample_res, num_threads=num_threads) #downsampled np
        # downsample_time = (time.time() - starttime) * 1000 
        # self.get_logger().info(f"Small GICP downsample time: {downsample_time:.2f} ms")
        
        pts4 = downsampled.points()                 # Nx4 float64 (view into C++ memory)
        pts3 = np.array(pts4[:, :3], dtype=np.float64) 
        
        curr_scan_down_pts = o3d.geometry.PointCloud()
        curr_scan_down_pts.points = o3d.utility.Vector3dVector(pts3)
        #o3d.utility.Vector3dVector(np.asarray(downsampled).astype(np.float64))

        # First frame initialization
        if self.frame_idx == 0:
            # self.target = (downsampled, tree)
            self.target.insert(downsampled)
            # self.Map = self.MapManager.update_point_map(self.Map, curr_scan_down_pts, np.eye(4))
            # self.Map = copy.deepcopy(curr_scan_down_pts)
            self.current_pose_stamped_.header.stamp = stamp
            self.update_map_array(curr_scan_down_pts, self.current_pose_stamped_)
            self.frame_idx += 1
            return

        # Scan-to-map matching
        # scan_matching_start = time.time()
        normal = self.T_world_lidar @ self.T_last_current
        with_odom = self.sim_trans
        result = small_gicp.align(self.target, downsampled, with_odom, num_threads=num_threads)
        # scan_matching_end = time.time()
        # self.scan_matching_time = (scan_matching_end - scan_matching_start) * 1000
        # self.get_logger().info(f"Small GICP scan matching time: {self.scan_matching_time:.2f} ms")

        self.scanmap_converged = result.converged
        self.scanmap_iterations = result.iterations
        self.scanmap_error = result.error
        self.scanmap_num_inliers = result.num_inliers
        
        # self.get_logger().info(f"Frame {self.frame_idx} - Small GICP scan-to-map error: {self.scanmap_error:.4f}")
        # # Warn if low fitness score
        # if self.scanmap_error > 300:
        #     self.get_logger().warn(f"Low fitness score: {self.scanmap_error:.3f} at frame {self.frame_idx}. Skipping frame.\n")
        #     self.T_world_lidar = with_odom
        # else:
        #     self.T_last_current = np.linalg.inv(self.T_world_lidar) @ result.T_target_source
        #     self.T_world_lidar = result.T_target_source
        #     transformation = self.T_world_lidar

        self.T_world_lidar = result.T_target_source
        self.T_world_lidar = result.T_target_source
        self.target.insert(downsampled, self.T_world_lidar)
        
        
        # starttime = time.time()
        transformation = self.T_world_lidar
        map_update_start = time.time()
        # self.Map = self.MapManager.update_point_map(self.Map, current_pcd, transformation)
        map_update_end = time.time()
        self.map_update_time = (map_update_end - map_update_start) * 1000 

        #---------------------------------------
        # # Small GICP based SLAM pipeline scan-to-scan matching
        # num_threads = os.cpu_count() or 1
        # downsampled, tree = small_gicp.preprocess_points(np.asarray(current_pcd.points), self.voxel_downsample_res, num_threads=num_threads)
        
        # # First frame initialization
        # if self.frame_idx == 0:
        #     self.target = (downsampled, tree)
        #     self.Map = self.MapManager.update_point_map(self.Map, curr_scan_down_pts, np.eye(4))
        #     self.Map = copy.deepcopy(curr_scan_down_pts)
        #     self.update_map(curr_scan_down_pts, np.eye(4), self.current_pose_stamped_)
        #     self.frame_idx += 1
        #     return
        
        # result = small_gicp.align(self.target[0], downsampled, self.target[1], self.T_last_current, num_threads=num_threads)
        # self.scanscan_converged = result.converged
        # self.scanscan_iterations = result.iterations
        # self.scanscan_error = result.error
        # self.scanscan_num_inliers = result.num_inliers
        
        
        # self.T_last_current = result.T_target_source
        # self.T_world_lidar = self.T_world_lidar @ result.T_target_source
        # self.target = (downsampled, tree)
        
        # transformation = self.T_world_lidar
        # map_update_start = time.time()
        # self.Map = self.MapManager.update_point_map(self.Map, current_pcd, transformation)
        # map_update_end = time.time()

        #---------------------------------------
        # # Open3D based SLAM pipeline scan-to-scan matching
        # downsampling_start = time.time()
        # curr_scan_down_pts = current_pcd.voxel_down_sample(self.voxel_downsample_res)
        # downsampling_end = time.time()
        # self.downsampling_time = (downsampling_end - downsampling_start) * 1000
        
        # # First frame initialization
        # if self.frame_idx == 0:
        #     self.prev_scan_pts = curr_scan_down_pts
        #     self.Map = self.MapManager.update_point_map(self.Map, curr_scan_down_pts, np.eye(4))
        #     self.Map = copy.deepcopy(curr_scan_down_pts)
        #     self.update_map(curr_scan_down_pts, np.eye(4), self.current_pose_stamped_)
        #     self.frame_idx += 1
        #     return
        
        # scan_matching_start = time.time()
        # result_transformation, self.scanscan_error, self.scanscan_RMSE, self.scanscan_num_inliers = self.match_scans(curr_scan_down_pts, self.prev_scan_pts, self.T_last_current)
        # self.scanscan_converged = True if self.scanscan_error > 0.1 else False
        # self.scanscan_iterations = 30  # Default Open3D ICP iterations
        
        # self.T_last_current = result_transformation
        # self.T_world_lidar = self.T_world_lidar @ result_transformation
        # self.prev_scan_pts = curr_scan_down_pts
        
        # transformation = self.T_world_lidar
        # map_update_start = time.time()
        # self.Map = self.MapManager.update_point_map(self.Map, current_pcd, transformation)
        # map_update_end = time.time()
        # scan_matching_end = time.time()
        # self.scan_matching_time = (scan_matching_end - scan_matching_start) * 1000
        # self.map_update_time = (map_update_end - map_update_start) * 1000
        
        #---------------------------------------
        
        # self.publish_map()
        self.publish_pose_and_pointcloud(self.T_world_lidar, curr_scan_down_pts, stamp)
        self.get_logger().info(f"Processed frame {self.frame_idx}, map size: {len(self.Map.points)}")
        self.frame_idx += 1

    def receive_odom(self, odom_msg: Odometry):
        if not self.use_imu_thight:
            return
        self.odom_ptr_last_ = (self.odom_ptr_last_ + 1) % self.odom_que_length_
        self.odom_que_[self.odom_ptr_last_] = odom_msg

        # self.get_logger().info(f"Received odometry data: {odom_msg.pose.pose.position.x}, {odom_msg.pose.pose.position.y}, {odom_msg.pose.pose.position.z}")
        # If buffer is full, move front pointer forward
        if (self.odom_ptr_last_ + 1) % self.odom_que_length_ == self.odom_ptr_front_:
            self.odom_ptr_front_ = (self.odom_ptr_front_ + 1) % self.odom_que_length_
    
    def receiveIMU(self, msg: Imu):
        
        if not self.use_imu:
            return

        # Extract quaternion and convert to roll, pitch, yaw
        quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(quat)

        # Gravity compensation
        g = 9.81
        acc_x = msg.linear_acceleration.x + np.sin(pitch) * g
        acc_y = msg.linear_acceleration.y - np.cos(pitch) * np.sin(roll) * g
        acc_z = msg.linear_acceleration.z - np.cos(pitch) * np.cos(roll) * g

        # Build numpy arrays
        angular_velo = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ], dtype=np.float32)

        acc = np.array([acc_x, acc_y, acc_z], dtype=np.float32)

        quat_np = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ], dtype=np.float32)  # x, y, z, w format as expected in get_imu

        imu_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # self.get_logger().info(f"IMU Data Received at {imu_time:.3f}")
        self.lidar_undistortion.get_imu(angular_velo, acc, quat_np, imu_time)
      
    # Filters
    def filter_outliers(self, pointcloud):
        """
        Apply outlier filtering to remove noisy points from the point cloud.
        
        Args:
            pointcloud: Open3D PointCloud object
            
        Returns:
            filtered_pointcloud: Open3D PointCloud object with outliers removed
        """
        if not self.enable_outlier_filtering:
            return pointcloud
            
        filtered_pcd = pointcloud
        original_points = len(filtered_pcd.points)
        
        # 1. Range filtering - remove points too close or too far
        filtered_pcd = self.range_filter(filtered_pcd)
        
        # 2. Statistical outlier removal
        filtered_pcd = self.statistical_outlier_removal(filtered_pcd)
        
        # 3. Radius outlier removal (optional, for very noisy data)
        # filtered_pcd = self.radius_outlier_removal(filtered_pcd)
        
        filtered_points = len(filtered_pcd.points)
        if original_points > 0:
            removal_percentage = ((original_points - filtered_points) / original_points) * 100
            self.get_logger().debug(f"Outlier filtering: {original_points} -> {filtered_points} points ({removal_percentage:.1f}% removed)")
        
        return filtered_pcd

    def range_filter(self, pointcloud):
        """Remove points outside the specified range."""
        points = np.asarray(pointcloud.points)
        if len(points) == 0:
            return pointcloud
            
        # Calculate distance from origin (sensor)
        distances = np.linalg.norm(points, axis=1)
        
        # Keep points within range
        valid_mask = (distances >= self.range_filter_min) & (distances <= self.range_filter_max)
        
        filtered_points = points[valid_mask]
        
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        
        return filtered_pcd

    def statistical_outlier_removal(self, pointcloud):
        """Remove statistical outliers based on neighboring points."""
        if len(pointcloud.points) < self.statistical_nb_neighbors:
            return pointcloud
            
        filtered_pcd, _ = pointcloud.remove_statistical_outlier(
            nb_neighbors=self.statistical_nb_neighbors,
            std_ratio=self.statistical_std_ratio
        )
        
        return filtered_pcd

    def radius_outlier_removal(self, pointcloud):
        """Remove outliers based on radius search."""
        if len(pointcloud.points) < self.radius_outlier_nb_points:
            return pointcloud
            
        filtered_pcd, _ = pointcloud.remove_radius_outlier(
            nb_points=self.radius_outlier_nb_points,
            radius=self.radius_outlier_radius
        )
        
        return filtered_pcd

    # Utility functions
    def rebuild_xyz_msg_fast(self, msg):
        # Zero-copy structured view with just x,y,z
        struct = pc2.read_points(msg, field_names=['x','y','z'], skip_nans=False)
        # Dense (N,3) array; cast once to float32 (no copy if already f32)
        xyz = structured_to_unstructured(struct).astype(np.float32, copy=False)
        # Single vectorized NaN filter
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        # Build canonical XYZ PointCloud2 (float32)
        return pc2.create_cloud_xyz32(msg.header, xyz)
    
    def transform_pointcloud_msg_optimized(self, cloud: PointCloud2, transform: TransformStamped) -> PointCloud2:
        """
        Most efficient point cloud transformation using tf2_sensor_msgs (C++ optimized)
        or vectorized numpy operations as fallback.
        """
        try:
            # Use tf2_sensor_msgs for maximum performance (C++ implementation)
            transformed_cloud = tf2_sensor_msgs.do_transform_cloud(cloud, transform)
            return transformed_cloud
        except Exception as e:
            self.get_logger().info(f"tf2_sensor_msgs transform failed: {e}")

    def ros2_to_open3d(self, msg):
        
        # Pull only XYZ as a NumPy array (shape: [N, 3])
        xyz = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        
        # Open3D prefers float64; this is a no-copy if already float64
        if xyz.dtype != np.float64:
            xyz = xyz.astype(np.float64, copy=False)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        
        return pcd
    
    def publish_pose_and_pointcloud(self, final_transformation: np.ndarray, pointcloud, stamp: Time):
        
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
        position = final_transformation[0:3, 3].astype(np.float64)
        quat = tf_transformations.quaternion_from_matrix(final_transformation.astype(np.float64))
        quat_msg = geometry_msgs.msg.Quaternion(
            x=quat[0],
            y=quat[1],
            z=quat[2],
            w=quat[3]
        )
        self.estimated_pose[0] = position[0]
        self.estimated_pose[1] = position[1]
        self.estimated_pose[2] = position[2]
        self.estimated_pose[3], self.estimated_pose[4], self.estimated_pose[5] = tf_transformations.euler_from_quaternion(quat)
        
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
        
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.global_frame_id
        odom.pose.pose = self.current_pose_stamped_.pose
        odom.twist = self.odom_que_[self.odom_ptr_front_].twist
        self.odom_pub_.publish(odom)

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
        if (self.sub_map_displacement >= self.trans_for_mapupdate_) and (not self.mapping_flag_) and (self.publish_map_array_msg):
            
            # Make a local copy of the pose stamped for passing into the thread
            current_pose_copy = PoseStamped()
            current_pose_copy.header = self.current_pose_stamped_.header
            current_pose_copy.pose = self.current_pose_stamped_.pose

            # Update previous_position_ immediately
            self.sub_map_previous_position = position.copy()
            self.get_logger().info(f"Map update triggered with displacement: {self.sub_map_displacement:.3f} m")

            # Launch the map-updating task on a new thread using the ThreadPoolExecutor
            def mapping_job():
                self.update_map_array(pointcloud, current_pose_copy)

            # Submit the mapping job to the executor
            self.mapping_flag_ = True
            self.loop_executor.submit(mapping_job)
            
        # Create a PointCloud2 message from Open3D point cloud
        header = Header()
        header.stamp = stamp
        header.frame_id = "map"  # Set the frame ID for the map #TODO: Make sensor frame
        points = np.asarray(pointcloud.points)
        pointcloud_msg = pc2.create_cloud_xyz32(header, points)
        
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

        # Create SubMap message
        current_pose_stamped_copy = copy.deepcopy(current_pose_stamped)
        points = np.asarray(downsampled_cloud.points)
        header = Header()
        header.stamp = current_pose_stamped_copy.header.stamp
        header.frame_id = self.global_frame_id  # or another appropriate frame
        cloud_msg = pc2.create_cloud_xyz32(header, points)
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

        # Publish the updated Maparray message
        self.map_array_pub.publish(self.map_array_msg)
        self.get_logger().info("Map updated successfully.")
        self.mapping_flag_ = False

        # # Conditional map publishing
        # map_time = self.get_clock().now()
        # dt = (map_time - self.last_map_time_).nanoseconds * 1e-9
        # if dt > self.map_publish_period_:
        #     self.publish_map(self.map_array_msg, self.global_frame_id)
        #     self.last_map_time_ = map_time
        # pass
        
        if self.publish_unmodified_map:
            self.Map_unmodified = self.create_map_cloud_from_map_array(self.map_array_msg)
            self.unmodified_map_pub.publish(self.Map_unmodified)
        
    def create_map_cloud_from_map_array(self, map_array_msg: MapArrayT):
            
        # Create unmodified map cloud
        self.get_logger().info("Creating unmodified map cloud")
        unmodified_map = o3d.geometry.PointCloud()
        submap_size = len(map_array_msg.submaps)
        for i in range(submap_size):
            
            # Get the transformation matrix from the submap pose
            position = map_array_msg.submaps[i].pose.position
            orientation = map_array_msg.submaps[i].pose.orientation
            trans = tf_transformations.translation_matrix([position.x, position.y, position.z])
            rot = tf_transformations.quaternion_matrix([orientation.x, orientation.y, orientation.z, orientation.w])
            T_unoptimised = np.dot(trans, rot) 
            
            # Convert PointCloud2 to Open3D PointCloud
            cloud_pcl2 = map_array_msg.submaps[i].cloud
            cloud_open3d = self.ros2_to_open3d(cloud_pcl2)
            
            ## Create map cloud
            transformed_cloud = self.transform_scan_open3d(cloud_open3d, T_unoptimised)
            unmodified_map += transformed_cloud
            
        # Downsample the map point cloud
        self.get_logger().info(f"Unmodified map contains {len(unmodified_map.points)} points")
        unmodified_map = unmodified_map.voxel_down_sample(self.map_global_downsample_res)
        self.get_logger().info(f"Downsampled Unmodified map contains {len(unmodified_map.points)} points")
        
        ## Create map point cloud msg
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.global_frame_id
        points = np.asarray(unmodified_map.points)
        map_cloud = point_cloud2.create_cloud_xyz32(header, points)
        
        return map_cloud
    
    def transform_scan_open3d(self, scan, transformation):
        
        # Check if the input is an Open3D PointCloud object
        if not isinstance(scan, o3d.geometry.PointCloud):
                raise TypeError("Input scan must be an Open3D PointCloud object.")

        # Apply the transformation to the point cloud
        points = np.asarray(scan.points)                                                
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))         
        transformed_points = (transformation @ homogeneous_points.T).T                  
        transformed_pcd = o3d.geometry.PointCloud()                                     
        transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points[:, :3])  

        return transformed_pcd
            
    def publish_map(self):
        
        if self.Map is None or len(self.Map.points) == 0:
            self.get_logger().warn("Map is empty. Nothing to publish.")
            return

        # Convert Open3D point cloud to a list of points
        points = np.asarray(self.Map.points)

        # Create a PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map" 
        pointcloud_msg = pc2.create_cloud_xyz32(header, points)

        # Publish the PointCloud2 message
        self.map_publisher.publish(pointcloud_msg)
     
    def get_transformation_matrix_from_pose(self, pose: Pose):
        
        # Convert ROS pose to Eigen-like 4x4 transformation matrix (as numpy)
        trans = tf_transformations.translation_matrix([
            pose.position.x,
            pose.position.y,
            pose.position.z
        ])
        rot = tf_transformations.quaternion_matrix([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ])
        return np.dot(trans, rot)

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
        tx, ty, tz = pose[0], pose[1], pose[2]
        
        # Convert roll, pitch, yaw to quaternion
        roll, pitch, yaw = pose[3], pose[4], pose[5]
        qx, qy, qz, qw = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
        
        # Write to file in TUM format
        with open(filename, 'a') as f:
            f.write(f"{time_sec:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
    
    def match_scans(self, source, target, trans_init=None):
        
        source_down = source
        target_down = target
        
        threshold = 0.4
        if trans_init is None:
            trans_init = np.eye(4)
        
        #Generalised ICP 
        source_down.estimate_covariances()
        target_down.estimate_covariances()
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_down, target_down, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        
        # # #Point_to_plane ICP
        # target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
        
        # # build a robust Tukey kernel: residuals beyond width=0.05m are completely ignored
        # # tukey_loss = o3d.pipelines.registration.TukeyLoss(k=0.05)
        # # tukey = o3d.pipelines.registration.loss.RobustKernel(type=o3d.pipelines.registration.loss.RobustKernel.Tukey, width=0.05)
        # # estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss=tukey)

        # # # set a convergence criteria (max 40 iters here)
        # criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40)
        
        # reg_p2p = o3d.pipelines.registration.registration_icp(
        #     source_down, 
        #     target_down, 
        #     threshold, 
        #     trans_init,
        #     o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        #     # estimator, 
        #     criteria
        # )
        
        # # Point_to_point ICP
        # reg_p2p = o3d.pipelines.registration.registration_icp(
        #     source_down, target_down, threshold, trans_init,
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint()
        # )
        
        return reg_p2p.transformation, reg_p2p.fitness, reg_p2p.inlier_rmse, len(reg_p2p.correspondence_set)
    
def main(args=None):
    rclpy.init(args=args)
    node = PointCloudOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
