#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from collections import deque
from gazebo_msgs.msg import ModelStates
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import TransformStamped
from tf_transformations import quaternion_from_euler
import tf_transformations
from sensor_msgs_py import point_cloud2
from builtin_interfaces.msg import Time
from slam_interfaces.msg import SubMapT, MapArrayT

from lidarslam_msgs.msg import SubMap, MapArray

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_srvs.srv import Empty

import os
import sys
import copy
import numpy as np
import open3d as o3d
from pathlib import Path as dir_path
import csv
import threading

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)  # Add the script directory to the system path for importing custom modules

# from utils.ScanContextManager import ScanContextManager
# from utils.PoseGraphManager import PoseGraphManager, PoseGraphResultSaver
# from utils.UtilsMisc  import ScanContextResultSaver
# from utils.UtilsMisc import yawdeg2se3

from mod_slam.utils.ScanContextManager import ScanContextManager
from mod_slam.utils.PoseGraphManager import PoseGraphManager, PoseGraphResultSaver
from mod_slam.utils.UtilsMisc  import ScanContextResultSaver
from mod_slam.utils.UtilsMisc import yawdeg2se3, savePoseGraph
from mod_slam.utils.occupancy_grid import ProbabilisticOccupancyGrid

class BackendNode(Node):
    def __init__(self):
        super().__init__('backend_node')
        
        # Declare parameters
        
        # Frame parameters
        self.declare_parameter('global_frame_id', 'map')
        self.declare_parameter('robot_frame_id', 'base_link')
        
        # Data saving parameters
        self.declare_parameter('test_name', 'real_lab_sc')  # Name of the test for saving results
        self.declare_parameter('save_dir', '/home/ruan/dev_ws/src/lidar_slam_3d/results')
        
        # Scan context parameters
        self.declare_parameter('use_scan_context', True)  # Use Scan Context for loop closure
        self.declare_parameter('SCM_num_rings', 20)#20 #40
        self.declare_parameter('SCM_num_sectors', 60)#60 #80
        self.declare_parameter('SCM_max_radius', 8.0)  # Maximum radius of the scan context 8
        self.declare_parameter('SCM_num_candidates', 10)
        self.declare_parameter('SCM_loop_threshold', 0.16)
        self.declare_parameter("SCM_recent_node_exclusion", 500)  # Number of recent nodes to exclude in loop detection
        self.declare_parameter('SCM_try_loop_detection_gap', 30) 
        self.declare_parameter('SCM_map_construct_gap', 30)  # Gap for constructing the map
        
        # Euclidean loop closure parameters
        self.declare_parameter('loop_detection_period', 3.0)  # Period for loop detection in seconds
        self.declare_parameter('number_of_adjacent_constraints', 5)  # Number of adjacent constraints to consider
        self.declare_parameter('loop_closure_distance', 12.0)  # Distance threshold for loop closure 4 , 6
        self.declare_parameter('loop_closure_search_radius', 2.0) # lab 0.6 #Other 2  # Search radius for loop closure 2 , 3
        self.declare_parameter('search_submap_num', 3)  # Number of submaps to search for loop closure
        self.declare_parameter('loop_submap_clouds_downsample_res', 0.01)  # Downsample resolution for loop submap clouds
        self.declare_parameter('min_fitness_score', 1.0)  # Minimum fitness score for loop closure (Loop closure threshold)
        self.declare_parameter('map_global_downsample_res', 0.1)  # Search radius for loop closure

        # Pose graph parameters
        self.declare_parameter('PGM_prior_covariance', [1.0e-6, 1.0e-6, 1.0e-6, 1.0e-4, 1.0e-4, 1.0e-4])
        self.declare_parameter('PGM_odom_covariance', [0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
        self.declare_parameter('PGM_loop_covariance', [0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
        
        # Occupancy grid map parameters
        self.declare_parameter('occupancy_grid_publish', False)
        self.declare_parameter("publish_unmodified_map", False) # Warning: Will slow down the system if True
        self.declare_parameter('trans_for_unmodified_mapupdate', 2.0)  # Distance threshold for map update # 1
        self.declare_parameter("map_resolution", 0.1)  # Resolution of the occupancy grid map
        self.declare_parameter("map_initial_size", 10)  # Initial size of the occupancy grid map
        self.declare_parameter("p_occ", 0.7)  # Probability of occupancy
        self.declare_parameter("p_free", 0.3)  # Probability of free space
        self.declare_parameter("uniform_expand", True)  # Uniformly expand the map
        self.declare_parameter("z_threshold", 0.6)  # Height threshold for occupancy grid update
        
        # Get parameters
        self.use_scan_context = self.get_parameter('use_scan_context').get_parameter_value().bool_value
        self.get_logger().info(f"Use Scan Context: {self.use_scan_context}")
        self.num_rings = self.get_parameter('SCM_num_rings').get_parameter_value().integer_value
        self.num_sectors = self.get_parameter('SCM_num_sectors').get_parameter_value().integer_value
        self.num_candidates = self.get_parameter('SCM_num_candidates').get_parameter_value().integer_value
        self.try_gap_loop_detection = self.get_parameter('SCM_try_loop_detection_gap').get_parameter_value().integer_value
        self.loop_threshold = self.get_parameter('SCM_loop_threshold').get_parameter_value().double_value
        self.max_radius = self.get_parameter('SCM_max_radius').get_parameter_value().double_value
        self.map_construct_gap = self.get_parameter('SCM_map_construct_gap').get_parameter_value().integer_value
        self.SCM_recent_node_exclusion = self.get_parameter('SCM_recent_node_exclusion').get_parameter_value().integer_value
        self.save_directory = self.get_parameter('save_dir').get_parameter_value().string_value
        self.save_dir_map = f"{self.save_directory}/Maps"
        self.save_dir_trajectory = f"{self.save_directory}/Data"
        self.save_dir_pose = f"{self.save_directory}/Data/Posegraph"
        self.test_name = self.get_parameter('test_name').get_parameter_value().string_value
        self.global_frame_id = self.get_parameter('global_frame_id').get_parameter_value().string_value
        self.robot_frame_id = self.get_parameter('robot_frame_id').get_parameter_value().string_value
        self.loop_closure_distance = self.get_parameter('loop_closure_distance').get_parameter_value().double_value
        self.loop_closure_search_radius = self.get_parameter('loop_closure_search_radius').get_parameter_value().double_value
        self.search_submap_num = self.get_parameter('search_submap_num').get_parameter_value().integer_value
        self.loop_submap_clouds_downsample_res = self.get_parameter('loop_submap_clouds_downsample_res').get_parameter_value().double_value
        self.min_fitness_score = self.get_parameter('min_fitness_score').get_parameter_value().double_value
        self.loop_detection_period = self.get_parameter('loop_detection_period').get_parameter_value().double_value
        self.number_of_adjacent_constraints = self.get_parameter('number_of_adjacent_constraints').get_parameter_value().integer_value
        self.map_global_downsample_res = self.get_parameter('map_global_downsample_res').get_parameter_value().double_value
        self.PGM_prior_covariance = np.array(self.get_parameter('PGM_prior_covariance').get_parameter_value().double_array_value)
        self.PGM_odom_covariance = np.array(self.get_parameter('PGM_odom_covariance').get_parameter_value().double_array_value)
        self.PGM_loop_covariance = np.array(self.get_parameter('PGM_loop_covariance').get_parameter_value().double_array_value)
        self.occupancy_grid_publish = self.get_parameter('occupancy_grid_publish').get_parameter_value().bool_value
        self.publish_unmodified_map = self.get_parameter('publish_unmodified_map').get_parameter_value().bool_value
        self.trans_for_unmodified_mapupdate = self.get_parameter('trans_for_unmodified_mapupdate').get_parameter_value().double_value
        self.map_resolution = self.get_parameter('map_resolution').get_parameter_value().double_value
        self.map_initial_size = self.get_parameter('map_initial_size').get_parameter_value().integer_value
        self.p_occ = self.get_parameter('p_occ').get_parameter_value().double_value
        self.p_free = self.get_parameter('p_free').get_parameter_value().double_value
        self.uniform_expand = self.get_parameter('uniform_expand').get_parameter_value().bool_value
        self.z_threshold = self.get_parameter('z_threshold').get_parameter_value().double_value

        # Initialize managers
        self.PGM = PoseGraphManager(self.PGM_prior_covariance,self.PGM_odom_covariance, self.PGM_loop_covariance)
        self.PGM.addPriorFactor()
        self.PGM.curr_se3 = np.eye(4)
        self.ResultSaver = PoseGraphResultSaver(
            init_pose = self.PGM.curr_se3,
            save_gap = 150,
            num_frames = 999999,  # Unknown in live mode
            seq_idx = '_',
            save_dir = self.save_dir_pose
        )
        self.SCM = ScanContextManager(
            shape = [self.num_rings, self.num_sectors],
            num_candidates = self.num_candidates,
            max_radius = self.max_radius,  # Maximum length/radius of the scan context
            threshold = self.loop_threshold,
            recent_node_exclusion = self.SCM_recent_node_exclusion
        )
        self.sc_saver = ScanContextResultSaver(save_gap = 5, save_dir = self.save_dir_pose)
        self.og_unoptimised_map = ProbabilisticOccupancyGrid(self.map_resolution, self.map_initial_size, self.p_occ, self.p_free, self.uniform_expand)
        self.og_optimised_map = ProbabilisticOccupancyGrid(self.map_resolution, self.map_initial_size, self.p_occ, self.p_free, self.uniform_expand)
        
        # Variables
        ## Logistic
        self.first_loop_closure = False
        self.graph_frame_idx = 0
        self.graph_frame_idx_opt = 0
        self.timestamp_list = []
        self.timestamp = None
        self.loop_edges = []
        self.map_array_msg_counter = 0
        self.initial_map_recieved = False
        self.latest_moving_distance_prev = 0.0
        self.map_moving_distance_prev = 0.0
        self.prev_odom_transform_estimate = np.eye(4)

        ## Ros msgs
        self.subMap = SubMap()
        self.MapArray_msg = MapArray()
        self.MapArray_modified = MapArray()
        self.Path_modified = Path()
        self.Map_unmodified = PointCloud2()
        self.Map_modified = PointCloud2()
        
        ## Threading
        self.map_thread_lock = threading.Lock()
        self.map_publishing_thread = None
        
        ## GPS values and integration
        # self.gps_queue = deque()  # Buffer for GPS messages # maxlen=100
        # self.pose_cov_threshold = 25.0
        # self.gps_cov_threshold = 100.0
        # self.use_gps_elevation = False
        # self.cloud_key_poses_3d: List[PointType] = []
        # self.transform_tobe_mapped = [0.0] * 6
        # self.pose_covariance = np.eye(6)
        # self.gt_sam_graph = gtsam.NonlinearFactorGraph()
        # self.a_loop_is_closed = False
        # self.time_laser_info_cur = 0.0

        # Subscribers
        if not self.use_scan_context:
            self.mapArray_sub = self.create_subscription(MapArray, '/map_array', self.mapArray_CB, 10)
        else:
            self.subMap_sub = self.create_subscription(SubMap, '/submap', self.subMap_callback, 10)
        # self.sub_gps = self.create_subscription(Odometry, '/gps/odom', self.gps_handler, 200)
        
        # Publishers
        self.modified_mapArray_pub = self.create_publisher(MapArray, '/modified_map_array', 10)
        self.unmodified_map = self.create_publisher(PointCloud2, '/unmodified_map_2', 10)
        self.modified_path_pub = self.create_publisher(Path, '/modified_path', 10)
        self.modified_map = self.create_publisher(PointCloud2, '/modified_map', 10)
        self.loop_submap_publisher = self.create_publisher(PointCloud2, '/loop_submap_pointcloud', 10)
        self.latest_submap_pointcloud = self.create_publisher(PointCloud2, '/latest_submap_pointcloud', 10)
        self.unmodified_og_map_pub = self.create_publisher(OccupancyGrid, '/unmodified_og_map', 10)
        self.optimised_og_map_pub = self.create_publisher(OccupancyGrid, '/optimised_og_map', 10)
        # self.test_submap_pointcloud = self.create_publisher(PointCloud2, '/test_submap_pointcloud', 10)

        # Service
        self.map_save_srv = self.create_service(Empty, 'map_save', self.map_save_callback)

        # Buffers
        self.odom_buffer = deque(maxlen=100)
        self.pcd_buffer = deque(maxlen=100)
        self.subMap_buffer = deque(maxlen=100)
        self.odometry_history = deque(maxlen=self.number_of_adjacent_constraints + 1)

        # Timers 
        if not self.use_scan_context:
            self.timer_loop_search = self.create_timer(self.loop_detection_period, self.loop_check)
        if self.occupancy_grid_publish:
            self.timer_occupancy_grid = self.create_timer(1.0, self.publish_unoptimised_occupancy_grid)
        
        # Create save directories if it doesn't exist
        if not os.path.exists(self.save_dir_pose):
            os.makedirs(self.save_dir_pose)
        if not os.path.exists(self.save_dir_map):
            os.makedirs(self.save_dir_map)
            
        # Log files
        # TUM format trajectory files for benchmarking
        self.unoptimised_tum_log_file = os.path.join(self.save_dir_trajectory, f'Sim_unoptimised_{self.test_name}.txt')
        open(self.unoptimised_tum_log_file, 'w').close()
        self.optimised_tum_log_file = os.path.join(self.save_dir_trajectory, f'Sim_optimised_{self.test_name}.txt')
        open(self.optimised_tum_log_file, 'w').close()

        self.get_logger().info('Backend Node initialized')

    # Callback functions
    def mapArray_CB(self, msg: MapArray):
        
        self.get_logger().info(f"Recieved map array message {self.map_array_msg_counter}")
        self.map_array_msg_counter += 1
        self.initial_map_recieved = True
        self.MapArray_msg = copy.deepcopy(msg)
        
        if self.publish_unmodified_map:
            self.get_logger().info("Publishing unmodified map")
            self.create_and_pub_map_cloud(self.MapArray_msg)
    
    def subMap_callback(self, msg: SubMap):
        self.subMap_buffer.append(msg)
        distance = msg.distance
        pose = msg.pose
        pointcloud = msg.cloud
        
        self.timestamp_list.append(msg.header.stamp)
        
        # Test: Publish the pointcloud from submap message
        # test_pointcloud = pointcloud
        # test_pointcloud.header.frame_id = self.global_frame_id  # Ensure correct frame_id
        # test_pointcloud.header.stamp = self.get_clock().now().to_msg()
        # self.test_submap_pointcloud.publish(test_pointcloud)
        # self.get_logger().info(f"Published test submap pointcloud with {len(test_pointcloud.data)} bytes of data")
        
        self.pipeline(pointcloud, pose, distance)
        
        # latest_pcd = None
        # latest_odom = None
        
        # if self.odom_buffer:
        #     latest_odom = self.odom_buffer[-1]
        #     pos = latest_odom.pose.pose.position
        #     ori = latest_odom.pose.pose.orientation
        #     # self.get_logger().info(
        #     #     f"[ODOM] Latest position: x={pos.x:.2f}, y={pos.y:.2f}, z={pos.z:.2f} | "
        #     #     f"Orientation (quat): x={ori.x:.2f}, y={ori.y:.2f}, z={ori.z:.2f}, w={ori.w:.2f}"
        #     # )
            
        # if self.pcd_buffer:
        #     latest_pcd = self.pcd_buffer[-1]
        #     num_points = sum(1 for _ in point_cloud2.read_points(latest_pcd, skip_nans=True))
        #     # self.get_logger().info(f"[PCD] Latest point cloud contains {num_points} points.")
        
        # if self.subMap_buffer:
        #     latest_subMap = self.subMap_buffer[-1]
        #     self.subMap = copy.deepcopy(latest_subMap)
        #     self.get_logger().info(f"Processing {len(self.subMap_buffer)} messages. Last position: {self.subMap.pose.position.x}, {self.subMap.pose.position.y}, {self.subMap.pose.position.z}")
        #     # self.get_logger().info(f"[SubMap] Latest submap received with {len(self.subMap.points)} points.")
        #     # self.get_logger().info(f"[SubMap] Latest submap timestamp: {self.subMap.header.stamp.sec}.{self.subMap.header.stamp.nanosec}")
        # else:
        #     self.get_logger().warn("No submap data received yet.")
        #     return
        
        # self.pipeline(latest_pcd, latest_odom)

        # Optionally clear buffers (depends on desired behavior)
        # self.odom_buffer.clear()
        # self.pcd_buffer.clear()
        # self.subMap_buffer.clear()    
        
        
        # self.pcd_buffer.append(msg) 
    
    def map_save_callback(self, request, response):
        self.get_logger().info("Received a request to save the map")
        
        # Check if there is a map to save
        if not self.initial_map_recieved:
            self.get_logger().warn("Initial map not received yet")
            return response
        if not self.first_loop_closure:
            self.get_logger().warn("No loop closure has been detected yet")

        if self.use_scan_context == True and self.initial_map_recieved:
            self.create_and_pub_sc_unmodified_map(save_map=True) # Unoptimised map
            if self.first_loop_closure:
                self.update_map_and_path(save_map=True) # Optimised map

        if  self.use_scan_context == False and self.initial_map_recieved:
            self.create_and_pub_map_cloud(self.MapArray_msg, save_map=True) # Unoptimised map
            if self.first_loop_closure:
                self.do_pose_adjustment(self.MapArray_msg, save_map=True) # Optimised map
        
        return response
    
    def gps_handler(self, msg: Odometry):
        self.gps_queue.append(msg)
    
    # Euclidean method
    def loop_check(self):
        msg = copy.deepcopy(self.MapArray_msg)
        self.search_loop_closure(msg)
        
    def search_loop_closure(self, map_array_msg: MapArray):
        
        if not self.initial_map_recieved: 
            self.get_logger().info("Initial map not yet recieved")
            return
        # if not self.map_array_updated : 
        
        num_submaps = len(map_array_msg.submaps)
        min_fitness_score = float('inf')
        distance_min_fitness_score = 0
        is_candidate = False
        
        # Extract latest pointcloud and transform to global frame
        self.get_logger().info("Extracting and transforming to global frame")
        latest_submap = map_array_msg.submaps[-1]
        latest_pose = latest_submap.pose
        latest_cloud_pcl = latest_submap.cloud
        
        # Update occupancy grid map
        if self.occupancy_grid_publish:
            self.og_unoptimised_map.update_from_pointcloud(latest_pose, latest_cloud_pcl, z_threshold=self.z_threshold)

        ## Convert Odometry pose to transformation matrix
        position = latest_pose.position
        orientation = latest_pose.orientation
        trans = tf_transformations.translation_matrix([position.x, position.y, position.z])
        rot = tf_transformations.quaternion_matrix([orientation.x, orientation.y, orientation.z, orientation.w])
        T_latest = np.dot(trans, rot)  # 4x4 transformation matrix
        latest_cloud_open3d = self.ros2_to_open3d(latest_cloud_pcl)
        latest_transformed_cloud = self.transform_scan_open3d(latest_cloud_open3d, T_latest)
        
        # Scan through prevoius frame position to see which cloud is close enough to latest frame
        self.get_logger().info("Scan through previous frame positions")
        latest_moving_distance = latest_submap.distance
        latest_submap_pos = np.array([position.x, position.y, position.z])
        id_min = 0
        min_distance = float('inf')
        for i in range(num_submaps - 1):
            submap = map_array_msg.submaps[i]
            pose = submap.pose
            pos = np.array([pose.position.x, pose.position.y, pose.position.z])
            distance = np.linalg.norm(latest_submap_pos - pos)
            
            ## Check if loop closure distance is far enough and within the search radius
            moving_distance = latest_moving_distance - submap.distance
            self.get_logger().info(f"Subamp number compare: {i}")
            self.get_logger().info(f"Moving distance: {moving_distance}")
            self.get_logger().info(f"Radius distance: {distance}")

            if (latest_moving_distance - submap.distance > self.loop_closure_distance) and (distance < self.loop_closure_search_radius):
                self.get_logger().info("Candidate found")
                is_candidate = True
                ## Store minimum distance for loop closure candidate
                if distance < min_distance:
                    min_distance = distance
                    id_min = i
                    min_submap = submap
            
        # If candidate found, perform ICP to find the best transformation
        if is_candidate:
            ## Combine a set number of submaps to global frame\
            self.get_logger().info("Combining cloud to check for loop closure")
            submap_clouds = o3d.geometry.PointCloud()
            for j in range(2 * self.search_submap_num):
                if (id_min + j - self.search_submap_num) < 0:
                    self.get_logger().info("Submaps still below search max")
                    continue
                if (len(map_array_msg.submaps) <= id_min + j - self.search_submap_num):
                    self.get_logger().info("Warning: Few submaps available for loop closure search")
                    continue
                self.get_logger().info(f'Length of submaps: {len(map_array_msg.submaps)}')
                self.get_logger().info(f'{j}')
                self.get_logger().info(f"Submap number: {id_min + j - self.search_submap_num}")
                near_submap = map_array_msg.submaps[id_min + j - self.search_submap_num]
                near_submap_cloud_pcl = near_submap.cloud
                near_submap_pose = near_submap.pose
                position = near_submap_pose.position
                orientation = near_submap_pose.orientation
                trans = tf_transformations.translation_matrix([position.x, position.y, position.z])
                rot = tf_transformations.quaternion_matrix([orientation.x, orientation.y, orientation.z, orientation.w])
                T_submap = np.dot(trans, rot)  # 4x4 transformation matrix
                near_submap_cloud_open3d = self.ros2_to_open3d(near_submap_cloud_pcl)
                transformed_cloud = self.transform_scan_open3d(near_submap_cloud_open3d, T_submap)
                submap_clouds += transformed_cloud
                
            ## Downsample subset of pointclouds
            submap_clouds = submap_clouds.voxel_down_sample(self.loop_submap_clouds_downsample_res)
            
            ## Create latest submap point cloud
            self.get_logger().info(f"Submap cloud contains {len(latest_transformed_cloud.points)} points")
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.global_frame_id
            points = np.asarray(latest_transformed_cloud.points)
            latest_cloud = point_cloud2.create_cloud_xyz32(header, points)
            self.latest_submap_pointcloud.publish(latest_cloud)
            
            ## Create submap point cloud
            self.get_logger().info(f"Submap cloud contains {len(submap_clouds.points)} points")
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.global_frame_id
            points = np.asarray(submap_clouds.points)
            loop_clouds = point_cloud2.create_cloud_xyz32(header, points)
            self.loop_submap_publisher.publish(loop_clouds)

            ## ICP latest scan to candidate submap
            self.get_logger().info("Performing ICP")
            icp_result = o3d.pipelines.registration.registration_icp(
                source=latest_transformed_cloud,
                target=submap_clouds,
                max_correspondence_distance=0.5,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80)
            )
            
            ## Check fitness score for valid loop closure
            self.get_logger().info(f"Checking fitness {icp_result.fitness}")
            if icp_result.fitness <= self.min_fitness_score:
                
                self.first_loop_closure = True
            
                # Build loop closure edge
                T_latest_affine = T_latest
                min_pose = min_submap.pose
                T_min_trans = tf_transformations.translation_matrix([min_pose.position.x, min_pose.position.y, min_pose.position.z])
                T_min_rot = tf_transformations.quaternion_matrix([min_pose.orientation.x, min_pose.orientation.y, min_pose.orientation.z, min_pose.orientation.w])
                T_min_affine = np.dot(T_min_trans, T_min_rot)

                T_icp = icp_result.transformation

                # Relative transformation from min_submap to latest_submap via ICP result
                T_relative = np.linalg.inv(T_min_affine) @ (T_icp @ T_latest_affine)

                self.get_logger().info("Storing loop edge")
                loop_edge = {
                    'pair_id': (id_min, num_submaps - 1),
                    'relative_pose': T_relative
                }
                self.loop_edges.append(loop_edge)

                print(f"Loop_list: {self.loop_edges}")
                print(f"PoseAdjustment - distance: {min_submap.distance}, score: {icp_result.fitness}")
                print(f"id_loop_point 1: {id_min}, id_loop_point 2: {num_submaps - 1}")
                print("Final transformation:\n", T_icp)

                self.do_pose_adjustment(map_array_msg)

                return

    def do_pose_adjustment(self, map_array_msg: MapArray, save_map = False):
        
        # Initialise solver
        self.get_logger().info("Initialising")
        PGM = PoseGraphManager()
        
        # Add edges and adjacent constraints to the solver
        self.get_logger().info("Adding edges and constraints to the solver")
        submap_size = len(map_array_msg.submaps)
        self.get_logger().info(f"Number of submaps: {submap_size}")
        T_odom = np.eye(4)
        T_odom_prev = np.eye(4)
        self.timestamp_list = []
        for i in range(submap_size):
            self.get_logger().info(f"Processing iteration {i}")
            submap = map_array_msg.submaps[i]
            pose = submap.pose
            self.timestamp_list.append(submap.header.stamp)

            ## First frame initialization
            if i == 0:
                PGM.addPriorFactor()
                PGM.curr_se3 = np.eye(4)
                continue
            
            ## Convert PoseStamped to transformation matrix
            position = pose.position
            orientation = pose.orientation
            trans = tf_transformations.translation_matrix([position.x, position.y, position.z])
            rot = tf_transformations.quaternion_matrix([orientation.x, orientation.y, orientation.z, orientation.w])
            T_odom = np.dot(trans, rot) 
            
            ## Update pose graph with odometry
            PGM.curr_node_idx = i
            PGM.curr_se3 = T_odom
            # T_odom_relative = np.linalg.inv(T_odom_prev) @ T_odom
            # PGM.addOdometryFactor(T_odom_relative)
            PGM.addOdometry_initial()
            if i > self.number_of_adjacent_constraints:
                for j in range(self.number_of_adjacent_constraints):
                    
                    adjacent_pose = map_array_msg.submaps[i - self.number_of_adjacent_constraints + j].pose
        
                    position = adjacent_pose.position
                    orientation = adjacent_pose.orientation
                    trans = tf_transformations.translation_matrix([position.x, position.y, position.z])
                    rot = tf_transformations.quaternion_matrix([orientation.x, orientation.y, orientation.z, orientation.w])
                    T_prev = np.dot(trans, rot) 
                    
                    T_rel = np.linalg.inv(T_prev) @ T_odom
                    
                    prev_idx = i - self.number_of_adjacent_constraints + j
                    PGM.addOdometryFactor_adv(T_rel, prev_idx)

            T_odom_prev = T_odom
            PGM.prev_node_idx = PGM.curr_node_idx
        
        # Add loop edges to the solver
        self.get_logger().info("Adding loop edges to the solver")
        for edge in self.loop_edges:
            id_min, id_latest = edge['pair_id']
            T_loop = edge['relative_pose']
            
            # Log the type and shape of T_loop
            self.get_logger().info(f"T_loop type: {type(T_loop)}, shape: {getattr(T_loop, 'shape', 'N/A')}")
            self.get_logger().info(f"T_loop content:\n{T_loop}")
            
            PGM.addLoopFactor_adv(T_loop, id_min, id_latest)
        
        # Optimize the pose graph
        self.get_logger().info("Optimizing the pose graph")
        PGM.optimizePoseGraph()
        
        ## Save unoptimized graph first
        filename_unopt = "pose_unoptimized_" + str(PGM.curr_node_idx) + ".csv"
        filename_unopt = os.path.join(self.save_dir_pose, filename_unopt)
        savePoseGraph(PGM.curr_node_idx+1, PGM.graph_initials, filename_unopt)

        ## Save optimized poses
        filename_opt = "pose_optimized_" + str(PGM.curr_node_idx) + ".csv"
        filename_opt = os.path.join(self.save_dir_pose, filename_opt)
        savePoseGraph(PGM.curr_node_idx+1, PGM.graph_optimized, filename_opt)
        
        # Create modified map and path messages
        self.get_logger().info("Creating modified MapArray message")
        self.MapArray_modified.header.stamp = self.get_clock().now().to_msg()
        self.MapArray_modified.header.frame_id = self.global_frame_id
        self.Path_modified.poses.clear()
        modified_map = o3d.geometry.PointCloud()
        open(self.unoptimised_tum_log_file, 'w').close()
        open(self.optimised_tum_log_file, 'w').close()
        for i in range(submap_size):
            
            T_optimized = PGM.get_optimised_transform(i)
            pose_stamped = self.Tmatrix_to_pose_stamped(T_optimized, frame_id=self.global_frame_id)
            distance = map_array_msg.submaps[i].distance
            cloud_pcl2 = map_array_msg.submaps[i].cloud
            unoptimised_pose = map_array_msg.submaps[i].pose
            cloud_open3d = self.ros2_to_open3d(cloud_pcl2)
            
            ## Create optimised occupancy grid map
            if self.occupancy_grid_publish:
                self.get_logger().info(f"Updating optimised occupancy grid map: submap {i}")
                self.og_optimised_map.update_from_open3d(pose_stamped, cloud_open3d, z_threshold=self.z_threshold)
            
            ## Create MapArray message
            submap = SubMap()
            submap.header.frame_id = self.robot_frame_id
            submap.header.stamp = self.get_clock().now().to_msg()
            submap.distance = distance
            submap.pose = pose_stamped.pose
            submap.cloud = cloud_pcl2
            self.MapArray_modified.submaps.append(submap)
            
            ## Save unoptimised pose to TUM log file
            self.save_pose_to_tum_from_pose(self.timestamp_list[i], unoptimised_pose, self.unoptimised_tum_log_file)

            ## Optimised path
            self.Path_modified.header.stamp = pose_stamped.header.stamp
            self.Path_modified.header.frame_id = self.global_frame_id
            self.Path_modified.poses.append(pose_stamped)
            self.save_pose_to_tum_from_pose(self.timestamp_list[i], pose_stamped.pose, self.optimised_tum_log_file)

            ## Map cloud
            transformed_cloud = self.transform_scan_open3d(cloud_open3d, T_optimized)
            modified_map += transformed_cloud

        ## Create map point cloud
        self.get_logger().info(f"Modified map contains {len(modified_map.points)} points")
        modified_map = modified_map.voxel_down_sample(self.map_global_downsample_res)
        self.get_logger().info(f"Downsampled Modified map contains {len(modified_map.points)} points")
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.global_frame_id
        points = np.asarray(modified_map.points)
        self.Map_modified = point_cloud2.create_cloud_xyz32(header, points)

        # Publish the modified MapArray message and modified path
        self.get_logger().info("Publishing modified MapArray and Path messages")
        self.modified_path_pub.publish(self.Path_modified)
        self.modified_map.publish(self.Map_modified)
        self.modified_mapArray_pub.publish(self.MapArray_modified)
        if self.occupancy_grid_publish:
            self.publish_optimised_occupancy_grid()

        # Check if map needs to be saved
        if (save_map):
            
            # Save point cloud
            o3d.io.write_point_cloud(f"{self.save_dir_map}/{self.test_name}_map_optimised.pcd", modified_map, write_ascii=True)
            self.get_logger().info(f"Saved map to {self.save_dir_map}/{self.test_name}_map_optimised.pcd")

            if self.occupancy_grid_publish:
                # Save unoptimised occupancy grid map
                self.og_unoptimised_map.save_map(f"{self.save_dir_map}/{self.test_name}_unoptimised_og_map")
                self.get_logger().info(f"Saved unoptimised occupancy grid map to {self.save_dir_map}/{self.test_name}_unoptimised_og_map")
                
                # Save optimised occupancy grid map
                self.og_optimised_map.save_map(f"{self.save_dir_map}/{self.test_name}_optimised_og_map")
                self.get_logger().info(f"Saved optimised occupancy grid map to {self.save_dir_map}/{self.test_name}_optimised_og_map")


       
            # # Save path as CSV
            # with open("path.csv", mode="w", newline="") as file:
            #     writer = csv.writer(file)
            #     writer.writerow(["index", "x", "y", "z", "qx", "qy", "qz", "qw"])
            #     for i, pose_stamped in enumerate(self.Path_modified.poses):
            #         pos = pose_stamped.pose.position
            #         ori = pose_stamped.pose.orientation
            #         writer.writerow([i, pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])
            # self.get_logger().info("Saved path to path.csv")
        
    # SC method
    def pipeline(self, pointcloud: PointCloud2,  pose: PoseStamped, latest_moving_distance):
        
        # Convert Odometry pose to transformation matrix and pointcloud to open3d
        position = pose.position
        orientation = pose.orientation
        trans = tf_transformations.translation_matrix([position.x, position.y, position.z])
        rot = tf_transformations.quaternion_matrix([orientation.x, orientation.y, orientation.z, orientation.w])
        odom_transform_estimate = np.dot(trans, rot)  # 4x4 transformation matrix
        self.received_estimate = odom_transform_estimate
        curr_scan_down_pts = self.ros2_to_open3d(pointcloud)
        
        # Update occupancy grid map
        if self.occupancy_grid_publish:
            self.og_unoptimised_map.update_from_pointcloud(pose, pointcloud, z_threshold=self.z_threshold)
        
        # Add data to Scan Context Manager
        self.SCM.addNode(node_idx = self.graph_frame_idx, ptcloud = curr_scan_down_pts)
        
        # --------------------------------------------------------------------
        # # # Add odomtry without adjacent constraints
        # if self.graph_frame_idx == 0:
        #     self.PGM.curr_node_idx = self.graph_frame_idx
        #     self.prev_odom_transform_estimate = odom_transform_estimate
        #     self.graph_frame_idx += 1
        #     return
    
        # # Update pose graph with odometry and incremental scan-to-map transform
        # self.PGM.curr_node_idx = self.graph_frame_idx
        # self.PGM.curr_se3 = odom_transform_estimate
        # relative_odom_transform = np.linalg.inv(self.prev_odom_transform_estimate) @ odom_transform_estimate
        
        # self.PGM.addOdometryFactor(relative_odom_transform)

        # self.prev_odom_transform_estimate = odom_transform_estimate
        # self.PGM.prev_node_idx = self.PGM.curr_node_idx
        
        # --------------------------------------------------------------------
        # Add odomtry with adjacent constraints
        if self.graph_frame_idx == 0:
            self.PGM.curr_node_idx = self.graph_frame_idx
            self.odometry_history.append(np.eye(4))
            self.graph_frame_idx += 1
            self.initial_map_recieved = True
            return
        
        self.PGM.curr_node_idx = self.graph_frame_idx
        T_odom = odom_transform_estimate
        self.PGM.curr_se3 = T_odom
         
        self.odometry_history.append(T_odom)
        
        self.PGM.addOdometry_initial()
        for i in range(self.number_of_adjacent_constraints):
            
            if (self.graph_frame_idx - self.number_of_adjacent_constraints + i < 0):
                continue
            
            T_prev = self.odometry_history[- 1 - self.number_of_adjacent_constraints + i]
            T_rel = np.linalg.inv(T_prev) @ T_odom
            
            prev_idx = self.graph_frame_idx - self.number_of_adjacent_constraints + i
            self.PGM.addOdometryFactor_adv(T_rel, prev_idx)
        
        # --------------------------------------------------------------------
        ## Save desciptors and ring keys
        # self.sc_saver.saveScanContext(self.graph_frame_idx, self.SCM.scancontexts[self.graph_frame_idx])
        # self.sc_saver.saveRingKey(self.graph_frame_idx, self.SCM.ringkeys[self.graph_frame_idx])
        # self.sc_saver.saveScanContextSVG(self.SCM.scancontexts[self.graph_frame_idx], self.graph_frame_idx)
        # self.sc_saver.saveRingKeySVG(self.SCM.ringkeys[self.graph_frame_idx],  self.graph_frame_idx)
        # self.sc_saver.saveScanContextHeatmapSVG(self.SCM.scancontexts[self.graph_frame_idx], self.graph_frame_idx)
        
        # ------------------------------------------------------------------------------------------
        # Unmodified map update
        distance_moved = latest_moving_distance - self.map_moving_distance_prev
        if self.publish_unmodified_map and (distance_moved > self.trans_for_unmodified_mapupdate):
            self.map_moving_distance_prev = latest_moving_distance
            self.get_logger().info("Publishing unmodified map")
            self.create_and_pub_sc_unmodified_map()
        
        # ------------------------------------------------------------------------------------------

        # # Loop detection
        # move_check = latest_moving_distance - self.latest_moving_distance_prev
        # if (latest_moving_distance - self.latest_moving_distance_prev > self.loop_closure_distance):
        #     self.latest_moving_distance_prev = latest_moving_distance
        #      #self.get_logger().info(f"Moving distance: {move_check}")
            
        # self.ResultSaver.saveUnoptimizedPoseGraphResult(self.PGM.curr_se3, self.graph_frame_idx)
        if self.graph_frame_idx > 1 and self.graph_frame_idx % self.try_gap_loop_detection == 0:
            
            ## Find candidate
            self.get_logger().info(f"Checking for loop closure at frame {self.graph_frame_idx}")
            loop_idx, loop_dist, yaw_diff_deg = self.SCM.detectLoop()
            
            ## If candidate found, perform ICP to find the best transformation
            if loop_idx is not None:
                self.get_logger().info(f"Loop closure detected at {self.graph_frame_idx} <-> {loop_idx}")
                self.first_loop_closure = True
                loop_pts = self.SCM.getPtcloud(loop_idx)
                test_points = copy.deepcopy(curr_scan_down_pts)
                
                # Best working s = curr_scan_down_pts, t = loop_pts_o3d and -yaw_diff_deg
                self.get_logger().info(f"Yaw angle: {yaw_diff_deg}")
                loop_pts_o3d = loop_pts                
                icp_result = o3d.pipelines.registration.registration_icp(
                    source=curr_scan_down_pts,
                    target=loop_pts_o3d,
                    max_correspondence_distance=10,
                    init=yawdeg2se3(-yaw_diff_deg),
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)
                )

                ## Check fitness score for valid loop closure
                self.get_logger().info(f"Checking fitness {icp_result.fitness}")
                
                ## Create latest submap point cloud
                self.get_logger().info(f"Submap cloud contains {len(curr_scan_down_pts.points)} points")
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = self.global_frame_id
                # points = np.asarray(curr_scan_down_pts.points)
                transformed_cloud = self.transform_scan_open3d(test_points, yawdeg2se3(-yaw_diff_deg))
                points = np.asarray(transformed_cloud.points)
                latest_cloud = point_cloud2.create_cloud_xyz32(header, points)
                self.latest_submap_pointcloud.publish(latest_cloud)
                
                ## Create loop submap point cloud
                self.get_logger().info(f"Submap cloud contains {len(loop_pts_o3d.points)} points")
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = self.global_frame_id
                points = np.asarray(loop_pts_o3d.points)
                loop_clouds = point_cloud2.create_cloud_xyz32(header, points)
                self.loop_submap_publisher.publish(loop_clouds)
                
                ### Add loop factor
                self.PGM.addLoopFactor(icp_result.transformation, loop_idx)
                
                ### Optimise graph
                self.PGM.optimizePoseGraph()
                
                self.get_logger().info(f"Graphs saved at frame {self.graph_frame_idx}")
                ## Save unoptimized graph first
                filename_unopt = "pose_unoptimized_" + str(self.PGM.curr_node_idx) + ".csv"
                filename_unopt = os.path.join(self.save_dir_pose, filename_unopt)
                savePoseGraph(self.PGM.curr_node_idx+1, self.PGM.graph_initials, filename_unopt) #+1 is to accomodate function

                ## Save optimized poses
                filename_opt = "pose_optimized_" + str(self.PGM.curr_node_idx) + ".csv"
                filename_opt = os.path.join(self.save_dir_pose, filename_opt)
                savePoseGraph(self.PGM.curr_node_idx+1, self.PGM.graph_optimized, filename_opt)
                
                ## Save loop closure
                self.ResultSaver.logLoopClosure(loop_idx, self.graph_frame_idx)
                
                ### Update map and path
                self.graph_frame_idx_opt = self.graph_frame_idx-1
                self.update_map_and_path()
        # ------------------------------------------------------------------------------------------
        
        self.graph_frame_idx += 1
        self.ResultSaver.filecount = self.graph_frame_idx
        self.get_logger().info(f"Processed frame {self.graph_frame_idx}")
    
    def update_map_and_path(self, save_map = False):
        
        # Optimised path creation
        self.Path_modified.poses.clear()
        self.Path_modified.header.frame_id = self.global_frame_id
        for i in range(self.graph_frame_idx_opt):
            T_optimized = self.PGM.get_optimised_transform(i)
            pose_stamped = self.Tmatrix_to_pose_stamped(T_optimized, frame_id=self.global_frame_id)
            self.Path_modified.header.stamp = pose_stamped.header.stamp
            self.save_pose_to_tum_from_pose(self.timestamp_list[i], pose_stamped.pose, self.optimised_tum_log_file)
            self.Path_modified.poses.append(pose_stamped)
            
        # Optimised map creation
        modified_map = o3d.geometry.PointCloud()
        for i in range(self.graph_frame_idx_opt):
            
            ## Sparse submaps for map construction
            if i % self.map_construct_gap != 0:
                continue
            
            ## Extract pose and point cloud for each submap
            # self.get_logger().info(f"Processing submap {i} for modified map and path")
            T_optimized = self.PGM.get_optimised_transform(i)
            pose_stamped = self.Tmatrix_to_pose_stamped(T_optimized, frame_id=self.global_frame_id)
            cloud_open3d = self.SCM.getPtcloud(i)
            
            ## Create optimised occupancy grid map
            if self.occupancy_grid_publish:
                self.get_logger().info(f"Updating optimised occupancy grid map: submap {i}")
                self.og_optimised_map.update_from_open3d(pose_stamped, cloud_open3d, z_threshold=self.z_threshold)
                
            ## Append map cloud
            transformed_cloud = self.transform_scan_open3d(cloud_open3d, T_optimized)
            modified_map += transformed_cloud

        ## Downsample the modified map
        self.get_logger().info(f"Modified map (Raw):  {len(modified_map.points)} points")
        modified_map = modified_map.voxel_down_sample(self.map_global_downsample_res)
        self.get_logger().info(f"Modified map (Downsampled): {len(modified_map.points)} points")
        
        ## Create map point cloud msg
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.global_frame_id
        points = np.asarray(modified_map.points)
        self.Map_modified = point_cloud2.create_cloud_xyz32(header, points)

        # Publish the modified map and modified path
        self.modified_path_pub.publish(self.Path_modified)
        self.modified_map.publish(self.Map_modified) 
        if self.occupancy_grid_publish:
            self.publish_optimised_occupancy_grid()
        
        # Check if map needs to be saved
        if (save_map):
            # Save point cloud
            o3d.io.write_point_cloud(f"{self.save_dir_map}/{self.test_name}_map_optimised.pcd", modified_map, write_ascii=True)
            self.get_logger().info(f"Saved map to {self.save_dir_map}/{self.test_name}_map_optimised.pcd")
            
            if self.occupancy_grid_publish:
                # Save unoptimised occupancy grid map
                self.og_unoptimised_map.save_map(f"{self.save_dir_map}/{self.test_name}_unoptimised_og_map")
                self.get_logger().info(f"Saved unoptimised occupancy grid map to {self.save_dir_map}/{self.test_name}_unoptimised_og_map")
                
                # Save optimised occupancy grid map
                self.og_optimised_map.save_map(f"{self.save_dir_map}/{self.test_name}_optimised_og_map")
                self.get_logger().info(f"Saved optimised occupancy grid map to {self.save_dir_map}/{self.test_name}_optimised_og_map")

            # # Save path as CSV
            # with open("path.csv", mode="w", newline="") as file:
            #     writer = csv.writer(file)
            #     writer.writerow(["index", "x", "y", "z", "qx", "qy", "qz", "qw"])
            #     for i, pose_stamped in enumerate(self.Path_modified.poses):
            #         pos = pose_stamped.pose.position
            #         ori = pose_stamped.pose.orientation
            #         writer.writerow([i, pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])
            # self.get_logger().info("Saved path to path.csv")

    
    # Utility functions
    def add_gps_factor(self):
        if not self.gps_queue:
            return

        if not self.cloud_key_poses_3d:
            return

        if np.linalg.norm(self.cloud_key_poses_3d[0] - self.cloud_key_poses_3d[-1]) < 5.0:
            return

        if self.pose_covariance[3, 3] < self.pose_cov_threshold and \
           self.pose_covariance[4, 4] < self.pose_cov_threshold:
            return

        while self.gps_queue:
            gps_msg = self.gps_queue[0]
            gps_time = gps_msg.header.stamp.sec + gps_msg.header.stamp.nanosec * 1e-9

            if gps_time < self.time_laser_info_cur - 0.2:
                self.gps_queue.popleft()
            elif gps_time > self.time_laser_info_cur + 0.2:
                break
            else:
                self.gps_queue.popleft()

                noise_x = gps_msg.pose.covariance[0]
                noise_y = gps_msg.pose.covariance[7]
                noise_z = gps_msg.pose.covariance[14]

                if noise_x > self.gps_cov_threshold or noise_y > self.gps_cov_threshold:
                    continue

                gps_x = gps_msg.pose.pose.position.x
                gps_y = gps_msg.pose.pose.position.y
                gps_z = gps_msg.pose.pose.position.z

                if not self.use_gps_elevation:
                    gps_z = self.transform_tobe_mapped[5]
                    noise_z = 0.01

                if abs(gps_x) < 1e-6 and abs(gps_y) < 1e-6:
                    continue

                cur_gps = np.array([gps_x, gps_y, gps_z])

                if np.linalg.norm(cur_gps - self.last_gps_point) < 5.0:
                    continue
                else:
                    self.last_gps_point = cur_gps

                # noise_vec = np.maximum([noise_x, noise_y, noise_z], 1.0)
                # gps_noise = noiseModel.Diagonal.Variances(noise_vec)
                # gps_factor = gtsam.GPSFactor(
                #     gtsam.symbol('x', len(self.cloud_key_poses_3d)),
                #     gtsam.Point3(*cur_gps),
                #     gps_noise
                # )
                # self.gt_sam_graph.add(gps_factor)
                # self.a_loop_is_closed = True
                break
            
    def publish_unoptimised_occupancy_grid(self):

        # Get probability map
        prob_map = self.og_unoptimised_map.get_probability_map().T
        
        # Convert probability map to integer occupancy values (-1 unknown, 0 free, 100 occupied)
        occupancy_values = np.clip((prob_map * 100).astype(int), 0, 100).flatten()
        
        # Create OccupancyGrid message
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.resolution = self.og_unoptimised_map.resolution
        msg.info.width = int(self.og_unoptimised_map.extents[0])
        msg.info.height = int(self.og_unoptimised_map.extents[1])
        origin_pose = Pose()
        origin_pose.position.x = -self.og_unoptimised_map.origin[0] * self.og_unoptimised_map.resolution
        origin_pose.position.y = -self.og_unoptimised_map.origin[1] * self.og_unoptimised_map.resolution
        msg.info.origin = origin_pose
        msg.data = occupancy_values.tolist()
        
        # Publish the occupancy grid
        self.unmodified_og_map_pub.publish(msg)
        self.get_logger().info("Published probability map")
    
    def publish_optimised_occupancy_grid(self):

        # Get probability map
        prob_map = self.og_optimised_map.get_probability_map().T

        # Convert probability map to integer occupancy values (-1 unknown, 0 free, 100 occupied)
        occupancy_values = np.clip((prob_map * 100).astype(int), 0, 100).flatten()
        
        # Create OccupancyGrid message
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
        
        # Publish the occupancy grid
        self.optimised_og_map_pub.publish(msg)
        self.get_logger().info("Published optimised probability map")

    def ros2_to_open3d(self, cloud_pcl2: PointCloud2):
        points = np.array([list(p)[:3] for p in point_cloud2.read_points(cloud_pcl2, skip_nans=True)])
        cloud_open3d = o3d.geometry.PointCloud()
        cloud_open3d.points = o3d.utility.Vector3dVector(points)
        
        # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcd, axis])
        
        # # Alternative conversion
        # Convert ROS PointCloud2 to a generator of (x, y, z)
        # points = point_cloud2.read_points(cloud_pcl2, field_names=("x", "y", "z"), skip_nans=True)
        
        # # Convert to numpy array
        # np_points = np.array(list(points), dtype=np.float32)

        # # Create Open3D point cloud
        # cloud_open3d = o3d.geometry.PointCloud()
        # cloud_open3d.points = o3d.utility.Vector3dVector(np_points)
    
        return cloud_open3d  
    
    def transform_scan_open3d(self, scan, transformation):
        if not isinstance(scan, o3d.geometry.PointCloud):
                raise TypeError("Input scan must be an Open3D PointCloud object.")

        points = np.asarray(scan.points)                                                # Convert Open3D PointCloud to NumPy array
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))         # Convert points to homogeneous coordinates (Nx4)
       
        transformed_points = (transformation @ homogeneous_points.T).T                  # Apply transformation matrix (4x4) to points (Nx4) 
        transformed_pcd = o3d.geometry.PointCloud()                                     # Create a new Open3D PointCloud object 
        transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points[:, :3])  # Keep only the 3D coordinates

        return transformed_pcd
    
    def Tmatrix_to_pose_stamped(self, matrix: np.array, frame_id="map") -> PoseStamped:
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = frame_id

        # Extract translation
        pose_stamped.pose.position.x = matrix[0, 3]
        pose_stamped.pose.position.y = matrix[1, 3]
        pose_stamped.pose.position.z = matrix[2, 3]

        # Extract rotation
        rotation = tf_transformations.quaternion_from_matrix(matrix)
        pose_stamped.pose.orientation.x = rotation[0]
        pose_stamped.pose.orientation.y = rotation[1]
        pose_stamped.pose.orientation.z = rotation[2]
        pose_stamped.pose.orientation.w = rotation[3]

        return pose_stamped

    def create_and_pub_map_cloud(self, map_array_msg: MapArray,  save_map = False):
            
        # Create unmodified map cloud
        self.get_logger().info("Creating unmodified map cloud")
        unmodified_map = o3d.geometry.PointCloud()
        submap_size = len(map_array_msg.submaps)
        for i in range(submap_size):
            
            position = map_array_msg.submaps[i].pose.position
            orientation = map_array_msg.submaps[i].pose.orientation
            trans = tf_transformations.translation_matrix([position.x, position.y, position.z])
            rot = tf_transformations.quaternion_matrix([orientation.x, orientation.y, orientation.z, orientation.w])
            T_unoptimised = np.dot(trans, rot) 
            
            distance = map_array_msg.submaps[i].distance
            cloud_pcl2 = map_array_msg.submaps[i].cloud
            cloud_open3d = self.ros2_to_open3d(cloud_pcl2)
            
            ## Create unmodified map cloud
            transformed_cloud = self.transform_scan_open3d(cloud_open3d, T_unoptimised)
            unmodified_map += transformed_cloud
            
        ## Create unmodified map point cloud
        self.get_logger().info(f"Unmodified map contains {len(unmodified_map.points)} points")
        unmodified_map = unmodified_map.voxel_down_sample(self.map_global_downsample_res)
        self.get_logger().info(f"Downsampled Unmodified map contains {len(unmodified_map.points)} points")
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.global_frame_id
        points = np.asarray(unmodified_map.points)
        self.Map_unmodified = point_cloud2.create_cloud_xyz32(header, points)
        
        self.unmodified_map.publish(self.Map_unmodified)
        
        if (save_map):
            # Save point cloud
            o3d.io.write_point_cloud(f"{self.save_dir_map}/{self.test_name}_map_unoptimised.pcd", unmodified_map, write_ascii=True)
            self.get_logger().info(f"Saved unoptimised map to {self.save_dir_map}/{self.test_name}_map_unoptimised.pcd")

    def create_and_pub_sc_unmodified_map(self, save_map = False):
        # Optimised path creation
        self.Path_modified.poses.clear()
        self.Path_modified.header.frame_id = self.global_frame_id
        for i in range(self.graph_frame_idx):
            T_optimized = self.PGM.get_unoptimised_transform(i)
            pose_stamped = self.Tmatrix_to_pose_stamped(T_optimized, frame_id=self.global_frame_id)
            self.Path_modified.header.stamp = pose_stamped.header.stamp
            self.Path_modified.poses.append(pose_stamped)
        
        # Optimised map creation
        unmodified_map = o3d.geometry.PointCloud()
        for i in range(self.graph_frame_idx):
            
            ## Sparse submaps for map construction
            if i % self.map_construct_gap != 0:
                continue
            
            ## Extract pose and point cloud for each submap
            self.get_logger().info(f"Processing submap {i} for modified map and path")
            T_optimized = self.PGM.get_unoptimised_transform(i)
            pose_stamped = self.Tmatrix_to_pose_stamped(T_optimized, frame_id=self.global_frame_id)
            cloud_open3d = self.SCM.getPtcloud(i)
            
            ## Append map cloud
            transformed_cloud = self.transform_scan_open3d(cloud_open3d, T_optimized)
            unmodified_map += transformed_cloud

        ## Downsample the unmodified map
        self.get_logger().info(f"Unmodified map (Raw):  {len(unmodified_map.points)} points")
        unmodified_map = unmodified_map.voxel_down_sample(self.map_global_downsample_res)
        self.get_logger().info(f"Unmodified map (Downsampled): {len(unmodified_map.points)} points")
        
        
        ## Create map point cloud msg
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.global_frame_id
        points = np.asarray(unmodified_map.points)
        self.Map_modified = point_cloud2.create_cloud_xyz32(header, points)

        # Publish the modified MapArray message and modified path
        #self.modified_path_pub.publish(self.Path_modified)
        self.unmodified_map.publish(self.Map_modified)
        
        if (save_map):
            # Save point cloud
            o3d.io.write_point_cloud(f"{self.save_dir_map}/{self.test_name}_map_unoptimised.pcd", unmodified_map, write_ascii=True)
            self.get_logger().info(f"Saved map to {self.save_dir_map}/{self.test_name}_map_unoptimised.pcd")

    def save_pose_to_tum_from_pose(self, timestamp, pose, filename):
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
        tx, ty, tz = pose.position.x, pose.position.y, pose.position.z
        
        # Convert roll, pitch, yaw to quaternion
        qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w

        # Write to file in TUM format
        with open(filename, 'a') as f:
            f.write(f"{time_sec:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

def main(args=None):
    rclpy.init(args=args)
    node = BackendNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown detected: Ctrl+C pressed.")
    finally:
        if node.initial_map_recieved == True and node.use_scan_context == False:
            node.get_logger().info("Performing pose adjustment before shutdown.")
            node.do_pose_adjustment(node.MapArray_msg, save_map=False)
            
        node.get_logger().info("Shutting down BackendNode.")
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
