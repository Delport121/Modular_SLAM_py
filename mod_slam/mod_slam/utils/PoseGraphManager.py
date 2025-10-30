import numpy as np
import gtsam
# from utils.UtilsMisc import *
from lidar_slam_3d.utils.UtilsMisc import *

np.set_printoptions(precision=4)
    
class PoseGraphManager:
    def __init__(self, prior_covariance=None, odom_covariance=None, loop_covariance=None):

        self.prior_cov = gtsam.noiseModel.Diagonal.Sigmas(prior_covariance if prior_covariance is not None else np.array([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]))
        self.odom_cov = gtsam.noiseModel.Diagonal.Sigmas(odom_covariance if odom_covariance is not None else np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1]))
        self.loop_cov = gtsam.noiseModel.Diagonal.Sigmas(loop_covariance if loop_covariance is not None else np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1]))

        self.graph_factors = gtsam.NonlinearFactorGraph()
        self.graph_initials = gtsam.Values()

        self.opt_param = gtsam.LevenbergMarquardtParams()
        self.opt = gtsam.LevenbergMarquardtOptimizer(self.graph_factors, self.graph_initials, self.opt_param)

        self.curr_se3 = None
        self.curr_node_idx = None
        self.prev_node_idx = None

        self.graph_optimized = None
        self.odomfactor_added = False

    def addPriorFactor(self):
        self.curr_node_idx = 0
        self.prev_node_idx = 0

        self.curr_se3 = np.eye(4)

        self.graph_initials.insert(gtsam.symbol('x', self.curr_node_idx), gtsam.Pose3(self.curr_se3))
        self.graph_factors.add(gtsam.PriorFactorPose3(
                                                gtsam.symbol('x', self.curr_node_idx), 
                                                gtsam.Pose3(self.curr_se3), 
                                                self.prior_cov))

    def addOdometryFactor(self, odom_transform):

        self.graph_initials.insert(gtsam.symbol('x', self.curr_node_idx), gtsam.Pose3(self.curr_se3))
        self.graph_factors.add(gtsam.BetweenFactorPose3(
                                                gtsam.symbol('x', self.prev_node_idx), 
                                                gtsam.symbol('x', self.curr_node_idx), 
                                                gtsam.Pose3(odom_transform), 
                                                self.odom_cov))
        print("Added odometry factor between node {} and node {}".format(self.prev_node_idx, self.curr_node_idx))
        
    def addOdometry_initial(self):
        self.graph_initials.insert(gtsam.symbol('x', self.curr_node_idx), gtsam.Pose3(self.curr_se3))
        
    def addOdometryFactor_adv(self, odom_transform, prev_node_idx):

        # For advance insertion, this must be commented out and addOdometry_initial must be used
        # self.graph_initials.insert(gtsam.symbol('x', self.curr_node_idx), gtsam.Pose3(self.curr_se3)) 
        self.graph_factors.add(gtsam.BetweenFactorPose3(
                                                gtsam.symbol('x', prev_node_idx), 
                                                gtsam.symbol('x', self.curr_node_idx), 
                                                gtsam.Pose3(odom_transform), 
                                                self.odom_cov))
        #print("Added odometry factor between node {} and node {}".format(prev_node_idx, self.curr_node_idx))

    def addLoopFactor(self, loop_transform, loop_idx):

        self.graph_factors.add(gtsam.BetweenFactorPose3(
                                        gtsam.symbol('x', loop_idx), 
                                        gtsam.symbol('x', self.curr_node_idx), 
                                        gtsam.Pose3(loop_transform), 
                                        self.odom_cov))
    
    
    def addLoopFactor_adv(self, loop_transform, loop_idx, frame_idx):

        self.graph_factors.add(gtsam.BetweenFactorPose3(
                                        gtsam.symbol('x', loop_idx), 
                                        gtsam.symbol('x', frame_idx), 
                                        gtsam.Pose3(loop_transform), 
                                        self.odom_cov))

    def optimizePoseGraph(self):

        self.opt = gtsam.LevenbergMarquardtOptimizer(self.graph_factors, self.graph_initials, self.opt_param)
        self.graph_optimized = self.opt.optimize()

        # status = self.opt.optimize(self.graph_factors, self.graph_initials, self.graph_optimized)
        # if status != minisam.NonlinearOptimizationStatus.SUCCESS:
            # print("optimization error: ", status)

        # correct current pose 
        pose_trans, pose_rot = getGraphNodePose(self.graph_optimized, self.curr_node_idx)
        # self.curr_se3[:3, :3] = pose_rot
        # self.curr_se3[:3, 3] = pose_trans
        
        new_se3 = self.curr_se3.copy()
        new_se3[:3, :3] = pose_rot
        new_se3[:3, 3] = pose_trans
        self.curr_se3 = new_se3

        
    def optimizePoseGraph2(self):

        self.opt = gtsam.LevenbergMarquardtOptimizer(self.graph_factors, self.graph_initials, self.opt_param)
        self.graph_optimized = self.opt.optimize()

        # status = self.opt.optimize(self.graph_factors, self.graph_initials, self.graph_optimized)
        # if status != minisam.NonlinearOptimizationStatus.SUCCESS:
            # print("optimization error: ", status)

        # # correct current pose 
        # while not self.odomfactor_added:
        #     pass  # Busy-wait until odomfactor_added is True
        # pose_trans, pose_rot = getGraphNodePose(self.graph_optimized, self.curr_node_idx-1)
        # self.curr_se3[:3, :3] = pose_rot
        # self.curr_se3[:3, 3] = pose_trans
    
    def get_optimised_transform(self, node_idx):
        if node_idx < 0 or node_idx >= self.graph_optimized.size():
            raise IndexError(f"Node index {node_idx} is out of bounds for the optimized graph.")
        
        pose = self.graph_optimized.atPose3(gtsam.symbol('x', node_idx))
        return pose.matrix()
    
    def get_unoptimised_transform(self, node_idx):
        if node_idx < 0 or node_idx >= self.graph_initials.size():
            raise IndexError(f"Node index {node_idx} is out of bounds for the unoptimized graph.")
        
        pose = self.graph_initials.atPose3(gtsam.symbol('x', node_idx))
        return pose.matrix()
        
        