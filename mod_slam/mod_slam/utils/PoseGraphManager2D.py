import numpy as np
np.set_printoptions(precision=4)

import gtsam
from utils.UtilsMisc2D import *

class PoseGraphManager:
    def __init__(self, prior_covariance=None, odom_covariance=None, loop_covariance=None):
        # Noise models for x, y, theta
        self.prior_cov = gtsam.noiseModel.Diagonal.Sigmas(prior_covariance if prior_covariance is not None else np.array([1e-6, 1e-6, 1e-6]))
        self.odom_cov = gtsam.noiseModel.Diagonal.Sigmas(odom_covariance if odom_covariance is not None else np.array([0.5, 0.5, 0.1]))
        self.loop_cov = gtsam.noiseModel.Diagonal.Sigmas(loop_covariance if loop_covariance is not None else np.array([0.5, 0.5, 0.1]))

        # Graph structures
        self.graph_factors = gtsam.NonlinearFactorGraph()
        self.graph_initials = gtsam.Values()

        # Optimization parameters
        self.opt_param = gtsam.LevenbergMarquardtParams()
        self.opt = gtsam.LevenbergMarquardtOptimizer(self.graph_factors, self.graph_initials, self.opt_param)

        # State tracking
        self.curr_se2 = np.eye(3)  # In 2D, this is the SE2 homogeneous transform (3x3)
        self.curr_node_idx = None
        self.prev_node_idx = None

        self.graph_optimized = None

    def addPriorFactor(self):
        self.curr_node_idx = 0
        self.prev_node_idx = 0

        self.curr_se2 = np.eye(3)

        pose2 = gtsam.Pose2(self.curr_se2[0, 2], self.curr_se2[1, 2], np.arctan2(self.curr_se2[1, 0], self.curr_se2[0, 0]))

        self.graph_initials.insert(gtsam.symbol('x', self.curr_node_idx), pose2)
        self.graph_factors.add(gtsam.PriorFactorPose2(
            gtsam.symbol('x', self.curr_node_idx),
            pose2,
            self.prior_cov))

    def addOdometryFactor(self, odom_transform):

        pose2 = gtsam.Pose2(self.curr_se2[0, 2], self.curr_se2[1, 2], np.arctan2(self.curr_se2[1, 0], self.curr_se2[0, 0]))
        rel_pose2 = gtsam.Pose2(odom_transform[0, 2], odom_transform[1, 2], np.arctan2(odom_transform[1, 0], odom_transform[0, 0]))

        self.graph_initials.insert(gtsam.symbol('x', self.curr_node_idx), pose2)
        self.graph_factors.add(gtsam.BetweenFactorPose2(
            gtsam.symbol('x', self.prev_node_idx),
            gtsam.symbol('x', self.curr_node_idx),
            rel_pose2,
            self.odom_cov))

    def addLoopFactor(self, loop_transform, loop_idx):

        rel_pose2 = gtsam.Pose2(loop_transform[0, 2], loop_transform[1, 2], np.arctan2(loop_transform[1, 0], loop_transform[0, 0]))

        self.graph_factors.add(gtsam.BetweenFactorPose2(
            gtsam.symbol('x', loop_idx),
            gtsam.symbol('x', self.curr_node_idx),
            rel_pose2,
            self.loop_cov))

    def optimizePoseGraph(self):
        
        self.opt = gtsam.LevenbergMarquardtOptimizer(self.graph_factors, self.graph_initials, self.opt_param)
        self.graph_optimized = self.opt.optimize()

        # Update current pose using optimized estimate
        pose_opt = self.graph_optimized.atPose2(gtsam.symbol('x', self.curr_node_idx))

        # Update SE2 matrix (3x3 homogeneous transform)
        x = pose_opt.x()
        y = pose_opt.y()
        theta = pose_opt.theta()
        self.curr_se2 = np.array([
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta),  np.cos(theta), y],
            [0,              0,             1]
        ])
        
    def get_optimized_pose_transform_at_node(self, node_idx):
        """Get the pose at a specific node index"""
        pose_opt = self.graph_optimized.atPose2(gtsam.symbol('x', node_idx))
        x = pose_opt.x()
        y = pose_opt.y()
        theta = pose_opt.theta()
        return np.array([
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta),  np.cos(theta), y],
            [0,              0,             1]
        ])

