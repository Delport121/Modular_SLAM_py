import os 
import csv
import copy
import time
import math

import numpy as np
import matplotlib.pyplot as plt

import gtsam

def getConstDigitsNumber(val, num_digits):
    return "{:.{}f}".format(val, num_digits)

def getUnixTime():
    return int(time.time())

# def eulerAnglesToRotationMatrix(theta) :
     
#     R_x = np.array([[1,         0,                  0                   ],
#                     [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
#                     [0,         math.sin(theta[0]), math.cos(theta[0])  ]
#                     ])
                     
#     R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
#                     [0,                     1,      0                   ],
#                     [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
#                     ])
                 
#     R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
#                     [math.sin(theta[2]),    math.cos(theta[2]),     0],
#                     [0,                     0,                      1]
#                     ])
                     
#     R = np.dot(R_z, np.dot( R_y, R_x ))
 
#     return R

# def yawdeg2so3(yaw_deg):
#     yaw_rad = np.deg2rad(yaw_deg)
#     return eulerAnglesToRotationMatrix([0, 0, yaw_rad])

# def yawdeg2se3(yaw_deg):
#     se3 = np.eye(4)
#     se3[:3, :3] = yawdeg2so3(yaw_deg)
#     return se3 


# def getGraphNodePose(graph, idx):

#     pose = graph.atPose3(gtsam.symbol('x', idx))
#     pose_trans = np.array([pose.x(), pose.y(), pose.z()])
#     pose_rot = pose.rotation().matrix()

#     return pose_trans, pose_rot

# def saveOptimizedGraphPose(curr_node_idx, graph_optimized, filename):

#     for opt_idx in range(curr_node_idx):
#         pose_trans, pose_rot = getGraphNodePose(graph_optimized, opt_idx)
#         pose_trans = np.reshape(pose_trans, (-1, 3)).squeeze()
#         pose_rot = np.reshape(pose_rot, (-1, 9)).squeeze()
#         optimized_pose_ith = np.array([ pose_rot[0], pose_rot[1], pose_rot[2], pose_trans[0], 
#                                         pose_rot[3], pose_rot[4], pose_rot[5], pose_trans[1], 
#                                         pose_rot[6], pose_rot[7], pose_rot[8], pose_trans[2],
#                                         0.0, 0.0, 0.0, 0.1 ])
#         if(opt_idx == 0):
#             optimized_pose_list = optimized_pose_ith
#         else:
#             optimized_pose_list = np.vstack((optimized_pose_list, optimized_pose_ith))

#     np.savetxt(filename, optimized_pose_list, delimiter=",")


# class PoseGraphResultSaver:
#     def __init__(self, init_pose, save_gap, num_frames, seq_idx, save_dir):
#         self.pose_list = np.reshape(init_pose, (-1, 16))
#         self.save_gap = save_gap
#         self.num_frames = num_frames

#         self.seq_idx = seq_idx
#         self.save_dir = save_dir

#     def saveUnoptimizedPoseGraphResult(self, cur_pose, cur_node_idx):
#         # save 
#         self.pose_list = np.vstack((self.pose_list, np.reshape(cur_pose, (-1, 16))))

#         # write
#         if(cur_node_idx % self.save_gap == 0 or cur_node_idx == self.num_frames):        
#             # save odometry-only poses
#             filename = "pose" + self.seq_idx + "unoptimized_" + str(getUnixTime()) + ".csv"
#             filename = os.path.join(self.save_dir, filename)
#             np.savetxt(filename, self.pose_list, delimiter=",")

#     def saveOptimizedPoseGraphResult(self, cur_node_idx, graph_optimized):
#         filename = "pose" + self.seq_idx + "optimized_" + str(getUnixTime()) + ".csv"
#         filename = os.path.join(self.save_dir, filename)
#         saveOptimizedGraphPose(cur_node_idx, graph_optimized, filename)    
    
#         optimized_pose_list = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
#         self.pose_list = optimized_pose_list # update with optimized pose 

#     def vizCurrentTrajectory(self, fig_idx):
#         x = self.pose_list[:,3]
#         y = self.pose_list[:,7]
#         z = self.pose_list[:,11]

#         fig = plt.figure(fig_idx)
#         plt.clf()
#         plt.plot(-y, x, color='blue') # kitti camera coord for clarity
#         plt.axis('equal')
#         plt.xlabel('x', labelpad=10)
#         plt.ylabel('y', labelpad=10)
#         plt.draw()
#         plt.pause(0.01) #is necessary for the plot to update for some reason


def eulerAngleToRotationMatrix2D(theta):
    """Returns 2D rotation matrix for yaw angle theta (in radians)"""
    return np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)]
    ])

def yawdeg2so2(yaw_deg):
    yaw_rad = np.deg2rad(yaw_deg)
    return eulerAngleToRotationMatrix2D(yaw_rad)

def yawdeg2se2(yaw_deg):
    yaw_rad = np.deg2rad(yaw_deg)
    se2 = np.eye(3)
    se2[:2, :2] = eulerAngleToRotationMatrix2D(yaw_rad)
    return se2

def getGraphNodePose2D(graph, idx):
    """Assuming gtsam.Pose2 is used in 2D"""
    pose = graph.atPose2(gtsam.symbol('x', idx))
    pose_trans = np.array([pose.x(), pose.y()])
    pose_rot = pose.theta()  # in radians
    return pose_trans, pose_rot

def saveOptimizedGraphPose2D(curr_node_idx, graph_optimized, filename):
    for opt_idx in range(curr_node_idx):
        pose_trans, pose_theta = getGraphNodePose2D(graph_optimized, opt_idx)
        cos_t = math.cos(pose_theta)
        sin_t = math.sin(pose_theta)

        optimized_pose_ith = np.array([
            cos_t, -sin_t, pose_trans[0],
            sin_t,  cos_t, pose_trans[1],
            0.0,    0.0,   0.1
        ])

        if opt_idx == 0:
            optimized_pose_list = optimized_pose_ith
        else:
            optimized_pose_list = np.vstack((optimized_pose_list, optimized_pose_ith))

    np.savetxt(filename, optimized_pose_list, delimiter=",")

class PoseGraphResultSaver2D:
        def __init__(self, init_pose, save_gap, num_frames, seq_idx, save_dir):
            self.pose_list = np.reshape(init_pose, (-1, 9))  # 2D pose matrix flattened
            self.save_gap = save_gap
            self.num_frames = num_frames
            self.seq_idx = seq_idx
            self.save_dir = save_dir
            self.filecount = 0
            self.loop_closure_log = os.path.join(self.save_dir, f"loop_closures_{self.seq_idx}.csv")
            # Initialize the loop closure log file with headers
            with open(self.loop_closure_log, 'w') as f:
                f.write("Index1,Index2\n")

        def saveUnoptimizedPoseGraphResult(self, cur_pose, cur_node_idx):
            self.pose_list = np.vstack((self.pose_list, np.reshape(cur_pose, (-1, 9))))
            if (cur_node_idx % self.save_gap == 0 or cur_node_idx == self.num_frames):
                filename = "pose" + self.seq_idx + "unoptimized_" + str(self.filecount) + ".csv"
                filename = os.path.join(self.save_dir, filename)
                np.savetxt(filename, self.pose_list, delimiter=",")
        
        def saveUnoptimizedPoseGraphResult_forced(self, cur_pose, cur_node_idx):
            self.pose_list = np.vstack((self.pose_list, np.reshape(cur_pose, (-1, 9))))
            filename = "pose" + self.seq_idx + "unoptimized_" + str(self.filecount) + ".csv"
            filename = os.path.join(self.save_dir, filename)
            np.savetxt(filename, self.pose_list, delimiter=",")

        def saveOptimizedPoseGraphResult(self, cur_node_idx, graph_optimized):
            filename = "pose" + self.seq_idx + "optimized_" + str(self.filecount) + ".csv"
            filename = os.path.join(self.save_dir, filename)
            saveOptimizedGraphPose2D(cur_node_idx, graph_optimized, filename)

            optimized_pose_list = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
            self.pose_list = optimized_pose_list

        def logLoopClosure(self, index1, index2):
            """Logs the two indexes for which a loop closure was detected."""
            with open(self.loop_closure_log, 'a') as f:
                f.write(f"{index1},{index2}\n")

        def vizCurrentTrajectory(self, fig_idx):
            x = self.pose_list[:, 2]
            y = self.pose_list[:, 5]

            # fig = plt.figure(fig_idx)
            plt.clf()
            plt.plot(x, y, color='blue')
            plt.axis('equal')
            plt.xlabel('x', labelpad=10)
            plt.ylabel('y', labelpad=10)
            plt.draw()
            plt.pause(0.01)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
class ScanContextResultSaver:
    def __init__(self, save_gap, save_dir):
        self.save_gap = save_gap
        self.save_dir = save_dir
        self.sc_dir = os.path.join(save_dir, "scan_contexts")
        self.rk_dir = os.path.join(save_dir, "ring_keys")
        ensure_dir(self.sc_dir)
        ensure_dir(self.rk_dir)

    def saveScanContext(self, node_idx, scan_context):
        if node_idx % self.save_gap == 0:
            filename = f"sc_{node_idx:04d}.npy"
            filepath = os.path.join(self.sc_dir, filename)
            np.save(filepath, scan_context)

    def saveRingKey(self, node_idx, ring_key):
        if node_idx % self.save_gap == 0:
            filename = f"rk_{node_idx:04d}.npy"
            filepath = os.path.join(self.rk_dir, filename)
            np.save(filepath, ring_key)

    def save(self, node_idx, scan_context, ring_key):
        self.saveScanContext(node_idx, scan_context)
        self.saveRingKey(node_idx, ring_key)

    def saveScanContextSVG(self, scancontext, node_idx):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))
        theta = np.linspace(0, 2 * np.pi, scancontext.shape[1])
        r = np.arange(scancontext.shape[0])
        theta_grid, r_grid = np.meshgrid(theta, r)
        cax = ax.pcolormesh(theta_grid, r_grid, scancontext, cmap='viridis', shading='auto')
        ax.set_title(f"ScanContext #{node_idx}")
        fig.colorbar(cax, ax=ax, orientation='vertical')
        plt.tight_layout()
        sc_path = os.path.join(self.sc_dir, f"scancontext_{node_idx:05d}.svg")
        fig.savefig(sc_path, format='svg')
        plt.close(fig)

    def saveRingKeySVG(self, ringkey, node_idx):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(ringkey, marker='o', linestyle='-', color='darkorange')
        ax.set_title(f"RingKey #{node_idx}")
        ax.set_xlabel("Ring Index")
        ax.set_ylabel("Mean Height")
        ax.grid(True)
        plt.tight_layout()
        rk_path = os.path.join(self.rk_dir, f"ringkey_{node_idx:05d}.svg")
        fig.savefig(rk_path, format='svg')
        plt.close(fig)
    
    def saveScanContextHeatmapSVG(self, scancontext, node_idx):
        """
        Save the scan context as a heatmap SVG with a colorbar.
        
        Parameters:
        - scancontext: 2D numpy array representing the scan context
        - node_idx: Index number used for naming the output file
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(scancontext, cmap='viridis', aspect='auto')
        ax.set_title(f"ScanContext Heatmap #{node_idx}")
        plt.colorbar(cax, ax=ax)
        plt.tight_layout()
        
        sc_path = os.path.join(self.sc_dir, f"scancontext_heatmap_{node_idx:05d}.svg")
        plt.savefig(sc_path, format='svg')
        plt.close()
