import numpy as np
np.set_printoptions(precision=4)

import time
from scipy import spatial

def xy2theta(x, y):
    if x == 0: x = 0.001
    if y == 0: y = 0.001

    if x >= 0 and y >= 0: 
        theta = 180/np.pi * np.arctan(y/x)
    elif x < 0 and y >= 0: 
        theta = 180 - (180/np.pi) * np.arctan(y/(-x))
    elif x < 0 and y < 0: 
        theta = 180 + (180/np.pi) * np.arctan(y/x)
    else:  # x >= 0 and y < 0
        theta = 360 - (180/np.pi) * np.arctan((-y)/x)
    return theta


def pt2rs(point, gap_ring, gap_sector, num_ring, num_sector):
    x = point[0]
    y = point[1]
    
    theta = xy2theta(x, y)
    faraway = np.sqrt(x*x + y*y)
    
    idx_ring = np.divmod(faraway, gap_ring)[0]       
    idx_sector = np.divmod(theta, gap_sector)[0]

    if idx_ring >= num_ring:
        idx_ring = num_ring - 1
    
    return int(idx_ring), int(idx_sector)


def ptcloud2sc(ptcloud, sc_shape, max_length):
    num_ring, num_sector = sc_shape

    gap_ring = max_length / num_ring
    gap_sector = 360 / num_sector
    
    sc = np.zeros([num_ring, num_sector])

    for pt in ptcloud:
        idx_ring, idx_sector = pt2rs(pt, gap_ring, gap_sector, num_ring, num_sector)
        sc[idx_ring, idx_sector] = 1  # mark presence

    return sc


def sc2rk(sc):
    return np.sum(sc, axis=1)


def distance_sc(sc1, sc2):
    num_sectors = sc1.shape[1]
    sim_for_each_cols = np.zeros(num_sectors)
    
    for i in range(num_sectors):
        sc1_shifted = np.roll(sc1, i, axis=1)

        sim = 0
        count = 0
        for j in range(num_sectors):
            col1 = sc1_shifted[:, j]
            col2 = sc2[:, j]
            if np.any(col1) and np.any(col2):
                dot = np.dot(col1, col2)
                norm_product = np.linalg.norm(col1) * np.linalg.norm(col2)
                if norm_product > 0:
                    sim += dot / norm_product
                    count += 1

        sim_for_each_cols[i] = sim / count if count > 0 else 0

    yaw_diff = np.argmax(sim_for_each_cols)
    sim = np.max(sim_for_each_cols)
    dist = 1 - sim

    return dist, yaw_diff


class ScanContextManager:
    def __init__(self, shape=[20, 60], num_candidates=10, threshold=0.15, pointcloud_radius = 8, recent_node_exclusion=50): 
        self.shape = shape
        self.num_candidates = num_candidates
        self.threshold = threshold
        self.max_length = pointcloud_radius
        self.recent_node_exclusion = recent_node_exclusion
        self.nn_dist = None

        self.ENOUGH_LARGE = 15000
        self.ptclouds = [None] * self.ENOUGH_LARGE
        self.scancontexts = [None] * self.ENOUGH_LARGE
        self.ringkeys = [None] * self.ENOUGH_LARGE

        self.curr_node_idx = 0

    def addNode(self, node_idx, ptcloud):
        sc = ptcloud2sc(ptcloud, self.shape, self.max_length)
        rk = sc2rk(sc)

        self.curr_node_idx = node_idx
        self.ptclouds[node_idx] = ptcloud
        self.scancontexts[node_idx] = sc
        self.ringkeys[node_idx] = rk

    def getPtcloud(self, node_idx):
        return self.ptclouds[node_idx]

    def detectLoop(self):        
        valid_recent_node_idx = self.curr_node_idx - self.recent_node_exclusion

        if valid_recent_node_idx < 1:
            return None, None, None
        else:
            ringkey_history = np.array(self.ringkeys[:valid_recent_node_idx])
            ringkey_tree = spatial.KDTree(ringkey_history)

            ringkey_query = self.ringkeys[self.curr_node_idx]
            _, nncandidates_idx = ringkey_tree.query(ringkey_query, k=self.num_candidates)

            query_sc = self.scancontexts[self.curr_node_idx]

            nn_dist = 1.0
            nn_idx = None
            nn_yawdiff = None

            for ith in range(self.num_candidates):
                candidate_idx = nncandidates_idx[ith]
                candidate_sc = self.scancontexts[candidate_idx]
                dist, yaw_diff = distance_sc(candidate_sc, query_sc)
                if dist < nn_dist:
                    nn_dist = dist
                    nn_yawdiff = yaw_diff
                    nn_idx = candidate_idx

            if nn_dist < self.threshold:
                nn_yawdiff_deg = nn_yawdiff * (360 / self.shape[1])
                return nn_idx, nn_dist, nn_yawdiff_deg
            else:
                return None, None, None
