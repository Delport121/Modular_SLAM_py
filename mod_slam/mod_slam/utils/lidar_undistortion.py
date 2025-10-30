# import numpy as np
# import open3d as o3d
# from scipy.spatial.transform import Rotation

# class LidarUndistortion:
#     def __init__(self, scan_period=0.125, imu_que_length=200):
#         self.scan_period = scan_period
#         self.imu_que_length = imu_que_length
#         self.imu_ptr_front = 0
#         self.imu_ptr_last = -1
#         self.imu_ptr_last_iter = 0

#         self.imu_time = np.zeros(imu_que_length)
#         self.imu_roll = np.zeros(imu_que_length)
#         self.imu_pitch = np.zeros(imu_que_length)
#         self.imu_yaw = np.zeros(imu_que_length)
#         self.imu_acc = np.zeros((imu_que_length, 3))
#         self.imu_velo = np.zeros((imu_que_length, 3))
#         self.imu_shift = np.zeros((imu_que_length, 3))
#         self.imu_angular_velo = np.zeros((imu_que_length, 3))
#         self.imu_angular_rot = np.zeros((imu_que_length, 3))

#     def set_scan_period(self, period):
#         self.scan_period = period

#     def get_imu(self, angular_velo, acc, quat, imu_time):
#         # quat should be [x, y, z, w]
#         r = Rotation.from_quat(quat)
#         roll, pitch, yaw = r.as_euler('xyz')

#         # Update IMU buffer pointers
#         self.imu_ptr_last = (self.imu_ptr_last + 1) % self.imu_que_length
#         if (self.imu_ptr_last + 1) % self.imu_que_length == self.imu_ptr_front:
#             self.imu_ptr_front = (self.imu_ptr_front + 1) % self.imu_que_length

#         # Store IMU data in buffers
#         i = self.imu_ptr_last
#         self.imu_time[i] = imu_time
#         self.imu_roll[i] = roll
#         self.imu_pitch[i] = pitch
#         self.imu_yaw[i] = yaw
#         self.imu_acc[i] = acc
#         self.imu_angular_velo[i] = angular_velo

#         # Transform IMU acceleration to world frame
#         acc_world = r.as_matrix() @ acc
        
#         # Time delta calculation
#         imu_ptr_back = (i - 1 + self.imu_que_length) % self.imu_que_length
#         dt = self.imu_time[i] - self.imu_time[imu_ptr_back]

#         # Integrate to update shift, velocity and rotation
#         if dt < self.scan_period:
#             self.imu_shift[i] = self.imu_shift[imu_ptr_back] + self.imu_velo[imu_ptr_back] * dt + 0.5 * acc_world * dt**2
#             self.imu_velo[i] = self.imu_velo[imu_ptr_back] + acc_world * dt
#             self.imu_angular_rot[i] = self.imu_angular_rot[imu_ptr_back] + angular_velo * dt

#     def adjust_distortion(self, cloud: o3d.geometry.PointCloud, scan_time: float):
#         points = np.asarray(cloud.points)
#         if len(points) == 0:
#             return

#         # Angle calculation
#         start_ori = -np.arctan2(points[0][1], points[0][0])
#         end_ori = -np.arctan2(points[-1][1], points[-1][0])
#         if end_ori - start_ori > 3 * np.pi:
#             end_ori -= 2 * np.pi
#         elif end_ori - start_ori < np.pi:
#             end_ori += 2 * np.pi
#         ori_diff = end_ori - start_ori

#         # Iterate over each point
#         half_passed = False
#         shift_from_start = np.zeros(3)
#         shift_start = np.zeros(3)
#         velo_start = np.zeros(3)
#         r_s_i = np.eye(3)  # Initial rotation matrix for the first point
#         for i in range(points.shape[0]):
            
#             px, py, pz = points[i]
#             ori_h = -np.arctan2(py, px) # Get point angle in horizontal plane

#             # Estimate when each point was captured 
#             if not half_passed:
#                 if ori_h < start_ori - 0.5 * np.pi:
#                     ori_h += 2 * np.pi
#                 elif ori_h > start_ori + 1.5 * np.pi:
#                     ori_h -= 2 * np.pi
#                 if ori_h - start_ori > np.pi:
#                     half_passed = True
#             else:
#                 ori_h += 2 * np.pi
#                 if ori_h < end_ori - 1.5 * np.pi:
#                     ori_h += 2 * np.pi
#                 elif ori_h > end_ori + 0.5 * np.pi:
#                     ori_h -= 2 * np.pi

#             rel_time = (ori_h - start_ori) / ori_diff * self.scan_period
#             t_point = scan_time + rel_time #?????

#             # Find or interpolate IMU pose at Point's time (Searches in the IMU buffer for nearest two timestamp)
#             if self.imu_ptr_last > 0:
#                 imu_front = self.imu_ptr_last_iter
#                 while imu_front != self.imu_ptr_last:
#                     if t_point < self.imu_time[imu_front]:
#                         break
#                     imu_front = (imu_front + 1) % self.imu_que_length

#                 if abs(t_point - self.imu_time[imu_front]) > self.scan_period:
#                     continue
                
#                 # #???????
#                 # imu_back = (imu_front - 1 + self.imu_que_length) % self.imu_que_length
#                 # ratio_front = (t_point - self.imu_time[imu_back]) / (self.imu_time[imu_front] - self.imu_time[imu_back])
#                 # ratio_back = 1.0 - ratio_front

#                 # rpy_cur = self.imu_roll[imu_front] * ratio_front + self.imu_roll[imu_back] * ratio_back, \
#                 #       self.imu_pitch[imu_front] * ratio_front + self.imu_pitch[imu_back] * ratio_back, \
#                 #       self.imu_yaw[imu_front] * ratio_front + self.imu_yaw[imu_back] * ratio_back
#                 # shift_cur = self.imu_shift[imu_front] * ratio_front + self.imu_shift[imu_back] * ratio_back
#                 # velo_cur = self.imu_velo[imu_front] * ratio_front + self.imu_velo[imu_back] * ratio_back
                
#                 if t_point > self.imu_time[imu_front]:
#                     rpy_cur = (
#                         self.imu_roll[imu_front],
#                         self.imu_pitch[imu_front],
#                         self.imu_yaw[imu_front]
#                     )
#                     shift_cur = self.imu_shift[imu_front]
#                     velo_cur = self.imu_velo[imu_front]
#                 else:
#                     imu_back = (imu_front - 1 + self.imu_que_length) % self.imu_que_length
#                     ratio_front = (t_point - self.imu_time[imu_back]) / (self.imu_time[imu_front] - self.imu_time[imu_back])
#                     ratio_back = 1.0 - ratio_front

#                     rpy_cur = (
#                         self.imu_roll[imu_front] * ratio_front + self.imu_roll[imu_back] * ratio_back,
#                         self.imu_pitch[imu_front] * ratio_front + self.imu_pitch[imu_back] * ratio_back,
#                         self.imu_yaw[imu_front] * ratio_front + self.imu_yaw[imu_back] * ratio_back
#                     )
#                     shift_cur = self.imu_shift[imu_front] * ratio_front + self.imu_shift[imu_back] * ratio_back
#                     velo_cur = self.imu_velo[imu_front] * ratio_front + self.imu_velo[imu_back] * ratio_back


#                 # Reconstruct rotaion matrix from roll, pitch, yaw
#                 r_c = Rotation.from_euler('xyz', rpy_cur).as_matrix()

#                 # Apply motion compensation
#                 if i == 0:
#                     shift_start = shift_cur
#                     velo_start = velo_cur
#                     r_s_i = np.linalg.inv(r_c)   # Transpose = inverse for rotation matrices
#                 else:
#                     shift_from_start = shift_cur - shift_start - velo_start * rel_time
#                     points[i] = r_s_i @ (r_c @ points[i] + shift_from_start)

#                 self.imu_ptr_last_iter = imu_front
                
                
#         print(f"[LidarUndistortion] Adjusted {points.shape[0]} points with scan time {scan_time:.3f}s")
#         cloud.points = o3d.utility.Vector3dVector(points)

import numpy as np
import open3d as o3d
from typing import Optional
from scipy.spatial.transform import Rotation, Slerp


class LidarUndistortion:
    """
    Drop-in replacement with:
      - Quaternion storage + SLERP orientation interpolation
      - Vectorized point deskew
      - Safe ring-buffer linearization
      - Optional gravity/bias compensation
      - Optional LiDAR->IMU extrinsics (R_li, t_li)

    Notes:
      * By default, gravity compensation is OFF to match your original behavior.
      * If your accelerometer reports *specific force* (common), set compensate_gravity=True.
    """

    def __init__(
        self,
        scan_period: float = 0.125,
        imu_que_length: int = 200,
        compensate_gravity: bool = False,
        g_world: np.ndarray = np.array([0.0, 0.0, -9.81], dtype=np.float64),
        acc_bias: Optional[np.ndarray] = None,
        gyro_bias: Optional[np.ndarray] = None,
        R_li: Optional[np.ndarray] = None,     # 3x3 LiDAR->IMU rotation
        t_li: Optional[np.ndarray] = None,     # 3   LiDAR->IMU translation
        max_dt_factor: float = 5.0             # ignore dt > max_dt_factor * scan_period
    ):
        self.scan_period = float(scan_period)
        self.imu_que_length = int(imu_que_length)

        # Pointers for ring buffer
        self.imu_ptr_front = 0
        self.imu_ptr_last = -1

        # Buffers (float64)
        N = self.imu_que_length
        self.imu_time   = np.zeros(N, dtype=np.float64)
        self.imu_quat   = np.zeros((N, 4), dtype=np.float64)  # [x, y, z, w]
        self.imu_acc    = np.zeros((N, 3), dtype=np.float64)
        self.imu_gyro   = np.zeros((N, 3), dtype=np.float64)
        self.imu_velo   = np.zeros((N, 3), dtype=np.float64)
        self.imu_shift  = np.zeros((N, 3), dtype=np.float64)

        # Keep Euler for optional debugging parity (not used for interpolation)
        self.imu_roll   = np.zeros(N, dtype=np.float64)
        self.imu_pitch  = np.zeros(N, dtype=np.float64)
        self.imu_yaw    = np.zeros(N, dtype=np.float64)

        # Options / calibration
        self.compensate_gravity = bool(compensate_gravity)
        self.g_world = np.asarray(g_world, dtype=np.float64).reshape(3)
        self.acc_bias = np.zeros(3, dtype=np.float64) if acc_bias is None else np.asarray(acc_bias, dtype=np.float64).reshape(3)
        self.gyro_bias = np.zeros(3, dtype=np.float64) if gyro_bias is None else np.asarray(gyro_bias, dtype=np.float64).reshape(3)

        # Extrinsics (defaults: identity)
        self.R_li = np.eye(3, dtype=np.float64) if R_li is None else np.asarray(R_li, dtype=np.float64).reshape(3, 3)
        self.t_li = np.zeros(3, dtype=np.float64) if t_li is None else np.asarray(t_li, dtype=np.float64).reshape(3)

        # Integration guard
        self.max_dt = float(max_dt_factor) * self.scan_period

    def set_scan_period(self, period: float):
        self.scan_period = float(period)
        self.max_dt = 5.0 * self.scan_period  # keep proportional

    # ------------------------- IMU ingest & integration -------------------------

    def get_imu(self, angular_velo: np.ndarray, acc: np.ndarray, quat: np.ndarray, imu_time: float):
        """
        Push one IMU sample into the ring buffer and integrate simple kinematics.
        quat must be [x, y, z, w]. All vectors in sensor/IMU frame.
        """
        quat = np.asarray(quat, dtype=np.float64).reshape(4)
        acc  = np.asarray(acc, dtype=np.float64).reshape(3)
        gyro = np.asarray(angular_velo, dtype=np.float64).reshape(3)

        # Normalize quaternion defensively
        qn = quat / max(1e-12, np.linalg.norm(quat))
        Rb2w = Rotation.from_quat(qn)  # IMU->World rotation at this stamp

        roll, pitch, yaw = Rb2w.as_euler('xyz', degrees=False)

        # Ring buffer push
        self.imu_ptr_last = (self.imu_ptr_last + 1) % self.imu_que_length
        if (self.imu_ptr_last + 1) % self.imu_que_length == self.imu_ptr_front:
            # overwrite oldest when full
            self.imu_ptr_front = (self.imu_ptr_front + 1) % self.imu_que_length

        i = self.imu_ptr_last
        self.imu_time[i]  = float(imu_time)
        self.imu_quat[i]  = qn
        self.imu_acc[i]   = acc
        self.imu_gyro[i]  = gyro
        self.imu_roll[i]  = roll
        self.imu_pitch[i] = pitch
        self.imu_yaw[i]   = yaw

        # Integrate velocity/shift (very simple dead-reckon; assumes modest dt)
        back = (i - 1 + self.imu_que_length) % self.imu_que_length
        dt = self.imu_time[i] - self.imu_time[back]

        if 0.0 < dt < self.max_dt:
            # accel handling
            acc_lin_body = acc - self.acc_bias
            acc_world = Rb2w.apply(acc_lin_body)
            if self.compensate_gravity:
                # If IMU gives specific force f = a - g, then a = f + g_world
                acc_world = acc_world + self.g_world

            self.imu_velo[i]  = self.imu_velo[back] + acc_world * dt
            self.imu_shift[i] = self.imu_shift[back] + self.imu_velo[back] * dt + 0.5 * acc_world * dt * dt
        else:
            # On bad dt, carry forward last state (no integration step)
            self.imu_velo[i]  = self.imu_velo[back]
            self.imu_shift[i] = self.imu_shift[back]

    # ---------------------------- Deskew / adjustment ----------------------------

    def adjust_distortion(self, cloud: o3d.geometry.PointCloud, scan_time: float):
        pts = np.asarray(cloud.points)
        if pts.size == 0:
            return
        pts = pts.astype(np.float64, copy=True)

        # Need at least two IMU samples to interpolate
        idxs = self._contiguous_indices()
        if idxs is None or idxs.size < 2:
            # No motion compensation possible; leave points as-is
            cloud.points = o3d.utility.Vector3dVector(pts)
            print("[LidarUndistortion] Not enough IMU samples; skipped deskew.")
            return

        t_imu = self.imu_time[idxs]
        q_imu = self.imu_quat[idxs]
        S_imu = self.imu_shift[idxs]
        V_imu = self.imu_velo[idxs]

        # ---------------- Azimuth -> per-point capture time (vectorized) ----------------
        az = -np.arctan2(pts[:, 1], pts[:, 0])     # [-π, π]
        az = np.unwrap(az)                          # monotonize across the scan

        span = az[-1] - az[0]
        if np.isclose(span, 0.0):
            cloud.points = o3d.utility.Vector3dVector(pts)
            print("[LidarUndistortion] Degenerate azimuth span; skipped deskew.")
            return

        rel = (az - az[0]) / span
        t_points = float(scan_time) + rel * self.scan_period

        # Restrict to points covered by IMU time span
        tmin, tmax = t_imu[0], t_imu[-1]
        eps = 1e-6
        valid = (t_points >= tmin - eps) & (t_points <= tmax + eps)
        if not np.any(valid):
            cloud.points = o3d.utility.Vector3dVector(pts)
            print("[LidarUndistortion] IMU time window does not cover scan; skipped deskew.")
            return

        # ---------------- Interpolate pose/velocity/shift at point times ----------------
        # Orientation via SLERP
        R_key = Rotation.from_quat(q_imu)
        slerp = Slerp(t_imu, R_key)

        tq = t_points[valid]
        R_pts = slerp(tq)                    # Rotation object, length Nv
        Rm = R_pts.as_matrix()               # (Nv, 3, 3)

        # Linear interp for shift & velocity (per axis)
        Sv = np.column_stack([np.interp(tq, t_imu, S_imu[:, j]) for j in range(3)])  # (Nv,3)
        Vv = np.column_stack([np.interp(tq, t_imu, V_imu[:, j]) for j in range(3)])  # (Nv,3)

        # Reference time = first valid point's time
        t0 = tq[0]
        R0 = slerp([t0]).as_matrix()[0]
        R0_inv = R0.T
        S0 = Sv[0]
        V0 = Vv[0]
        dt_vec = (tq - t0)[:, None]

        # Shift relative to start (constant-velocity first-order approx)
        shift_from_start = (Sv - S0) - V0 * dt_vec  # (Nv,3)

        # ---------------- Apply deskew (vectorized) ----------------
        pts_valid = pts[valid]  # (Nv,3)

        # If extrinsics are identity, fast path
        if np.allclose(self.R_li, np.eye(3)) and np.allclose(self.t_li, 0.0):
            # p_out = R0^T ( R(t) p + Δs )
            p_rot = (Rm @ pts_valid[..., None]).squeeze(-1)          # (Nv,3)
            p_rel = p_rot + shift_from_start                         # (Nv,3)
            p_out = (R0_inv @ p_rel.T).T                             # (Nv,3)
        else:
            # Full extrinsic handling:
            # p_I = R_li * p_L + t_li
            p_I = (self.R_li @ pts_valid.T).T + self.t_li[None, :]   # (Nv,3)
            # World at t: R(t)*p_I + s(t)
            p_rot = (Rm @ p_I[..., None]).squeeze(-1)                # (Nv,3)
            p_rel = p_rot + shift_from_start                         # (Nv,3)
            # Into IMU start frame:
            p_I0 = (R0_inv @ p_rel.T).T                              # (Nv,3)
            # Back to LiDAR start frame: subtract t_li
            p_out = p_I0 - self.t_li[None, :]

        # Scatter back
        pts[valid] = p_out
        cloud.points = o3d.utility.Vector3dVector(pts)
        print(f"[LidarUndistortion] Deskewed {p_out.shape[0]}/{pts.shape[0]} points "
              f"using IMU window [{tmin:.6f}, {tmax:.6f}] s.")

    # ----------------------------- Helpers -----------------------------

    def _contiguous_indices(self) -> Optional[np.ndarray]:
        """Return a time-sorted, strictly increasing view of the IMU ring buffer."""
        if self.imu_ptr_last == -1:
            return None

        start = self.imu_ptr_front
        end = self.imu_ptr_last
        if start <= end:
            idxs = np.arange(start, end + 1, dtype=int)
        else:
            idxs = np.concatenate([
                np.arange(start, self.imu_que_length, dtype=int),
                np.arange(0, end + 1, dtype=int)
            ])

        t = self.imu_time[idxs]
        if t.size < 2:
            return None

        # Ensure strictly increasing times for interpolation/SLERP
        t_unique, keep = np.unique(t, return_index=True)
        if t_unique.size < 2:
            return None
        return idxs[keep]
