import numpy as np
import time
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

class ScansegmentationManager2D:
    
    def __init__(self):
        self.alpha_max = np.pi/4				# Relative angular threshold
        self.eta = 1.5							# Reletive lenght threshold	
        self.min_point_in_segment = 4 			# Sets the minimum allowable points in a segment to avoid segments with only 1 point
        self.min_segmentation_threshold = 0.1 	# Set minimum distance to avoid segmetation between sets of points that are too close to each other
        self.min_point_in_angle_check = 10		# Number of points to consider in angle check #10

    def segment_scan(self, ranges, angles, alpha_max=np.pi/4, eta=1.5, min_point_in_segment=4, min_segment_threshold = 0.1, min_point_in_angle_check = 10):
        start_time = time.time()  # Start timing

        x_coords = np.array(ranges) * np.cos(angles)
        y_coords = np.array(ranges) * np.sin(angles)
        points = np.vstack((x_coords, y_coords)).T 

        segments = []
        range_bearing_segments = []
        current_segment = [points[0]]  # Start with the first point
        current_range_bearing = [(ranges[0], angles[0])]  # Corresponding range and bearing
        distance_variance_detected = False
        lengths = []
        cos_alphas = []

        for i in range(1, len(points)):
            # Vector from previous point to current point
            p_i = points[i] - points[i - 1]
            p_next = points[(i + 1) % len(points)] - points[i]  # Wrap-around for the last point

            # Distance between consecutive points
            d_i = np.linalg.norm(p_i)
            d_next = np.linalg.norm(p_next)

            # Calculate angle between consecutive vectors
            cos_alpha_i = np.dot(p_i, p_next) / (np.linalg.norm(p_i) * np.linalg.norm(p_next))

            lengths.append(d_next)
            cos_alphas.append(cos_alpha_i)

            # #Pure angle check
            # # Determine whether to continue or start a new segment
            # if cos_alpha_i >= np.cos(self.alpha_max):
            # 	current_segment.append(points[i])
            # else:
            # 	segments.append(np.array(current_segment))
            # 	current_segment = [points[i]]

            # # # Pure distance check 
            # if max(d_i, d_next) <= self.eta * min(d_i, d_next):
            # 	current_segment.append(points[i])
            # else:
            # 	segments.append(np.array(current_segment))
            # 	current_segment = [points[i]]

            # Pure distance check with flag to detect variance
            # Check relative distance between consecutive points but not if the distances are too small
            if (max(d_i, d_next) <= eta * min(d_i, d_next)) or (d_i <= min_segment_threshold and d_next <= min_segment_threshold):
                current_segment.append(points[i])
                current_range_bearing.append((ranges[i], angles[i]))
            else:     
                if not distance_variance_detected:
                    current_segment.append(points[i])
                    current_range_bearing.append((ranges[i], angles[i]))
                    distance_variance_detected = True
                else:
                    # Store the completed segment
                    segments.append(np.array(current_segment))
                    range_bearing_segments.append(np.array(current_range_bearing))

                    # Start a new segment cleanly
                    current_segment = [points[i]]
                    current_range_bearing = [(ranges[i], angles[i])]
                    distance_variance_detected = False  # Ensure reset here

        # Check if the last segment should wrap around and join with the first segment
        if current_segment:
            # Calculate the angle and distance between the last and first points
            p_last = points[0] - points[-1]
            cos_alpha_last = np.dot(p_last, points[1] - points[0]) / (np.linalg.norm(p_last) * np.linalg.norm(points[1] - points[0]))

            if cos_alpha_last >= np.cos(alpha_max):
                segments[0] = np.vstack((current_segment, segments[0]))  # Merge with the first segment
                range_bearing_segments[0] = np.vstack((current_range_bearing, range_bearing_segments[0]))
            elif len(current_segment) >= min_point_in_segment:
                segments.append(np.array(current_segment))  # Add as a separate segment if it meets the minimum length
                range_bearing_segments.append(np.array(current_range_bearing))


        # Additional segmentation based on angle within each segment
        final_segments = []
        final_range_bearing_segments = []
        for segment, rb_segment in zip(segments, range_bearing_segments):
            sub_segment = [segment[0]]  # Start with the first point of the current segment
            sub_range_bearing = [rb_segment[0]]  # Start with the first range and bearing


            

            # for j in range(1, len(segment) - 1):
            # 	# Vector between consecutive points within the segment
            # 	p_j = segment[j] - segment[j - 1]
            # 	p_next_j = segment[j + 1] - segment[j]
            # 	# Lengt of vectors
            # 	# d_j = np.linalg.norm(segment[j + 1]-segment[j-1])


            # for j in range(n, len(segment) - n):
            # 	# Vector between consecutive points within the segment
            # 	p_j = segment[j] - segment[j - n]
            # 	p_next_j = segment[j + n] - segment[j]

            n = min_point_in_angle_check # Number of points to consider in angle check
            for j in range(1, len(segment) - 1):
                if j < n or j >= len(segment) - n:
                    p_j = segment[j] - segment[j - 1]
                    p_next_j = segment[j + 1] - segment[j]
                else:
                    # Vector between consecutive points within the segment
                    p_j = segment[j] - segment[j - n]
                    p_next_j = segment[j + n] - segment[j]

                # Calculate angle between vectors within the segment
                cos_alpha_j = np.dot(p_j, p_next_j) / (np.linalg.norm(p_j) * np.linalg.norm(p_next_j))

                # Lengt of vectors
                d_j = np.linalg.norm(segment[j + 1]-segment[j-1])

                # Check angle condition
                # if (cos_alpha_j >= np.cos(alpha_max)) or (d_j <= min_segment_threshold):
                if (cos_alpha_j >= np.cos(alpha_max)):
                    sub_segment.append(segment[j])
                    sub_range_bearing.append(rb_segment[j])
                else:
                    # End current sub-segment and start a new one
                    if len(sub_segment) >= min_point_in_segment:
                        final_segments.append(np.array(sub_segment))
                        final_range_bearing_segments.append(np.array(sub_range_bearing))
                    sub_segment = [segment[j]]
                    sub_range_bearing = [rb_segment[j]]

            # Add the last point of the current segment to the sub-segment ()
            sub_segment.append(segment[-1])
            sub_range_bearing.append(rb_segment[-1])

            # Check if the sub-segment meets the minimum length
            if len(sub_segment) >= min_point_in_segment:
                final_segments.append(np.array(sub_segment))
                final_range_bearing_segments.append(np.array(sub_range_bearing))

        # Check for outliers at the end of each segment
        final_segments, final_range_bearing_segments = self.remove_edge_outliers(final_segments, final_range_bearing_segments)

        # End timing
        end_time = time.time()
        execution_time = end_time - start_time

        # return segments, range_bearing_segments, execution_time	# Use this ln for the pure distance segmentation
        return final_segments, final_range_bearing_segments, execution_time #Use the line for the angle segmentation
    
    def remove_edge_outliers(self, segments, range_bearing_segments, threshold_factor=1.5):
        for i in range(len(segments)):
            current_segment = segments[i]
            current_range_bearing = range_bearing_segments[i]

            if len(current_segment) > 3:  # Need at least 4 points to have inner distances
                # Compute distances between consecutive points
                distances = np.linalg.norm(np.diff(current_segment, axis=0), axis=1)

                # Exclude edge distances when computing average
                avg_distance = np.mean(distances[1:-1]) if len(distances) > 2 else np.mean(distances)

                # Check start point
                d_start = distances[0]  # Distance from first to second point
                if d_start > threshold_factor * avg_distance:
                    current_segment = current_segment[1:]
                    current_range_bearing = current_range_bearing[1:]
                    distances = distances[1:]  # Update distances after removal

                # Check end point
                d_end = distances[-1]  # Distance from last to second-last point
                if d_end > threshold_factor * avg_distance:
                    current_segment = current_segment[:-1]
                    current_range_bearing = current_range_bearing[:-1]

            # Update the segments
            segments[i] = current_segment
            range_bearing_segments[i] = current_range_bearing

        return segments, range_bearing_segments