# Modular SLAM (Python)

A modular SLAM (Simultaneous Localization and Mapping) implementation in Python for ROS 2.

## Overview

This package provides a flexible SLAM system with both 2D and 3D capabilities, featuring:
- Front-end processing for both 2D and 3D LiDAR data
- Back-end optimization using pose graph optimization
- Loop closure detection using Scan Context

## Package Structure

- `mod_slam/` - Main SLAM implementation
  - `front_end_2d.py` - 2D LiDAR front-end processing
  - `front_end_3d.py` - 3D LiDAR front-end processing
  - `back_end.py` - Back-end optimization
  - `utils/` - Utility modules for ICP, map management, scan context, etc.
- `launch/` - ROS 2 launch files
- `config/` - Configuration files for different robot platforms
- `slam_interfaces/` - Custom ROS 2 message and service definitions

## Dependencies

- ROS 2 (tested with Humble/Iron)
- Python 3.8+
- NumPy
- Open3D
- SciPy
- SmallGICP
- GTSAM

## Installation

1. Clone this repository into your ROS 2 workspace:
```bash
cd ~/dev_ws/src
git clone https://github.com/Delport121/Modular_SLAM_py.git mod_slam_python
```

2. Install dependencies:
```bash
cd ~/dev_ws
rosdep install --from-paths src --ignore-src -r -y
```

3. Build the workspace:
```bash
colcon build --packages-select mod_slam slam_interfaces
```

4. Source the workspace:
```bash
source install/setup.bash
```

## Usage

### 2D SLAM
```bash
ros2 launch mod_slam slam_2d.launch.py
```

### 3D SLAM
```bash
ros2 launch mod_slam slam_3d.launch.py
```

### Voyager Platform
```bash
ros2 launch mod_slam slam_voyager.launch.py
```

## Configuration

Configuration files are located in the `config/` directory. You can modify parameters for:
- ICP matching
- Loop closure detection
- Map resolution
- Sensor topics

## License



## Authors

P.J. Delport

## Acknowledgments


