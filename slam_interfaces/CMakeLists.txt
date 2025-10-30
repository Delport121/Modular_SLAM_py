cmake_minimum_required(VERSION 3.8)
project(slam_interfaces)

# Enable strict compilation options for better code quality
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)

find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Add custom message files
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BsplineSegment.msg"
  "msg/BsplineArray.msg"
  "msg/TargetCoordinates.msg"
  "msg/SubMapT.msg"
  "msg/MapArrayT.msg"
  # "msg/SubMap.msg"
  # "msg/MapArray.msg"
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs sensor_msgs
)

install(DIRECTORY
  ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_py/
  DESTINATION lib/python3.10/site-packages
)


# Export runtime dependencies
ament_export_dependencies(rosidl_default_runtime)

# Ensure package is configured as a ROS 2 package
ament_package()
