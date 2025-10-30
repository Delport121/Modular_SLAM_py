import os
from ament_index_python.packages import get_package_share_directory
import launch
import launch_ros.actions
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node

def generate_launch_description():

    share_dir = get_package_share_directory('mod_slam')
    main_param_dir = LaunchConfiguration(
        'main_param_dir',
        default=os.path.join(
            share_dir,
            'config',
            'voyager.yaml'))
    
    params_declare = DeclareLaunchArgument(
        'main_param_dir',
        default_value=os.path.join(share_dir, 'config', 'voyager.yaml'),
        description='Full path to main parameter file to load')
    
    rviz_config_file = os.path.join(share_dir, 'config', 'voyager.rviz')
    
    frontend = TimerAction(
        period=5.0,  # delay in seconds
        actions=[Node(
                package='mod_slam',
                executable='front_end_3d.py',
                parameters=[main_param_dir],
                name='frontend_node',
                output='screen',
            )
        ]
    )
    
    backend = Node(
        package='mod_slam',
        executable='back_end.py',
        parameters=[main_param_dir],
        name='backend_node',
        output='screen',
    )
    
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file]
    )

    return launch.LaunchDescription([
        params_declare,
        backend,
        frontend,
        rviz,
            ])