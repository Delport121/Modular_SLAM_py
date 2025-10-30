from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'mod_slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Add message definitions
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
        # Add launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Add configuration files (if any)
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools','tf2_sensor_msgs'],
    zip_safe=True,
    maintainer='ruan',
    maintainer_email='delport121@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "frontend_2d = mod_slam.front_end_2d:main",
            "frontend_3d = mod_slam.front_end_3d:main",
            "backend = mod_slam.back_end:main",
            
        ],
    },
)