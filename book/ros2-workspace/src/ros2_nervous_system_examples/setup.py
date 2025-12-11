from setuptools import setup
import os
from glob import glob

package_name = 'ros2_nervous_system_examples'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
        # Include all config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maintainer',
    maintainer_email='maintainer@todo.todo',
    description='Examples for the ROS 2 Robotic Nervous System module',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = ros2_nervous_system_examples.talker:main',
            'listener = ros2_nervous_system_examples.listener:main',
            'service_server = ros2_nervous_system_examples.service_server:main',
            'service_client = ros2_nervous_system_examples.service_client:main',
            'action_server = ros2_nervous_system_examples.action_server:main',
            'action_client = ros2_nervous_system_examples.action_client:main',
            'ai_perception_node = ros2_nervous_system_examples.ai_perception_node:main',
        ],
    },
)