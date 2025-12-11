from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros2_nervous_system_examples',
            executable='talker',
            name='talker',
            output='screen'
        ),
        Node(
            package='ros2_nervous_system_examples',
            executable='listener',
            name='listener',
            output='screen'
        )
    ])