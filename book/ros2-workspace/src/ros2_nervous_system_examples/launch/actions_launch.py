from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros2_nervous_system_examples',
            executable='action_server',
            name='action_server',
            output='screen'
        ),
        Node(
            package='ros2_nervous_system_examples',
            executable='action_client',
            name='action_client',
            output='screen'
        )
    ])