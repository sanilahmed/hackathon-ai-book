from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros2_nervous_system_examples',
            executable='service_server',
            name='service_server',
            output='screen'
        ),
        Node(
            package='ros2_nervous_system_examples',
            executable='service_client',
            name='service_client',
            output='screen',
            arguments=['10', '20']  # Example arguments for the service call
        )
    ])