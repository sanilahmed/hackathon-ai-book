from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros2_nervous_system_examples',
            executable='ai_perception_node',
            name='ai_perception_node',
            output='screen',
            # Remap topics to match expected sensor topics
            remappings=[
                ('camera/image_raw', '/camera/image_raw'),
                ('scan', '/scan'),
                ('cmd_vel', '/cmd_vel')
            ]
        )
    ])