from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cpp_examples',
            executable='talker_cpp',
            name='talker_cpp',
            output='screen'
        ),
        Node(
            package='cpp_examples',
            executable='listener_cpp',
            name='listener_cpp',
            output='screen'
        )
    ])