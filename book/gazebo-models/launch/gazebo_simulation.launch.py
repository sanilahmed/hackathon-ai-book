import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directories
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_gazebo_models = get_package_share_directory('humanoid_description')  # Assuming we'll create this

    # Get world file path
    world_file = os.path.join(
        get_package_share_directory('humanoid_description'),
        'worlds',
        'humanoid_world.sdf'
    )

    # Gazebo simulation
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': f'-r -v 4 {world_file}',
        }.items(),
    )

    # Spawn the robot in Gazebo
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'humanoid_robot',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '1.0',
        ],
        output='screen',
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': True}
        ],
        remappings=[
            ('/joint_states', 'joint_states'),
        ]
    )

    # Joint state publisher (for GUI)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[
            {'use_sim_time': True}
        ],
    )

    # Create launch description
    ld = LaunchDescription()

    # Add actions
    ld.add_action(gazebo)
    ld.add_action(spawn_robot)
    ld.add_action(robot_state_publisher)
    ld.add_action(joint_state_publisher)

    return ld