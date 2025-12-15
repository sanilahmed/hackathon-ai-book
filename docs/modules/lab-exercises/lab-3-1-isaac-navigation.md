---
sidebar_label: 'Lab 3.1: Isaac Navigation'
---

# Lab Exercise 3.1: NVIDIA Isaac Navigation

This lab exercise covers setting up and using NVIDIA Isaac for robot navigation tasks.

## Objectives

- Install and configure NVIDIA Isaac navigation stack
- Set up costmaps and local/global planners
- Implement navigation behaviors
- Test navigation in simulation

## Prerequisites

- NVIDIA GPU with CUDA support
- ROS 2 Humble Hawksbill
- Isaac Sim (optional for simulation)

## Isaac Navigation Overview

NVIDIA Isaac Navigation provides:
- GPU-accelerated path planning
- Optimized obstacle avoidance
- Real-time performance
- Integration with Isaac ecosystem

## Installation and Setup

### Isaac ROS Navigation Installation

1. Install Isaac ROS packages:
   ```bash
   sudo apt update
   sudo apt install ros-humble-isaac-ros-nav2-bringup
   sudo apt install ros-humble-isaac-ros-occupancy-grid-localizer
   sudo apt install ros-humble-isaac-ros-visual-slam
   ```

2. Install navigation dependencies:
   ```bash
   sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
   ```

### Environment Setup

Add to your `.bashrc`:
```bash
export ISAAC_ROS_WS=/opt/isaac_ros
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$ISAAC_ROS_WS
```

## Navigation Stack Configuration

### Costmap Configuration

Create `costmap_params.yaml`:
```yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_delay: 0.5
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

amcl_map_client:
  ros__parameters:
    use_sim_time: True

amcl_rclcpp_node:
  ros__parameters:
    use_sim_time: True

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    # Specify the path to the Behavior Tree XML file
    bt_xml_filename: "navigate_w_replanning_and_recovery.xml"
    # 'default_nav_through_poses_bt_xml' and 'default_nav_to_pose_bt_xml' are set automatically based on 'bt_xml_filename'
    # Rate limits the frequency of calling the next command
    navigators: ["navigate_to_pose", "navigate_through_poses"]
    navigate_to_pose: ["navigation"]
    navigate_through_poses: ["navigation"]
    navigation: ["RateController", "GoalChecker", "Controller", "GoalReacher", "Recovery", "GoalChecker"]
    GoalReacher:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      costmap_topic: "local_costmap/costmap_raw"
      footprint_topic: "local_costmap/published_footprint"
      goal_checker: "goal_checker"
      replanner: "recovery"
    RateController:
      plugin: "nav2_regulated_pure_pursuit_controller/RegulatedPurePursuitController"
      speed_topic: "cmd_vel"
    Recovery:
      plugin: "nav2_recoveries/RecoveryServer"
      costmap_topic: "local_costmap/costmap_raw"
      footprint_topic: "local_costmap/published_footprint"
    GoalChecker:
      plugin: "nav2_controller::SimpleGoalChecker"
      costmap_topic: "local_costmap/costmap_raw"
      footprint_topic: "local_costmap/published_footprint"
```

### Local Costmap Configuration

Create `local_costmap_params.yaml`:
```yaml
local_costmap:
  ros__parameters:
    update_frequency: 10.0
    publish_frequency: 10.0
    global_frame: odom
    robot_base_frame: base_link
    use_sim_time: true
    rolling_window: true
    width: 3
    height: 3
    resolution: 0.05
    robot_radius: 0.22
    plugins: ["voxel_layer", "inflation_layer"]
    inflation_layer:
      plugin: "nav2_costmap_2d::InflationLayer"
      cost_scaling_factor: 3.0
      inflation_radius: 0.55
    voxel_layer:
      plugin: "nav2_costmap_2d::VoxelLayer"
      enabled: True
      publish_voxel_map: True
      origin_z: 0.0
      z_resolution: 0.05
      z_voxels: 16
      max_obstacle_height: 2.0
      mark_threshold: 0
      observation_sources: scan
      scan:
        topic: /scan
        max_obstacle_height: 2.0
        clearing: True
        marking: True
        data_type: "LaserScan"
        raytrace_max_range: 3.0
        raytrace_min_range: 0.0
        obstacle_max_range: 2.5
        obstacle_min_range: 0.0
```

## Isaac-Specific Navigation Features

### GPU-Accelerated Path Planning

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import numpy as np

class IsaacPathPlanner(Node):
    def __init__(self):
        super().__init__('isaac_path_planner')

        # Publisher for planned path
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)

        # Subscriber for goal poses
        self.goal_sub = self.create_subscription(
            PoseStamped, 'goal_pose', self.goal_callback, 10
        )

        # Initialize GPU-accelerated planner (simulated)
        self.gpu_planner_initialized = self.initialize_gpu_planner()

    def initialize_gpu_planner(self):
        # This would initialize NVIDIA's GPU-accelerated planning
        # For simulation purposes, return True
        self.get_logger().info('GPU-accelerated path planner initialized')
        return True

    def goal_callback(self, goal_msg):
        if self.gpu_planner_initialized:
            # Plan path using GPU acceleration
            path = self.plan_path_gpu(goal_msg)
            self.path_pub.publish(path)

    def plan_path_gpu(self, goal_msg):
        # Simulate GPU-accelerated path planning
        # In real implementation, this would use Isaac's GPU planners
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        # Create a simple path (in real implementation, this would be the result of GPU planning)
        start_x, start_y = 0.0, 0.0
        goal_x, goal_y = goal_msg.pose.position.x, goal_msg.pose.position.y

        # Generate intermediate poses
        steps = 20
        for i in range(steps + 1):
            t = i / steps
            x = start_x + t * (goal_x - start_x)
            y = start_y + t * (goal_y - start_y)

            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0

            path_msg.poses.append(pose)

        return path_msg
```

### Visual SLAM Integration

```python
class IsaacVisualSLAM(Node):
    def __init__(self):
        super().__init__('isaac_visual_slam')

        # Publishers and subscribers
        self.odom_pub = self.create_publisher(Odometry, 'visual_odom', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, 'visual_map', 10)

        # Camera subscriber
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )

        # Initialize Isaac Visual SLAM
        self.slam_initialized = self.initialize_visual_slam()

    def initialize_visual_slam(self):
        # Initialize Isaac's visual SLAM pipeline
        # This would typically involve CUDA initialization
        self.get_logger().info('Isaac Visual SLAM initialized')
        return True

    def image_callback(self, image_msg):
        if self.slam_initialized:
            # Process image for visual SLAM
            self.process_visual_slam(image_msg)

    def process_visual_slam(self, image_msg):
        # Process image for pose estimation and map building
        # This would use Isaac's GPU-accelerated visual SLAM
        pass
```

## Navigation Launch Configuration

### Isaac Navigation Launch File

Create `isaac_navigation.launch.py`:
```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    map_yaml_file = LaunchConfiguration('map')

    # Navigation launch
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('isaac_ros_nav2_bringup'),
                'launch',
                'navigation.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # SLAM node
    visual_slam_node = Node(
        package='isaac_ros_visual_slam',
        executable='visual_slam_node',
        name='visual_slam',
        parameters=[{
            'use_sim_time': use_sim_time,
            'enable_rectified_pose': True,
            'map_frame': 'map',
            'odom_frame': 'odom',
            'base_frame': 'base_link',
            'enable_fisheye': False,
            'input_width': 1920,
            'input_height': 1080,
            'publish_odom_tf': True
        }],
        remappings=[
            ('/visual_slam/image_raw', '/camera/image_raw'),
            ('/visual_slam/camera_info', '/camera/camera_info')
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),
        DeclareLaunchArgument(
            'map',
            default_value='',
            description='Full path to map file to load'
        ),
        navigation_launch,
        visual_slam_node
    ])
```

## Testing Navigation

### Navigation Test Node

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient

class NavigationTest(Node):
    def __init__(self):
        super().__init__('navigation_test')

        # Action client for navigation
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Timer to send navigation goals
        self.timer = self.create_timer(10.0, self.send_navigation_goal)

    def send_navigation_goal(self):
        # Wait for action server
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation action server not available')
            return

        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = 2.0
        goal_msg.pose.pose.position.y = 2.0
        goal_msg.pose.pose.orientation.w = 1.0

        # Send goal
        self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        ).add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        goal_handle.get_result_async().add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Navigation progress: {feedback_msg.feedback.distance_remaining:.2f}m remaining')

    def get_result_callback(self, future):
        result = future.result().result
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Navigation succeeded!')
        else:
            self.get_logger().info(f'Navigation failed with status: {status}')
```

## Performance Optimization

### GPU Memory Management

```python
class IsaacNavigationOptimizer:
    def __init__(self):
        self.gpu_memory_limit = 0.8  # Use 80% of available GPU memory
        self.plan_cache_size = 10    # Cache recent plans

    def optimize_gpu_usage(self):
        # Monitor GPU memory usage
        # Adjust planning parameters based on available memory
        # Implement plan caching to reduce repeated computation
        pass
```

## Exercise Tasks

1. Install Isaac ROS navigation packages
2. Configure costmaps and planners for your robot
3. Implement a simple path planner using Isaac's GPU acceleration (simulated)
4. Create a launch file for Isaac navigation
5. Test navigation in simulation or with a real robot
6. Measure and optimize navigation performance

## Troubleshooting

### Common Issues

- **GPU initialization failures**: Check CUDA installation and GPU drivers
- **Navigation timeouts**: Verify sensor data and costmap configuration
- **Performance issues**: Monitor GPU utilization and memory usage
- **Map building problems**: Check camera calibration and visual features

## Summary

In this lab, you learned to set up and use NVIDIA Isaac for robot navigation. You configured the navigation stack, implemented GPU-accelerated planning, and tested navigation performance. Isaac's GPU acceleration provides significant performance improvements for complex navigation tasks.