# Gazebo Setup for Humanoid Robotics

## Introduction to Gazebo

Gazebo is a 3D simulation environment that enables accurate and efficient testing of robotics algorithms, robot design, and software integration. For humanoid robotics, Gazebo provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces that allow for rapid development and testing of complex robotic systems.

## Installing Gazebo Garden

Gazebo Garden is the recommended version for use with ROS 2 Humble Hawksbill. The installation process involves setting up the simulation environment with proper physics engines and rendering capabilities.

### Prerequisites

Before installing Gazebo, ensure you have:
- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill installed
- Graphics drivers properly configured (especially if using GPU acceleration)
- At least 4GB of RAM (8GB+ recommended for complex simulations)

### Installation Steps

```bash
# Update package list
sudo apt update

# Install Gazebo Garden
sudo apt install gz-garden

# Install additional Gazebo plugins and tools
sudo apt install libgazebo-dev
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-gazebo-ros2-control
sudo apt install ros-humble-gazebo-dev
```

### Verification

After installation, verify Gazebo is working correctly:

```bash
# Check Gazebo version
gz --version

# Launch Gazebo GUI
gz sim

# Launch with a simple world
gz sim -r -v 4 shapes.sdf
```

## Configuring Gazebo for Humanoid Robots

### Physics Engine Configuration

Gazebo supports multiple physics engines including ODE, Bullet, DART, and SimBody. For humanoid robots, ODE (Open Dynamics Engine) is often the best choice due to its balance of accuracy and performance.

```xml
<!-- Example physics configuration in SDF -->
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
</physics>
```

### World Setup

Create a world file for humanoid robot simulation:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_world">
    <!-- Include a ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include a sun for lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Physics configuration -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Your humanoid robot will be inserted here -->
  </world>
</sdf>
```

## Converting URDF to SDF

Since Gazebo works with SDF (Simulation Description Format) but ROS 2 typically uses URDF (Unified Robot Description Format), you'll need to convert your humanoid robot model:

### Method 1: Direct URDF Integration

Gazebo can directly load URDF files through the `libgazebo_ros_factory.so` plugin:

```xml
<!-- In your launch file -->
<node name="spawn_urdf" pkg="gazebo_ros" type="spawn_entity.py"
      args="-entity my_robot -topic robot_description -x 0 -y 0 -z 1"/>
```

### Method 2: Conversion to SDF

Convert your URDF to SDF format:

```bash
# Install the conversion tool
sudo apt install ros-humble-xacro

# Convert URDF to SDF
ros2 run xacro xacro --inorder /path/to/your/robot.urdf.xacro > robot.sdf
```

## Setting Up Humanoid Robot Simulation

### Creating a Robot Spawn Launch File

```python
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('humanoid_description')

    # Get URDF file path
    urdf_path = os.path.join(pkg_share, 'urdf', 'humanoid.urdf.xacro')

    # Use xacro to process the file
    robot_description = Command(['xacro ', urdf_path])

    # Gazebo server and client
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch'), '/gazebo.launch.py']),
    )

    # Spawn the robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description',
                   '-entity', 'humanoid_robot',
                   '-x', '0.0',
                   '-y', '0.0',
                   '-z', '1.0'],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'publish_frequency': 50.0
        }]
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
    ])
```

## Adding Sensors to the Humanoid Robot

### LiDAR Sensor Configuration

Add a LiDAR sensor to the humanoid robot head:

```xml
<!-- In your URDF/Xacro file -->
<gazebo reference="head">
  <sensor name="lidar_sensor" type="ray">
    <pose>0.1 0 0 0 0 0</pose>
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/humanoid</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>lidar_frame</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### IMU Sensor Configuration

Add IMU sensors to torso and limbs:

```xml
<gazebo reference="torso">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>true</visualize>
    <plugin name="imu_controller" filename="libgazebo_ros_imu_sensor.so">
      <ros>
        <namespace>/humanoid</namespace>
        <remapping>~/out:=imu</remapping>
      </ros>
      <frame_name>imu_frame</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### Camera Sensor Configuration

Add a depth camera to the head:

```xml
<gazebo reference="head">
  <sensor name="camera" type="depth">
    <always_on>true</always_on>
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>/humanoid</namespace>
        <remapping>image_raw:=camera/image_raw</remapping>
        <remapping>camera_info:=camera/camera_info</remapping>
      </ros>
      <frame_name>camera_frame</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

## Running the Simulation

### Launching the Complete Simulation

```bash
# Source your ROS 2 workspace
source install/setup.bash

# Launch the simulation with your humanoid robot
ros2 launch humanoid_description gazebo.launch.py
```

### Controlling the Robot in Simulation

Once the simulation is running, you can control the robot using ROS 2 topics:

```bash
# Send joint position commands
ros2 topic pub /humanoid/joint_trajectory_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "header:
  stamp:
    sec: 0
    nanosec: 0
  frame_id: 'base_link'
joint_names: ['joint1', 'joint2']  # Replace with actual joint names
points:
- positions: [0.0, 0.0]
  time_from_start:
    sec: 1
    nanosec: 0"
```

## Troubleshooting Common Issues

### Gazebo Not Starting

If Gazebo fails to start:
1. Check that your graphics drivers are properly installed
2. Verify X11 forwarding if running remotely
3. Ensure sufficient RAM is available
4. Try running with software rendering: `gazebo --verbose --render-engine=ogre`

### Robot Falls Through Ground

If your robot falls through the ground:
1. Verify that collision geometries are properly defined in URDF
2. Check that the robot has proper mass and inertia values
3. Ensure joint limits and safety controllers are configured

### Sensor Data Not Publishing

If sensor data is not publishing:
1. Verify that Gazebo ROS plugins are properly configured
2. Check that the sensor topics are being published: `ros2 topic list`
3. Confirm that the robot state publisher is running

## Performance Optimization

For complex humanoid robots with multiple sensors:

1. **Adjust Physics Update Rate**: Balance accuracy and performance
2. **Use Simplified Collision Models**: Use simpler geometries for collision detection
3. **Limit Sensor Update Rates**: Reduce sensor update rates where possible
4. **Disable Unnecessary Rendering**: Use headless mode for batch simulations

## Best Practices

1. **Model Validation**: Always validate URDF/SDF models before simulation
2. **Realistic Parameters**: Use realistic mass, inertia, and friction values
3. **Incremental Testing**: Test simple models before adding complexity
4. **Consistent Units**: Maintain consistent units throughout the model
5. **Documentation**: Document all simulation parameters for reproducibility

## Summary

Gazebo provides a powerful simulation environment for humanoid robotics, enabling realistic physics simulation and sensor modeling. Proper setup involves configuring physics parameters, converting URDF models to SDF format, and integrating with ROS 2 for seamless control and perception. With careful configuration, Gazebo enables effective testing and validation of humanoid robot systems before deployment to physical hardware.

## Learning Check

After studying this section, you should be able to:
- Install and configure Gazebo Garden for ROS 2 Humble
- Convert URDF models to SDF format for simulation
- Add various sensor types to humanoid robot models
- Launch and control humanoid robots in simulation
- Troubleshoot common simulation issues