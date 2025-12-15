---
sidebar_label: 'Lab 2.1: Gazebo Setup'
---

# Lab Exercise 2.1: Gazebo Setup for Digital Twin

This lab exercise guides you through setting up Gazebo for creating a digital twin environment.

## Objectives

- Install and configure Gazebo Garden
- Create a basic simulation environment
- Integrate with ROS 2
- Test basic robot simulation

## Prerequisites

- ROS 2 Humble Hawksbill installed
- Ubuntu 22.04
- NVIDIA GPU (recommended for rendering)

## Gazebo Installation

### Installing Gazebo Garden

1. Add the ignition repository:
   ```bash
   sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
   wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
   sudo apt update
   ```

2. Install Gazebo Garden:
   ```bash
   sudo apt install gz-garden
   ```

3. Install ROS 2 Gazebo packages:
   ```bash
   sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
   ```

## Basic Gazebo Environment

### Launching Gazebo

1. Start Gazebo with an empty world:
   ```bash
   gz sim -r empty.sdf
   ```

2. Or launch with ROS 2 integration:
   ```bash
   ros2 launch gazebo_ros empty_world.launch.py
   ```

### Creating a Custom World

Create a custom world file `my_world.sdf`:
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="my_world">
    <!-- Include sun for lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Include ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Add a simple box -->
    <model name="box">
      <pose>1 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
            <specular>1 0 0 1</specular>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## ROS 2 Integration

### Gazebo ROS Packages

The main packages for ROS 2 integration:
- `gazebo_ros`: Core ROS 2 plugins for Gazebo
- `gazebo_plugins`: Various sensor and actuator plugins
- `gazebo_dev`: Development files for creating custom plugins

### Launch File for Gazebo

Create a launch file `gazebo_launch.py`:
```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from launch_ros.actions import Node

def generate_launch_description():
    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'worlds',
                'my_world.sdf'
            ])
        }.items()
    )

    return LaunchDescription([
        gazebo
    ])
```

## Robot Model in Gazebo

### Adding a Robot Model

1. Create a URDF model for your robot
2. Convert to SDF if needed (Gazebo can read URDF directly)
3. Add Gazebo-specific tags to your URDF:

```xml
<gazebo reference="link_name">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
</gazebo>

<!-- For joints -->
<gazebo reference="joint_name">
  <implicitSpringDamper>1</implicitSpringDamper>
</gazebo>
```

### Robot Spawn Launch

Create a launch file to spawn your robot in Gazebo:
```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            'robot_description',
            default_value='[path_to_robot_description]/urdf/robot.urdf',
            description='Robot description file'
        )
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[{'robot_description': Command(['xacro ', LaunchConfiguration('robot_description')])}],
    )

    # Spawn entity
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description',
                  '-entity', 'my_robot'],
        output='screen'
    )

    return LaunchDescription(declared_arguments + [
        robot_state_publisher,
        spawn_entity,
    ])
```

## Sensor Integration

### Adding Sensors to Robot

Example of adding a camera sensor to your robot:
```xml
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <camera>
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <always_on>1</always_on>
    <update_rate>30</update_rate>
    <visualize>true</visualize>
  </sensor>
</gazebo>
```

## Testing the Setup

### Launch Everything

1. Start Gazebo with your world:
   ```bash
   gz sim -r worlds/my_world.sdf
   ```

2. Or use the ROS 2 launch:
   ```bash
   ros2 launch my_robot_gazebo launch.py
   ```

### Verify Integration

1. Check that Gazebo is running:
   ```bash
   gz topic -l
   ```

2. Verify ROS 2 topics:
   ```bash
   ros2 topic list | grep gazebo
   ```

3. Check TF transforms:
   ```bash
   ros2 run tf2_tools view_frames
   ```

## Troubleshooting

### Common Issues

- **Rendering issues**: Check graphics drivers and OpenGL support
- **Connection problems**: Verify Gazebo and ROS 2 network configuration
- **Model loading**: Ensure all mesh files and materials are properly located

### Performance Tips

- Reduce physics update rate for better performance
- Use simpler collision geometries
- Limit the number of active sensors during testing

## Exercise Tasks

1. Install Gazebo Garden and ROS 2 integration packages
2. Create a simple world file with basic objects
3. Create a launch file to start Gazebo with your world
4. Add a simple robot model to the simulation
5. Verify that you can see the robot in Gazebo and ROS 2

## Summary

In this lab, you learned to set up Gazebo for digital twin applications, integrate it with ROS 2, and create basic simulation environments. You now have the foundation for more complex simulation scenarios.