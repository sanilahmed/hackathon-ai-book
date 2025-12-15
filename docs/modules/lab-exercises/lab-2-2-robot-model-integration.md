---
sidebar_label: 'Lab 2.2: Robot Model Integration'
---

# Lab Exercise 2.2: Robot Model Integration in Gazebo

This lab exercise covers integrating robot models with Gazebo simulation and ROS 2.

## Objectives

- Import and configure robot models in Gazebo
- Set up physics properties and materials
- Configure sensors and actuators
- Test robot simulation in Gazebo

## Prerequisites

- Gazebo Garden installed
- ROS 2 Humble with Gazebo packages
- Basic URDF knowledge

## Robot Model Preparation

### URDF to SDF Conversion

Gazebo can work with both URDF and SDF formats. For complex robots, you may want to convert or enhance your URDF:

```xml
<!-- Enhanced URDF with Gazebo-specific tags -->
<?xml version="1.0"?>
<robot name="my_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.2" radius="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Gazebo-specific properties -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
    <kp>1000000.0</kp>
    <kd>1.0</kd>
  </gazebo>
</robot>
```

### Physics Configuration

Configure realistic physics properties:
- Mass and inertia tensors
- Friction coefficients
- Spring and damping parameters

## Gazebo-Specific Tags

### Material Properties

```xml
<gazebo reference="link_name">
  <material>Gazebo/Orange</material>
  <mu1>0.8</mu1>  <!-- Primary friction coefficient -->
  <mu2>0.8</mu2>  <!-- Secondary friction coefficient -->
  <kp>1000000.0</kp>  <!-- Contact stiffness -->
  <kd>1000000.0</kd>  <!-- Contact damping -->
  <max_vel>100.0</max_vel>  <!-- Maximum contact penetration velocity -->
  <min_depth>0.001</min_depth>  <!-- Minimum contact depth -->
</gazebo>
```

### Inertial Properties

For better simulation stability:
```xml
<inertial>
  <mass value="1.0"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
</inertial>
```

## Joint Configuration

### Joint Transmission

Configure joint transmissions for ROS 2 control:
```xml
<transmission name="joint1_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="joint1">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="joint1_motor">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Gazebo Joint Plugins

```xml
<gazebo reference="joint1">
  <implicitSpringDamper>1</implicitSpringDamper>
  <provideFeedback>1</provideFeedback>
</gazebo>
```

## Sensor Integration

### Camera Sensor

```xml
<link name="camera_link">
  <visual>
    <geometry>
      <box size="0.02 0.02 0.02"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <box size="0.02 0.02 0.02"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.1"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<joint name="camera_joint" type="fixed">
  <parent link="base_link"/>
  <child link="camera_link"/>
  <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
</joint>

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

### IMU Sensor

```xml
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>false</visualize>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

## ROS 2 Control Integration

### ROS 2 Control Configuration

Create a `ros2_control.xacro` file:
```xml
<xacro:macro name="ros2_control" params="name">
  <ros2_control name="${name}" type="system">
    <hardware>
      <plugin>gazebo_ros2_control/GazeboSystem</plugin>
    </hardware>
    <joint name="joint1">
      <command_interface name="position">
        <param name="min">-3.14</param>
        <param name="max">3.14</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
  </ros2_control>
</xacro:macro>
```

### Controller Configuration

Create a `controllers.yaml` file:
```yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    forward_position_controller:
      type: position_controllers/JointGroupPositionController

forward_position_controller:
  ros__parameters:
    joints:
      - joint1
      - joint2
      - joint3
```

## Launch Configuration

### Complete Launch File

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare arguments
    pkg_share = FindPackageShare('my_robot_description').find('my_robot_description')

    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='my_robot',
        description='Name of the robot'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[
            {'use_sim_time': True},
            {'robot_description': Command([
                'xacro ',
                PathJoinSubstitution([
                    FindPackageShare('my_robot_description'),
                    'urdf',
                    'robot.urdf.xacro'
                ])
            ])}
        ]
    )

    # Spawn entity in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', LaunchConfiguration('robot_name'),
            '-x', '0', '-y', '0', '-z', '1.0'
        ],
        output='screen'
    )

    # ROS 2 control
    ros2_control = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ])
    )

    return LaunchDescription([
        robot_name_arg,
        gazebo,
        robot_state_publisher,
        spawn_entity,
        ros2_control
    ])
```

## Testing and Validation

### Launch the Robot

1. Build your package:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select my_robot_description
   source install/setup.bash
   ```

2. Launch the simulation:
   ```bash
   ros2 launch my_robot_gazebo robot.launch.py
   ```

### Verification Steps

1. Check robot in Gazebo GUI
2. Verify joint states:
   ```bash
   ros2 topic echo /joint_states
   ```
3. Test joint control:
   ```bash
   ros2 control list_controllers
   ```
4. Verify sensor data:
   ```bash
   ros2 topic echo /camera/image_raw
   ```

## Exercise Tasks

1. Create a simple robot URDF with at least 3 joints
2. Add Gazebo-specific tags for physics properties
3. Configure at least one sensor (camera, IMU, or LIDAR)
4. Set up ROS 2 control for the joints
5. Test the robot in Gazebo simulation
6. Verify that you can command joint movements

## Troubleshooting

### Common Issues

- **Robot falls through ground**: Check inertial properties and mass
- **Joints don't respond**: Verify transmission configuration
- **Sensors not publishing**: Check Gazebo sensor configuration
- **Controller not loading**: Verify controller configuration YAML

## Summary

In this lab, you learned to integrate robot models with Gazebo simulation, configure physics properties, add sensors, and set up ROS 2 control. You now have a complete simulation pipeline for testing robot behavior.