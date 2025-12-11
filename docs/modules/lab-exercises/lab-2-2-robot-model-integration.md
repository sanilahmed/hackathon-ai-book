# Lab 2.2: Robot Model Creation and Integration

## Overview

In this lab, you will learn how to create detailed robot models using both URDF and SDF formats, integrate them with Gazebo simulation, and ensure proper physics properties and sensor configurations. You'll develop a complete humanoid robot model with joints, links, and sensors.

## Objectives

By the end of this lab, you will be able to:
- Create detailed robot models in URDF and SDF formats
- Integrate robot models with Gazebo physics simulation
- Configure proper joint limits, dynamics, and collision properties
- Add sensors to robot models for simulation
- Validate robot models and troubleshoot common issues
- Create a complete humanoid robot model with proper kinematics

## Prerequisites

- Completion of Lab 2.1: Gazebo Simulation Environment Setup
- Understanding of coordinate systems and transformations
- Basic knowledge of robot kinematics
- Familiarity with XML/SDF file formats

## Duration

4-5 hours

## Exercise 1: Creating a URDF Robot Model

### Step 1: Create a URDF package structure

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python robot_modeling_py --dependencies rclpy xacro
```

### Step 2: Create URDF directory structure

```bash
mkdir -p ~/ros2_ws/src/robot_modeling_py/robot_modeling_py/urdf
mkdir -p ~/ros2_ws/src/robot_modeling_py/robot_modeling_py/meshes
mkdir -p ~/ros2_ws/src/robot_modeling_py/robot_modeling_py/config
```

### Step 3: Create a simple wheeled robot in URDF

Create `~/ros2_ws/src/robot_modeling_py/robot_modeling_py/urdf/simple_robot.urdf`:

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Base footprint for ground contact -->
  <link name="base_footprint">
    <visual>
      <geometry>
        <cylinder radius="0.25" length="0.01"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 0.8"/>
      </material>
    </visual>
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Joint connecting base_footprint to base_link -->
  <joint name="base_joint" type="fixed">
    <parent link="base_footprint"/>
    <child link="base_link"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>

  <!-- Left wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0.15 0.15 -0.1" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Right wheel -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0.15 -0.15 -0.1" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Camera link -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.2 0 0.05" rpy="0 0 0"/>
  </joint>
</robot>
```

## Exercise 2: Using Xacro for Complex Robot Models

### Step 1: Create a Xacro version of the robot

Create `~/ros2_ws/src/robot_modeling_py/robot_modeling_py/urdf/simple_robot.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="simple_robot_xacro">
  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.05" />
  <xacro:property name="base_length" value="0.5" />
  <xacro:property name="base_width" value="0.3" />
  <xacro:property name="base_height" value="0.2" />

  <!-- Macro for wheel -->
  <xacro:macro name="wheel" params="prefix parent xyz rpy">
    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.2"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 0 1"/>
    </joint>
  </xacro:macro>

  <!-- Base footprint -->
  <link name="base_footprint">
    <visual>
      <geometry>
        <cylinder radius="0.25" length="0.01"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 0.8"/>
      </material>
    </visual>
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Base joint -->
  <joint name="base_joint" type="fixed">
    <parent link="base_footprint"/>
    <child link="base_link"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>

  <!-- Wheels using macro -->
  <xacro:wheel prefix="left" parent="base_link" xyz="0.15 0.15 -0.1" rpy="${M_PI/2} 0 0"/>
  <xacro:wheel prefix="right" parent="base_link" xyz="0.15 -0.15 -0.1" rpy="${M_PI/2} 0 0"/>

  <!-- Camera -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.2 0 0.05" rpy="0 0 0"/>
  </joint>
</robot>
```

### Step 2: Create a launch file to display the robot

Create `~/ros2_ws/src/robot_modeling_py/robot_modeling_py/launch/display_robot.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    urdf_model_path = os.path.join(
        get_package_share_directory('robot_modeling_py'),
        'urdf',
        'simple_robot.xacro'
    )

    # Launch arguments
    model_arg = DeclareLaunchArgument(
        name='model',
        default_value=urdf_model_path,
        description='Absolute path to robot urdf file'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': open(urdf_model_path).read()
        }]
    )

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher'
    )

    # RViz node
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(get_package_share_directory('robot_modeling_py'), 'config', 'robot.rviz')]
    )

    return LaunchDescription([
        model_arg,
        robot_state_publisher,
        joint_state_publisher,
        rviz
    ])
```

### Step 3: Create RViz configuration

Create the config directory and file:

```bash
mkdir -p ~/ros2_ws/src/robot_modeling_py/robot_modeling_py/config
```

Create `~/ros2_ws/src/robot_modeling_py/robot_modeling_py/config/robot.rviz`:

```yaml
Panels:
  - Class: rviz_common/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /RobotModel1
      Splitter Ratio: 0.5
    Tree Height: 617
  - Class: rviz_common/Selection
    Name: Selection
  - Class: rviz_common/Tool Properties
    Expanded:
      - /2D Goal Pose1
      - /Publish Point1
    Name: Tool Properties
    Splitter Ratio: 0.5886790156364441
  - Class: rviz_common/Views
    Expanded:
      - /Current View1
    Name: Views
    Splitter Ratio: 0.5
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz_default_plugins/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true
    - Alpha: 1
      Class: rviz_default_plugins/RobotModel
      Collision Enabled: false
      Description File: ""
      Description Source: Topic
      Description Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /robot_description
      Enabled: true
      Links:
        All Links Enabled: true
        Expand Joint Details: false
        Expand Link Details: false
        Expand Tree: false
        Link Tree Style: Links in Alphabetic Order
      Name: RobotModel
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: base_footprint
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
    - Class: rviz_default_plugins/SetInitialPose
    - Class: rviz_default_plugins/SetGoal
    - Class: rviz_default_plugins/PublishPoint
  Transformation:
    Current:
      Class: rviz_default_plugins/TF
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 3.0
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05000000074505806
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.009999999776482582
      Pitch: 0.5
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz)
      Yaw: 0.5
    Saved: ~
Window Geometry:
  Displays:
    collapsed: false
  Height: 846
  Hide Left Dock: false
  Hide Right Dock: false
  QMainWindow State: 000000ff00000000fd000000040000000000000156000002f4fc0200000008fb0000001200530065006c0065006300740069006f006e00000001e10000009b0000005c00fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c006100790073010000003d000002f4000000c900fffffffb0000002000730065006c0065006300740069006f006e00200062007500660066006500720200000138000000aa0000023a00000294fb00000014005700690064006500530074006500720065006f02000000e6000000d2000003ee0000030bfb0000000c004b0069006e0065006300740200000186000001060000030c00000261000000010000010f000002f4fc0200000003fb0000001e0054006f006f006c002000500072006f00700065007200740069006500730100000041000000780000000000000000fb0000000a00560069006500770073010000003d000002f4000000a400fffffffb0000001200530065006c0065006300740069006f006e010000025a000000b200000000000000000000000200000490000000a9fc0100000001fb0000000a00560069006500770073030000004e00000080000002e10000019700000003000004420000003efc0100000002fb0000000800540069006d00650100000000000004420000000000000000fb0000000800540069006d0065010000000000000450000000000000000000000234000002f400000004000000040000000800000008fc0000000100000002000000010000000a0054006f006f006c00730100000000ffffffff0000000000000000
  Width: 1200
  X: 72
  Y: 60
```

## Exercise 3: Create a Humanoid Robot Model

### Step 1: Create a complex humanoid URDF

Create `~/ros2_ws/src/robot_modeling_py/robot_modeling_py/urdf/humanoid_robot.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_robot">
  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="torso_height" value="0.5" />
  <xacro:property name="torso_width" value="0.2" />
  <xacro:property name="torso_depth" value="0.15" />
  <xacro:property name="head_radius" value="0.1" />
  <xacro:property name="arm_length" value="0.4" />
  <xacro:property name="arm_radius" value="0.05" />
  <xacro:property name="leg_length" value="0.6" />
  <xacro:property name="leg_radius" value="0.07" />
  <xacro:property name="foot_size" value="0.15 0.08 0.05" />

  <!-- Materials -->
  <material name="red">
    <color rgba="1 0 0 1"/>
  </material>
  <material name="green">
    <color rgba="0 1 0 1"/>
  </material>
  <material name="blue">
    <color rgba="0 0 1 1"/>
  </material>
  <material name="white">
    <color rgba="1 1 1 1"/>
  </material>
  <material name="black">
    <color rgba="0 0 0 1"/>
  </material>

  <!-- Base footprint -->
  <link name="base_footprint">
    <visual>
      <geometry>
        <box size="0.01 0.01 0.01"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.01 0.01 0.01"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.001"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="${torso_height} ${torso_width} ${torso_depth}"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <box size="${torso_height} ${torso_width} ${torso_depth}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.5" ixy="0" ixz="0" iyy="0.5" iyz="0" izz="0.5"/>
    </inertial>
  </link>

  <!-- Base to torso joint -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_footprint"/>
    <child link="torso"/>
    <origin xyz="0 0 0.7" rpy="0 0 0"/>
  </joint>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="${head_radius}"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="${head_radius}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="${torso_height/2} 0 ${torso_depth/2}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="1.0"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="${arm_length}" radius="${arm_radius}"/>
      </geometry>
      <material name="red"/>
      <origin xyz="0 0 ${arm_length/2}" rpy="${M_PI/2} 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="${arm_length}" radius="${arm_radius}"/>
      </geometry>
      <origin xyz="0 0 ${arm_length/2}" rpy="${M_PI/2} 0 0"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="${torso_height/2} ${torso_width/2} 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="50" velocity="2.0"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <cylinder length="${arm_length}" radius="${arm_radius}"/>
      </geometry>
      <material name="red"/>
      <origin xyz="0 0 ${arm_length/2}" rpy="${M_PI/2} 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="${arm_length}" radius="${arm_radius}"/>
      </geometry>
      <origin xyz="0 0 ${arm_length/2}" rpy="${M_PI/2} 0 0"/>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 ${arm_length}" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="${M_PI/2}" effort="30" velocity="2.0"/>
  </joint>

  <!-- Right Arm -->
  <link name="right_upper_arm">
    <visual>
      <geometry>
        <cylinder length="${arm_length}" radius="${arm_radius}"/>
      </geometry>
      <material name="red"/>
      <origin xyz="0 0 ${arm_length/2}" rpy="${M_PI/2} 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="${arm_length}" radius="${arm_radius}"/>
      </geometry>
      <origin xyz="0 0 ${arm_length/2}" rpy="${M_PI/2} 0 0"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="${torso_height/2} ${-torso_width/2} 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="50" velocity="2.0"/>
  </joint>

  <link name="right_lower_arm">
    <visual>
      <geometry>
        <cylinder length="${arm_length}" radius="${arm_radius}"/>
      </geometry>
      <material name="red"/>
      <origin xyz="0 0 ${arm_length/2}" rpy="${M_PI/2} 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="${arm_length}" radius="${arm_radius}"/>
      </geometry>
      <origin xyz="0 0 ${arm_length/2}" rpy="${M_PI/2} 0 0"/>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 ${arm_length}" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="${M_PI/2}" effort="30" velocity="2.0"/>
  </joint>

  <!-- Left Leg -->
  <link name="left_upper_leg">
    <visual>
      <geometry>
        <cylinder length="${leg_length}" radius="${leg_radius}"/>
      </geometry>
      <material name="green"/>
      <origin xyz="0 0 ${-leg_length/2}" rpy="${M_PI/2} 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="${leg_length}" radius="${leg_radius}"/>
      </geometry>
      <origin xyz="0 0 ${-leg_length/2}" rpy="${M_PI/2} 0 0"/>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
  </link>

  <joint name="left_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_leg"/>
    <origin xyz="${-torso_height/4} ${torso_width/2} ${-torso_depth/2}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="1.0"/>
  </joint>

  <link name="left_lower_leg">
    <visual>
      <geometry>
        <cylinder length="${leg_length}" radius="${leg_radius}"/>
      </geometry>
      <material name="green"/>
      <origin xyz="0 0 ${-leg_length/2}" rpy="${M_PI/2} 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="${leg_length}" radius="${leg_radius}"/>
      </geometry>
      <origin xyz="0 0 ${-leg_length/2}" rpy="${M_PI/2} 0 0"/>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.04" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.04"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_upper_leg"/>
    <child link="left_lower_leg"/>
    <origin xyz="0 0 ${-leg_length}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="${M_PI/2}" effort="80" velocity="1.0"/>
  </joint>

  <link name="left_foot">
    <visual>
      <geometry>
        <box size="${foot_size[0]} ${foot_size[1]} ${foot_size[2]}"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <box size="${foot_size[0]} ${foot_size[1]} ${foot_size[2]}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_lower_leg"/>
    <child link="left_foot"/>
    <origin xyz="0 0 ${-leg_length}" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="${-M_PI/4}" upper="${M_PI/4}" effort="40" velocity="1.0"/>
  </joint>

  <!-- Right Leg -->
  <link name="right_upper_leg">
    <visual>
      <geometry>
        <cylinder length="${leg_length}" radius="${leg_radius}"/>
      </geometry>
      <material name="green"/>
      <origin xyz="0 0 ${-leg_length/2}" rpy="${M_PI/2} 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="${leg_length}" radius="${leg_radius}"/>
      </geometry>
      <origin xyz="0 0 ${-leg_length/2}" rpy="${M_PI/2} 0 0"/>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
  </link>

  <joint name="right_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_leg"/>
    <origin xyz="${-torso_height/4} ${-torso_width/2} ${-torso_depth/2}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="1.0"/>
  </joint>

  <link name="right_lower_leg">
    <visual>
      <geometry>
        <cylinder length="${leg_length}" radius="${leg_radius}"/>
      </geometry>
      <material name="green"/>
      <origin xyz="0 0 ${-leg_length/2}" rpy="${M_PI/2} 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="${leg_length}" radius="${leg_radius}"/>
      </geometry>
      <origin xyz="0 0 ${-leg_length/2}" rpy="${M_PI/2} 0 0"/>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.04" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.04"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_upper_leg"/>
    <child link="right_lower_leg"/>
    <origin xyz="0 0 ${-leg_length}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="${M_PI/2}" effort="80" velocity="1.0"/>
  </joint>

  <link name="right_foot">
    <visual>
      <geometry>
        <box size="${foot_size[0]} ${foot_size[1]} ${foot_size[2]}"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <box size="${foot_size[0]} ${foot_size[1]} ${foot_size[2]}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_lower_leg"/>
    <child link="right_foot"/>
    <origin xyz="0 0 ${-leg_length}" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="${-M_PI/4}" upper="${M_PI/4}" effort="40" velocity="1.0"/>
  </joint>

  <!-- IMU sensor in head -->
  <link name="imu_link">
    <visual>
      <geometry>
        <box size="0.01 0.01 0.01"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.01 0.01 0.01"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="imu_joint" type="fixed">
    <parent link="head"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <!-- Camera in head -->
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
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="head"/>
    <child link="camera_link"/>
    <origin xyz="${head_radius} 0 0" rpy="0 0 0"/>
  </joint>
</robot>
```

## Exercise 4: Add Gazebo-Specific Elements to Robot Model

### Step 1: Create Gazebo-extended URDF

Create `~/ros2_ws/src/robot_modeling_py/robot_modeling_py/urdf/humanoid_robot_gazebo.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_robot_gazebo">
  <!-- Include the basic humanoid model -->
  <xacro:include filename="humanoid_robot.xacro" />

  <!-- Gazebo-specific elements -->
  <gazebo reference="torso">
    <material>Gazebo/Blue</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
  </gazebo>

  <gazebo reference="head">
    <material>Gazebo/White</material>
    <mu1>0.1</mu1>
    <mu2>0.1</mu2>
  </gazebo>

  <gazebo reference="left_upper_arm">
    <material>Gazebo/Red</material>
    <mu1>0.1</mu1>
    <mu2>0.1</mu2>
  </gazebo>

  <gazebo reference="left_lower_arm">
    <material>Gazebo/Red</material>
    <mu1>0.1</mu1>
    <mu2>0.1</mu2>
  </gazebo>

  <gazebo reference="right_upper_arm">
    <material>Gazebo/Red</material>
    <mu1>0.1</mu1>
    <mu2>0.1</mu2>
  </gazebo>

  <gazebo reference="right_lower_arm">
    <material>Gazebo/Red</material>
    <mu1>0.1</mu1>
    <mu2>0.1</mu2>
  </gazebo>

  <gazebo reference="left_upper_leg">
    <material>Gazebo/Green</material>
    <mu1>0.3</mu1>
    <mu2>0.3</mu2>
  </gazebo>

  <gazebo reference="left_lower_leg">
    <material>Gazebo/Green</material>
    <mu1>0.3</mu1>
    <mu2>0.3</mu2>
  </gazebo>

  <gazebo reference="left_foot">
    <material>Gazebo/Black</material>
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
  </gazebo>

  <gazebo reference="right_upper_leg">
    <material>Gazebo/Green</material>
    <mu1>0.3</mu1>
    <mu2>0.3</mu2>
  </gazebo>

  <gazebo reference="right_lower_leg">
    <material>Gazebo/Green</material>
    <mu1>0.3</mu1>
    <mu2>0.3</mu2>
  </gazebo>

  <gazebo reference="right_foot">
    <material>Gazebo/Black</material>
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
  </gazebo>

  <!-- IMU Sensor -->
  <gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>true</visualize>
      <topic>imu</topic>
      <plugin filename="libgazebo_ros_imu_sensor.so" name="imu_plugin">
        <ros>
          <namespace>/humanoid</namespace>
          <remapping>~/out:=imu_data</remapping>
        </ros>
        <initial_orientation_as_reference>false</initial_orientation_as_reference>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Camera Sensor -->
  <gazebo reference="camera_link">
    <sensor name="camera" type="camera">
      <always_on>true</always_on>
      <visualize>true</visualize>
      <update_rate>30</update_rate>
      <camera name="head_camera">
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <plugin filename="libgazebo_ros_camera.so" name="camera_plugin">
        <ros>
          <namespace>/humanoid</namespace>
          <remapping>image_raw:=camera/image_raw</remapping>
          <remapping>camera_info:=camera/camera_info</remapping>
        </ros>
        <camera_name>camera</camera_name>
        <frame_name>camera_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- LiDAR Sensor -->
  <gazebo reference="head">
    <sensor name="lidar" type="ray">
      <always_on>true</always_on>
      <visualize>true</visualize>
      <update_rate>10</update_rate>
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
      <plugin filename="libgazebo_ros_laser.so" name="lidar_plugin">
        <ros>
          <namespace>/humanoid</namespace>
          <remapping>scan:=scan</remapping>
        </ros>
        <frame_name>head</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Control plugins for joints -->
  <gazebo>
    <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
      <ros>
        <namespace>/humanoid</namespace>
        <remapping>~/out:=joint_states</remapping>
      </ros>
      <update_rate>30</update_rate>
      <joint_name>left_shoulder_joint</joint_name>
      <joint_name>left_elbow_joint</joint_name>
      <joint_name>right_shoulder_joint</joint_name>
      <joint_name>right_elbow_joint</joint_name>
      <joint_name>left_hip_joint</joint_name>
      <joint_name>left_knee_joint</joint_name>
      <joint_name>left_ankle_joint</joint_name>
      <joint_name>right_hip_joint</joint_name>
      <joint_name>right_knee_joint</joint_name>
      <joint_name>right_ankle_joint</joint_name>
      <joint_name>neck_joint</joint_name>
    </plugin>
  </gazebo>

  <!-- Position control for joints -->
  <gazebo>
    <plugin name="position_controller" filename="libgazebo_ros_joint_trajectory.so">
      <ros>
        <namespace>/humanoid</namespace>
      </ros>
      <update_rate>100</update_rate>
    </plugin>
  </gazebo>
</robot>
```

## Exercise 5: Create SDF Version of Robot Model

### Step 1: Create SDF model directory

```bash
mkdir -p ~/gazebo_ws/models/humanoid_robot
```

### Step 2: Create SDF model files

Create `~/gazebo_ws/models/humanoid_robot/model.config`:

```xml
<?xml version="1.0"?>
<model>
  <name>humanoid_robot</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>
  <description>A humanoid robot model for Gazebo simulation.</description>
</model>
```

Create `~/gazebo_ws/models/humanoid_robot/model.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="humanoid_robot">
    <!-- Torso -->
    <link name="torso">
      <pose>0 0 0.7 0 0 0</pose>
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>0.5</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.5</iyy>
          <iyz>0.0</iyz>
          <izz>0.5</izz>
        </inertia>
      </inertial>

      <collision name="torso_collision">
        <geometry>
          <box>
            <size>0.5 0.2 0.15</size>
          </box>
        </geometry>
      </collision>

      <visual name="torso_visual">
        <geometry>
          <box>
            <size>0.5 0.2 0.15</size>
          </box>
        </geometry>
        <material>
          <ambient>0.0 0.0 1.0 1.0</ambient>
          <diffuse>0.0 0.0 1.0 1.0</diffuse>
          <specular>0.0 0.0 1.0 1.0</specular>
        </material>
      </visual>
    </link>

    <!-- Head -->
    <link name="head">
      <pose>0.25 0 0.85 0 0 0</pose>
      <inertial>
        <mass>2.0</mass>
        <inertia>
          <ixx>0.02</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.02</iyy>
          <iyz>0.0</iyz>
          <izz>0.02</izz>
        </inertia>
      </inertial>

      <collision name="head_collision">
        <geometry>
          <sphere>
            <radius>0.1</radius>
          </sphere>
        </geometry>
      </collision>

      <visual name="head_visual">
        <geometry>
          <sphere>
            <radius>0.1</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>1.0 1.0 1.0 1.0</ambient>
          <diffuse>1.0 1.0 1.0 1.0</diffuse>
          <specular>1.0 1.0 1.0 1.0</specular>
        </material>
      </visual>
    </link>

    <joint name="neck_joint" type="revolute">
      <parent>torso</parent>
      <child>head</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>100</effort>
          <velocity>1.0</velocity>
        </limit>
      </axis>
      <pose>0.25 0 0.15 0 0 0</pose>
    </joint>

    <!-- Left Arm -->
    <link name="left_upper_arm">
      <pose>0.25 0.1 0.7 0 0 0</pose>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.01</iyy>
          <iyz>0.0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>

      <collision name="left_upper_arm_collision">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
      </collision>

      <visual name="left_upper_arm_visual">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>1.0 0.0 0.0 1.0</ambient>
          <diffuse>1.0 0.0 0.0 1.0</diffuse>
          <specular>1.0 0.0 0.0 1.0</specular>
        </material>
      </visual>
    </link>

    <joint name="left_shoulder_joint" type="revolute">
      <parent>torso</parent>
      <child>left_upper_arm</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>50</effort>
          <velocity>2.0</velocity>
        </limit>
      </axis>
      <pose>0 0.1 0 0 0 0</pose>
    </joint>

    <link name="left_lower_arm">
      <pose>0.25 0.1 0.3 0 0 0</pose>
      <inertial>
        <mass>0.8</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.01</iyy>
          <iyz>0.0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>

      <collision name="left_lower_arm_collision">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
      </collision>

      <visual name="left_lower_arm_visual">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>1.0 0.0 0.0 1.0</ambient>
          <diffuse>1.0 0.0 0.0 1.0</diffuse>
          <specular>1.0 0.0 0.0 1.0</specular>
        </material>
      </visual>
    </link>

    <joint name="left_elbow_joint" type="revolute">
      <parent>left_upper_arm</parent>
      <child>left_lower_arm</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>0</lower>
          <upper>1.57</upper>
          <effort>30</effort>
          <velocity>2.0</velocity>
        </limit>
      </axis>
      <pose>0 0 -0.4 0 0 0</pose>
    </joint>

    <!-- Right Arm -->
    <link name="right_upper_arm">
      <pose>0.25 -0.1 0.7 0 0 0</pose>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.01</iyy>
          <iyz>0.0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>

      <collision name="right_upper_arm_collision">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
      </collision>

      <visual name="right_upper_arm_visual">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>1.0 0.0 0.0 1.0</ambient>
          <diffuse>1.0 0.0 0.0 1.0</diffuse>
          <specular>1.0 0.0 0.0 1.0</specular>
        </material>
      </visual>
    </link>

    <joint name="right_shoulder_joint" type="revolute">
      <parent>torso</parent>
      <child>right_upper_arm</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>50</effort>
          <velocity>2.0</velocity>
        </limit>
      </axis>
      <pose>0 -0.1 0 0 0 0</pose>
    </joint>

    <link name="right_lower_arm">
      <pose>0.25 -0.1 0.3 0 0 0</pose>
      <inertial>
        <mass>0.8</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.01</iyy>
          <iyz>0.0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>

      <collision name="right_lower_arm_collision">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
      </collision>

      <visual name="right_lower_arm_visual">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>1.0 0.0 0.0 1.0</ambient>
          <diffuse>1.0 0.0 0.0 1.0</diffuse>
          <specular>1.0 0.0 0.0 1.0</specular>
        </material>
      </visual>
    </link>

    <joint name="right_elbow_joint" type="revolute">
      <parent>right_upper_arm</parent>
      <child>right_lower_arm</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>0</lower>
          <upper>1.57</upper>
          <effort>30</effort>
          <velocity>2.0</velocity>
        </limit>
      </axis>
      <pose>0 0 -0.4 0 0 0</pose>
    </joint>

    <!-- Left Leg -->
    <link name="left_upper_leg">
      <pose>-0.125 0.1 0.55 0 0 0</pose>
      <inertial>
        <mass>2.0</mass>
        <inertia>
          <ixx>0.05</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.05</iyy>
          <iyz>0.0</iyz>
          <izz>0.05</izz>
        </inertia>
      </inertial>

      <collision name="left_upper_leg_collision">
        <geometry>
          <cylinder>
            <radius>0.07</radius>
            <length>0.6</length>
          </cylinder>
        </geometry>
      </collision>

      <visual name="left_upper_leg_visual">
        <geometry>
          <cylinder>
            <radius>0.07</radius>
            <length>0.6</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.0 1.0 0.0 1.0</ambient>
          <diffuse>0.0 1.0 0.0 1.0</diffuse>
          <specular>0.0 1.0 0.0 1.0</specular>
        </material>
      </visual>
    </link>

    <joint name="left_hip_joint" type="revolute">
      <parent>torso</parent>
      <child>left_upper_leg</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>100</effort>
          <velocity>1.0</velocity>
        </limit>
      </axis>
      <pose>-0.375 0.1 -0.075 0 0 0</pose>
    </joint>

    <link name="left_lower_leg">
      <pose>-0.125 0.1 0.25 0 0 0</pose>
      <inertial>
        <mass>1.5</mass>
        <inertia>
          <ixx>0.04</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.04</iyy>
          <iyz>0.0</iyz>
          <izz>0.04</izz>
        </inertia>
      </inertial>

      <collision name="left_lower_leg_collision">
        <geometry>
          <cylinder>
            <radius>0.07</radius>
            <length>0.6</length>
          </cylinder>
        </geometry>
      </collision>

      <visual name="left_lower_leg_visual">
        <geometry>
          <cylinder>
            <radius>0.07</radius>
            <length>0.6</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.0 1.0 0.0 1.0</ambient>
          <diffuse>0.0 1.0 0.0 1.0</diffuse>
          <specular>0.0 1.0 0.0 1.0</specular>
        </material>
      </visual>
    </link>

    <joint name="left_knee_joint" type="revolute">
      <parent>left_upper_leg</parent>
      <child>left_lower_leg</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>0</lower>
          <upper>1.57</upper>
          <effort>80</effort>
          <velocity>1.0</velocity>
        </limit>
      </axis>
      <pose>0 0 -0.6 0 0 0</pose>
    </joint>

    <link name="left_foot">
      <pose>-0.125 0.1 -0.05 0 0 0</pose>
      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.001</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.001</iyy>
          <iyz>0.0</iyz>
          <izz>0.001</izz>
        </inertia>
      </inertial>

      <collision name="left_foot_collision">
        <geometry>
          <box>
            <size>0.15 0.08 0.05</size>
          </box>
        </geometry>
      </collision>

      <visual name="left_foot_visual">
        <geometry>
          <box>
            <size>0.15 0.08 0.05</size>
          </box>
        </geometry>
        <material>
          <ambient>0.0 0.0 0.0 1.0</ambient>
          <diffuse>0.0 0.0 0.0 1.0</diffuse>
          <specular>0.0 0.0 0.0 1.0</specular>
        </material>
      </visual>
    </link>

    <joint name="left_ankle_joint" type="revolute">
      <parent>left_lower_leg</parent>
      <child>left_foot</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-0.785</lower>
          <upper>0.785</upper>
          <effort>40</effort>
          <velocity>1.0</velocity>
        </limit>
      </axis>
      <pose>0 0 -0.6 0 0 0</pose>
    </joint>

    <!-- Right Leg -->
    <link name="right_upper_leg">
      <pose>-0.125 -0.1 0.55 0 0 0</pose>
      <inertial>
        <mass>2.0</mass>
        <inertia>
          <ixx>0.05</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.05</iyy>
          <iyz>0.0</iyz>
          <izz>0.05</izz>
        </inertia>
      </inertial>

      <collision name="right_upper_leg_collision">
        <geometry>
          <cylinder>
            <radius>0.07</radius>
            <length>0.6</length>
          </cylinder>
        </geometry>
      </collision>

      <visual name="right_upper_leg_visual">
        <geometry>
          <cylinder>
            <radius>0.07</radius>
            <length>0.6</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.0 1.0 0.0 1.0</ambient>
          <diffuse>0.0 1.0 0.0 1.0</diffuse>
          <specular>0.0 1.0 0.0 1.0</specular>
        </material>
      </visual>
    </link>

    <joint name="right_hip_joint" type="revolute">
      <parent>torso</parent>
      <child>right_upper_leg</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>100</effort>
          <velocity>1.0</velocity>
        </limit>
      </axis>
      <pose>-0.375 -0.1 -0.075 0 0 0</pose>
    </joint>

    <link name="right_lower_leg">
      <pose>-0.125 -0.1 0.25 0 0 0</pose>
      <inertial>
        <mass>1.5</mass>
        <inertia>
          <ixx>0.04</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.04</iyy>
          <iyz>0.0</iyz>
          <izz>0.04</izz>
        </inertia>
      </inertial>

      <collision name="right_lower_leg_collision">
        <geometry>
          <cylinder>
            <radius>0.07</radius>
            <length>0.6</length>
          </cylinder>
        </geometry>
      </collision>

      <visual name="right_lower_leg_visual">
        <geometry>
          <cylinder>
            <radius>0.07</radius>
            <length>0.6</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.0 1.0 0.0 1.0</ambient>
          <diffuse>0.0 1.0 0.0 1.0</diffuse>
          <specular>0.0 1.0 0.0 1.0</specular>
        </material>
      </visual>
    </link>

    <joint name="right_knee_joint" type="revolute">
      <parent>right_upper_leg</parent>
      <child>right_lower_leg</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>0</lower>
          <upper>1.57</upper>
          <effort>80</effort>
          <velocity>1.0</velocity>
        </limit>
      </axis>
      <pose>0 0 -0.6 0 0 0</pose>
    </joint>

    <link name="right_foot">
      <pose>-0.125 -0.1 -0.05 0 0 0</pose>
      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.001</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.001</iyy>
          <iyz>0.0</iyz>
          <izz>0.001</izz>
        </inertia>
      </inertial>

      <collision name="right_foot_collision">
        <geometry>
          <box>
            <size>0.15 0.08 0.05</size>
          </box>
        </geometry>
      </collision>

      <visual name="right_foot_visual">
        <geometry>
          <box>
            <size>0.15 0.08 0.05</size>
          </box>
        </geometry>
        <material>
          <ambient>0.0 0.0 0.0 1.0</ambient>
          <diffuse>0.0 0.0 0.0 1.0</diffuse>
          <specular>0.0 0.0 0.0 1.0</specular>
        </material>
      </visual>
    </link>

    <joint name="right_ankle_joint" type="revolute">
      <parent>right_lower_leg</parent>
      <child>right_foot</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-0.785</lower>
          <upper>0.785</upper>
          <effort>40</effort>
          <velocity>1.0</velocity>
        </limit>
      </axis>
      <pose>0 0 -0.6 0 0 0</pose>
    </joint>

    <!-- Sensors -->
    <sensor name="imu_sensor" type="imu">
      <pose>0.25 0 0.85 0 0 0</pose>
      <topic>imu</topic>
      <always_on>true</always_on>
      <update_rate>100</update_rate>
    </sensor>

    <sensor name="camera" type="camera">
      <pose>0.35 0 0.85 0 0 0</pose>
      <camera name="head_camera">
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <always_on>true</always_on>
      <update_rate>30</update_rate>
    </sensor>
  </model>
</sdf>
```

## Exercise 6: Build and Test the Robot Models

### Step 1: Update setup.py

Edit `~/ros2_ws/src/robot_modeling_py/setup.py`:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'robot_modeling_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Robot modeling examples',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
```

### Step 2: Build the package

```bash
cd ~/ros2_ws
colcon build --packages-select robot_modeling_py
source ~/ros2_ws/install/setup.bash
```

### Step 3: Test URDF models

```bash
# Check URDF validity
xacro check_args ~/ros2_ws/src/robot_modeling_py/robot_modeling_py/urdf/humanoid_robot.xacro

# Launch RViz to view the robot
ros2 launch robot_modeling_py display_robot.launch.py
```

### Step 4: Test with Gazebo

```bash
# Set Gazebo model path
export GAZEBO_MODEL_PATH=~/gazebo_ws/models:$GAZEBO_MODEL_PATH

# Create a test world with the humanoid robot
echo '<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_test">
    <include>
      <uri>model://sun</uri>
    </include>
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>
    <include>
      <uri>model://humanoid_robot</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>
  </world>
</sdf>' > ~/gazebo_ws/worlds/humanoid_test.sdf

# Launch Gazebo with the humanoid robot
gz sim ~/gazebo_ws/worlds/humanoid_test.sdf
```

## Exercise 7: Model Validation and Troubleshooting

### Step 1: Validate URDF models

```bash
# Install urdf validation tools
sudo apt install ros-humble-urdfdom-py

# Validate URDF
python3 -c "from urdf_parser_py.urdf import URDF; robot = URDF.from_xml_file('/path/to/robot.urdf'); print('Valid URDF')"
```

### Step 2: Common validation checks

Create a validation script `~/ros2_ws/src/robot_modeling_py/robot_modeling_py/validate_robot.py`:

```python
#!/usr/bin/env python3
import xml.etree.ElementTree as ET
from urdf_parser_py.urdf import URDF
import sys

def validate_urdf_model(urdf_path):
    """Validate URDF model for common issues."""
    try:
        # Parse URDF
        robot = URDF.from_xml_file(urdf_path)
        print(f"✓ Successfully parsed URDF: {urdf_path}")

        # Check for common issues
        issues = []

        # Check if robot has a base link
        if not robot.get_root():
            issues.append("No root link found")

        # Check joint limits
        for joint in robot.joints:
            if joint.type in ['revolute', 'prismatic']:
                if joint.limit is None:
                    issues.append(f"Joint {joint.name} has no limits")
                elif joint.limit.effort == 0:
                    issues.append(f"Joint {joint.name} has zero effort limit")

        # Check for links without visuals or collisions
        for link in robot.links:
            if not link.visual and not link.collision:
                issues.append(f"Link {link.name} has no visual or collision geometry")

        # Check inertial properties
        for link in robot.links:
            if link.inertial is None:
                issues.append(f"Link {link.name} has no inertial properties")
            elif link.inertial.mass <= 0:
                issues.append(f"Link {link.name} has invalid mass: {link.inertial.mass}")

        if issues:
            print("⚠ Issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✓ No major issues found")

        return len(issues) == 0

    except Exception as e:
        print(f"✗ Error validating URDF: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 validate_robot.py <urdf_file>")
        sys.exit(1)

    urdf_file = sys.argv[1]
    success = validate_urdf_model(urdf_file)
    sys.exit(0 if success else 1)
```

## Troubleshooting

### Common Issues and Solutions

1. **URDF parsing errors**:
   - Check XML syntax and proper closing tags
   - Ensure all referenced materials are defined
   - Verify joint types and limits are properly specified

2. **Gazebo model not appearing**:
   - Verify GAZEBO_MODEL_PATH includes model directory
   - Check model.config has correct name and SDF reference
   - Ensure SDF file is valid and properly formatted

3. **Physics simulation issues**:
   - Check inertial properties are realistic
   - Verify mass values are positive and reasonable
   - Ensure collision geometries are properly defined

4. **Joint limits not working**:
   - Verify joint types match intended motion
   - Check limit values are in radians for revolute joints
   - Ensure effort and velocity limits are set appropriately

5. **Sensors not publishing data**:
   - Check sensor plugins are properly configured
   - Verify Gazebo plugin paths are correct
   - Ensure sensor topics are accessible

## Assessment Questions

1. What are the key differences between URDF and SDF formats?
2. Why are inertial properties important in robot models?
3. How do you add sensors to a robot model in Gazebo?
4. What are the advantages of using Xacro for complex robots?
5. How do you validate a robot model before simulation?

## Extension Exercises

1. Add a gripper to one of the robot arms with proper kinematics
2. Create a complete walking controller for the humanoid robot
3. Add more sensors (LiDAR, force/torque sensors) to the robot
4. Implement a control interface for the humanoid robot
5. Create a custom mesh for one of the robot parts and integrate it

## Summary

In this lab, you successfully:
- Created detailed robot models in both URDF and SDF formats
- Integrated sensors into robot models for simulation
- Used Xacro to create parameterized and modular robot models
- Validated robot models and addressed common issues
- Created a complete humanoid robot model with proper kinematics

These skills are fundamental for robotics development, as accurate robot models are essential for simulation, control system development, and testing before deployment on physical robots. The ability to create detailed, physically accurate models is crucial for realistic simulation and successful transfer to real-world applications.