# URDF Modeling for Humanoid Robots

## Introduction to URDF

Unified Robot Description Format (URDF) is an XML-based format used in ROS to describe robot models. It defines the physical and visual properties of a robot, including its kinematic structure, inertial properties, and visual appearance. For humanoid robots, URDF is essential for simulation, visualization, and motion planning.

## URDF Structure

A URDF file consists of:
- A single `<robot>` root element
- Multiple `<link>` elements describing rigid bodies
- Multiple `<joint>` elements describing connections between links
- Optional `<gazebo>` elements for simulation-specific properties
- Optional `<material>`, `<visual>`, and `<collision>` elements for appearance and physics

## Basic URDF Elements

### Robot Element
```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Robot content goes here -->
</robot>
```

### Link Element
A link represents a rigid body with physical properties:

```xml
<link name="torso">
  <inertial>
    <mass value="10.0" />
    <origin xyz="0 0 0.2" rpy="0 0 0" />
    <inertia ixx="0.4" ixy="0.0" ixz="0.0" iyy="0.4" iyz="0.0" izz="0.2" />
  </inertial>
  <visual>
    <origin xyz="0 0 0.2" rpy="0 0 0" />
    <geometry>
      <cylinder length="0.4" radius="0.15" />
    </geometry>
    <material name="blue">
      <color rgba="0 0 0.8 1.0" />
    </material>
  </visual>
  <collision>
    <origin xyz="0 0 0.2" rpy="0 0 0" />
    <geometry>
      <cylinder length="0.4" radius="0.15" />
    </geometry>
  </collision>
</link>
```

### Joint Element
A joint connects two links with specific kinematic properties:

```xml
<joint name="torso_head_joint" type="revolute">
  <parent link="torso" />
  <child link="head" />
  <origin xyz="0.0 0.0 0.4" rpy="0 0 0" />
  <axis xyz="0 1 0" />
  <limit lower="-1.57" upper="1.57" effort="30" velocity="1.0" />
</joint>
```

## Humanoid Robot Kinematic Structure

A typical humanoid robot has the following kinematic chains:
- Torso to head (neck joint)
- Torso to left/right arm (shoulder, elbow, wrist joints)
- Torso to left/right leg (hip, knee, ankle joints)

### Complete Humanoid URDF Example

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base Link -->
  <link name="base_link" />

  <!-- Torso -->
  <link name="torso">
    <inertial>
      <mass value="10.0" />
      <origin xyz="0 0 0.2" rpy="0 0 0" />
      <inertia ixx="0.4" ixy="0.0" ixz="0.0" iyy="0.4" iyz="0.0" izz="0.2" />
    </inertial>
    <visual>
      <origin xyz="0 0 0.2" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.4" radius="0.15" />
      </geometry>
      <material name="light_grey">
        <color rgba="0.7 0.7 0.7 1.0" />
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.2" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.4" radius="0.15" />
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <link name="head">
    <inertial>
      <mass value="2.0" />
      <origin xyz="0 0 0.1" rpy="0 0 0" />
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02" />
    </inertial>
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0" />
      <geometry>
        <sphere radius="0.1" />
      </geometry>
      <material name="white">
        <color rgba="1.0 1.0 1.0 1.0" />
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0" />
      <geometry>
        <sphere radius="0.1" />
      </geometry>
    </collision>
  </link>

  <!-- Neck Joint -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso" />
    <child link="head" />
    <origin xyz="0.0 0.0 0.4" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-0.785" upper="0.785" effort="10" velocity="2.0" />
  </joint>

  <!-- Left Arm (Simplified) -->
  <link name="left_upper_arm">
    <inertial>
      <mass value="1.5" />
      <origin xyz="0 0 -0.15" rpy="0 0 0" />
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005" />
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.3" radius="0.05" />
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1.0" />
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.3" radius="0.05" />
      </geometry>
    </collision>
  </link>

  <!-- Shoulder Joint -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso" />
    <child link="left_upper_arm" />
    <origin xyz="0.15 0.1 0.3" rpy="0 0 0" />
    <axis xyz="0 0 1" />
    <limit lower="-1.57" upper="1.57" effort="15" velocity="1.5" />
  </joint>

  <!-- Right Arm (Simplified) -->
  <link name="right_upper_arm">
    <inertial>
      <mass value="1.5" />
      <origin xyz="0 0 -0.15" rpy="0 0 0" />
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005" />
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.3" radius="0.05" />
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1.0" />
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.3" radius="0.05" />
      </geometry>
    </collision>
  </link>

  <!-- Right Shoulder Joint -->
  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso" />
    <child link="right_upper_arm" />
    <origin xyz="0.15 -0.1 0.3" rpy="0 0 0" />
    <axis xyz="0 0 1" />
    <limit lower="-1.57" upper="1.57" effort="15" velocity="1.5" />
  </joint>
</robot>
```

## Joint Types in URDF

- `revolute`: Rotational joint with limited range
- `continuous`: Rotational joint without limits
- `prismatic`: Linear sliding joint with limits
- `fixed`: No movement between links
- `floating`: 6-DOF movement (rarely used)
- `planar`: Movement in a plane (rarely used)

## Xacro for Complex Models

For complex humanoid robots, Xacro (XML Macros) is used to avoid repetition:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">

  <xacro:property name="M_PI" value="3.1415926535897931" />

  <xacro:macro name="simple_arm" params="prefix parent *origin">
    <link name="${prefix}_upper_arm">
      <inertial>
        <mass value="1.5" />
        <origin xyz="0 0 -0.15" />
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005" />
      </inertial>
      <visual>
        <origin xyz="0 0 -0.15" />
        <geometry>
          <cylinder length="0.3" radius="0.05" />
        </geometry>
      </visual>
      <collision>
        <origin xyz="0 0 -0.15" />
        <geometry>
          <cylinder length="0.3" radius="0.05" />
        </geometry>
      </collision>
    </link>

    <joint name="${prefix}_shoulder_joint" type="revolute">
      <parent link="${parent}" />
      <child link="${prefix}_upper_arm" />
      <xacro:insert_block name="origin" />
      <axis xyz="0 0 1" />
      <limit lower="-1.57" upper="1.57" effort="15" velocity="1.5" />
    </joint>
  </xacro:macro>

  <link name="torso">
    <inertial>
      <mass value="10.0" />
      <origin xyz="0 0 0.2" />
      <inertia ixx="0.4" ixy="0.0" ixz="0.0" iyy="0.4" iyz="0.0" izz="0.2" />
    </inertial>
    <visual>
      <origin xyz="0 0 0.2" />
      <geometry>
        <cylinder length="0.4" radius="0.15" />
      </geometry>
    </visual>
  </link>

  <!-- Use the macro for left arm -->
  <xacro:simple_arm prefix="left" parent="torso">
    <origin xyz="0.15 0.1 0.3" rpy="0 0 0" />
  </xacro:simple_arm>

  <!-- Use the macro for right arm -->
  <xacro:simple_arm prefix="right" parent="torso">
    <origin xyz="0.15 -0.1 0.3" rpy="0 0 0" />
  </xacro:simple_arm>
</robot>
```

## Visualization and Debugging

URDF models can be visualized using:
- RViz for ROS visualization
- Gazebo for physics simulation
- `check_urdf` command-line tool for validation
- `urdf_to_graphiz` for kinematic chain visualization

## Best Practices

1. **Mass and Inertia**: Accurately define mass and inertia properties for realistic simulation
2. **Joint Limits**: Set appropriate joint limits to prevent self-collision and damage
3. **Collision vs Visual**: Use simpler geometries for collision detection than for visual representation
4. **Hierarchy**: Maintain a clear parent-child hierarchy for kinematic chains
5. **Xacro**: Use Xacro macros to reduce redundancy in complex models
6. **Units**: Use consistent units (typically meters for length, kilograms for mass)

## Integration with ROS 2

URDF models are typically loaded using the `robot_state_publisher` node, which publishes joint states to TF transforms:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class StatePublisher(Node):
    def __init__(self):
        super().__init__('state_publisher')
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.broadcaster = TransformBroadcaster(self)
        # Implementation to publish joint states and transforms
```

## Summary

URDF modeling is fundamental for representing humanoid robots in ROS 2. Understanding how to create accurate kinematic models with proper physical properties is essential for simulation, visualization, and motion planning. Xacro macros help manage complex models by reducing redundancy and improving maintainability.

## Learning Check

After studying this section, you should be able to:
- Create URDF files for simple and complex robot models
- Define links with proper inertial, visual, and collision properties
- Create joints with appropriate types and limits
- Use Xacro macros to simplify complex models
- Integrate URDF models with ROS 2 visualization tools