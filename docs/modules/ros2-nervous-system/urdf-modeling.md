---
sidebar_label: 'URDF Modeling'
---

# URDF Modeling in ROS 2

URDF (Unified Robot Description Format) is used in ROS 2 to describe robot models.

## Overview

URDF is an XML format that describes robots in terms of:
- Links: Rigid bodies with visual and collision properties
- Joints: Connections between links
- Transmissions: Mapping between actuators and joints
- Materials: Visual appearance properties

## Basic Structure

A basic URDF file includes:

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Links -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Joints -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0.25 0" rpy="0 0 0"/>
  </joint>

  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.1"/>
      </geometry>
    </visual>
  </link>
</robot>
```

## Robot State Publisher

The robot_state_publisher node:
- Reads the URDF from the parameter server
- Publishes the joint states as TF transforms
- Makes the robot's kinematic structure available to other nodes

## Xacro

Xacro is an XML macro language that allows:
- Variable definitions
- Macros
- Mathematical expressions
- Inclusion of other xacro files

This makes complex robot descriptions more manageable.