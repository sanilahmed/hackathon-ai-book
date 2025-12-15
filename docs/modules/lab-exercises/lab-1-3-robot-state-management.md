---
sidebar_label: 'Lab 1.3: Robot State Management'
---

# Lab Exercise 1.3: Robot State Management in ROS 2

This lab exercise covers managing robot states in ROS 2, including joint states, transforms, and robot state publishing.

## Objectives

- Understand the robot state representation in ROS 2
- Implement a robot state publisher
- Work with TF transforms
- Monitor and visualize robot states

## Prerequisites

- ROS 2 Humble Hawksbill installed
- Basic knowledge of URDF
- Completed Lab 1.1 and 1.2

## Robot State Overview

The robot state includes:
- Joint positions, velocities, and efforts
- Transform relationships between links
- End-effector poses
- Sensor states

## Robot State Publisher

### Implementation

Create a robot state publisher node:

```cpp
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "tf2_msgs/msg/tf_message.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include <vector>
#include <string>

class RobotStatePublisher : public rclcpp::Node
{
public:
    RobotStatePublisher()
    : Node("robot_state_publisher")
    {
        joint_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
        tf_pub_ = this->create_publisher<tf2_msgs::msg::TFMessage>("tf", 10);

        timer_ = this->create_wall_timer(
            50ms, std::bind(&RobotStatePublisher::publish_states, this));
    }

private:
    void publish_states()
    {
        // Create joint state message
        auto joint_msg = sensor_msgs::msg::JointState();
        joint_msg.header.stamp = this->now();
        joint_msg.name = {"joint1", "joint2", "joint3"};
        joint_msg.position = {0.1, 0.2, 0.3};
        joint_msg.velocity = {0.0, 0.0, 0.0};
        joint_msg.effort = {0.0, 0.0, 0.0};

        joint_pub_->publish(joint_msg);

        // Create TF transforms
        tf2_msgs::msg::TFMessage tf_msg;
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = this->now();
        transform.header.frame_id = "base_link";
        transform.child_frame_id = "link1";
        transform.transform.translation.x = 0.1;
        transform.transform.translation.y = 0.0;
        transform.transform.translation.z = 0.1;
        transform.transform.rotation.x = 0.0;
        transform.transform.rotation.y = 0.0;
        transform.transform.rotation.z = 0.0;
        transform.transform.rotation.w = 1.0;

        tf_msg.transforms.push_back(transform);
        tf_pub_->publish(tf_msg);
    }

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
    rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr tf_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};
```

## TF (Transforms) System

### Understanding TF

The TF system manages coordinate frame relationships:
- Static transforms (unchanging)
- Dynamic transforms (changing over time)
- Transform trees for complex robots

### Creating TF Publishers

```python
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class DynamicTfBroadcaster(Node):
    def __init__(self):
        super().__init__('dynamic_tf_broadcaster')
        self.tf_broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(0.1, self.broadcast_transform)
        self.offset = 0

    def broadcast_transform(self):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'turtle1'
        t.child_frame_id = 'carrot1'

        # Define transform with changing values
        t.transform.translation.x = 10 * math.sin(self.offset)
        t.transform.translation.y = 10 * math.cos(self.offset)
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)
        self.offset += 0.05
```

## Working with URDF

### URDF Integration

The robot state publisher works with URDF files to:
- Publish joint states to TF
- Maintain kinematic chains
- Provide collision information

### Example URDF Integration

```xml
<robot name="my_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </visual>
  </link>

  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0.25 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
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

## Visualization

### Using RViz2

Visualize robot states in RViz2:
1. Add RobotModel display
2. Set Fixed Frame to robot's base frame
3. Set Robot Description to your URDF parameter

### Command-line Tools

Monitor robot states:
```bash
# View joint states
ros2 topic echo /joint_states

# View transforms
ros2 run tf2_tools view_frames

# Check transform tree
ros2 run tf2_ros tf2_echo base_link tool0
```

## Advanced State Management

### Robot State Server

For complex robots, use the robot state server:
- Maintains complete robot state
- Provides state queries
- Handles multiple joint state sources

### State Estimation

Implement state estimation for:
- Sensor fusion
- Kalman filtering
- Particle filtering

## Exercise Tasks

1. Create a URDF for a simple 3-DOF robot arm
2. Implement a robot state publisher that publishes joint states
3. Create TF broadcasters for dynamic transforms
4. Visualize the robot in RViz2
5. Add sensor data integration to the state publisher

## Summary

In this lab, you learned to manage robot states in ROS 2, including joint states, transforms, and visualization. You implemented state publishers and learned to work with URDF files for complete robot representation.