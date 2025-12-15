---
sidebar_label: 'Lab 2.4: Multi-Environment Synchronization'
---

# Lab Exercise 2.4: Multi-Environment Synchronization

This lab exercise covers synchronizing robot states and behaviors across multiple simulation environments (Gazebo and Unity).

## Objectives

- Understand multi-environment synchronization challenges
- Implement state synchronization between Gazebo and Unity
- Create a unified representation of robot state
- Handle time synchronization across environments

## Prerequisites

- Gazebo simulation environment set up
- Unity robotics environment set up
- Basic ROS 2 knowledge

## Synchronization Architecture

### Multi-Environment Overview

In a digital twin system, we need to synchronize:
- Robot joint states
- Sensor data
- Environmental conditions
- Time progression

### Synchronization Patterns

#### Centralized Synchronization

One central node manages synchronization between all environments:
```
        +------------------+
        | Synchronization  |
        |    Node          |
        +--------+---------+
                 |
        +--------+---------+
        |                |
    +---v----+      +----v---+
    | Gazebo |      | Unity  |
    +--------+      +--------+
```

#### Distributed Synchronization

Each environment maintains its own synchronization logic:
```
    +--------+      +--------+
    | Gazebo |<---->| Unity  |
    +--------+      +--------+
         |              |
         +--------------+
```

## Time Synchronization

### Time Management Strategies

#### Real-time Synchronization

```python
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from std_msgs.msg import Header
from rclpy.time import Time as ROSTime

class TimeSynchronizer(Node):
    def __init__(self):
        super().__init__('time_synchronizer')

        # Publishers for time synchronization
        self.gazebo_time_pub = self.create_publisher(Time, 'gazebo/time', 10)
        self.unity_time_pub = self.create_publisher(Time, 'unity/time', 10)

        # Timer for time synchronization
        self.timer = self.create_timer(0.01, self.sync_time)  # 100Hz sync

        # Store reference time
        self.start_time = self.get_clock().now()

    def sync_time(self):
        # Get current ROS time
        current_time = self.get_clock().now()

        # Create time message
        time_msg = current_time.to_msg()

        # Publish to both environments
        self.gazebo_time_pub.publish(time_msg)
        self.unity_time_pub.publish(time_msg)

        self.get_logger().debug(f'Synchronized time: {current_time}')
```

#### Simulation Time Synchronization

```python
class SimulationTimeSynchronizer:
    def __init__(self):
        self.gazebo_time = 0.0
        self.unity_time = 0.0
        self.last_sync_time = 0.0
        self.time_scale = 1.0  # 1x, 2x, 0.5x, etc.

    def synchronize_simulation_time(self):
        # Calculate elapsed time since last sync
        elapsed = self.get_current_time() - self.last_sync_time

        # Apply time scaling
        scaled_elapsed = elapsed * self.time_scale

        # Update both simulation times
        self.gazebo_time += scaled_elapsed
        self.unity_time += scaled_elapsed

        # Send time updates to both environments
        self.send_time_to_gazebo(self.gazebo_time)
        self.send_time_to_unity(self.unity_time)

        # Update last sync time
        self.last_sync_time = self.get_current_time()
```

## State Synchronization

### Robot State Publisher

```python
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros

class MultiEnvStateSync(Node):
    def __init__(self):
        super().__init__('multi_env_state_sync')

        # Publishers for both environments
        self.gazebo_joint_pub = self.create_publisher(JointState, '/gazebo/joint_states', 10)
        self.unity_joint_pub = self.create_publisher(JointState, '/unity/joint_states', 10)
        self.gazebo_odom_pub = self.create_publisher(Odometry, '/gazebo/odom', 10)
        self.unity_odom_pub = self.create_publisher(Odometry, '/unity/odom', 10)

        # Subscribers for robot state
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # State storage
        self.current_joint_state = None
        self.current_odom = None

    def joint_state_callback(self, msg):
        # Store current state
        self.current_joint_state = msg

        # Publish to both environments
        self.publish_to_gazebo_joint(msg)
        self.publish_to_unity_joint(msg)

    def odom_callback(self, msg):
        # Store current state
        self.current_odom = msg

        # Publish to both environments
        self.publish_to_gazebo_odom(msg)
        self.publish_to_unity_odom(msg)

    def publish_to_gazebo_joint(self, joint_state):
        # Modify message for Gazebo-specific requirements
        gazebo_msg = JointState()
        gazebo_msg.header = joint_state.header
        gazebo_msg.name = joint_state.name
        gazebo_msg.position = joint_state.position
        gazebo_msg.velocity = joint_state.velocity
        gazebo_msg.effort = joint_state.effort

        self.gazebo_joint_pub.publish(gazebo_msg)

    def publish_to_unity_joint(self, joint_state):
        # Modify message for Unity-specific requirements
        unity_msg = JointState()
        unity_msg.header = joint_state.header
        unity_msg.name = joint_state.name
        unity_msg.position = joint_state.position
        unity_msg.velocity = joint_state.velocity
        unity_msg.effort = joint_state.effort

        self.unity_joint_pub.publish(unity_msg)

    def publish_to_gazebo_odom(self, odom):
        # Publish to Gazebo
        self.gazebo_odom_pub.publish(odom)

    def publish_to_unity_odom(self, odom):
        # Publish to Unity
        self.unity_odom_pub.publish(odom)
```

### Sensor Data Synchronization

```python
from sensor_msgs.msg import LaserScan, Image, Imu

class SensorSync(Node):
    def __init__(self):
        super().__init__('sensor_sync')

        # Publishers for both environments
        self.gazebo_laser_pub = self.create_publisher(LaserScan, '/gazebo/laser_scan', 10)
        self.unity_laser_pub = self.create_publisher(LaserScan, '/unity/laser_scan', 10)
        self.gazebo_camera_pub = self.create_publisher(Image, '/gazebo/camera/image_raw', 10)
        self.unity_camera_pub = self.create_publisher(Image, '/unity/camera/image_raw', 10)

        # Subscribers for sensor data
        self.laser_sub = self.create_subscription(LaserScan, '/laser_scan', self.laser_callback, 10)
        self.camera_sub = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)

    def laser_callback(self, msg):
        # Publish to both environments
        self.gazebo_laser_pub.publish(msg)
        self.unity_laser_pub.publish(msg)

    def camera_callback(self, msg):
        # Publish to both environments
        self.gazebo_camera_pub.publish(msg)
        self.unity_camera_pub.publish(msg)
```

## Gazebo-Specific Synchronization

### Gazebo State Publisher

```cpp
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "gazebo_msgs/msg/model_states.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"

class GazeboStateSync : public rclcpp::Node
{
public:
    GazeboStateSync() : Node("gazebo_state_sync")
    {
        // Publishers
        joint_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "/gazebo/joint_states", 10);
        model_state_pub_ = this->create_publisher<gazebo_msgs::msg::ModelStates>(
            "/gazebo/model_states", 10);

        // Subscribers
        joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&GazeboStateSync::joint_callback, this, std::placeholders::_1));

        // TF broadcaster
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
    }

private:
    void joint_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        // Forward joint states to Gazebo
        joint_pub_->publish(*msg);

        // Create and publish model states
        publish_model_states(msg);

        // Publish transforms
        publish_transforms(msg);
    }

    void publish_model_states(const sensor_msgs::msg::JointState::SharedPtr joint_msg)
    {
        gazebo_msgs::msg::ModelStates model_states;

        // Convert joint states to model states
        // This would depend on your specific robot model
        model_states.name = {"robot_model"};  // Model names
        model_states.pose = {create_pose_from_joints(joint_msg)};
        model_states.twist = {create_twist_from_joints(joint_msg)};

        model_state_pub_->publish(model_states);
    }

    geometry_msgs::msg::Pose create_pose_from_joints(const sensor_msgs::msg::JointState::SharedPtr joint_msg)
    {
        // Create pose based on joint configuration
        geometry_msgs::msg::Pose pose;
        // Implementation depends on kinematics
        return pose;
    }

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
    rclcpp::Publisher<gazebo_msgs::msg::ModelStates>::SharedPtr model_state_pub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};
```

## Unity-Specific Synchronization

### Unity State Synchronization Script

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Nav;
using RosMessageTypes.Geometry;

public class UnityStateSync : MonoBehaviour
{
    ROSConnection ros;
    public GameObject robot;
    private float lastSyncTime;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Subscribe to state updates from ROS
        ros.Subscribe<JointStateMsg>("unity/joint_states", OnJointStateUpdate);
        ros.Subscribe<OdometryMsg>("unity/odom", OnOdomUpdate);

        // Initialize sync time
        lastSyncTime = Time.time;
    }

    void OnJointStateUpdate(JointStateMsg jointState)
    {
        // Update robot joints in Unity based on received state
        for (int i = 0; i < jointState.name.Length && i < jointState.position.Length; i++)
        {
            UpdateRobotJoint(jointState.name[i], jointState.position[i]);
        }
    }

    void OnOdomUpdate(OdometryMsg odom)
    {
        // Update robot position and orientation in Unity
        robot.transform.position = new Vector3(
            (float)odom.pose.pose.position.x,
            (float)odom.pose.pose.position.y,
            (float)odom.pose.pose.position.z
        );

        robot.transform.rotation = new Quaternion(
            (float)odom.pose.pose.orientation.x,
            (float)odom.pose.pose.orientation.y,
            (float)odom.pose.pose.orientation.z,
            (float)odom.pose.pose.orientation.w
        );
    }

    void UpdateRobotJoint(string jointName, double position)
    {
        // Find the joint in the robot hierarchy and update its position
        Transform joint = robot.transform.Find(jointName);
        if (joint != null)
        {
            // Update joint position based on joint type
            // For revolute joints: apply rotation
            // For prismatic joints: apply translation
            joint.localEulerAngles = new Vector3(0, 0, (float)position * Mathf.Rad2Deg);
        }
    }

    void Update()
    {
        // Periodically send Unity state back to ROS
        if (Time.time - lastSyncTime > 0.05f) // 20Hz update
        {
            PublishUnityState();
            lastSyncTime = Time.time;
        }
    }

    void PublishUnityState()
    {
        // Publish current Unity robot state to ROS
        var jointState = new JointStateMsg();
        jointState.header = new std_msgs.HeaderMsg();
        jointState.header.stamp = new TimeStamp(0, (uint)(Time.time * 1e9));
        jointState.header.frame_id = "unity_base";

        // Get joint names and positions from Unity robot
        var jointNames = GetJointNames();
        var jointPositions = GetJointPositions();

        jointState.name = jointNames;
        jointState.position = jointPositions;

        ros.Publish("unity/current_joint_states", jointState);
    }

    string[] GetJointNames()
    {
        // Return array of joint names in the robot
        // This should match the joint names in your URDF
        return new string[] { "joint1", "joint2", "joint3" };
    }

    double[] GetJointPositions()
    {
        // Return current joint positions from Unity robot
        // This would involve traversing the robot hierarchy
        return new double[] { 0.1, 0.2, 0.3 };
    }
}
```

## Synchronization Validation

### Validation Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

class SyncValidator(Node):
    def __init__(self):
        super().__init__('sync_validator')

        # Subscribers for both environments
        self.gazebo_sub = self.create_subscription(JointState, '/gazebo/joint_states', self.gazebo_callback, 10)
        self.unity_sub = self.create_subscription(JointState, '/unity/joint_states', self.unity_callback, 10)

        # Publisher for validation results
        self.validation_pub = self.create_publisher(Float64MultiArray, '/sync_validation', 10)

        # Storage for comparison
        self.gazebo_state = None
        self.unity_state = None
        self.sync_errors = []

    def gazebo_callback(self, msg):
        self.gazebo_state = msg
        self.validate_sync()

    def unity_callback(self, msg):
        self.unity_state = msg
        self.validate_sync()

    def validate_sync(self):
        if self.gazebo_state is None or self.unity_state is None:
            return

        # Compare joint states
        if len(self.gazebo_state.position) != len(self.unity_state.position):
            self.get_logger().warn('Joint count mismatch between environments')
            return

        # Calculate synchronization error
        errors = []
        for i in range(len(self.gazebo_state.position)):
            error = abs(self.gazebo_state.position[i] - self.unity_state.position[i])
            errors.append(error)

        avg_error = np.mean(errors)
        max_error = np.max(errors)

        # Log validation results
        self.get_logger().info(f'Sync validation - Avg error: {avg_error:.4f}, Max error: {max_error:.4f}')

        # Store for trend analysis
        self.sync_errors.append(avg_error)
        if len(self.sync_errors) > 100:
            self.sync_errors.pop(0)

        # Publish validation metrics
        validation_msg = Float64MultiArray()
        validation_msg.data = [avg_error, max_error, len(self.sync_errors)]
        self.validation_pub.publish(validation_msg)
```

## Exercise Tasks

1. Create a synchronization node that publishes robot states to both Gazebo and Unity
2. Implement time synchronization between the environments
3. Add sensor data synchronization (at least one sensor type)
4. Create a validation node to measure synchronization accuracy
5. Test the synchronization with a moving robot
6. Analyze the synchronization errors and identify improvement areas

## Performance Considerations

### Optimization Strategies

- Use appropriate update rates for different data types
- Implement data compression for large messages
- Use efficient data structures for state storage
- Consider network bandwidth limitations

### Quality of Service (QoS)

Configure appropriate QoS settings for different types of data:
- Joint states: Reliable delivery with moderate frequency
- Sensor data: Best effort with high frequency
- Control commands: Reliable delivery with high priority

## Troubleshooting

### Common Issues

- **Time drift**: Implement periodic time corrections
- **State desynchronization**: Use state validation and correction
- **Network delays**: Implement predictive synchronization
- **Data format mismatches**: Standardize message formats

## Summary

In this lab, you learned to synchronize robot states across multiple simulation environments. You implemented time synchronization, state synchronization, and validation mechanisms to ensure consistency between Gazebo and Unity environments.