# Lab 2.4: Multi-Environment Synchronization

## Overview

In this lab, you will learn how to synchronize robot states and data across multiple simulation environments (Gazebo, Unity) and ROS. You'll implement time synchronization, TF tree coordination, and state broadcasting to create a cohesive digital twin system where all environments reflect the same robot state.

## Objectives

By the end of this lab, you will be able to:
- Implement time synchronization between Gazebo, Unity, and ROS
- Create and maintain consistent TF trees across environments
- Broadcast robot states from one environment to others
- Handle coordinate system transformations between environments
- Validate synchronization accuracy and consistency
- Create a unified digital twin system

## Prerequisites

- Completion of Lab 2.1-2.3 (Gazebo, robot modeling, Unity integration)
- Understanding of ROS TF trees and coordinate transformations
- Basic knowledge of time synchronization concepts
- Experience with both Gazebo and Unity environments

## Duration

3-4 hours

## Exercise 1: Time Synchronization Fundamentals

### Step 1: Understand time synchronization concepts

Time synchronization is crucial for digital twin systems. We need to ensure all environments operate on the same timeline:

- **Real-time simulation**: All environments run at 1x real-time speed
- **Simulated time**: All environments use the same simulated time source
- **Clock synchronization**: All environments publish and subscribe to `/clock` topic

### Step 2: Create a time synchronization node

Create a ROS package for synchronization:

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python time_sync_py --dependencies rclpy builtin_interfaces std_msgs rosgraph_msgs
```

Create `~/ros2_ws/src/time_sync_py/time_sync_py/time_synchronizer.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Header
import time


class TimeSynchronizer(Node):
    def __init__(self):
        super().__init__('time_synchronizer')

        # Parameters
        self.declare_parameter('use_sim_time', True)
        self.declare_parameter('time_step', 0.01)  # 10ms time step
        self.declare_parameter('real_time_factor', 1.0)

        # Publishers
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)

        # Subscribers
        self.gazebo_clock_sub = self.create_subscription(
            Clock,
            '/gazebo_clock',
            self.gazebo_clock_callback,
            10
        )

        self.unity_clock_sub = self.create_subscription(
            Clock,
            '/unity_clock',
            self.unity_clock_callback,
            10
        )

        # Initialize time
        self.current_time = self.get_clock().now().nanoseconds / 1e9
        self.time_step = self.get_parameter('time_step').value
        self.real_time_factor = self.get_parameter('real_time_factor').value

        # Timer for publishing synchronized time
        self.timer = self.create_timer(self.time_step, self.publish_synchronized_time)

        self.get_logger().info('Time Synchronizer started')

    def gazebo_clock_callback(self, msg):
        """Receive time from Gazebo simulation."""
        self.current_time = msg.clock.sec + msg.clock.nanosec / 1e9
        self.get_logger().debug(f'Received time from Gazebo: {self.current_time}')

    def unity_clock_callback(self, msg):
        """Receive time from Unity simulation."""
        self.current_time = msg.clock.sec + msg.clock.nanosec / 1e9
        self.get_logger().debug(f'Received time from Unity: {self.current_time}')

    def publish_synchronized_time(self):
        """Publish synchronized time to all environments."""
        clock_msg = Clock()

        # Update time based on time step
        self.current_time += self.time_step

        # Convert to ROS time format
        total_seconds = int(self.current_time)
        nanoseconds = int((self.current_time - total_seconds) * 1e9)

        clock_msg.clock.sec = total_seconds
        clock_msg.clock.nanosec = nanoseconds

        self.clock_pub.publish(clock_msg)
        self.get_logger().debug(f'Published synchronized time: {self.current_time}')

    def get_synchronized_time(self):
        """Get the current synchronized time."""
        return self.current_time


def main(args=None):
    rclpy.init(args=args)
    time_sync = TimeSynchronizer()

    try:
        rclpy.spin(time_sync)
    except KeyboardInterrupt:
        pass
    finally:
        time_sync.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Exercise 2: TF Tree Synchronization

### Step 1: Create TF synchronization node

Create `~/ros2_ws/src/time_sync_py/time_sync_py/tf_synchronizer.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
import threading
import time


class TFSynchronizer(Node):
    def __init__(self):
        super().__init__('tf_synchronizer')

        # TF buffer and broadcaster
        self.tf_buffer = Buffer()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers for environment-specific TFs
        self.gazebo_tf_pub = self.create_publisher(TFMessage, '/gazebo_tf', 10)
        self.unity_tf_pub = self.create_publisher(TFMessage, '/unity_tf', 10)

        # Subscribers for environment-specific TFs
        self.gazebo_tf_sub = self.create_subscription(
            TFMessage,
            '/gazebo_tf',
            self.gazebo_tf_callback,
            10
        )

        self.unity_tf_sub = self.create_subscription(
            TFMessage,
            '/unity_tf',
            self.unity_tf_callback,
            10
        )

        # Joint states for synchronization
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Timer for broadcasting synchronized TFs
        self.timer = self.create_timer(0.05, self.broadcast_synchronized_tfs)  # 20 Hz

        # Store transforms for synchronization
        self.transforms = {}
        self.joint_states = {}
        self.lock = threading.Lock()

        self.get_logger().info('TF Synchronizer started')

    def gazebo_tf_callback(self, msg):
        """Receive TF transforms from Gazebo."""
        with self.lock:
            for transform in msg.transforms:
                key = f"{transform.header.frame_id}_{transform.child_frame_id}"
                self.transforms[f"gazebo_{key}"] = transform
                # Also broadcast to ROS TF tree
                self.tf_broadcaster.sendTransform(transform)

    def unity_tf_callback(self, msg):
        """Receive TF transforms from Unity."""
        with self.lock:
            for transform in msg.transforms:
                key = f"{transform.header.frame_id}_{transform.child_frame_id}"
                self.transforms[f"unity_{key}"] = transform
                # Also broadcast to ROS TF tree
                self.tf_broadcaster.sendTransform(transform)

    def joint_state_callback(self, msg):
        """Receive joint states."""
        with self.lock:
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.joint_states[name] = {
                        'position': msg.position[i],
                        'velocity': msg.velocity[i] if i < len(msg.velocity) else 0.0,
                        'effort': msg.effort[i] if i < len(msg.effort) else 0.0
                    }

    def broadcast_synchronized_tfs(self):
        """Broadcast synchronized transforms to all environments."""
        with self.lock:
            # Create synchronized TF message
            synchronized_tf_msg = TFMessage()
            synchronized_tf_msg.transforms = []

            # Add all known transforms
            for key, transform in self.transforms.items():
                # Update timestamp to current time
                transform.header.stamp = self.get_clock().now().to_msg()
                synchronized_tf_msg.transforms.append(transform)

            # Publish to all environments
            if synchronized_tf_msg.transforms:
                self.gazebo_tf_pub.publish(synchronized_tf_msg)
                self.unity_tf_pub.publish(synchronized_tf_msg)

    def transform_frame(self, transform, from_env, to_env):
        """Transform coordinates from one environment's frame to another."""
        # In a real implementation, this would handle coordinate system differences
        # between Gazebo (right-handed) and Unity (left-handed)
        new_transform = TransformStamped()
        new_transform.header = transform.header
        new_transform.child_frame_id = transform.child_frame_id
        new_transform.transform = transform.transform

        # Apply coordinate transformation if needed
        # This is a simplified version - real implementation would be more complex
        if from_env == 'unity' and to_env == 'gazebo':
            # Unity to Gazebo coordinate transformation
            new_transform.transform.translation.y, new_transform.transform.translation.z = \
                new_transform.transform.translation.z, new_transform.transform.translation.y
            new_transform.transform.rotation.y, new_transform.transform.rotation.z = \
                new_transform.transform.rotation.z, new_transform.transform.rotation.y

        return new_transform

    def get_transform(self, target_frame, source_frame, time=0):
        """Get transform between frames with error handling."""
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time()
            )
            return transform
        except Exception as e:
            self.get_logger().warn(f'Could not transform {source_frame} to {target_frame}: {e}')
            return None


def main(args=None):
    rclpy.init(args=args)
    tf_sync = TFSynchronizer()

    try:
        rclpy.spin(tf_sync)
    except KeyboardInterrupt:
        pass
    finally:
        tf_sync.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Exercise 3: Robot State Synchronization

### Step 1: Create robot state synchronization

Create `~/ros2_ws/src/time_sync_py/time_sync_py/robot_state_sync.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Header
import threading
import time
import numpy as np


class RobotStateSynchronizer(Node):
    def __init__(self):
        super().__init__('robot_state_synchronizer')

        # Robot state storage
        self.robot_states = {
            'gazebo': {'joint_states': {}, 'odometry': None, 'timestamp': 0},
            'unity': {'joint_states': {}, 'odometry': None, 'timestamp': 0},
            'ros': {'joint_states': {}, 'odometry': None, 'timestamp': 0}
        }

        self.lock = threading.Lock()

        # Publishers for each environment
        self.gazebo_joint_pub = self.create_publisher(JointState, '/gazebo/joint_states', 10)
        self.unity_joint_pub = self.create_publisher(JointState, '/unity/joint_states', 10)
        self.ros_joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        self.gazebo_odom_pub = self.create_publisher(Odometry, '/gazebo/odom', 10)
        self.unity_odom_pub = self.create_publisher(Odometry, '/unity/odom', 10)
        self.ros_odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # Subscribers from each environment
        self.gazebo_joint_sub = self.create_subscription(
            JointState,
            '/gazebo/joint_states',
            self.gazebo_joint_callback,
            10
        )

        self.unity_joint_sub = self.create_subscription(
            JointState,
            '/unity/joint_states',
            self.unity_joint_callback,
            10
        )

        self.ros_joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.ros_joint_callback,
            10
        )

        self.gazebo_odom_sub = self.create_subscription(
            Odometry,
            '/gazebo/odom',
            self.gazebo_odom_callback,
            10
        )

        self.unity_odom_sub = self.create_subscription(
            Odometry,
            '/unity/odom',
            self.unity_odom_callback,
            10
        )

        self.ros_odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.ros_odom_callback,
            10
        )

        # Timer for synchronization
        self.sync_timer = self.create_timer(0.05, self.synchronize_states)  # 20 Hz

        self.get_logger().info('Robot State Synchronizer started')

    def gazebo_joint_callback(self, msg):
        """Receive joint states from Gazebo."""
        with self.lock:
            joint_state = {}
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    joint_state[name] = {
                        'position': msg.position[i],
                        'velocity': msg.velocity[i] if i < len(msg.velocity) else 0.0,
                        'effort': msg.effort[i] if i < len(msg.effort) else 0.0
                    }
            self.robot_states['gazebo']['joint_states'] = joint_state
            self.robot_states['gazebo']['timestamp'] = time.time()

    def unity_joint_callback(self, msg):
        """Receive joint states from Unity."""
        with self.lock:
            joint_state = {}
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    joint_state[name] = {
                        'position': msg.position[i],
                        'velocity': msg.velocity[i] if i < len(msg.velocity) else 0.0,
                        'effort': msg.effort[i] if i < len(msg.effort) else 0.0
                    }
            self.robot_states['unity']['joint_states'] = joint_state
            self.robot_states['unity']['timestamp'] = time.time()

    def ros_joint_callback(self, msg):
        """Receive joint states from ROS."""
        with self.lock:
            joint_state = {}
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    joint_state[name] = {
                        'position': msg.position[i],
                        'velocity': msg.velocity[i] if i < len(msg.velocity) else 0.0,
                        'effort': msg.effort[i] if i < len(msg.effort) else 0.0
                    }
            self.robot_states['ros']['joint_states'] = joint_state
            self.robot_states['ros']['timestamp'] = time.time()

    def gazebo_odom_callback(self, msg):
        """Receive odometry from Gazebo."""
        with self.lock:
            self.robot_states['gazebo']['odometry'] = msg
            self.robot_states['gazebo']['timestamp'] = time.time()

    def unity_odom_callback(self, msg):
        """Receive odometry from Unity."""
        with self.lock:
            self.robot_states['unity']['odometry'] = msg
            self.robot_states['unity']['timestamp'] = time.time()

    def ros_odom_callback(self, msg):
        """Receive odometry from ROS."""
        with self.lock:
            self.robot_states['ros']['odometry'] = msg
            self.robot_states['ros']['timestamp'] = time.time()

    def synchronize_states(self):
        """Synchronize robot states across all environments."""
        with self.lock:
            # Determine the most recent state (or use a specific source)
            latest_env = self.get_latest_environment()

            if latest_env:
                # Update all environments with the latest state
                self.update_all_environments(latest_env)

    def get_latest_environment(self):
        """Get the environment with the most recent state update."""
        latest_time = 0
        latest_env = None

        for env, state in self.robot_states.items():
            if state['timestamp'] > latest_time:
                latest_time = state['timestamp']
                latest_env = env

        return latest_env

    def update_all_environments(self, source_env):
        """Update all environments with the source environment's state."""
        source_state = self.robot_states[source_env]

        # Update joint states for all environments
        for env in self.robot_states:
            if env != source_env and source_state['joint_states']:
                self.publish_joint_states(env, source_state['joint_states'])

        # Update odometry for all environments
        if source_state['odometry']:
            for env in self.robot_states:
                if env != source_env:
                    self.publish_odometry(env, source_state['odometry'])

    def publish_joint_states(self, env, joint_states):
        """Publish joint states to a specific environment."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"{env}_base_link"

        # Convert joint states dict to message format
        for joint_name, joint_data in joint_states.items():
            msg.name.append(joint_name)
            msg.position.append(joint_data['position'])
            msg.velocity.append(joint_data['velocity'])
            msg.effort.append(joint_data['effort'])

        # Publish to appropriate topic
        if env == 'gazebo':
            self.gazebo_joint_pub.publish(msg)
        elif env == 'unity':
            self.unity_joint_pub.publish(msg)
        else:  # ros
            self.ros_joint_pub.publish(msg)

    def publish_odometry(self, env, odometry):
        """Publish odometry to a specific environment."""
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"{env}_odom"
        msg.child_frame_id = f"{env}_base_link"

        # Copy odometry data
        msg.pose = odometry.pose
        msg.twist = odometry.twist

        # Publish to appropriate topic
        if env == 'gazebo':
            self.gazebo_odom_pub.publish(msg)
        elif env == 'unity':
            self.unity_odom_pub.publish(msg)
        else:  # ros
            self.ros_odom_pub.publish(msg)

    def coordinate_transform(self, pose, from_env, to_env):
        """Transform coordinates between different environments."""
        # This would handle coordinate system differences
        # For example: Unity (left-handed) to ROS (right-handed)
        new_pose = Pose()
        new_pose.position = pose.position
        new_pose.orientation = pose.orientation

        if from_env == 'unity' and to_env == 'ros':
            # Transform Unity coordinates to ROS coordinates
            new_pose.position.y, new_pose.position.z = pose.position.z, pose.position.y
            new_pose.orientation.y, new_pose.orientation.z = pose.orientation.z, pose.orientation.y

        return new_pose


def main(args=None):
    rclpy.init(args=args)
    robot_sync = RobotStateSynchronizer()

    try:
        rclpy.spin(robot_sync)
    except KeyboardInterrupt:
        pass
    finally:
        robot_sync.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Exercise 4: Gazebo-Unity Bridge Configuration

### Step 1: Create Gazebo-Unity bridge launch file

Create `~/ros2_ws/src/time_sync_py/time_sync_py/launch/digital_twin_sync.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package directories
    time_sync_pkg = get_package_share_directory('time_sync_py')
    gazebo_pkg = get_package_share_directory('gazebo_ros')

    # Launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    # Time synchronizer node
    time_sync_node = Node(
        package='time_sync_py',
        executable='time_synchronizer',
        name='time_synchronizer',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    # TF synchronizer node
    tf_sync_node = Node(
        package='time_sync_py',
        executable='tf_synchronizer',
        name='tf_synchronizer',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    # Robot state synchronizer node
    robot_sync_node = Node(
        package='time_sync_py',
        executable='robot_state_sync',
        name='robot_state_synchronizer',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    # Joint state broadcaster
    joint_state_broadcaster = Node(
        package='joint_state_broadcaster',
        executable='joint_state_broadcaster',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'robot_description': '<robot name="test"/>',  # Will be replaced with actual URDF
        }],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time,
        time_sync_node,
        tf_sync_node,
        robot_sync_node,
        joint_state_broadcaster,
        robot_state_publisher,
    ])
```

## Exercise 5: Unity Synchronization Scripts

### Step 1: Create Unity synchronization scripts

Create `Assets/Scripts/UnityGazeboSync.cs`:

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Tf2_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Nav_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;

public class UnityGazeboSync : MonoBehaviour
{
    ROSConnection ros;

    // ROS topics
    string clockTopic = "/clock";
    string jointStatesTopic = "/unity/joint_states";
    string gazeboJointStatesTopic = "/gazebo/joint_states";
    string unityTfTopic = "/unity_tf";
    string gazeboTfTopic = "/gazebo_tf";
    string unityOdomTopic = "/unity/odom";
    string gazeboOdomTopic = "/gazebo/odom";

    // Robot components
    public Transform[] robotJoints;
    public string[] jointNames;

    // Robot transforms for TF
    public Transform[] tfTransforms;
    public string[] tfFrameIds;
    public string[] tfParentFrameIds;

    // Synchronization timing
    float lastSyncTime = 0f;
    float syncInterval = 0.05f; // 20 Hz

    void Start()
    {
        ros = ROSConnection.instance;

        // Subscribe to Gazebo joint states
        ros.Subscribe<JointStateMsg>(gazeboJointStatesTopic, OnGazeboJointStatesReceived);

        // Subscribe to Gazebo TF
        ros.Subscribe<TFMessage>(gazeboTfTopic, OnGazeboTfReceived);

        // Subscribe to Gazebo odometry
        ros.Subscribe<OdometryMsg>(gazeboOdomTopic, OnGazeboOdomReceived);
    }

    void Update()
    {
        // Periodically publish Unity states to synchronize
        if (Time.time - lastSyncTime >= syncInterval)
        {
            PublishUnityStates();
            lastSyncTime = Time.time;
        }
    }

    void OnGazeboJointStatesReceived(JointStateMsg jointState)
    {
        // Update Unity robot joints based on Gazebo states
        for (int i = 0; i < jointNames.Length; i++)
        {
            if (i < jointState.name.Length)
            {
                string jointName = jointState.name[i];

                // Find matching joint
                for (int j = 0; j < jointNames.Length; j++)
                {
                    if (jointNames[j] == jointName && j < jointState.position.Length)
                    {
                        // Apply joint position (simplified - you'd need proper joint mapping)
                        if (robotJoints[j] != null)
                        {
                            // For revolute joints, apply rotation
                            robotJoints[j].localRotation = Quaternion.Euler(
                                0,
                                (float)jointState.position[j] * Mathf.Rad2Deg,
                                0
                            );
                        }
                        break;
                    }
                }
            }
        }
    }

    void OnGazeboTfReceived(TFMessage tfMsg)
    {
        // Update Unity transforms based on Gazebo TF
        foreach (var transform in tfMsg.transforms)
        {
            string childFrame = transform.child_frame_id;

            // Find corresponding Unity transform
            for (int i = 0; i < tfFrameIds.Length; i++)
            {
                if (tfFrameIds[i] == childFrame)
                {
                    if (tfTransforms[i] != null)
                    {
                        // Apply transform (with coordinate conversion)
                        Vector3 position = new Vector3(
                            (float)transform.transform.translation.x,
                            (float)transform.transform.translation.z, // Gazebo Z -> Unity Y
                            (float)transform.transform.translation.y  // Gazebo Y -> Unity Z
                        );

                        Quaternion rotation = new Quaternion(
                            (float)transform.transform.rotation.x,
                            (float)transform.transform.rotation.z, // Gazebo Z -> Unity Y
                            (float)transform.transform.rotation.y, // Gazebo Y -> Unity Z
                            (float)transform.transform.rotation.w
                        );

                        tfTransforms[i].position = position;
                        tfTransforms[i].rotation = rotation;
                    }
                    break;
                }
            }
        }
    }

    void OnGazeboOdomReceived(OdometryMsg odom)
    {
        // Update Unity robot position based on Gazebo odometry
        Vector3 position = new Vector3(
            (float)odom.pose.pose.position.x,
            (float)odom.pose.pose.position.z, // Gazebo Z -> Unity Y
            (float)odom.pose.pose.position.y  // Gazebo Y -> Unity Z
        );

        // Apply position to robot base
        if (transform != null)
        {
            transform.position = position;

            // Apply orientation
            Quaternion rotation = new Quaternion(
                (float)odom.pose.pose.orientation.x,
                (float)odom.pose.pose.orientation.z, // Gazebo Z -> Unity Y
                (float)odom.pose.pose.orientation.y, // Gazebo Y -> Unity Z
                (float)odom.pose.pose.orientation.w
            );
            transform.rotation = rotation;
        }
    }

    void PublishUnityStates()
    {
        // Publish Unity joint states
        PublishJointStates();

        // Publish Unity TF
        PublishTf();

        // Publish Unity odometry
        PublishOdometry();
    }

    void PublishJointStates()
    {
        JointStateMsg jointState = new JointStateMsg();
        jointState.header.stamp = new TimeStamp(Time.time);
        jointState.header.frame_id = "unity_base_link";

        List<string> names = new List<string>();
        List<double> positions = new List<double>();
        List<double> velocities = new List<double>();
        List<double> efforts = new List<double>();

        for (int i = 0; i < robotJoints.Length; i++)
        {
            if (robotJoints[i] != null && i < jointNames.Length)
            {
                names.Add(jointNames[i]);
                // Get joint position (simplified)
                positions.Add(robotJoints[i].localEulerAngles.y * Mathf.Deg2Rad);
                velocities.Add(0); // Simplified
                efforts.Add(0);    // Simplified
            }
        }

        jointState.name = names.ToArray();
        jointState.position = positions.ToArray();
        jointState.velocity = velocities.ToArray();
        jointState.effort = efforts.ToArray();

        ros.Publish(jointStatesTopic, jointState);
    }

    void PublishTf()
    {
        TFMessage tfMsg = new TFMessage();

        List<Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs.TransformStamped> transforms =
            new List<Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs.TransformStamped>();

        for (int i = 0; i < tfTransforms.Length; i++)
        {
            if (tfTransforms[i] != null)
            {
                var transformStamped = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs.TransformStamped();
                transformStamped.header.stamp = new TimeStamp(Time.time);
                transformStamped.header.frame_id = tfParentFrameIds[i];
                transformStamped.child_frame_id = tfFrameIds[i];

                // Convert Unity coordinates to ROS coordinates
                transformStamped.transform.translation.x = tfTransforms[i].position.x;
                transformStamped.transform.translation.y = tfTransforms[i].position.z; // Unity Z -> ROS Y
                transformStamped.transform.translation.z = tfTransforms[i].position.y; // Unity Y -> ROS Z

                transformStamped.transform.rotation.x = tfTransforms[i].rotation.x;
                transformStamped.transform.rotation.y = tfTransforms[i].rotation.z; // Unity Z -> ROS Y
                transformStamped.transform.rotation.z = tfTransforms[i].rotation.y; // Unity Y -> ROS Z
                transformStamped.transform.rotation.w = tfTransforms[i].rotation.w;

                transforms.Add(transformStamped);
            }
        }

        tfMsg.transforms = transforms.ToArray();
        ros.Publish(unityTfTopic, tfMsg);
    }

    void PublishOdometry()
    {
        OdometryMsg odom = new OdometryMsg();
        odom.header.stamp = new TimeStamp(Time.time);
        odom.header.frame_id = "unity_odom";
        odom.child_frame_id = "unity_base_link";

        // Position
        odom.pose.pose.position.x = transform.position.x;
        odom.pose.pose.position.y = transform.position.z; // Unity Z -> ROS Y
        odom.pose.pose.position.z = transform.position.y; // Unity Y -> ROS Z

        // Orientation
        odom.pose.pose.orientation.x = transform.rotation.x;
        odom.pose.pose.orientation.y = transform.rotation.z; // Unity Z -> ROS Y
        odom.pose.pose.orientation.z = transform.rotation.y; // Unity Y -> ROS Z
        odom.pose.pose.orientation.w = transform.rotation.w;

        // Velocity (simplified)
        odom.twist.twist.linear.x = 0;
        odom.twist.twist.linear.y = 0;
        odom.twist.twist.linear.z = 0;
        odom.twist.twist.angular.x = 0;
        odom.twist.twist.angular.y = 0;
        odom.twist.twist.angular.z = 0;

        ros.Publish(unityOdomTopic, odom);
    }
}
```

## Exercise 6: Validation and Testing

### Step 1: Create validation tools

Create `~/ros2_ws/src/time_sync_py/time_sync_py/sync_validator.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float32
import time
import numpy as np


class SyncValidator(Node):
    def __init__(self):
        super().__init__('sync_validator')

        # Data storage for validation
        self.gazebo_data = {'joint_states': {}, 'odometry': None, 'timestamp': 0}
        self.unity_data = {'joint_states': {}, 'odometry': None, 'timestamp': 0}
        self.ros_data = {'joint_states': {}, 'odometry': None, 'timestamp': 0}

        # Validation metrics
        self.sync_accuracy = 0.0
        self.time_drift = 0.0
        self.position_error = 0.0

        # Subscribers for all environments
        self.gazebo_joint_sub = self.create_subscription(
            JointState,
            '/gazebo/joint_states',
            self.gazebo_joint_callback,
            10
        )

        self.unity_joint_sub = self.create_subscription(
            JointState,
            '/unity/joint_states',
            self.unity_joint_callback,
            10
        )

        self.ros_joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.ros_joint_callback,
            10
        )

        self.gazebo_odom_sub = self.create_subscription(
            Odometry,
            '/gazebo/odom',
            self.gazebo_odom_callback,
            10
        )

        self.unity_odom_sub = self.create_subscription(
            Odometry,
            '/unity/odom',
            self.unity_odom_callback,
            10
        )

        self.ros_odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.ros_odom_callback,
            10
        )

        # Publishers for validation metrics
        self.sync_accuracy_pub = self.create_publisher(Float32, '/sync_accuracy', 10)
        self.time_drift_pub = self.create_publisher(Float32, '/time_drift', 10)
        self.position_error_pub = self.create_publisher(Float32, '/position_error', 10)

        # Timer for validation
        self.validation_timer = self.create_timer(1.0, self.validate_synchronization)

        self.get_logger().info('Synchronization Validator started')

    def gazebo_joint_callback(self, msg):
        """Receive joint states from Gazebo."""
        joint_state = {}
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                joint_state[name] = msg.position[i]
        self.gazebo_data['joint_states'] = joint_state
        self.gazebo_data['timestamp'] = time.time()

    def unity_joint_callback(self, msg):
        """Receive joint states from Unity."""
        joint_state = {}
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                joint_state[name] = msg.position[i]
        self.unity_data['joint_states'] = joint_state
        self.unity_data['timestamp'] = time.time()

    def ros_joint_callback(self, msg):
        """Receive joint states from ROS."""
        joint_state = {}
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                joint_state[name] = msg.position[i]
        self.ros_data['joint_states'] = joint_state
        self.ros_data['timestamp'] = time.time()

    def gazebo_odom_callback(self, msg):
        """Receive odometry from Gazebo."""
        self.gazebo_data['odometry'] = msg
        self.gazebo_data['timestamp'] = time.time()

    def unity_odom_callback(self, msg):
        """Receive odometry from Unity."""
        self.unity_data['odometry'] = msg
        self.unity_data['timestamp'] = time.time()

    def ros_odom_callback(self, msg):
        """Receive odometry from ROS."""
        self.ros_data['odometry'] = msg
        self.ros_data['timestamp'] = time.time()

    def validate_synchronization(self):
        """Validate synchronization between all environments."""
        # Calculate time synchronization accuracy
        time_drift = self.calculate_time_drift()

        # Calculate position synchronization accuracy
        position_error = self.calculate_position_error()

        # Calculate overall sync accuracy
        sync_accuracy = self.calculate_sync_accuracy(time_drift, position_error)

        # Publish validation metrics
        accuracy_msg = Float32()
        accuracy_msg.data = sync_accuracy
        self.sync_accuracy_pub.publish(accuracy_msg)

        drift_msg = Float32()
        drift_msg.data = time_drift
        self.time_drift_pub.publish(drift_msg)

        error_msg = Float32()
        error_msg.data = position_error
        self.position_error_pub.publish(error_msg)

        # Log validation results
        self.get_logger().info(
            f'Sync Validation - Accuracy: {sync_accuracy:.3f}, '
            f'Time Drift: {time_drift:.3f}s, Position Error: {position_error:.3f}m'
        )

    def calculate_time_drift(self):
        """Calculate time drift between environments."""
        timestamps = [
            self.gazebo_data['timestamp'],
            self.unity_data['timestamp'],
            self.ros_data['timestamp']
        ]

        if any(ts == 0 for ts in timestamps):
            return float('inf')  # Not enough data yet

        # Calculate standard deviation of timestamps
        mean_time = sum(timestamps) / len(timestamps)
        variance = sum((ts - mean_time) ** 2 for ts in timestamps) / len(timestamps)
        time_drift = variance ** 0.5

        return time_drift

    def calculate_position_error(self):
        """Calculate position error between environments."""
        positions = []

        # Extract positions from odometry data
        for data in [self.gazebo_data, self.unity_data, self.ros_data]:
            if data['odometry'] is not None:
                pos = data['odometry'].pose.pose.position
                positions.append(np.array([pos.x, pos.y, pos.z]))

        if len(positions) < 2:
            return float('inf')  # Not enough data

        # Calculate pairwise distances and return mean
        total_error = 0
        pairs = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.linalg.norm(positions[i] - positions[j])
                total_error += distance
                pairs += 1

        return total_error / pairs if pairs > 0 else 0

    def calculate_sync_accuracy(self, time_drift, position_error):
        """Calculate overall synchronization accuracy."""
        # Normalize metrics to 0-1 scale (lower is better)
        max_acceptable_drift = 0.1  # 100ms
        max_acceptable_error = 0.1  # 10cm

        drift_score = max(0, 1 - time_drift / max_acceptable_drift)
        error_score = max(0, 1 - position_error / max_acceptable_error)

        # Weighted average
        accuracy = 0.5 * drift_score + 0.5 * error_score

        return min(1.0, max(0.0, accuracy))


def main(args=None):
    rclpy.init(args=args)
    validator = SyncValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        pass
    finally:
        validator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Exercise 7: Build and Test the Synchronization System

### Step 1: Update setup.py

Edit `~/ros2_ws/src/time_sync_py/setup.py`:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'time_sync_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Time and state synchronization for digital twin',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'time_synchronizer = time_sync_py.time_synchronizer:main',
            'tf_synchronizer = time_sync_py.tf_synchronizer:main',
            'robot_state_sync = time_sync_py.robot_state_sync:main',
            'sync_validator = time_sync_py.sync_validator:main',
        ],
    },
)
```

### Step 2: Build the package

```bash
cd ~/ros2_ws
colcon build --packages-select time_sync_py
source ~/ros2_ws/install/setup.bash
```

### Step 3: Test the synchronization system

```bash
# Run the complete synchronization system
ros2 launch time_sync_py digital_twin_sync.launch.py
```

### Step 4: Test with validation

```bash
# In another terminal, run the validation node
source ~/ros2_ws/install/setup.bash
ros2 run time_sync_py sync_validator
```

## Exercise 8: Advanced Synchronization Features

### Step 1: Create a coordinator node

Create `~/ros2_ws/src/time_sync_py/time_sync_py/sync_coordinator.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
import time


class SyncCoordinator(Node):
    def __init__(self):
        super().__init__('sync_coordinator')

        # State management
        self.sync_enabled = True
        self.primary_environment = 'gazebo'  # Which environment leads
        self.last_sync_time = time.time()

        # Publishers
        self.sync_control_pub = self.create_publisher(Bool, '/sync_control', 10)
        self.sync_status_pub = self.create_publisher(String, '/sync_status', 10)

        # Subscribers
        self.gazebo_sync_sub = self.create_subscription(
            String,
            '/gazebo/sync_status',
            self.gazebo_sync_callback,
            10
        )

        self.unity_sync_sub = self.create_subscription(
            String,
            '/unity/sync_status',
            self.unity_sync_callback,
            10
        )

        # Timer for coordination
        self.coordination_timer = self.create_timer(0.1, self.coordinate_sync)

        self.get_logger().info('Synchronization Coordinator started')

    def gazebo_sync_callback(self, msg):
        """Receive sync status from Gazebo."""
        self.get_logger().debug(f'Gazebo sync status: {msg.data}')

    def unity_sync_callback(self, msg):
        """Receive sync status from Unity."""
        self.get_logger().debug(f'Unity sync status: {msg.data}')

    def coordinate_sync(self):
        """Coordinate synchronization activities."""
        # Check if sync should be enabled/disabled
        if self.should_adjust_sync():
            self.toggle_sync()

        # Check if primary environment should be changed
        if self.should_change_primary():
            self.change_primary_environment()

        # Publish sync status
        status_msg = String()
        status_msg.data = f"Primary: {self.primary_environment}, Enabled: {self.sync_enabled}"
        self.sync_status_pub.publish(status_msg)

    def should_adjust_sync(self):
        """Determine if sync should be adjusted."""
        # Implement logic to determine if sync needs adjustment
        # For example, based on validation metrics
        return False

    def toggle_sync(self):
        """Toggle synchronization on/off."""
        self.sync_enabled = not self.sync_enabled
        control_msg = Bool()
        control_msg.data = self.sync_enabled
        self.sync_control_pub.publish(control_msg)
        self.get_logger().info(f'Synchronization {"enabled" if self.sync_enabled else "disabled"}')

    def should_change_primary(self):
        """Determine if primary environment should be changed."""
        # Implement logic to determine if primary environment should change
        # For example, based on which environment has the most recent/accurate data
        return False

    def change_primary_environment(self):
        """Change which environment is considered primary."""
        if self.primary_environment == 'gazebo':
            self.primary_environment = 'unity'
        else:
            self.primary_environment = 'gazebo'
        self.get_logger().info(f'Primary environment changed to: {self.primary_environment}')


def main(args=None):
    rclpy.init(args=args)
    coordinator = SyncCoordinator()

    try:
        rclpy.spin(coordinator)
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Troubleshooting

### Common Issues and Solutions

1. **Time synchronization drift**:
   - Ensure all environments use the same time source
   - Check that `use_sim_time` parameter is properly set
   - Verify that simulation rates are consistent

2. **TF tree conflicts**:
   - Use environment-specific frame prefixes (gazebo_*, unity_*)
   - Ensure TF trees are properly connected
   - Check for circular dependencies in TF

3. **Joint state mapping issues**:
   - Verify joint names match across environments
   - Check joint limits and ranges are consistent
   - Ensure proper coordinate system transformations

4. **Performance problems**:
   - Reduce synchronization frequency for less critical updates
   - Use appropriate data types and message sizes
   - Consider using latching for static transforms

5. **Coordinate system mismatches**:
   - Implement proper coordinate transformation functions
   - Verify axis mappings between environments
   - Test with simple geometric shapes first

## Assessment Questions

1. How do you handle coordinate system differences between Gazebo and Unity?
2. What is the role of the `/clock` topic in time synchronization?
3. How would you validate that all environments are properly synchronized?
4. What are the challenges in synchronizing joint states across environments?
5. How can you optimize the synchronization frequency for performance?

## Extension Exercises

1. Implement fault tolerance in the synchronization system
2. Add network latency compensation for distributed environments
3. Create a visualization tool to monitor synchronization quality
4. Implement selective synchronization for different robot parts
5. Add machine learning-based prediction for smoother synchronization

## Summary

In this lab, you successfully:
- Implemented time synchronization between Gazebo, Unity, and ROS
- Created TF tree synchronization across environments
- Developed robot state synchronization mechanisms
- Validated synchronization accuracy and consistency
- Created a unified digital twin coordination system

These skills are essential for creating robust digital twin systems where multiple simulation environments maintain consistent robot states. The synchronization mechanisms you've implemented ensure that actions in one environment are properly reflected in others, enabling seamless integration between different simulation and visualization platforms.