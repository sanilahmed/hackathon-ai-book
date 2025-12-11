# Lab 1.3: Robot State Management

## Overview

In this lab, you will learn about robot state management in ROS 2, focusing on TF (Transforms) trees, robot state publisher, and joint state management. You'll implement systems to track and broadcast the robot's state, including positions, orientations, and joint angles.

## Objectives

By the end of this lab, you will be able to:
- Understand the TF tree concept and its importance in robotics
- Implement robot state publisher for broadcasting transforms
- Create joint state publisher for tracking joint positions
- Use TF tools for debugging and visualization
- Integrate robot state management with URDF models
- Transform coordinates between different frames

## Prerequisites

- Completion of Lab 1.1 and Lab 1.2
- Understanding of ROS 2 topics, services, and actions
- Basic knowledge of coordinate systems and transformations
- Understanding of robot kinematics concepts

## Duration

3 hours

## Exercise 1: Understanding TF Trees

### Step 1: Install TF2 tools

```bash
sudo apt update
sudo apt install ros-humble-tf2-tools ros-humble-tf2-ros
```

### Step 2: Create a robot state management package

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python robot_state_py --dependencies rclpy std_msgs sensor_msgs geometry_msgs tf2_ros tf2_geometry_msgs tf2_msgs
```

### Step 3: Create a simple URDF robot model

Create the URDF directory:

```bash
mkdir -p ~/ros2_ws/src/robot_state_py/robot_state_py/urdf
```

Create `~/ros2_ws/src/robot_state_py/robot_state_py/urdf/simple_robot.urdf`:

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
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
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
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.2 0 0.05" rpy="0 0 0"/>
  </joint>
</robot>
```

## Exercise 2: Implement Robot State Publisher

### Step 1: Create the robot state publisher node

Create `~/ros2_ws/src/robot_state_py/robot_state_py/robot_state_publisher.py`:

```python
#!/usr/bin/env python3
# robot_state_publisher.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import math


class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        # Create publisher for joint states
        self.joint_state_publisher = self.create_publisher(
            JointState, 'joint_states', 10
        )

        # Create transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Create timer to publish states
        timer_period = 0.1  # 10 Hz
        self.timer = self.create_timer(timer_period, self.publish_states)

        # Initialize joint positions
        self.time = 0.0

        self.get_logger().info('Robot State Publisher started')

    def publish_states(self):
        # Create joint state message
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.header.frame_id = 'base_footprint'

        # Set joint names
        joint_state.name = ['left_wheel_joint', 'right_wheel_joint']

        # Simulate wheel rotation
        self.time += 0.1
        left_wheel_pos = math.sin(self.time) * 0.5
        right_wheel_pos = math.cos(self.time) * 0.5

        # Set joint positions
        joint_state.position = [left_wheel_pos, right_wheel_pos]
        joint_state.velocity = [math.cos(self.time) * 0.5, -math.sin(self.time) * 0.5]
        joint_state.effort = [0.0, 0.0]

        # Publish joint states
        self.joint_state_publisher.publish(joint_state)

        # Broadcast transforms
        self.broadcast_transforms(joint_state)

    def broadcast_transforms(self, joint_state):
        # Base footprint to base link transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'

        # Simulate robot movement in a circle
        t.transform.translation.x = math.sin(self.time * 0.5) * 2.0
        t.transform.translation.y = math.cos(self.time * 0.5) * 2.0
        t.transform.translation.z = 0.0

        # Calculate orientation for facing direction of movement
        angle = self.time * 0.5 + math.pi/2
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(angle/2)
        t.transform.rotation.w = math.cos(angle/2)

        self.tf_broadcaster.sendTransform(t)

        # Base link to left wheel transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'left_wheel'

        t.transform.translation.x = 0.15
        t.transform.translation.y = 0.15
        t.transform.translation.z = -0.1

        # Apply wheel rotation
        wheel_rotation = joint_state.position[0]
        t.transform.rotation.x = math.sin(wheel_rotation/2) * math.sin(math.pi/2)
        t.transform.rotation.y = math.sin(wheel_rotation/2) * 0.0
        t.transform.rotation.z = math.sin(wheel_rotation/2) * math.cos(math.pi/2)
        t.transform.rotation.w = math.cos(wheel_rotation/2)

        self.tf_broadcaster.sendTransform(t)

        # Base link to right wheel transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'right_wheel'

        t.transform.translation.x = 0.15
        t.transform.translation.y = -0.15
        t.transform.translation.z = -0.1

        # Apply wheel rotation
        wheel_rotation = joint_state.position[1]
        t.transform.rotation.x = math.sin(wheel_rotation/2) * math.sin(math.pi/2)
        t.transform.rotation.y = math.sin(wheel_rotation/2) * 0.0
        t.transform.rotation.z = math.sin(wheel_rotation/2) * math.cos(math.pi/2)
        t.transform.rotation.w = math.cos(wheel_rotation/2)

        self.tf_broadcaster.sendTransform(t)

        # Base link to camera transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'camera_link'

        t.transform.translation.x = 0.2
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.05

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = RobotStatePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Exercise 3: Create Joint State Publisher

### Step 1: Create a separate joint state publisher

Create `~/ros2_ws/src/robot_state_py/robot_state_py/joint_state_publisher.py`:

```python
#!/usr/bin/env python3
# joint_state_publisher.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math


class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')

        # Create publisher for joint states
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Create timer to publish joint states
        timer_period = 0.05  # 20 Hz
        self.timer = self.create_timer(timer_period, self.publish_joint_states)

        # Initialize time
        self.time = 0.0

        self.get_logger().info('Joint State Publisher started')

    def publish_joint_states(self):
        # Create joint state message
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Set joint names
        msg.name = ['left_wheel_joint', 'right_wheel_joint']

        # Simulate joint positions (oscillating wheels)
        self.time += 0.05
        left_pos = math.sin(self.time) * 0.5
        right_pos = math.cos(self.time) * 0.5

        # Set positions, velocities, and efforts
        msg.position = [left_pos, right_pos]
        msg.velocity = [math.cos(self.time) * 0.5, -math.sin(self.time) * 0.5]
        msg.effort = [0.0, 0.0]

        # Publish joint states
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Exercise 4: TF Transform Utilities

### Step 1: Create TF listener and transform utilities

Create `~/ros2_ws/src/robot_state_py/robot_state_py/tf_utils.py`:

```python
#!/usr/bin/env python3
# tf_utils.py
import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
from tf2_geometry_msgs import PointStamped
import tf2_ros


class TFUtils(Node):
    def __init__(self):
        super().__init__('tf_utils')

        # Create TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Create timer to perform TF operations
        timer_period = 1.0  # 1 Hz
        self.timer = self.create_timer(timer_period, self.perform_tf_operations)

        self.get_logger().info('TF Utilities node started')

    def perform_tf_operations(self):
        try:
            # Lookup transform from base_link to camera_link
            trans = self.tf_buffer.lookup_transform(
                'base_link',
                'camera_link',
                rclpy.time.Time()
            )

            self.get_logger().info(
                f'Camera position relative to base: '
                f'x={trans.transform.translation.x:.2f}, '
                f'y={trans.transform.translation.y:.2f}, '
                f'z={trans.transform.translation.z:.2f}'
            )

            # Transform a point from camera frame to base frame
            point_camera = PointStamped()
            point_camera.header.frame_id = 'camera_link'
            point_camera.header.stamp = self.get_clock().now().to_msg()
            point_camera.point.x = 1.0  # 1m in front of camera
            point_camera.point.y = 0.0
            point_camera.point.z = 0.0

            # Transform the point to base frame
            point_base = self.tf_buffer.transform(point_camera, 'base_link')

            self.get_logger().info(
                f'Point in base frame: '
                f'x={point_base.point.x:.2f}, '
                f'y={point_base.point.y:.2f}, '
                f'z={point_base.point.z:.2f}'
            )

        except tf2_ros.TransformException as ex:
            self.get_logger().info(f'Could not transform: {ex}')


def main(args=None):
    rclpy.init(args=args)
    node = TFUtils()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Exercise 5: URDF Robot State Publisher Integration

### Step 1: Create a URDF-based robot state publisher

Create `~/ros2_ws/src/robot_state_py/robot_state_py/urdf_robot_publisher.py`:

```python
#!/usr/bin/env python3
# urdf_robot_publisher.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from rcl_interfaces.msg import ParameterDescriptor
import math


class URDFRobotPublisher(Node):
    def __init__(self):
        super().__init__('urdf_robot_publisher')

        # Declare parameters
        self.declare_parameter(
            'robot_description',
            '<robot name="simple_robot">...</robot>',
            ParameterDescriptor(description='Robot URDF description')
        )

        # Create publisher for joint states
        self.joint_state_publisher = self.create_publisher(
            JointState, 'joint_states', 10
        )

        # Create transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Create timer to publish states
        timer_period = 0.05  # 20 Hz
        self.timer = self.create_timer(timer_period, self.publish_states)

        # Initialize joint positions
        self.time = 0.0

        # Joint names for our robot
        self.joint_names = ['left_wheel_joint', 'right_wheel_joint']

        self.get_logger().info('URDF Robot Publisher started')

    def publish_states(self):
        # Create joint state message
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.header.frame_id = 'base_footprint'

        # Set joint names
        joint_state.name = self.joint_names

        # Simulate joint movement
        self.time += 0.05
        left_wheel_pos = math.sin(self.time) * 0.5
        right_wheel_pos = math.cos(self.time) * 0.5

        # Set joint positions, velocities, and efforts
        joint_state.position = [left_wheel_pos, right_wheel_pos]
        joint_state.velocity = [math.cos(self.time) * 0.5, -math.sin(self.time) * 0.5]
        joint_state.effort = [0.0, 0.0]

        # Publish joint states
        self.joint_state_publisher.publish(joint_state)

        # Broadcast transforms
        self.broadcast_transforms(joint_state)

    def broadcast_transforms(self, joint_state):
        # Odom to base_footprint transform (robot moving in a circle)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'

        # Simulate robot movement
        t.transform.translation.x = math.sin(self.time * 0.3) * 3.0
        t.transform.translation.y = math.cos(self.time * 0.3) * 3.0
        t.transform.translation.z = 0.0

        # Calculate orientation
        angle = self.time * 0.3 + math.pi/2
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(angle/2)
        t.transform.rotation.w = math.cos(angle/2)

        self.tf_broadcaster.sendTransform(t)

        # Base_footprint to base_link transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_footprint'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.1  # Base link is 0.1m above footprint

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

        # Base link to left wheel
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'left_wheel'

        t.transform.translation.x = 0.15
        t.transform.translation.y = 0.15
        t.transform.translation.z = -0.1

        # Apply wheel rotation
        wheel_rotation = joint_state.position[0]
        t.transform.rotation.x = math.sin(wheel_rotation/2) * math.sin(math.pi/2)
        t.transform.rotation.y = math.sin(wheel_rotation/2) * 0.0
        t.transform.rotation.z = math.sin(wheel_rotation/2) * math.cos(math.pi/2)
        t.transform.rotation.w = math.cos(wheel_rotation/2)

        self.tf_broadcaster.sendTransform(t)

        # Base link to right wheel
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'right_wheel'

        t.transform.translation.x = 0.15
        t.transform.translation.y = -0.15
        t.transform.translation.z = -0.1

        # Apply wheel rotation
        wheel_rotation = joint_state.position[1]
        t.transform.rotation.x = math.sin(wheel_rotation/2) * math.sin(math.pi/2)
        t.transform.rotation.y = math.sin(wheel_rotation/2) * 0.0
        t.transform.rotation.z = math.sin(wheel_rotation/2) * math.cos(math.pi/2)
        t.transform.rotation.w = math.cos(wheel_rotation/2)

        self.tf_broadcaster.sendTransform(t)

        # Base link to camera
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'camera_link'

        t.transform.translation.x = 0.2
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.05

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = URDFRobotPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Exercise 6: Update setup.py

Edit `~/ros2_ws/src/robot_state_py/setup.py`:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'robot_state_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Robot state management examples',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_state_publisher = robot_state_py.robot_state_publisher:main',
            'joint_state_publisher = robot_state_py.joint_state_publisher:main',
            'tf_utils = robot_state_py.tf_utils:main',
            'urdf_robot_publisher = robot_state_py.urdf_robot_publisher:main',
        ],
    },
)
```

## Exercise 7: Build and Test the Robot State System

### Step 1: Build the package

```bash
cd ~/ros2_ws
colcon build --packages-select robot_state_py
```

### Step 2: Source the workspace

```bash
source ~/ros2_ws/install/setup.bash
```

### Step 3: Launch the robot state publisher

```bash
ros2 run robot_state_py urdf_robot_publisher
```

### Step 4: In another terminal, check the joint states

```bash
source ~/ros2_ws/install/setup.bash
ros2 topic echo /joint_states
```

### Step 5: Check the TF tree

```bash
source ~/ros2_ws/install/setup.bash
ros2 run tf2_tools view_frames
```

This will generate a PDF showing the current TF tree structure.

### Step 6: Use TF echo to check specific transforms

```bash
source ~/ros2_ws/install/setup.bash
ros2 run tf2_ros tf2_echo odom base_link
```

## Exercise 8: Robot State Visualization

### Step 1: Create a visualization node

Create `~/ros2_ws/src/robot_state_py/robot_state_py/robot_visualizer.py`:

```python
#!/usr/bin/env python3
# robot_visualizer.py
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import math


class RobotVisualizer(Node):
    def __init__(self):
        super().__init__('robot_visualizer')

        # Create publisher for visualization markers
        self.marker_pub = self.create_publisher(MarkerArray, 'robot_markers', 10)

        # Create timer to publish markers
        timer_period = 0.1  # 10 Hz
        self.timer = self.create_timer(timer_period, self.publish_markers)

        self.get_logger().info('Robot Visualizer started')

    def publish_markers(self):
        marker_array = MarkerArray()

        # Robot body marker
        body_marker = Marker()
        body_marker.header.frame_id = 'base_link'
        body_marker.header.stamp = self.get_clock().now().to_msg()
        body_marker.ns = 'robot'
        body_marker.id = 0
        body_marker.type = Marker.CUBE
        body_marker.action = Marker.ADD
        body_marker.pose.position.x = 0.0
        body_marker.pose.position.y = 0.0
        body_marker.pose.position.z = 0.0
        body_marker.pose.orientation.x = 0.0
        body_marker.pose.orientation.y = 0.0
        body_marker.pose.orientation.z = 0.0
        body_marker.pose.orientation.w = 1.0
        body_marker.scale.x = 0.5
        body_marker.scale.y = 0.3
        body_marker.scale.z = 0.2
        body_marker.color.a = 0.8
        body_marker.color.r = 0.0
        body_marker.color.g = 0.0
        body_marker.color.b = 1.0
        marker_array.markers.append(body_marker)

        # Left wheel marker
        left_wheel_marker = Marker()
        left_wheel_marker.header.frame_id = 'left_wheel'
        left_wheel_marker.header.stamp = self.get_clock().now().to_msg()
        left_wheel_marker.ns = 'robot'
        left_wheel_marker.id = 1
        left_wheel_marker.type = Marker.CYLINDER
        left_wheel_marker.action = Marker.ADD
        left_wheel_marker.pose.position.x = 0.0
        left_wheel_marker.pose.position.y = 0.0
        left_wheel_marker.pose.position.z = 0.0
        left_wheel_marker.pose.orientation.x = 0.0
        left_wheel_marker.pose.orientation.y = 0.0
        left_wheel_marker.pose.orientation.z = 0.0
        left_wheel_marker.pose.orientation.w = 1.0
        left_wheel_marker.scale.x = 0.2
        left_wheel_marker.scale.y = 0.2
        left_wheel_marker.scale.z = 0.05
        left_wheel_marker.color.a = 0.8
        left_wheel_marker.color.r = 0.0
        left_wheel_marker.color.g = 0.0
        left_wheel_marker.color.b = 0.0
        marker_array.markers.append(left_wheel_marker)

        # Right wheel marker
        right_wheel_marker = Marker()
        right_wheel_marker.header.frame_id = 'right_wheel'
        right_wheel_marker.header.stamp = self.get_clock().now().to_msg()
        right_wheel_marker.ns = 'robot'
        right_wheel_marker.id = 2
        right_wheel_marker.type = Marker.CYLINDER
        right_wheel_marker.action = Marker.ADD
        right_wheel_marker.pose.position.x = 0.0
        right_wheel_marker.pose.position.y = 0.0
        right_wheel_marker.pose.position.z = 0.0
        right_wheel_marker.pose.orientation.x = 0.0
        right_wheel_marker.pose.orientation.y = 0.0
        right_wheel_marker.pose.orientation.z = 0.0
        right_wheel_marker.pose.orientation.w = 1.0
        right_wheel_marker.scale.x = 0.2
        right_wheel_marker.scale.y = 0.2
        right_wheel_marker.scale.z = 0.05
        right_wheel_marker.color.a = 0.8
        right_wheel_marker.color.r = 0.0
        right_wheel_marker.color.g = 0.0
        right_wheel_marker.color.b = 0.0
        marker_array.markers.append(right_wheel_marker)

        # Camera marker
        camera_marker = Marker()
        camera_marker.header.frame_id = 'camera_link'
        camera_marker.header.stamp = self.get_clock().now().to_msg()
        camera_marker.ns = 'robot'
        camera_marker.id = 3
        camera_marker.type = Marker.CUBE
        camera_marker.action = Marker.ADD
        camera_marker.pose.position.x = 0.0
        camera_marker.pose.position.y = 0.0
        camera_marker.pose.position.z = 0.0
        camera_marker.pose.orientation.x = 0.0
        camera_marker.pose.orientation.y = 0.0
        camera_marker.pose.orientation.z = 0.0
        camera_marker.pose.orientation.w = 1.0
        camera_marker.scale.x = 0.05
        camera_marker.scale.y = 0.05
        camera_marker.scale.z = 0.05
        camera_marker.color.a = 0.8
        camera_marker.color.r = 1.0
        camera_marker.color.g = 0.0
        camera_marker.color.b = 0.0
        marker_array.markers.append(camera_marker)

        # Publish marker array
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = RobotVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

Update the setup.py to include the visualizer:

```python
entry_points={
    'console_scripts': [
        'robot_state_publisher = robot_state_py.robot_state_publisher:main',
        'joint_state_publisher = robot_state_py.joint_state_publisher:main',
        'tf_utils = robot_state_py.tf_utils:main',
        'urdf_robot_publisher = robot_state_py.urdf_robot_publisher:main',
        'robot_visualizer = robot_state_py.robot_visualizer:main',
    ],
},
```

Rebuild the package:

```bash
cd ~/ros2_ws
colcon build --packages-select robot_state_py
source ~/ros2_ws/install/setup.bash
```

## Exercise 9: Testing with RViz2

### Step 1: Launch the robot state publisher and visualizer

Terminal 1:
```bash
source ~/ros2_ws/install/setup.bash
ros2 run robot_state_py urdf_robot_publisher
```

Terminal 2:
```bash
source ~/ros2_ws/install/setup.bash
ros2 run robot_state_py robot_visualizer
```

### Step 2: Launch RViz2 to visualize the robot

Terminal 3:
```bash
source ~/ros2_ws/install/setup.bash
rviz2
```

In RViz2:
1. Set the "Fixed Frame" to "odom"
2. Add a "RobotModel" display and set the description topic to "robot_description" (if using robot state publisher)
3. Add a "TF" display to visualize the transform tree
4. Add a "MarkerArray" display and set the topic to "/robot_markers"

## Troubleshooting

### Common Issues and Solutions

1. **TF tree not showing up**:
   - Ensure transform broadcaster is properly initialized
   - Check that frame names are consistent
   - Verify that timestamps are current

2. **Joint states not updating**:
   - Check that joint names in the message match URDF
   - Verify that the joint state publisher is running
   - Ensure proper frame relationships

3. **RViz2 not displaying robot**:
   - Verify that robot_description parameter is set
   - Check that joint states are being published
   - Ensure proper TF relationships

4. **Transform lookup failures**:
   - Increase TF buffer size if looking up old transforms
   - Ensure transforms are being broadcasted regularly
   - Check for circular dependencies in TF tree

## Assessment Questions

1. What is the purpose of the TF tree in ROS 2?
2. How do joint states and transforms work together to represent robot state?
3. What is the difference between robot_state_publisher and joint_state_publisher?
4. Why is it important to have a consistent TF tree structure?
5. How can you transform coordinates between different frames?

## Extension Exercises

1. Create a robot with more complex kinematics (e.g., a robotic arm)
2. Implement forward kinematics to calculate end-effector position
3. Add sensor frames to the TF tree (IMU, camera, LiDAR)
4. Create a node that publishes robot state based on odometry input
5. Implement inverse kinematics for a simple 2-DOF arm

## Summary

In this lab, you successfully:
- Implemented robot state publishers for broadcasting transforms
- Created joint state publishers for tracking joint positions
- Used TF tools for debugging and visualization
- Integrated robot state management with URDF models
- Transformed coordinates between different frames

These skills are fundamental to robotics development, as proper state management is crucial for navigation, manipulation, perception, and control systems. The TF tree provides a unified way to understand spatial relationships in the robot's environment, while joint states enable precise control and monitoring of the robot's configuration.