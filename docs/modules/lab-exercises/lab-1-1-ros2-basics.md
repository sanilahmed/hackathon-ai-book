# Lab 1.1: ROS 2 Fundamentals for Humanoid Robotics

## Overview

This lab exercise introduces the fundamental concepts of ROS 2 (Robot Operating System 2) specifically applied to humanoid robotics. Students will learn to create nodes, topics, services, and actions while understanding how they apply to controlling humanoid robots with multiple degrees of freedom.

## Learning Objectives

By the end of this lab, students will be able to:
1. Create and run basic ROS 2 nodes in Python and C++
2. Implement publisher-subscriber communication patterns
3. Use services for request-response communication
4. Work with actions for goal-oriented communication
5. Understand ROS 2 workspace structure and package management
6. Apply ROS 2 concepts to humanoid robot control

## Prerequisites

- Basic knowledge of Python or C++
- Understanding of Linux command line
- Familiarity with version control systems (Git)
- ROS 2 Humble Hawksbill installed on Ubuntu 22.04

## Theory Background

ROS 2 is a middleware framework that provides services designed for robot applications, including hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more. Key concepts include:

- **Nodes**: Processes that perform computation
- **Topics**: Named buses over which nodes exchange messages
- **Publishers/Subscribers**: Communication pattern for one-to-many message passing
- **Services**: Request-response communication pattern
- **Actions**: Goal-oriented communication with feedback
- **Packages**: Organizational unit containing code and resources

## Lab Exercise

### Part 1: Creating Your First ROS 2 Node

Let's start by creating a simple ROS 2 node that publishes humanoid joint states:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math
import time

class HumanoidJointPublisher(Node):
    def __init__(self):
        super().__init__('humanoid_joint_publisher')

        # Create publisher for joint states
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)

        # Create timer to publish at 50 Hz
        self.timer = self.create_timer(0.02, self.publish_joint_states)

        # Initialize joint names for a 14-DOF humanoid
        self.joint_names = [
            'torso_joint',
            'head_joint',
            'left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint',
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
        ]

        # Initialize joint positions (starting pose)
        self.joint_positions = [0.0] * len(self.joint_names)

        # Initialize time for trajectory generation
        self.start_time = time.time()

        self.get_logger().info('Humanoid Joint Publisher node initialized')

    def publish_joint_states(self):
        """Publish joint state messages"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.name = self.joint_names

        # Generate a simple oscillating motion for demonstration
        current_time = time.time() - self.start_time
        amplitude = 0.2
        frequency = 0.5  # Hz

        # Apply oscillating motion to some joints
        for i in range(len(self.joint_positions)):
            # Apply different oscillation patterns to different joints
            if i < 3:  # Torso and head
                self.joint_positions[i] = amplitude * math.sin(2 * math.pi * frequency * current_time + i)
            elif i < 6:  # Left arm
                self.joint_positions[i] = amplitude * math.sin(2 * math.pi * frequency * current_time + i + 1)
            elif i < 9:  # Right arm
                self.joint_positions[i] = amplitude * math.sin(2 * math.pi * frequency * current_time + i + 2)
            else:  # Legs
                self.joint_positions[i] = amplitude * math.sin(2 * math.pi * frequency * current_time + i + 3)

        msg.position = self.joint_positions
        msg.velocity = [0.0] * len(self.joint_positions)  # Zero velocity for now
        msg.effort = [0.0] * len(self.joint_positions)    # Zero effort for now

        self.publisher_.publish(msg)
        self.get_logger().debug(f'Published joint states: {self.joint_positions[:3]}...')

def main(args=None):
    rclpy.init(args=args)

    joint_publisher = HumanoidJointPublisher()

    try:
        rclpy.spin(joint_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        joint_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 2: Creating a Joint State Subscriber

Now let's create a subscriber that listens to joint states and processes them:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import numpy as np

class HumanoidJointSubscriber(Node):
    def __init__(self):
        super().__init__('humanoid_joint_subscriber')

        # Create subscriber for joint states
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10)

        # Create publisher for processed data
        self.status_publisher = self.create_publisher(String, 'robot_status', 10)

        # Store the latest joint state
        self.latest_joint_state = None

        self.get_logger().info('Humanoid Joint Subscriber node initialized')

    def joint_state_callback(self, msg):
        """Process incoming joint state messages"""
        self.latest_joint_state = msg

        # Calculate some basic metrics
        avg_position = np.mean(msg.position) if msg.position else 0.0
        max_position = max(msg.position) if msg.position else 0.0
        min_position = min(msg.position) if msg.position else 0.0

        # Check for joint limits (simplified)
        joint_limits_exceeded = any(abs(pos) > 2.0 for pos in msg.position)  # 2.0 rad limit

        # Create status message
        status_msg = String()
        status_msg.data = (
            f'Robot Status - Avg Pos: {avg_position:.3f}, '
            f'Max Pos: {max_position:.3f}, '
            f'Min Pos: {min_position:.3f}, '
            f'Joints: {len(msg.position)}, '
            f'Limit Exceeded: {joint_limits_exceeded}'
        )

        self.status_publisher.publish(status_msg)

        # Log if any joint limits are exceeded
        if joint_limits_exceeded:
            self.get_logger().warn('Joint limit exceeded - check robot configuration!')

def main(args=None):
    rclpy.init(args=args)

    joint_subscriber = HumanoidJointSubscriber()

    try:
        rclpy.spin(joint_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        joint_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 3: Creating a ROS 2 Service

Let's implement a service that allows remote control of the humanoid robot:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from example_interfaces.srv import SetBool
import numpy as np

class HumanoidControlService(Node):
    def __init__(self):
        super().__init__('humanoid_control_service')

        # Create service
        self.srv = self.create_service(
            SetBool,
            'set_robot_mode',
            self.set_robot_mode_callback
        )

        # Create publisher for joint commands
        self.joint_command_publisher = self.create_publisher(
            Float32MultiArray,
            'joint_commands',
            10
        )

        # Robot state
        self.robot_enabled = False
        self.current_pose = [0.0] * 14  # 14 DOF humanoid

        self.get_logger().info('Humanoid Control Service initialized')

    def set_robot_mode_callback(self, request, response):
        """Handle robot mode change requests"""
        if request.data:
            self.robot_enabled = True
            response.success = True
            response.message = 'Robot enabled successfully'
            self.get_logger().info('Robot enabled')
        else:
            self.robot_enabled = False
            response.success = True
            response.message = 'Robot disabled successfully'
            self.get_logger().info('Robot disabled')

        return response

    def send_joint_command(self, joint_positions):
        """Send joint position commands"""
        if not self.robot_enabled:
            self.get_logger().warn('Robot not enabled, cannot send commands')
            return False

        if len(joint_positions) != 14:
            self.get_logger().error(f'Expected 14 joint positions, got {len(joint_positions)}')
            return False

        # Publish joint commands
        cmd_msg = Float32MultiArray()
        cmd_msg.data = joint_positions
        self.joint_command_publisher.publish(cmd_msg)

        self.current_pose = joint_positions
        return True

def main(args=None):
    rclpy.init(args=args)

    control_service = HumanoidControlService()

    try:
        rclpy.spin(control_service)
    except KeyboardInterrupt:
        pass
    finally:
        control_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 4: Creating a ROS 2 Action Server

Now let's implement an action server for complex humanoid behaviors:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import time
from example_interfaces.action import Fibonacci

class HumanoidActionServer(Node):
    def __init__(self):
        super().__init__('humanoid_action_server')

        # Create action server with reentrant callback group
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'move_humanoid',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        self.get_logger().info('Humanoid Action Server initialized')

    def goal_callback(self, goal_request):
        """Accept or reject goal requests"""
        self.get_logger().info(f'Received goal request with order: {goal_request.order}')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject cancel requests"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the action goal"""
        self.get_logger().info('Executing goal...')

        # Feedback and result
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        result = Fibonacci.Result()

        # Simulate humanoid movement steps
        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                result.sequence = feedback_msg.sequence
                return result

            # Update feedback
            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1]
            )

            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Feedback: {feedback_msg.sequence[-1]}')

            # Simulate movement time
            time.sleep(0.5)

        # Check if goal was canceled
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            self.get_logger().info('Goal canceled during execution')
            result.sequence = feedback_msg.sequence
            return result

        # Complete the goal
        goal_handle.succeed()
        result.sequence = feedback_msg.sequence

        self.get_logger().info(f'Result: {result.sequence}')
        return result

def main(args=None):
    rclpy.init(args=args)

    action_server = HumanoidActionServer()

    # Use MultiThreadedExecutor to handle callbacks in separate threads
    executor = MultiThreadedExecutor()
    executor.add_node(action_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 5: Creating a Parameter Server

Let's implement parameter management for humanoid robot configuration:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_system_default
import json

class HumanoidParameterServer(Node):
    def __init__(self):
        super().__init__('humanoid_parameter_server')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'humanoid_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('max_acceleration', 2.0)
        self.declare_parameter('joint_limits', [2.0] * 14)  # 14 DOF
        self.declare_parameter('control_mode', 'position')
        self.declare_parameter('safety_enabled', True)

        # Set up parameter change callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Timer to periodically check parameters
        self.timer = self.create_timer(1.0, self.check_parameters)

        self.get_logger().info('Humanoid Parameter Server initialized')

    def parameter_callback(self, params):
        """Handle parameter changes"""
        for param in params:
            if param.name == 'max_velocity':
                if param.value > 5.0:
                    self.get_logger().warn(f'Max velocity {param.value} may be unsafe')
            elif param.name == 'joint_limits':
                if len(param.value) != 14:
                    return rclpy.node.SetParametersResult(
                        successful=False,
                        reason='Joint limits must have 14 values for 14-DOF robot'
                    )

        return rclpy.node.SetParametersResult(successful=True)

    def check_parameters(self):
        """Periodically check and log parameter values"""
        robot_name = self.get_parameter('robot_name').value
        max_vel = self.get_parameter('max_velocity').value
        safety_enabled = self.get_parameter('safety_enabled').value

        self.get_logger().info(
            f'Robot: {robot_name}, Max Vel: {max_vel}, Safety: {safety_enabled}'
        )

def main(args=None):
    rclpy.init(args=args)

    param_server = HumanoidParameterServer()

    try:
        rclpy.spin(param_server)
    except KeyboardInterrupt:
        pass
    finally:
        param_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 6: Creating a Launch File

Let's create a launch file to bring up the complete humanoid system:

```python
#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    robot_name = LaunchConfiguration('robot_name', default='humanoid_robot')

    # Get package directory
    pkg_name = 'humanoid_control'

    # Joint state publisher node
    joint_publisher_node = Node(
        package='humanoid_control',
        executable='humanoid_joint_publisher',
        name='humanoid_joint_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_name': robot_name}
        ],
        output='screen'
    )

    # Joint state subscriber node
    joint_subscriber_node = Node(
        package='humanoid_control',
        executable='humanoid_joint_subscriber',
        name='humanoid_joint_subscriber',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Control service node
    control_service_node = Node(
        package='humanoid_control',
        executable='humanoid_control_service',
        name='humanoid_control_service',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Action server node
    action_server_node = Node(
        package='humanoid_control',
        executable='humanoid_action_server',
        name='humanoid_action_server',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Parameter server node
    param_server_node = Node(
        package='humanoid_control',
        executable='humanoid_parameter_server',
        name='humanoid_parameter_server',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # RViz2 node for visualization
    rviz_config = os.path.join(
        get_package_share_directory(pkg_name),
        'config',
        'humanoid_rviz_config.rviz'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'robot_name',
            default_value='humanoid_robot',
            description='Name of the robot'
        ),
        joint_publisher_node,
        joint_subscriber_node,
        control_service_node,
        action_server_node,
        param_server_node,
        rviz_node
    ])
```

### Part 7: Creating a C++ Node Example

Finally, let's create a C++ node for performance-critical humanoid control:

```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <vector>
#include <string>
#include <chrono>

class HumanoidController : public rclcpp::Node
{
public:
    HumanoidController() : Node("humanoid_controller_cpp")
    {
        // Create publisher for joint commands
        joint_command_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "joint_commands_cpp", 10);

        // Create subscriber for joint states
        joint_state_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "joint_states", 10,
            std::bind(&HumanoidController::joint_state_callback, this, std::placeholders::_1));

        // Create timer for control loop
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20), // 50 Hz
            std::bind(&HumanoidController::control_loop, this));

        // Initialize joint names
        joint_names_ = {
            "torso_joint", "head_joint",
            "left_shoulder_joint", "left_elbow_joint", "left_wrist_joint",
            "right_shoulder_joint", "right_elbow_joint", "right_wrist_joint",
            "left_hip_joint", "left_knee_joint", "left_ankle_joint",
            "right_hip_joint", "right_knee_joint", "right_ankle_joint"
        };

        // Initialize target positions
        target_positions_.resize(joint_names_.size(), 0.0);
        current_positions_.resize(joint_names_.size(), 0.0);

        RCLCPP_INFO(this->get_logger(), "Humanoid Controller C++ node initialized");
    }

private:
    void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        // Update current joint positions
        if (msg->position.size() == current_positions_.size()) {
            current_positions_ = msg->position;
        }
    }

    void control_loop()
    {
        // Simple PD controller for each joint
        std::vector<double> commands;

        for (size_t i = 0; i < target_positions_.size(); ++i) {
            double error = target_positions_[i] - current_positions_[i];
            double command = 10.0 * error; // Simple proportional control

            // Apply limits
            if (command > 2.0) command = 2.0;
            if (command < -2.0) command = -2.0;

            commands.push_back(command);
        }

        // Publish commands
        auto cmd_msg = std_msgs::msg::Float64MultiArray();
        cmd_msg.data = commands;
        joint_command_publisher_->publish(cmd_msg);

        // Update targets with a simple oscillating pattern
        static double time_counter = 0.0;
        time_counter += 0.02; // 50 Hz * 0.02s = 1 step per second

        for (size_t i = 0; i < target_positions_.size(); ++i) {
            target_positions_[i] = 0.5 * sin(time_counter + i * 0.1);
        }
    }

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_command_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscription_;
    rclcpp::TimerBase::SharedPtr control_timer_;

    std::vector<std::string> joint_names_;
    std::vector<double> target_positions_;
    std::vector<double> current_positions_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HumanoidController>());
    rclcpp::shutdown();
    return 0;
}
```

## Implementation Steps

1. Create a new ROS 2 package for humanoid control:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python humanoid_control
   ```

2. Create the Python nodes in the package:
   - `humanoid_joint_publisher.py`
   - `humanoid_joint_subscriber.py`
   - `humanoid_control_service.py`
   - `humanoid_action_server.py`
   - `humanoid_parameter_server.py`

3. Create the launch file in `launch/humanoid_system_launch.py`

4. Create the C++ node and add it to `CMakeLists.txt`

5. Build the package:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select humanoid_control
   source install/setup.bash
   ```

6. Run the system:
   ```bash
   ros2 launch humanoid_control humanoid_system_launch.py
   ```

## Expected Outcomes

After completing this lab, you should have:
1. A working ROS 2 workspace with a humanoid control package
2. Multiple nodes implementing different communication patterns
3. A launch file that brings up the complete system
4. Understanding of ROS 2 concepts applied to humanoid robotics
5. Experience with both Python and C++ ROS 2 development

## Troubleshooting Tips

- Ensure ROS 2 environment is sourced before running nodes
- Check that all dependencies are properly installed
- Verify topic names match between publishers and subscribers
- Use `ros2 topic list` and `ros2 node list` to verify system status
- Monitor system performance and adjust timer rates as needed

## Further Exploration

- Implement more complex humanoid behaviors using state machines
- Add TF (Transforms) for robot kinematics
- Create custom message types for humanoid-specific data
- Integrate with Gazebo simulation for realistic testing
- Implement safety systems and emergency stops