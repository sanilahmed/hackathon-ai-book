# Lab 1.2: ROS 2 Services and Actions

## Overview

In this lab, you will learn about ROS 2 services and actions, which provide synchronous request/response communication and goal-oriented asynchronous communication respectively. You'll implement custom services and actions and understand when to use each communication pattern.

## Objectives

By the end of this lab, you will be able to:
- Implement ROS 2 services for request/response communication
- Create and use ROS 2 actions for goal-oriented tasks
- Understand the differences between topics, services, and actions
- Choose the appropriate communication pattern for different scenarios
- Debug and monitor services and actions using ROS 2 tools

## Prerequisites

- Completion of Lab 1.1: ROS 2 Environment Setup and Basic Nodes
- Understanding of ROS 2 topics and messages
- Basic Python programming skills
- Familiarity with ROS 2 workspace setup

## Duration

2 hours

## Exercise 1: Creating a ROS 2 Service

### Step 1: Create a new package for services

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python robot_services_py --dependencies rclpy std_msgs example_interfaces
```

### Step 2: Create a custom service definition

Create the service directory:

```bash
mkdir -p ~/ros2_ws/src/robot_services_py/robot_services_py/srv
```

Create `~/ros2_ws/src/robot_services_py/robot_services_py/srv/AddTwoInts.srv`:

```
int64 a
int64 b
---
int64 sum
```

### Step 3: Create a service server

Create `~/ros2_ws/src/robot_services_py/robot_services_py/add_two_ints_server.py`:

```python
#!/usr/bin/env python3
# add_two_ints_server.py
import rclpy
from rclpy.node import Node
from robot_services_py.srv import AddTwoInts


class AddTwoIntsServer(Node):
    def __init__(self):
        super().__init__('add_two_ints_server')
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback
        )
        self.get_logger().info('Add Two Ints Server started')

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Request received: {request.a} + {request.b} = {response.sum}')
        return response


def main(args=None):
    rclpy.init(args=args)
    add_two_ints_server = AddTwoIntsServer()

    try:
        rclpy.spin(add_two_ints_server)
    except KeyboardInterrupt:
        pass
    finally:
        add_two_ints_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 4: Create a service client

Create `~/ros2_ws/src/robot_services_py/robot_services_py/add_two_ints_client.py`:

```python
#!/usr/bin/env python3
# add_two_ints_client.py
import sys
import rclpy
from rclpy.node import Node
from robot_services_py.srv import AddTwoInts


class AddTwoIntsClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main(args=None):
    rclpy.init(args=args)

    client = AddTwoIntsClient()

    # Get arguments from command line
    if len(sys.argv) != 3:
        print('Usage: python3 add_two_ints_client.py <int1> <int2>')
        return

    a = int(sys.argv[1])
    b = int(sys.argv[2])

    print(f'Requesting {a} + {b}')
    response = client.send_request(a, b)
    print(f'Result: {a} + {b} = {response.sum}')

    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 5: Update setup.py for services

Edit `~/ros2_ws/src/robot_services_py/setup.py`:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'robot_services_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'srv'), glob('srv/*.srv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='ROS 2 services example',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'add_two_ints_server = robot_services_py.add_two_ints_server:main',
            'add_two_ints_client = robot_services_py.add_two_ints_client:main',
        ],
    },
)
```

### Step 6: Update package.xml for services

Edit `~/ros2_ws/src/robot_services_py/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>robot_services_py</name>
  <version>0.0.0</version>
  <description>ROS 2 services example</description>
  <maintainer email="your.email@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>example_interfaces</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Exercise 2: Creating a ROS 2 Action

### Step 1: Create a new package for actions

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python robot_actions_py --dependencies rclpy std_msgs action_tutorials_interfaces
```

### Step 2: Create a custom action definition

Create the action directory:

```bash
mkdir -p ~/ros2_ws/src/robot_actions_py/robot_actions_py/action
```

Create `~/ros2_ws/src/robot_actions_py/robot_actions_py/action/Fibonacci.action`:

```
int32 order
---
int32[] sequence
---
int32 feedback
```

### Step 3: Create an action server

Create `~/ros2_ws/src/robot_actions_py/robot_actions_py/fibonacci_action_server.py`:

```python
#!/usr/bin/env python3
# fibonacci_action_server.py
import time
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from robot_actions_py.action import Fibonacci


class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        self.get_logger().info('Fibonacci Action Server started')

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the goal."""
        self.get_logger().info('Executing goal...')

        # Create feedback message
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.feedback = 0

        # Generate Fibonacci sequence
        sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.feedback = i
            goal_handle.publish_feedback(feedback_msg)

            sequence.append(sequence[i] + sequence[i-1])
            time.sleep(0.5)  # Simulate work

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = sequence

        self.get_logger().info(f'Result: {sequence}')
        return result


def main(args=None):
    rclpy.init(args=args)
    fibonacci_action_server = FibonacciActionServer()

    try:
        rclpy.spin(fibonacci_action_server)
    except KeyboardInterrupt:
        pass
    finally:
        fibonacci_action_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 4: Create an action client

Create `~/ros2_ws/src/robot_actions_py/robot_actions_py/fibonacci_action_client.py`:

```python
#!/usr/bin/env python3
# fibonacci_action_client.py
import sys
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from robot_actions_py.action import Fibonacci


class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci'
        )

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        self.get_logger().info(f'Sending goal with order: {order}')
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.feedback}')


def main(args=None):
    rclpy.init(args=args)

    action_client = FibonacciActionClient()

    if len(sys.argv) != 2:
        print('Usage: python3 fibonacci_action_client.py <order>')
        return

    order = int(sys.argv[1])
    action_client.send_goal(order)

    rclpy.spin(action_client)


if __name__ == '__main__':
    main()
```

### Step 5: Update setup.py for actions

Edit `~/ros2_ws/src/robot_actions_py/setup.py`:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'robot_actions_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'action'), glob('action/*.action')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='ROS 2 actions example',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fibonacci_action_server = robot_actions_py.fibonacci_action_server:main',
            'fibonacci_action_client = robot_actions_py.fibonacci_action_client:main',
        ],
    },
)
```

### Step 6: Update package.xml for actions

Edit `~/ros2_ws/src/robot_actions_py/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>robot_actions_py</name>
  <version>0.0.0</version>
  <description>ROS 2 actions example</description>
  <maintainer email="your.email@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>action_tutorials_interfaces</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Exercise 3: Build and Test Services

### Step 1: Build the packages

```bash
cd ~/ros2_ws
colcon build --packages-select robot_services_py robot_actions_py
```

### Step 2: Source the workspace

```bash
source ~/ros2_ws/install/setup.bash
```

### Step 3: Test the service

Open a new terminal and run the service server:

```bash
source ~/ros2_ws/install/setup.bash
ros2 run robot_services_py add_two_ints_server
```

In another terminal, run the service client:

```bash
source ~/ros2_ws/install/setup.bash
ros2 run robot_services_py add_two_ints_client 5 3
```

You should see the server respond with the sum of 5 and 3.

### Step 4: Test the action

Open a new terminal and run the action server:

```bash
source ~/ros2_ws/install/setup.bash
ros2 run robot_actions_py fibonacci_action_server
```

In another terminal, run the action client:

```bash
source ~/ros2_ws/install/setup.bash
ros2 run robot_actions_py fibonacci_action_client 5
```

You should see the action server generate a Fibonacci sequence and provide feedback during execution.

## Exercise 4: Using ROS 2 Tools for Services and Actions

### Step 1: List services

```bash
source ~/ros2_ws/install/setup.bash
ros2 service list
```

### Step 2: Get service information

```bash
ros2 service info /add_two_ints
```

### Step 3: Call service directly from command line

```bash
ros2 service call /add_two_ints robot_services_py/srv/AddTwoInts "{a: 10, b: 20}"
```

### Step 4: List actions

```bash
ros2 action list
```

### Step 5: Get action information

```bash
ros2 action info /fibonacci
```

## Exercise 5: Robot Control Service Example

Create a more practical service for robot control:

Create `~/ros2_ws/src/robot_services_py/robot_services_py/srv/RobotControl.srv`:

```
string command
float64[] parameters
---
bool success
string message
```

Create `~/ros2_ws/src/robot_services_py/robot_services_py/robot_control_server.py`:

```python
#!/usr/bin/env python3
# robot_control_server.py
import rclpy
from rclpy.node import Node
from robot_services_py.srv import RobotControl


class RobotControlServer(Node):
    def __init__(self):
        super().__init__('robot_control_server')
        self.srv = self.create_service(
            RobotControl,
            'robot_control',
            self.robot_control_callback
        )
        self.get_logger().info('Robot Control Server started')

    def robot_control_callback(self, request, response):
        command = request.command
        params = request.parameters

        self.get_logger().info(f'Received command: {command} with params: {params}')

        # Process different commands
        if command == 'move_to':
            if len(params) >= 2:
                x, y = params[0], params[1]
                response.success = True
                response.message = f'Moving to position ({x}, {y})'
            else:
                response.success = False
                response.message = 'Move command requires 2 parameters (x, y)'
        elif command == 'rotate':
            if len(params) >= 1:
                angle = params[0]
                response.success = True
                response.message = f'Rotating by {angle} radians'
            else:
                response.success = False
                response.message = 'Rotate command requires 1 parameter (angle)'
        elif command == 'stop':
            response.success = True
            response.message = 'Robot stopped'
        else:
            response.success = False
            response.message = f'Unknown command: {command}'

        return response


def main(args=None):
    rclpy.init(args=args)
    robot_control_server = RobotControlServer()

    try:
        rclpy.spin(robot_control_server)
    except KeyboardInterrupt:
        pass
    finally:
        robot_control_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

Update the entry points in `setup.py`:

```python
entry_points={
    'console_scripts': [
        'add_two_ints_server = robot_services_py.add_two_ints_server:main',
        'add_two_ints_client = robot_services_py.add_two_ints_client:main',
        'robot_control_server = robot_services_py.robot_control_server:main',
    ],
},
```

Build and test the robot control service:

```bash
cd ~/ros2_ws
colcon build --packages-select robot_services_py
source ~/ros2_ws/install/setup.bash
ros2 run robot_services_py robot_control_server
```

In another terminal:

```bash
source ~/ros2_ws/install/setup.bash
ros2 service call /robot_control robot_services_py/srv/RobotControl "{command: 'move_to', parameters: [1.0, 2.0]}"
```

## Troubleshooting

### Common Issues and Solutions

1. **Service/Action not found errors**:
   - Ensure the service/action server is running
   - Check that the service/action name matches exactly
   - Verify that the correct message/action type is used

2. **Build errors with custom messages/actions**:
   - Make sure the .srv/.action files are in the correct directory
   - Verify that package.xml includes the rosidl dependencies
   - Check that setup.py includes the message/action files in data_files

3. **Client hangs when calling service**:
   - Verify that the service server is running
   - Check network configuration if using multiple machines
   - Ensure both nodes are on the same ROS domain

4. **Action client doesn't receive feedback**:
   - Check that the action server is publishing feedback
   - Verify that the feedback callback is properly implemented
   - Ensure the action server is not completing too quickly

## Assessment Questions

1. What is the main difference between ROS 2 topics, services, and actions?
2. When would you use a service instead of a topic for communication?
3. What are the three parts of an action definition?
4. How does an action client receive feedback during execution?
5. What are the advantages of using actions over services for long-running tasks?

## Extension Exercises

1. Create a service that controls a robot's LED lights (on/off, color, blinking pattern)
2. Implement an action that moves a robot to multiple waypoints with feedback on progress
3. Create a service that queries robot battery status and other sensor readings
4. Implement a complex action with preemption (ability to cancel and start a new goal)

## Summary

In this lab, you successfully:
- Created and implemented ROS 2 services for request/response communication
- Developed ROS 2 actions for goal-oriented tasks with feedback
- Built and tested both services and actions
- Used ROS 2 tools to monitor and interact with services and actions
- Created practical examples for robot control

These communication patterns are essential for building complex robotic systems where different types of interactions are needed. Services are ideal for immediate responses to queries or commands, while actions are perfect for long-running tasks that require feedback and the ability to cancel.

The skills learned in this lab will be crucial as you develop more sophisticated robotic applications that require coordinated behavior between multiple nodes and systems.