# Lab 1.1: ROS 2 Environment Setup and Basic Nodes

## Overview

In this lab, you will set up the ROS 2 development environment and create your first ROS 2 nodes. You'll learn how to create packages, implement publisher/subscriber communication, and understand the ROS 2 architecture.

## Objectives

By the end of this lab, you will be able to:
- Install and configure ROS 2 Humble Hawksbill
- Create a ROS 2 workspace and packages
- Implement publisher and subscriber nodes
- Use ROS 2 tools for debugging and visualization
- Understand the basic concepts of ROS 2 topics and messages

## Prerequisites

- Ubuntu 20.04 or 22.04 LTS
- Basic knowledge of Linux command line
- Basic Python or C++ programming skills
- Administrative privileges for package installation

## Duration

2-3 hours

## Setup Instructions

### 1. Install ROS 2 Humble Hawksbill

```bash
# Set locale
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Add ROS 2 apt repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install ros-humble-ros-base
sudo apt install python3-colcon-common-extensions
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### 2. Set up ROS 2 environment

```bash
# Add sourcing to bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Initialize rosdep
sudo rosdep init
rosdep update
```

### 3. Create a workspace

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Source ROS 2 environment
source /opt/ros/humble/setup.bash
```

## Exercise 1: Create a Basic Publisher Node

### Step 1: Create a new package

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python robot_basics_py --dependencies rclpy std_msgs
```

### Step 2: Create the publisher node

Create `~/ros2_ws/src/robot_basics_py/robot_basics_py/talker.py`:

```python
#!/usr/bin/env python3
# talker.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TalkerNode(Node):
    def __init__(self):
        super().__init__('talker')

        # Create publisher
        self.publisher = self.create_publisher(String, 'chatter', 10)

        # Create timer to publish messages
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Counter for messages
        self.i = 0

        self.get_logger().info('Talker node started')

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    talker = TalkerNode()

    try:
        rclpy.spin(talker)
    except KeyboardInterrupt:
        pass
    finally:
        talker.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 3: Make the file executable and add to setup.py

```bash
cd ~/ros2_ws/src/robot_basics_py
chmod +x robot_basics_py/talker.py
```

Edit `~/ros2_ws/src/robot_basics_py/setup.py`:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'robot_basics_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Basic ROS 2 publisher/subscriber nodes',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = robot_basics_py.talker:main',
            'listener = robot_basics_py.listener:main',
        ],
    },
)
```

## Exercise 2: Create a Basic Subscriber Node

Create `~/ros2_ws/src/robot_basics_py/robot_basics_py/listener.py`:

```python
#!/usr/bin/env python3
# listener.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ListenerNode(Node):
    def __init__(self):
        super().__init__('listener')

        # Create subscriber
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.get_logger().info('Listener node started')

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    rclpy.init(args=args)
    listener = ListenerNode()

    try:
        rclpy.spin(listener)
    except KeyboardInterrupt:
        pass
    finally:
        listener.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Exercise 3: Build and Test the Nodes

### Step 1: Build the package

```bash
cd ~/ros2_ws
colcon build --packages-select robot_basics_py
```

### Step 2: Source the workspace

```bash
source ~/ros2_ws/install/setup.bash
```

### Step 3: Run the nodes

Open a new terminal and run the talker:

```bash
source ~/ros2_ws/install/setup.bash
ros2 run robot_basics_py talker
```

Open another terminal and run the listener:

```bash
source ~/ros2_ws/install/setup.bash
ros2 run robot_basics_py listener
```

You should see the talker publishing messages and the listener receiving them.

## Exercise 4: Use ROS 2 Tools

### Step 1: Check available topics

In a new terminal:

```bash
source ~/ros2_ws/install/setup.bash
ros2 topic list
```

### Step 2: Check topic information

```bash
ros2 topic info /chatter
```

### Step 3: Echo messages from topic

```bash
source ~/ros2_ws/install/setup.bash
ros2 topic echo /chatter
```

### Step 4: Use rqt_graph to visualize nodes

```bash
rqt_graph
```

## Exercise 5: Create a C++ Publisher (Optional)

Create a C++ package:

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake robot_basics_cpp --dependencies rclcpp std_msgs
```

Create `~/ros2_ws/src/robot_basics_cpp/src/talker.cpp`:

```cpp
#include <chrono>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class Talker : public rclcpp::Node
{
public:
    Talker() : Node("cpp_talker")
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("chatter", 10);
        timer_ = this->create_wall_timer(
            500ms, std::bind(&Talker::timer_callback, this));
        count_ = 0;
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Hello World: " + std::to_string(count_++);
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    size_t count_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Talker>());
    rclcpp::shutdown();
    return 0;
}
```

Edit `~/ros2_ws/src/robot_basics_cpp/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.8)
project(robot_basics_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(talker src/talker.cpp)
ament_target_dependencies(talker rclcpp std_msgs)

install(TARGETS
  talker
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
```

Build the C++ package:

```bash
cd ~/ros2_ws
colcon build --packages-select robot_basics_cpp
```

## Exercise 6: Custom Message Types

### Step 1: Create a custom message

Create a new package for custom messages:

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python robot_messages --dependencies rclpy std_msgs
```

Create the message directory and file:

```bash
mkdir -p ~/ros2_ws/src/robot_messages/robot_messages/msg
```

Create `~/ros2_ws/src/robot_messages/robot_messages/msg/RobotStatus.msg`:

```
string robot_name
int32 battery_level
float64 position_x
float64 position_y
float64 orientation
bool is_moving
```

Edit `~/ros2_ws/src/robot_messages/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>robot_messages</name>
  <version>0.0.0</version>
  <description>Custom messages for robot status</description>
  <maintainer email="your.email@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <build_depend>rosidl_default_generators</build_depend>
  <exec_depend>rosidl_default_runtime</exec_depend>
  <member_of_group>rosidl_interface_packages</member_of_group>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

Edit `~/ros2_ws/src/robot_messages/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.8)
project(robot_messages)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(msg_files
  "msg/RobotStatus.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES builtin_interfaces std_msgs
)

ament_package()
```

Build the custom message package:

```bash
cd ~/ros2_ws
colcon build --packages-select robot_messages
```

## Troubleshooting

### Common Issues and Solutions

1. **Permission denied errors during installation**:
   - Make sure you're using `sudo` for system package installations
   - Check that your user is in the appropriate groups

2. **Package not found errors**:
   - Ensure you've sourced the ROS 2 environment: `source /opt/ros/humble/setup.bash`
   - Verify workspace setup: `source ~/ros2_ws/install/setup.bash`

3. **Build errors**:
   - Check that all dependencies are installed
   - Verify correct Python version (3.8-3.10)
   - Ensure proper file permissions

4. **Nodes not communicating**:
   - Verify that both nodes are on the same ROS domain
   - Check topic names match exactly
   - Ensure nodes are in the same namespace or use correct namespace prefixes

## Assessment Questions

1. Explain the difference between a publisher and subscriber in ROS 2.
2. What is the purpose of the `colcon build` command?
3. How do you check what topics are currently active in ROS 2?
4. What are the advantages of using custom message types over standard types?

## Extension Exercises

1. Create a node that publishes sensor data (e.g., temperature, distance) at a specific rate
2. Implement a node that subscribes to multiple topics and combines the data
3. Add parameters to your nodes to make them configurable
4. Create a launch file to start multiple nodes simultaneously

## Summary

In this lab, you successfully:
- Set up the ROS 2 development environment
- Created publisher and subscriber nodes in Python
- Built and tested the nodes
- Used ROS 2 tools for monitoring and visualization
- Created custom message types (optional)

These fundamental skills form the basis for all subsequent ROS 2 development and will be essential as you progress through the robotics curriculum.