---
sidebar_label: 'Lab 1.1: ROS 2 Basics'
---

# Lab Exercise 1.1: ROS 2 Basics

This lab exercise introduces the fundamental concepts of ROS 2.

## Objectives

- Understand the basic concepts of ROS 2
- Create and run simple ROS 2 nodes
- Use topics for communication between nodes
- Use tools like ros2 topic and ros2 node

## Prerequisites

- ROS 2 Humble Hawksbill installed
- Basic knowledge of C++ or Python
- Terminal/command line familiarity

## Setup

1. Source your ROS 2 installation:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. Create a new workspace:
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws
   ```

## Creating a Simple Publisher

1. Create a new package:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_cmake cpp_pubsub
   ```

2. Navigate to the package source directory:
   ```bash
   cd cpp_pubsub/src
   ```

3. Create a publisher node:
   ```bash
   touch publisher_member_function.cpp
   ```

4. Add the following code to publisher_member_function.cpp:
   ```cpp
   #include "rclcpp/rclcpp.hpp"
   #include "std_msgs/msg/string.hpp"

   class MinimalPublisher : public rclcpp::Node
   {
   public:
       MinimalPublisher()
       : Node("minimal_publisher"), count_(0)
       {
           publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
           timer_ = this->create_wall_timer(
               500ms, std::bind(&MinimalPublisher::timer_callback, this));
       }

   private:
       void timer_callback()
       {
           auto message = std_msgs::msg::String();
           message.data = "Hello, world! " + std::to_string(count_++);
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
       rclcpp::spin(std::make_shared<MinimalPublisher>());
       rclcpp::shutdown();
       return 0;
   }
   ```

## Running the Publisher

1. Build the package:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select cpp_pubsub
   ```

2. Source the workspace:
   ```bash
   source install/setup.bash
   ```

3. Run the publisher:
   ```bash
   ros2 run cpp_pubsub minimal_publisher
   ```

## Expected Output

You should see messages being published every 500ms:
```
[INFO] [1617200000.123456789] [minimal_publisher]: Publishing: 'Hello, world! 0'
[INFO] [1617200000.623456789] [minimal_publisher]: Publishing: 'Hello, world! 1'
```

## Summary

In this lab, you learned how to create a simple publisher node in ROS 2. In the next lab, you'll create a subscriber to receive these messages.