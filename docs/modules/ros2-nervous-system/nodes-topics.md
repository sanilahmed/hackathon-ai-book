---
sidebar_label: 'Nodes and Topics'
---

# Nodes and Topics in ROS 2

This document explains how nodes and topics work in the ROS 2 system.

## Nodes

Nodes are the fundamental building blocks of a ROS 2 system. Each node:
- Performs a specific function within the robot system
- Communicates with other nodes through topics, services, and actions
- Can be written in different languages (C++, Python, etc.)

## Topics

Topics are the publish/subscribe communication mechanism in ROS 2:
- Publishers send data to topics
- Subscribers receive data from topics
- Multiple publishers and subscribers can exist for the same topic
- Communication is asynchronous

## Creating Nodes

To create a ROS 2 node, you typically:
1. Initialize the ROS 2 client library
2. Create a node instance
3. Create publishers, subscribers, services, or actions
4. Spin the node to process callbacks
5. Clean up resources

## Example

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
```