# Nodes and Topics in ROS 2

## Understanding Nodes

Nodes are the fundamental computational elements in ROS 2. Each node is an independent process that performs specific functions within the robotic system. Nodes are designed to be modular, allowing for flexible system architectures where each component has a single, well-defined responsibility.

### Creating a Node

In ROS 2, nodes are created by inheriting from the `Node` class provided by the client libraries (rclpy for Python, rclcpp for C++):

**Python Example:**
```python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1
```

**C++ Example:**
```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalPublisher : public rclcpp::Node
{
public:
    MinimalPublisher() : Node("minimal_publisher"), count_(0)
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
        timer_ = this->create_wall_timer(
            500ms, std::bind(&MinimalPublisher::timer_callback, this));
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
```

## Topics and Message Passing

Topics are named buses that allow nodes to exchange messages. The communication follows a publish-subscribe pattern where publishers send messages to topics and subscribers receive messages from topics.

### Message Types

ROS 2 defines standard message types in packages like `std_msgs`, `geometry_msgs`, and `sensor_msgs`. Custom message types can also be defined using `.msg` files.

### Publisher Example

A publisher node creates a publisher object and sends messages to a topic:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello from ROS 2 talker'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')
```

### Subscriber Example

A subscriber node creates a subscription object and registers a callback function:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: {msg.data}')
```

## Quality of Service (QoS) Settings

QoS settings allow fine-tuning of communication behavior:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Create a QoS profile for sensor data (best effort, keep last 10)
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

# Create a QoS profile for critical commands (reliable, keep all)
command_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_ALL
)

publisher = self.create_publisher(String, 'critical_topic', command_qos)
```

## Advanced Topic Patterns

### Latching (Transient Local Durability)

For static data that late-joining subscribers should receive immediately:

```python
from rclpy.qos import DurabilityPolicy, QoSProfile

latching_qos = QoSProfile(
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_ALL
)
latching_publisher = self.create_publisher(String, 'static_topic', latching_qos)
```

### Multiple Publishers and Subscribers

ROS 2 supports multiple publishers and subscribers on the same topic:

```python
# Multiple publishers can send to the same topic
pub1 = self.create_publisher(String, 'shared_topic', 10)
pub2 = self.create_publisher(String, 'shared_topic', 10)

# Multiple subscribers can receive from the same topic
sub1 = self.create_subscription(String, 'shared_topic', callback1, 10)
sub2 = self.create_subscription(String, 'shared_topic', callback2, 10)
```

## Best Practices

1. **Node Design**: Each node should have a single responsibility and be independently testable
2. **Topic Naming**: Use descriptive, consistent naming conventions (e.g., `/robot_name/sensor_type/data`)
3. **Message Frequency**: Balance between responsiveness and network load
4. **QoS Selection**: Choose appropriate QoS settings based on data importance and real-time requirements
5. **Error Handling**: Implement proper error handling for network disconnections and message failures

## Practical Exercise

Create a simple publisher-subscriber pair:
1. Implement a publisher that sends sensor readings at 10 Hz
2. Implement a subscriber that logs the received data
3. Configure appropriate QoS settings for sensor data
4. Test the communication using ROS 2 command-line tools

## Summary

Nodes and topics form the foundation of ROS 2 communication. Understanding how to create nodes, define publishers and subscribers, and configure QoS settings is essential for building robust robotic systems. The modular nature of nodes enables flexible system architectures that can be easily extended and maintained.

## Learning Check

After studying this section, you should be able to:
- Create nodes in both Python and C++
- Implement publishers and subscribers for message passing
- Configure appropriate QoS settings for different use cases
- Design effective topic naming conventions
- Understand the publish-subscribe communication pattern