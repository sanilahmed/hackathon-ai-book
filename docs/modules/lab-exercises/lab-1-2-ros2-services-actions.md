---
sidebar_label: 'Lab 1.2: Services and Actions'
---

# Lab Exercise 1.2: ROS 2 Services and Actions

This lab exercise covers services and actions in ROS 2.

## Objectives

- Understand the difference between topics, services, and actions
- Create a ROS 2 service server and client
- Create a ROS 2 action server and client
- Use command-line tools to interact with services and actions

## Services

Services provide synchronous request/response communication in ROS 2.

### Creating a Service Server

1. Create a service definition file `AddTwoInts.srv` in a `srv` directory:
   ```
   int64 a
   int64 b
   ---
   int64 sum
   ```

2. Create a service server in C++:
   ```cpp
   #include "rclcpp/rclcpp.hpp"
   #include "example_interfaces/srv/add_two_ints.hpp"
   #include <memory>

   void add(const std::shared_ptr<example_interfaces::srv::AddTwoInts::Request> request,
           std::shared_ptr<example_interfaces::srv::AddTwoInts::Response> response)
   {
       response->sum = request->a + request->b;
       RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Incoming request: %ld + %ld = %ld",
                   request->a, request->b, response->sum);
   }

   int main(int argc, char **argv)
   {
       rclcpp::init(argc, argv);

       std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("add_two_ints_server");

       auto service = node->create_service<example_interfaces::srv::AddTwoInts>("add_two_ints", &add);
       RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Ready to add two ints.");

       rclcpp::spin(node);
       rclcpp::shutdown();
       return 0;
   }
   ```

### Creating a Service Client

```cpp
#include "rclcpp/rclcpp.hpp"
#include "example_interfaces/srv/add_two_ints.hpp"
#include <chrono>
#include <memory>

using AddTwoInts = example_interfaces::srv::AddTwoInts;

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("add_two_ints_client");

    auto client = node->create_client<AddTwoInts>("add_two_ints");

    while (!client->wait_for_service(std::chrono::seconds(1))) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Interrupted while waiting for the service. Exiting.");
            return 0;
        }
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Service not available, waiting again...");
    }

    auto request = std::make_shared<AddTwoInts::Request>();
    request->a = 2;
    request->b = 3;

    auto result = client->async_send_request(request);
    if (rclcpp::spin_until_future_complete(node, result) == rclcpp::FutureReturnCode::SUCCESS) {
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Result of add_two_ints: %ld", result.get()->sum);
    } else {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Failed to call service add_two_ints");
    }

    rclcpp::shutdown();
    return 0;
}
```

## Actions

Actions provide asynchronous goal-oriented communication with feedback.

### Creating an Action Server

```cpp
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "example_interfaces/action/fibonacci.hpp"

class FibonacciActionServer : public rclcpp::Node
{
public:
    using Fibonacci = example_interfaces::action::Fibonacci;
    using GoalHandleFibonacci = rclcpp_action::ServerGoalHandle<Fibonacci>;

    explicit FibonacciActionServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
    : Node("fibonacci_action_server", options)
    {
        using namespace std::placeholders;

        this->action_server_ = rclcpp_action::create_server<Fibonacci>(
            this,
            "fibonacci",
            std::bind(&FibonacciActionServer::handle_goal, this, _1, _2),
            std::bind(&FibonacciActionServer::handle_cancel, this, _1),
            std::bind(&FibonacciActionServer::handle_accepted, this, _1));
    }

private:
    rclcpp_action::Server<Fibonacci>::SharedPtr action_server_;

    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID & uuid,
        std::shared_ptr<const Fibonacci::Goal> goal)
    {
        RCLCPP_INFO(this->get_logger(), "Received goal request with order %d", goal->order);
        (void)uuid;
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<GoalHandleFibonacci> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
        (void)goal_handle;
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(const std::shared_ptr<GoalHandleFibonacci> goal_handle)
    {
        using namespace std::placeholders;
        std::thread{std::bind(&FibonacciActionServer::execute, this, _1), goal_handle}.detach();
    }

    void execute(const std::shared_ptr<GoalHandleFibonacci> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Executing goal");
        rclcpp::Rate loop_rate(1);
        const auto goal = goal_handle->get_goal();
        auto feedback = std::make_shared<Fibonacci::Feedback>();
        auto & sequence = feedback->sequence;
        auto result = std::make_shared<Fibonacci::Result>();

        sequence.push_back(0);
        sequence.push_back(1);

        for (int i = 1; (i < goal->order) && rclcpp::ok(); ++i) {
            if (goal_handle->is_canceling()) {
                result->sequence = sequence;
                goal_handle->canceled(result);
                RCLCPP_INFO(this->get_logger(), "Goal canceled");
                return;
            }

            sequence.push_back(sequence[i] + sequence[i - 1]);
            feedback->sequence = sequence;
            goal_handle->publish_feedback(feedback);
            RCLCPP_INFO(this->get_logger(), "Publishing feedback");

            loop_rate.sleep();
        }

        if (rclcpp::ok()) {
            result->sequence = sequence;
            goal_handle->succeed(result);
            RCLCPP_INFO(this->get_logger(), "Goal succeeded");
        }
    }
};
```

## Command-line Tools

### Working with Services

- List services: `ros2 service list`
- Get service type: `ros2 service type <service_name>`
- Call a service: `ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 1, b: 2}"`

### Working with Actions

- List actions: `ros2 action list`
- Get action type: `ros2 action type <action_name>`
- Send action goal: `ros2 action send_goal /fibonacci example_interfaces/action/Fibonacci "{order: 5}"`

## Summary

In this lab, you learned how to create and use services and actions in ROS 2. Services are good for simple request/response interactions, while actions are better for long-running tasks that need feedback.