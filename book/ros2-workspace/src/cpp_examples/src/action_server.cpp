#include <memory>
#include <string>
#include <functional>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include "example_interfaces/action/fibonacci.hpp"

class ActionServer : public rclcpp::Node
{
public:
    using Fibonacci = example_interfaces::action::Fibonacci;
    using GoalHandleFibonacci = rclcpp_action::ServerGoalHandle<Fibonacci>;

    ActionServer() : Node("action_server_cpp")
    {
        using namespace std::placeholders;

        action_server_ = rclcpp_action::create_server<Fibonacci>(
            this,
            "fibonacci_cpp",
            std::bind(&ActionServer::handle_goal, this, _1, _2),
            std::bind(&ActionServer::handle_cancel, this, _1),
            std::bind(&ActionServer::handle_accepted, this, _1));
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
        RCLCPP_INFO(this->get_logger(), "Received cancel request");
        (void)goal_handle;
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(const std::shared_ptr<GoalHandleFibonacci> goal_handle)
    {
        using namespace std::placeholders;
        // this needs to return quickly to avoid blocking the executor, so spin up a new thread
        std::thread{std::bind(&ActionServer::execute, this, _1), goal_handle}.detach();
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
            // Check if there is a cancel request
            if (goal_handle->is_canceling()) {
                result->sequence = sequence;
                goal_handle->canceled(result);
                RCLCPP_INFO(this->get_logger(), "Goal canceled");
                return;
            }
            // Update sequence
            sequence.push_back(sequence[i] + sequence[i - 1]);
            // Publish feedback
            goal_handle->publish_feedback(feedback);
            RCLCPP_INFO(this->get_logger(), "Publishing feedback: %" PRId64, sequence[i]);

            loop_rate.sleep();
        }

        // Check if goal is done
        if (rclcpp::ok()) {
            result->sequence = sequence;
            goal_handle->succeed(result);
            RCLCPP_INFO(this->get_logger(), "Goal succeeded");
        }
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto action_server = std::make_shared<ActionServer>();
    rclcpp::spin(action_server);
    rclcpp::shutdown();
    return 0;
}