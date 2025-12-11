#include "rclcpp/rclcpp.hpp"
#include "example_interfaces/srv/add_two_ints.hpp"

class ServiceServer : public rclcpp::Node
{
public:
    ServiceServer() : Node("service_server_cpp")
    {
        service_ = this->create_service<example_interfaces::srv::AddTwoInts>(
            "add_two_ints_cpp",
            std::bind(&ServiceServer::add_callback, this,
                     std::placeholders::_1, std::placeholders::_2));
    }

private:
    void add_callback(
        const example_interfaces::srv::AddTwoInts::Request::SharedPtr request,
        example_interfaces::srv::AddTwoInts::Response::SharedPtr response)
    {
        response->sum = request->a + request->b;
        RCLCPP_INFO(this->get_logger(), "%ld + %ld = %ld",
                   request->a, request->b, response->sum);
    }
    rclcpp::Service<example_interfaces::srv::AddTwoInts>::SharedPtr service_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ServiceServer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}