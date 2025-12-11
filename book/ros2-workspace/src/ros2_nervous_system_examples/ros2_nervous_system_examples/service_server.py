import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class ServiceServerNode(Node):
    def __init__(self):
        super().__init__('service_server')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {request.a} + {request.b} = {response.sum}')
        return response


def main(args=None):
    rclpy.init(args=args)

    service_server = ServiceServerNode()

    try:
        rclpy.spin(service_server)
    except KeyboardInterrupt:
        pass
    finally:
        service_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()