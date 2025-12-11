import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class ServiceClientNode(Node):
    def __init__(self):
        super().__init__('service_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main(args=None):
    rclpy.init(args=args)

    service_client = ServiceClientNode()
    try:
        response = service_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
        service_client.get_logger().info(
            f'Result of add_two_ints: for {sys.argv[1]} + {sys.argv[2]} = {response.sum}')
    except Exception as e:
        service_client.get_logger().error(f'Service call failed: {e}')
    finally:
        service_client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()