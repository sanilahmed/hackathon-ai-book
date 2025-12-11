import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import cv2


class AIPerceptionNode(Node):
    def __init__(self):
        super().__init__('ai_perception_node')

        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )

        # Create publishers
        self.object_pub = self.create_publisher(
            String,
            'ai/detected_objects',
            10
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        # Initialize variables
        self.latest_image = None
        self.latest_scan = None

        # Create timer for processing
        self.timer = self.create_timer(0.1, self.process_data)

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def scan_callback(self, msg):
        # Store latest laser scan
        self.latest_scan = np.array(msg.ranges)
        # Replace invalid values
        self.latest_scan = np.nan_to_num(self.latest_scan, nan=10.0, posinf=10.0, neginf=0.0)

    def process_data(self):
        if self.latest_image is not None:
            # Simple color-based object detection (example)
            detected_objects = self.detect_objects(self.latest_image)

            if detected_objects:
                # Publish detected objects
                obj_msg = String()
                obj_msg.data = ', '.join(detected_objects)
                self.object_pub.publish(obj_msg)

                # Example: if we detect something, move the robot
                if 'red_object' in detected_objects:
                    cmd_msg = Twist()
                    cmd_msg.linear.x = 0.2  # Move forward
                    cmd_msg.angular.z = 0.0
                    self.cmd_pub.publish(cmd_msg)

        if self.latest_scan is not None:
            # Simple obstacle avoidance based on scan data
            self.avoid_obstacles()

    def detect_objects(self, image):
        """Simple color-based object detection example."""
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for red color (in HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = mask1 + mask2

        # Find contours
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small areas
                detected_objects.append('red_object')

        return detected_objects

    def avoid_obstacles(self):
        """Simple obstacle avoidance based on laser scan."""
        if self.latest_scan is None:
            return

        # Get distances in front, left, and right sectors
        size = len(self.latest_scan)
        front_sector = self.latest_scan[size//2 - size//8 : size//2 + size//8]
        left_sector = self.latest_scan[size//4 - size//16 : size//4 + size//16]
        right_sector = self.latest_scan[3*size//4 - size//16 : 3*size//4 + size//16]

        front_min = np.min(front_sector)
        left_min = np.min(left_sector)
        right_min = np.min(right_sector)

        cmd_msg = Twist()

        # If obstacle is close in front, turn
        if front_min < 1.0:
            if left_min > right_min:
                cmd_msg.angular.z = 0.5  # Turn left
            else:
                cmd_msg.angular.z = -0.5  # Turn right
        else:
            cmd_msg.linear.x = 0.3  # Move forward

        self.cmd_pub.publish(cmd_msg)


def main(args=None):
    rclpy.init(args=args)

    ai_perception_node = AIPerceptionNode()

    try:
        rclpy.spin(ai_perception_node)
    except KeyboardInterrupt:
        pass
    finally:
        ai_perception_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()