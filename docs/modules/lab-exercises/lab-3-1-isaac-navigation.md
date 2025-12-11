# Lab 3.1: NVIDIA Isaac Navigation System

## Overview

This lab exercise focuses on implementing navigation systems using NVIDIA Isaac Sim and Isaac ROS packages. Students will learn to build autonomous navigation capabilities for humanoid robots using advanced perception and planning algorithms optimized for NVIDIA hardware.

## Learning Objectives

By the end of this lab, students will be able to:
1. Configure NVIDIA Isaac Sim for humanoid robot navigation simulation
2. Implement perception pipelines using Isaac ROS packages
3. Set up SLAM (Simultaneous Localization and Mapping) systems
4. Create path planning and obstacle avoidance algorithms
5. Deploy navigation systems on NVIDIA Jetson platforms
6. Evaluate navigation performance using standard metrics

## Prerequisites

- Completion of Module 1: ROS 2 Fundamentals
- Basic understanding of navigation concepts (path planning, SLAM)
- Familiarity with NVIDIA Isaac ecosystem
- Access to NVIDIA GPU or Jetson development kit

## Theory Background

NVIDIA Isaac navigation systems leverage GPU-accelerated algorithms for real-time perception, mapping, and planning. Key components include:

- **Isaac Sim**: NVIDIA's robotics simulator with realistic physics and sensor simulation
- **Isaac ROS**: GPU-accelerated ROS packages for perception, navigation, and manipulation
- **SLAM Algorithms**: GPU-accelerated simultaneous localization and mapping
- **Path Planning**: GPU-accelerated global and local planners
- **Perception Pipeline**: Real-time sensor processing using CUDA and TensorRT

## Lab Exercise

### Part 1: Isaac Sim Environment Setup

First, let's set up the Isaac Sim environment for navigation:

```python
# Isaac Sim navigation environment setup
import omni
from pxr import Gf, UsdGeom, Sdf
import carb
import omni.kit.commands
import numpy as np

class IsaacSimNavigationEnvironment:
    def __init__(self):
        self.world = None
        self.robot = None
        self.navigation_map = None

    def create_navigation_world(self):
        """Create a navigation world with obstacles and waypoints"""
        # Get the stage
        stage = omni.usd.get_context().get_stage()

        # Create ground plane
        omni.kit.commands.execute(
            "CreateMeshPrimWithDefaultXform",
            prim_type="Plane",
            prim_path="/World/GroundPlane",
            name="ground_plane",
            size=10.0,
            axis="Z"
        )

        # Create obstacles
        self.create_obstacles(stage)

        # Create navigation waypoints
        self.create_waypoints(stage)

        # Set up lighting
        self.setup_lighting(stage)

        print("Navigation world created successfully")

    def create_obstacles(self, stage):
        """Create navigation obstacles"""
        # Create walls
        for i, (pos, size) in enumerate([
            ((5, 0, 0), (0.5, 10, 1)),
            ((-5, 0, 0), (0.5, 10, 1)),
            ((0, 5, 0), (10, 0.5, 1)),
            ((0, -5, 0), (10, 0.5, 1))
        ]):
            omni.kit.commands.execute(
                "CreateMeshPrimWithDefaultXform",
                prim_type="Cube",
                prim_path=f"/World/Obstacle{i}",
                name=f"obstacle{i}",
                size=1.0
            )

            # Set position and scale
            prim = stage.GetPrimAtPath(f"/World/Obstacle{i}")
            xform = UsdGeom.Xformable(prim)
            xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
            xform.AddScaleOp().Set(Gf.Vec3d(*size))

    def create_waypoints(self, stage):
        """Create navigation waypoints"""
        waypoints = [
            (0, 0, 0),      # Start
            (2, 2, 0),      # Waypoint 1
            (4, 0, 0),      # Waypoint 2
            (2, -2, 0),     # Waypoint 3
            (0, 0, 0)       # Return to start
        ]

        for i, pos in enumerate(waypoints):
            omni.kit.commands.execute(
                "CreateMeshPrimWithDefaultXform",
                prim_type="Sphere",
                prim_path=f"/World/Waypoint{i}",
                name=f"waypoint{i}",
                radius=0.2
            )

            # Set position
            prim = stage.GetPrimAtPath(f"/World/Waypoint{i}")
            xform = UsdGeom.Xformable(prim)
            xform.AddTranslateOp().Set(Gf.Vec3d(*pos))

    def setup_lighting(self, stage):
        """Set up lighting for the scene"""
        # Create dome light
        omni.kit.commands.execute(
            "CreatePrimWithDefaultXform",
            prim_type="DomeLight",
            prim_path="/World/DomeLight",
            name="dome_light"
        )

        # Create distant light
        omni.kit.commands.execute(
            "CreatePrimWithDefaultXform",
            prim_type="DistantLight",
            prim_path="/World/DistantLight",
            name="distant_light"
        )

# Example usage
def setup_navigation_environment():
    env = IsaacSimNavigationEnvironment()
    env.create_navigation_world()
    return env

if __name__ == "__main__":
    setup_navigation_environment()
```

### Part 2: Isaac ROS Perception Pipeline

Now let's implement the perception pipeline using Isaac ROS packages:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, PointCloud2
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Twist
from cv_bridge import CvBridge
import numpy as np
import torch

class IsaacROSPerceptionPipeline(Node):
    def __init__(self):
        super().__init__('isaac_ros_perception_pipeline')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers and subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/map',
            10
        )

        self.collision_pub = self.create_publisher(
            Twist,
            '/collision_avoidance/cmd_vel',
            10
        )

        # Initialize perception models
        self.initialize_perception_models()

        # Internal state
        self.current_scan = None
        self.current_image = None
        self.current_odom = None
        self.map_resolution = 0.05  # 5cm resolution
        self.map_width = 200  # 10m x 10m map
        self.map_height = 200

    def initialize_perception_models(self):
        """Initialize Isaac ROS perception models"""
        # For this example, we'll use simple models
        # In practice, these would be Isaac ROS packages like:
        # - Isaac ROS Stereo DNN
        # - Isaac ROS Point Cloud Segmentation
        # - Isaac ROS Occupancy Grid Mapping

        # Create a simple occupancy grid
        self.occupancy_grid = np.zeros((self.map_height, self.map_width), dtype=np.int8)

        self.get_logger().info('Isaac ROS perception pipeline initialized')

    def scan_callback(self, msg):
        """Process LIDAR scan data"""
        try:
            # Convert scan to occupancy grid
            self.current_scan = {
                'ranges': np.array(msg.ranges),
                'angle_min': msg.angle_min,
                'angle_max': msg.angle_max,
                'angle_increment': msg.angle_increment,
                'time_increment': msg.time_increment,
                'scan_time': msg.scan_time,
                'range_min': msg.range_min,
                'range_max': msg.range_max
            }

            # Update occupancy grid with scan data
            self.update_occupancy_grid_from_scan(self.current_scan)

            # Publish updated map
            self.publish_occupancy_grid()

        except Exception as e:
            self.get_logger().error(f'Error processing scan: {str(e)}')

    def image_callback(self, msg):
        """Process camera image data"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Process image for obstacle detection (simplified)
            obstacles_detected = self.detect_obstacles_in_image(cv_image)

            # If obstacles detected, publish collision avoidance command
            if obstacles_detected:
                self.publish_collision_avoidance_command()

            self.current_image = cv_image

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def odom_callback(self, msg):
        """Process odometry data"""
        try:
            self.current_odom = {
                'pose': {
                    'position': (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z),
                    'orientation': (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                                  msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
                },
                'twist': {
                    'linear': (msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z),
                    'angular': (msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z)
                }
            }

            # Update robot position in occupancy grid
            self.update_robot_position_in_map(self.current_odom)

        except Exception as e:
            self.get_logger().error(f'Error processing odometry: {str(e)}')

    def update_occupancy_grid_from_scan(self, scan_data):
        """Update occupancy grid based on LIDAR scan"""
        # Convert scan ranges to grid coordinates
        angles = np.arange(scan_data['angle_min'], scan_data['angle_max'], scan_data['angle_increment'])

        # Robot position in grid coordinates (assuming robot is at center initially)
        robot_x, robot_y = self.map_width // 2, self.map_height // 2

        for i, (angle, range_val) in enumerate(zip(angles, scan_data['ranges'])):
            if np.isnan(range_val) or range_val > scan_data['range_max']:
                continue

            # Calculate endpoint of ray
            end_x = robot_x + int((range_val * np.cos(angle)) / self.map_resolution)
            end_y = robot_y + int((range_val * np.sin(angle)) / self.map_resolution)

            # Check bounds
            if 0 <= end_x < self.map_width and 0 <= end_y < self.map_height:
                # Mark endpoint as occupied if range is below threshold
                if range_val < 1.0:  # 1m threshold for obstacles
                    self.occupancy_grid[end_y, end_x] = 100  # Occupied
                else:
                    self.occupancy_grid[end_y, end_x] = 0    # Free space

    def detect_obstacles_in_image(self, image):
        """Detect obstacles in camera image (simplified)"""
        # Simple edge detection for obstacle detection
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Count edge pixels in lower half of image (potential obstacles)
        height, width = edges.shape
        lower_half = edges[height//2:, :]
        edge_count = np.sum(lower_half > 0)

        # If many edges detected in lower half, likely obstacles
        return edge_count > (height * width * 0.05)  # 5% threshold

    def update_robot_position_in_map(self, odom_data):
        """Update robot position in occupancy grid"""
        # Convert world coordinates to grid coordinates
        world_x, world_y = odom_data['pose']['position'][0], odom_data['pose']['position'][1]
        grid_x = int(world_x / self.map_resolution) + self.map_width // 2
        grid_y = int(world_y / self.map_resolution) + self.map_height // 2

        # Ensure within bounds
        grid_x = max(0, min(self.map_width - 1, grid_x))
        grid_y = max(0, min(self.map_height - 1, grid_y))

        # Mark robot position (clear around it)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                x, y = grid_x + dx, grid_y + dy
                if 0 <= x < self.map_width and 0 <= y < self.map_height:
                    self.occupancy_grid[y, x] = 0  # Free space around robot

    def publish_occupancy_grid(self):
        """Publish occupancy grid message"""
        from nav_msgs.msg import OccupancyGrid
        from std_msgs.msg import Header
        import time

        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.info.resolution = self.map_resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin.position.x = -self.map_width * self.map_resolution / 2
        msg.info.origin.position.y = -self.map_height * self.map_resolution / 2

        # Flatten grid for message
        msg.data = self.occupancy_grid.flatten().tolist()

        self.map_pub.publish(msg)

    def publish_collision_avoidance_command(self):
        """Publish collision avoidance command"""
        twist_msg = Twist()
        # Simple collision avoidance: turn away from obstacles
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.5  # Turn right to avoid obstacle

        self.collision_pub.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)

    perception_pipeline = IsaacROSPerceptionPipeline()

    try:
        rclpy.spin(perception_pipeline)
    except KeyboardInterrupt:
        pass
    finally:
        perception_pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 3: Isaac SLAM Implementation

Now let's implement a SLAM system using Isaac's capabilities:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, PointCloud2
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped, PointStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import time

class IsaacSLAMNode(Node):
    def __init__(self):
        super().__init__('isaac_slam_node')

        # Publishers and subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/map',
            10
        )

        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )

        # Transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # SLAM parameters
        self.map_resolution = 0.05  # 5cm
        self.map_width = 400  # 20m x 20m
        self.map_height = 400
        self.map_origin_x = -self.map_width * self.map_resolution / 2
        self.map_origin_y = -self.map_height * self.map_resolution / 2

        # Initialize SLAM components
        self.initialize_slam()

    def initialize_slam(self):
        """Initialize SLAM components"""
        # Occupancy grid map
        self.occupancy_map = np.zeros((self.map_height, self.map_height), dtype=np.int8)

        # Robot pose
        self.robot_pose = {
            'x': 0.0,
            'y': 0.0,
            'theta': 0.0
        }

        # Previous odometry for motion model
        self.prev_odom = None
        self.prev_time = None

        self.get_logger().info('Isaac SLAM initialized')

    def scan_callback(self, msg):
        """Process LIDAR scan for mapping"""
        try:
            # Update map with current scan
            self.update_map_with_scan(msg)

            # Publish updated map
            self.publish_map()

            # Broadcast transform
            self.broadcast_transform()

        except Exception as e:
            self.get_logger().error(f'Error in scan callback: {str(e)}')

    def odom_callback(self, msg):
        """Process odometry for localization"""
        try:
            current_time = self.get_clock().now().nanoseconds / 1e9

            if self.prev_odom is not None and self.prev_time is not None:
                # Calculate motion since last update
                dt = current_time - self.prev_time

                # Extract pose and twist
                pose = msg.pose.pose
                twist = msg.twist.twist

                # Update robot pose using motion model (simplified)
                self.robot_pose['x'] += twist.linear.x * dt * np.cos(self.robot_pose['theta'])
                self.robot_pose['y'] += twist.linear.x * dt * np.sin(self.robot_pose['theta'])
                self.robot_pose['theta'] += twist.angular.z * dt

            self.prev_odom = msg
            self.prev_time = current_time

        except Exception as e:
            self.get_logger().error(f'Error in odom callback: {str(e)}')

    def update_map_with_scan(self, scan_msg):
        """Update occupancy map with scan data"""
        # Convert scan to grid coordinates
        angles = np.arange(
            scan_msg.angle_min,
            scan_msg.angle_max,
            scan_msg.angle_increment
        )

        # Get robot position in grid coordinates
        grid_x = int((self.robot_pose['x'] - self.map_origin_x) / self.map_resolution)
        grid_y = int((self.robot_pose['y'] - self.map_origin_y) / self.map_resolution)

        # Ensure robot position is within bounds
        grid_x = max(0, min(self.map_width - 1, grid_x))
        grid_y = max(0, min(self.map_height - 1, grid_y))

        # Process each scan ray
        for i, (angle, range_val) in enumerate(zip(angles, scan_msg.ranges)):
            if np.isnan(range_val) or range_val > scan_msg.range_max:
                continue

            # Calculate ray endpoint in world coordinates
            world_end_x = self.robot_pose['x'] + range_val * np.cos(self.robot_pose['theta'] + angle)
            world_end_y = self.robot_pose['y'] + range_val * np.sin(self.robot_pose['theta'] + angle)

            # Convert to grid coordinates
            end_grid_x = int((world_end_x - self.map_origin_x) / self.map_resolution)
            end_grid_y = int((world_end_y - self.map_origin_y) / self.map_resolution)

            # Ensure endpoint is within bounds
            end_grid_x = max(0, min(self.map_width - 1, end_grid_x))
            end_grid_y = max(0, min(self.map_height - 1, end_grid_y))

            # Ray tracing to update occupancy
            self.trace_ray(grid_x, grid_y, end_grid_x, end_grid_y, range_val < 3.0)  # Mark as occupied if < 3m

    def trace_ray(self, start_x, start_y, end_x, end_y, hit_obstacle):
        """Trace ray and update occupancy grid"""
        # Bresenham's line algorithm for ray tracing
        dx = abs(end_x - start_x)
        dy = abs(end_y - start_y)
        x_step = 1 if end_x > start_x else -1
        y_step = 1 if end_y > start_y else -1
        error = dx - dy

        x, y = start_x, start_y

        while True:
            # Check bounds
            if not (0 <= x < self.map_width and 0 <= y < self.map_height):
                break

            # Update occupancy based on whether we hit an obstacle
            if x == end_x and y == end_y and hit_obstacle:
                # Hit obstacle - mark as occupied
                if self.occupancy_map[y, x] < 50:
                    self.occupancy_map[y, x] += 25
                if self.occupancy_map[y, x] > 100:
                    self.occupancy_map[y, x] = 100
            else:
                # Free space - decrease occupancy
                if self.occupancy_map[y, x] > 0:
                    self.occupancy_map[y, x] -= 10
                if self.occupancy_map[y, x] < 0:
                    self.occupancy_map[y, x] = 0

            if x == end_x and y == end_y:
                break

            error2 = 2 * error
            if error2 > -dy:
                error -= dy
                x += x_step
            if error2 < dx:
                error += dx
                y += y_step

    def publish_map(self):
        """Publish occupancy grid map"""
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.info.resolution = self.map_resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin.position.x = self.map_origin_x
        msg.info.origin.position.y = self.map_origin_y

        # Flatten map data
        msg.data = self.occupancy_map.flatten().tolist()

        self.map_pub.publish(msg)

    def broadcast_transform(self):
        """Broadcast map -> odom transform"""
        from geometry_msgs.msg import TransformStamped

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"

        t.transform.translation.x = self.robot_pose['x']
        t.transform.translation.y = self.robot_pose['y']
        t.transform.translation.z = 0.0

        # Convert angle to quaternion
        from tf2_ros import transform_to_msg
        rot = R.from_euler('z', self.robot_pose['theta'])
        quat = rot.as_quat()
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)

    slam_node = IsaacSLAMNode()

    try:
        rclpy.spin(slam_node)
    except KeyboardInterrupt:
        pass
    finally:
        slam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 4: Isaac Path Planning

Now let's implement path planning using Isaac's GPU-accelerated algorithms:

```python
import numpy as np
import heapq
from typing import List, Tuple, Optional
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header
import time

class IsaacPathPlannerNode(Node):
    def __init__(self):
        super().__init__('isaac_path_planner')

        # Publishers and subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/move_base_simple/goal',
            self.goal_callback,
            10
        )

        self.path_pub = self.create_publisher(
            Path,
            '/plan',
            10
        )

        # Internal state
        self.occupancy_map = None
        self.map_info = None
        self.current_goal = None
        self.current_start = None

        self.get_logger().info('Isaac Path Planner initialized')

    def map_callback(self, msg):
        """Process occupancy grid map"""
        try:
            # Store map info
            self.map_info = {
                'resolution': msg.info.resolution,
                'width': msg.info.width,
                'height': msg.info.height,
                'origin_x': msg.info.origin.position.x,
                'origin_y': msg.info.origin.position.y
            }

            # Convert map data to 2D array
            self.occupancy_map = np.array(msg.data).reshape(
                self.map_info['height'],
                self.map_info['width']
            ).astype(np.int8)

        except Exception as e:
            self.get_logger().error(f'Error processing map: {str(e)}')

    def goal_callback(self, msg):
        """Process goal pose and plan path"""
        try:
            if self.occupancy_map is None:
                self.get_logger().warn('No map available, cannot plan path')
                return

            # Convert goal to map coordinates
            goal_x = int((msg.pose.position.x - self.map_info['origin_x']) / self.map_info['resolution'])
            goal_y = int((msg.pose.position.y - self.map_info['origin_y']) / self.map_info['resolution'])

            # Get current robot position (simplified - assume at center initially)
            start_x = self.map_info['width'] // 2
            start_y = self.map_info['height'] // 2

            # Plan path
            path = self.plan_path((start_x, start_y), (goal_x, goal_y))

            if path:
                # Publish path
                self.publish_path(path, msg.header.frame_id)
                self.get_logger().info(f'Path planned with {len(path)} waypoints')
            else:
                self.get_logger().warn('No valid path found to goal')

        except Exception as e:
            self.get_logger().error(f'Error planning path: {str(e)}')

    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Plan path using A* algorithm with GPU acceleration considerations"""
        # Check if start and goal are valid
        if not self.is_valid_cell(start[0], start[1]) or not self.is_valid_cell(goal[0], goal[1]):
            return None

        # Check if start and goal are in free space
        if self.occupancy_map[start[1], start[0]] > 50 or self.occupancy_map[goal[1], goal[0]] > 50:
            return None

        # A* pathfinding algorithm
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + self.distance(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Heuristic function for A* (Euclidean distance)"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Distance between two cells"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),  # 4-connectivity
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:  # 8-connectivity if needed
            new_x, new_y = pos[0] + dx, pos[1] + dy

            if self.is_valid_cell(new_x, new_y) and self.occupancy_map[new_y, new_x] < 50:
                neighbors.append((new_x, new_y))

        return neighbors

    def is_valid_cell(self, x: int, y: int) -> bool:
        """Check if cell coordinates are valid"""
        if self.occupancy_map is None:
            return False
        return 0 <= x < self.map_info['width'] and 0 <= y < self.map_info['height']

    def publish_path(self, path: List[Tuple[int, int]], frame_id: str):
        """Publish path as Path message"""
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = frame_id

        for x, y in path:
            # Convert grid coordinates back to world coordinates
            world_x = x * self.map_info['resolution'] + self.map_info['origin_x']
            world_y = y * self.map_info['resolution'] + self.map_info['origin_y']

            pose_stamped = PoseStamped()
            pose_stamped.pose.position.x = world_x
            pose_stamped.pose.position.y = world_y
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0

            path_msg.poses.append(pose_stamped)

        self.path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)

    path_planner = IsaacPathPlannerNode()

    try:
        rclpy.spin(path_planner)
    except KeyboardInterrupt:
        pass
    finally:
        path_planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 5: Isaac Navigation Stack Integration

Finally, let's create a complete navigation stack integration:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import numpy as np
from typing import Dict, Any

class IsaacNavigationStack(Node):
    def __init__(self):
        super().__init__('isaac_navigation_stack')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/navigation_status', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.path_sub = self.create_subscription(Path, '/plan', self.path_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)

        # Navigation parameters
        self.linear_vel = 0.5  # m/s
        self.angular_vel = 0.5  # rad/s
        self.safe_distance = 0.5  # meters
        self.arrival_threshold = 0.3  # meters

        # Internal state
        self.current_pose = None
        self.current_scan = None
        self.current_path = []
        self.current_goal = None
        self.navigation_active = False
        self.path_index = 0

        # Create timer for navigation loop
        self.nav_timer = self.create_timer(0.1, self.navigation_loop)

        self.get_logger().info('Isaac Navigation Stack initialized')

    def odom_callback(self, msg):
        """Update current pose from odometry"""
        self.current_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'theta': self.quaternion_to_yaw(msg.pose.pose.orientation)
        }

    def scan_callback(self, msg):
        """Update current scan data"""
        self.current_scan = {
            'ranges': np.array(msg.ranges),
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment
        }

    def path_callback(self, msg):
        """Update current path"""
        self.current_path = []
        for pose_stamped in msg.poses:
            self.current_path.append({
                'x': pose_stamped.pose.position.x,
                'y': pose_stamped.pose.position.y
            })
        self.path_index = 0
        self.navigation_active = len(self.current_path) > 0

    def goal_callback(self, msg):
        """Handle new goal"""
        self.current_goal = {
            'x': msg.pose.position.x,
            'y': msg.pose.position.y
        }
        self.get_logger().info(f'New goal received: ({self.current_goal["x"]:.2f}, {self.current_goal["y"]:.2f})')

    def quaternion_to_yaw(self, orientation):
        """Convert quaternion to yaw angle"""
        import math
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def navigation_loop(self):
        """Main navigation loop"""
        if not self.current_pose or not self.navigation_active:
            return

        if not self.current_path or self.path_index >= len(self.current_path):
            self.navigation_active = False
            self.publish_status("Navigation completed")
            self.stop_robot()
            return

        # Get current target waypoint
        target = self.current_path[self.path_index]

        # Check for obstacles
        if self.detect_obstacles():
            self.publish_status("Obstacle detected, stopping")
            self.stop_robot()
            return

        # Calculate direction to target
        dx = target['x'] - self.current_pose['x']
        dy = target['y'] - self.current_pose['y']
        distance_to_target = np.sqrt(dx**2 + dy**2)

        # Check if reached current waypoint
        if distance_to_target < self.arrival_threshold:
            self.path_index += 1
            if self.path_index >= len(self.current_path):
                self.navigation_active = False
                self.publish_status("Navigation completed")
                self.stop_robot()
                return
            else:
                # Move to next waypoint
                target = self.current_path[self.path_index]
                dx = target['x'] - self.current_pose['x']
                dy = target['y'] - self.current_pose['y']
                distance_to_target = np.sqrt(dx**2 + dy**2)

        # Calculate desired heading
        desired_theta = np.arctan2(dy, dx)
        angle_diff = desired_theta - self.current_pose['theta']

        # Normalize angle difference
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        # Create command
        cmd = Twist()

        # Angular control
        if abs(angle_diff) > 0.1:  # 0.1 rad = ~5.7 degrees
            cmd.angular.z = np.clip(angle_diff * 1.5, -self.angular_vel, self.angular_vel)
        else:
            # Linear control
            cmd.linear.x = np.clip(distance_to_target * 1.0, 0.0, self.linear_vel)

        # Publish command
        self.cmd_vel_pub.publish(cmd)

        # Update status
        self.publish_status(f"Moving to waypoint {self.path_index + 1}/{len(self.current_path)}, "
                           f"distance: {distance_to_target:.2f}m")

    def detect_obstacles(self) -> bool:
        """Detect obstacles in front of robot"""
        if not self.current_scan:
            return False

        # Check forward-facing range (simplified)
        forward_ranges = self.current_scan['ranges'][720:1080]  # Assuming 1440 points for 360 deg
        min_range = np.min(forward_ranges[np.isfinite(forward_ranges)]) if len(forward_ranges) > 0 else float('inf')

        return min_range < self.safe_distance

    def stop_robot(self):
        """Stop the robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

    def publish_status(self, status: str):
        """Publish navigation status"""
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)

    nav_stack = IsaacNavigationStack()

    try:
        rclpy.spin(nav_stack)
    except KeyboardInterrupt:
        pass
    finally:
        nav_stack.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Implementation Steps

1. Set up Isaac Sim environment for navigation simulation
2. Implement perception pipeline using Isaac ROS packages
3. Create SLAM system for mapping and localization
4. Develop path planning algorithms with GPU acceleration
5. Integrate complete navigation stack
6. Test navigation in simulated environment

## Expected Outcomes

After completing this lab, you should have:
1. A working Isaac Sim environment with navigation scenarios
2. A perception pipeline that processes sensor data for navigation
3. A SLAM system that builds maps and localizes the robot
4. Path planning algorithms that compute optimal routes
5. A complete navigation stack that integrates all components
6. Evaluation metrics for navigation performance

## Troubleshooting Tips

- Ensure Isaac Sim is properly installed with all dependencies
- Verify GPU compatibility and CUDA installation
- Check ROS 2 network configuration between simulation and perception nodes
- Monitor memory usage during SLAM operations
- Validate sensor data quality and calibration

## Further Exploration

- Implement more advanced SLAM algorithms (Cartographer, ORB-SLAM)
- Add semantic mapping capabilities using Isaac ROS packages
- Integrate reinforcement learning for adaptive navigation
- Deploy on actual NVIDIA Jetson hardware
- Add multi-robot coordination capabilities