# Perception Systems

## Overview

This section covers the implementation of perception systems using NVIDIA Isaac's computer vision capabilities. Perception is a critical component of the AI robot brain, enabling the humanoid robot to understand its environment through sensors like cameras, LiDAR, and IMU.

## Isaac Perception Pipeline

NVIDIA Isaac provides a comprehensive perception pipeline that includes:
- Sensor data processing
- Feature extraction
- Object detection and tracking
- SLAM (Simultaneous Localization and Mapping)
- Semantic segmentation

## Camera-Based Perception

### RGB-D Camera Setup

```python
# Example of setting up RGB-D camera in Isaac Sim
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.sensor import Camera

# Create camera sensor
camera = Camera(
    prim_path="/World/HumanoidRobot/Camera",
    frequency=30,
    resolution=(640, 480)
)

# Configure camera properties
camera.add_motion_vectors_to_frame()
camera.add_depth_to_frame()
camera.add_instance_segmentation_to_frame()

# Enable ROS 2 bridge for camera data
from omni.isaac.ros2_bridge import ROS2Bridge
ROS2Bridge().publish_camera(camera, topic_name="/humanoid/camera")
```

### Isaac ROS Perception Nodes

Isaac ROS provides optimized perception nodes:

```bash
# Image processing pipeline
ros2 launch isaac_ros_image_pipeline image_pipeline.launch.py

# AprilTag detection
ros2 launch isaac_ros_apriltag apriltag.launch.py

# DetectNet for object detection
ros2 launch isaac_ros_detectnet detectnet.launch.py

# Visual SLAM
ros2 launch isaac_ros_visual_slam visual_slam.launch.py
```

## LiDAR Perception

### LiDAR Sensor Configuration

```python
# Configure LiDAR sensor in Isaac Sim
from omni.isaac.range_sensor import LidarRtx

lidar = LidarRtx(
    prim_path="/World/HumanoidRobot/Lidar",
    translation=np.array([0.0, 0.0, 0.5]),
    orientation=rotations.gf_to_omni(gf.RotationTransform().value),
    config="Example_Rotary",
    range_resolution=0.005
)

# Set up LiDAR parameters
lidar.set_max_range(25.0)
lidar.set_horizontal_resolution(0.25)
lidar.set_vertical_resolution(0.4)
lidar.set_horizontal_fov(360)
```

### Point Cloud Processing

Isaac ROS provides optimized point cloud processing:

```python
# Example point cloud processing node
import rclpy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import numpy as np

class PointCloudProcessor:
    def __init__(self):
        self.node = rclpy.create_node('point_cloud_processor')
        self.subscription = self.node.create_subscription(
            PointCloud2,
            '/humanoid/scan',
            self.point_cloud_callback,
            10
        )
        self.publisher = self.node.create_publisher(
            PointCloud2,
            '/processed_point_cloud',
            10
        )

    def point_cloud_callback(self, msg):
        # Process point cloud data
        points = self.point_cloud2_to_array(msg)
        processed_points = self.process_points(points)

        # Publish processed point cloud
        processed_msg = self.array_to_point_cloud2(processed_points, msg.header)
        self.publisher.publish(processed_msg)

    def point_cloud2_to_array(self, cloud_msg):
        # Convert PointCloud2 message to numpy array
        # Implementation details...
        pass
```

## IMU and Sensor Fusion

### IMU Configuration in Isaac Sim

```python
# Configure IMU sensor
from omni.isaac.core.sensors import ImuSensor

imu = ImuSensor(
    prim_path="/World/HumanoidRobot/Imu",
    name="humanoid_imu",
    translation=np.array([0.0, 0.0, 0.8]),  # Mount on torso
    orientation=rotations.gf_to_omni(gf.RotationTransform().value)
)

# Enable ROS 2 bridge for IMU
ROS2Bridge().publish_imu(imu, topic_name="/humanoid/imu")
```

### Sensor Fusion with Isaac

```python
# Example sensor fusion using Isaac's capabilities
import numpy as np
from scipy.spatial.transform import Rotation as R

class SensorFusion:
    def __init__(self):
        self.orientation = R.from_quat([0, 0, 0, 1])
        self.position = np.array([0, 0, 0])
        self.velocity = np.array([0, 0, 0])

    def fuse_imu_camera(self, imu_data, camera_pose):
        # Fuse IMU data with camera pose estimation
        # Use complementary filter or Kalman filter
        imu_rotation = R.from_quat(imu_data.orientation)

        # Combine with camera-based pose estimation
        fused_rotation = self.complementary_filter(
            imu_rotation,
            camera_pose.rotation,
            alpha=0.9
        )

        return fused_rotation.as_quat()

    def complementary_filter(self, imu_rot, camera_rot, alpha):
        # Implement complementary filter for sensor fusion
        # Higher alpha gives more weight to IMU (more stable)
        # Lower alpha gives more weight to camera (more accurate)
        return R.slerp(imu_rot, camera_rot, 1 - alpha)
```

## Object Detection and Recognition

### Isaac DetectNet Integration

```bash
# Launch DetectNet for object detection
ros2 launch isaac_ros_detectnet detectnet.launch.py \
    model_name="detectnet_coco" \
    engine_file_path="/path/to/trt_engine.plan"
```

### Custom Object Detection Node

```python
# Custom object detection node using Isaac ROS
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from isaac_ros_detectnet_interfaces.msg import Detection2DArray
from vision_msgs.msg import Detection2D

class IsaacObjectDetector(Node):
    def __init__(self):
        super().__init__('isaac_object_detector')

        # Create subscribers and publishers
        self.image_sub = self.create_subscription(
            Image,
            '/humanoid/camera/image_raw',
            self.image_callback,
            10
        )

        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/object_detections',
            10
        )

    def image_callback(self, msg):
        # Process image through DetectNet
        # This would typically interface with TensorRT model
        detections = self.run_detection(msg)

        # Publish detections
        detection_msg = self.create_detection_message(detections, msg.header)
        self.detection_pub.publish(detection_msg)
```

## SLAM Implementation

### Visual SLAM with Isaac

```bash
# Launch Isaac Visual SLAM
ros2 launch isaac_ros_visual_slam visual_slam.launch.py \
    input_image_topic="/humanoid/camera/image_raw" \
    input_camera_info_topic="/humanoid/camera/camera_info" \
    map_frame="map" \
    odom_frame="odom" \
    base_frame="base_link"
```

### Occupancy Grid Mapping

```python
# Example occupancy grid mapping
from nav_msgs.msg import OccupancyGrid
import numpy as np

class OccupancyGridMapper:
    def __init__(self):
        self.resolution = 0.05  # 5cm per cell
        self.width = 400  # 20m x 20m map
        self.height = 400
        self.origin = [-10.0, -10.0, 0.0]  # Map origin

        # Initialize occupancy grid
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)

    def update_grid(self, laser_scan, robot_pose):
        # Update occupancy grid based on laser scan
        # Implementation of ray casting algorithm
        for i, range_reading in enumerate(laser_scan.ranges):
            if range_reading < laser_scan.range_min or range_reading > laser_scan.range_max:
                continue

            # Calculate point in robot frame
            angle = laser_scan.angle_min + i * laser_scan.angle_increment
            x_local = range_reading * np.cos(angle)
            y_local = range_reading * np.sin(angle)

            # Transform to map frame
            x_map = robot_pose.x + x_local
            y_map = robot_pose.y + y_local

            # Update grid cell
            grid_x = int((x_map - self.origin[0]) / self.resolution)
            grid_y = int((y_map - self.origin[1]) / self.resolution)

            if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                self.grid[grid_y, grid_x] = 100  # Occupied
```

## Semantic Segmentation

### Isaac Semantic Segmentation

```python
# Semantic segmentation using Isaac's segmentation capabilities
from omni.isaac.core.utils.semantics import add_semantic_data_to_stage

# Add semantic labels to objects in the scene
add_semantic_data_to_stage(
    prim_path="/World/Object",
    semantic_label="chair",
    type_label="class"
)

# Enable semantic segmentation on camera
camera.add_semantic_segmentation_to_frame()
```

## Performance Optimization

### TensorRT Integration

```python
# Optimize perception models with TensorRT
import tensorrt as trt
import pycuda.driver as cuda

class TensorRTOptimizer:
    def __init__(self):
        self.logger = trt.Logger(trt.Logger.WARNING)

    def build_engine(self, onnx_model_path):
        # Build TensorRT engine from ONNX model
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )

        parser = trt.OnnxParser(network, self.logger)

        with open(onnx_model_path, 'rb') as model:
            parser.parse(model.read())

        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB

        return builder.build_engine(network, config)
```

## Integration with AI Robot Brain

The perception system feeds processed sensor data to the AI decision-making components:

```python
# Perception data flow to AI brain
class PerceptionToAIInterface:
    def __init__(self):
        # Initialize perception components
        self.camera_processor = CameraProcessor()
        self.lidar_processor = LiDARProcessor()
        self.imu_processor = IMUProcessor()

        # Initialize AI brain interface
        self.ai_brain_input = AIBrainInput()

    def process_sensor_data(self):
        # Collect and process all sensor data
        camera_data = self.camera_processor.get_data()
        lidar_data = self.lidar_processor.get_data()
        imu_data = self.imu_processor.get_data()

        # Fuse sensor data
        fused_data = self.fuse_sensor_data(camera_data, lidar_data, imu_data)

        # Send to AI brain
        self.ai_brain_input.update_sensor_data(fused_data)

    def fuse_sensor_data(self, camera, lidar, imu):
        # Implement sensor fusion logic
        return {
            'environment_map': self.create_environment_map(lidar),
            'object_detections': camera['detections'],
            'robot_pose': self.estimate_pose(imu, camera)
        }
```

## Testing Perception Systems

### Perception Test Suite

```bash
# Run perception tests
ros2 launch isaac_ros_apriltag apriltag_test.launch.py
ros2 launch isaac_ros_visual_slam visual_slam_test.launch.py
ros2 launch isaac_ros_detectnet detectnet_test.launch.py
```

### Performance Benchmarks

```python
# Perception performance metrics
class PerceptionMetrics:
    def __init__(self):
        self.frame_rate = 0
        self.latency = 0
        self.detection_accuracy = 0
        self.tracking_precision = 0

    def calculate_fps(self, start_time, end_time, num_frames):
        elapsed = end_time - start_time
        return num_frames / elapsed

    def evaluate_detection_accuracy(self, predictions, ground_truth):
        # Calculate precision, recall, mAP
        pass
```

## Troubleshooting

### Common Perception Issues

1. **Low Detection Accuracy**
   - Verify sensor calibration
   - Check lighting conditions in simulation
   - Retrain models with domain randomization

2. **High Latency**
   - Optimize TensorRT models
   - Reduce sensor resolution temporarily
   - Use multi-threading for parallel processing

3. **False Positives**
   - Adjust detection thresholds
   - Implement temporal filtering
   - Use multiple sensor verification

---
[Next: Planning and Control](./planning-control.md) | [Previous: Isaac Sim Setup](./isaac-sim-setup.md)