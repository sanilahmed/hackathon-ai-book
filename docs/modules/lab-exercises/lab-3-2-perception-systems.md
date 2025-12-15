---
sidebar_label: 'Lab 3.2: Perception Systems'
---

# Lab Exercise 3.2: Perception Systems in AI-Robot Brain

This lab exercise covers implementing perception systems for robot AI using NVIDIA Isaac.

## Objectives

- Set up perception sensors in simulation
- Implement computer vision pipelines
- Integrate perception with ROS 2
- Test perception accuracy and performance

## Prerequisites

- Isaac Sim installed and configured
- ROS 2 Humble with perception packages
- Basic Python and OpenCV knowledge

## Perception System Overview

### Key Components

A perception system typically includes:
- **Cameras**: RGB, depth, stereo
- **LIDAR**: 2D and 3D scanning
- **IMU**: Inertial measurement
- **Other sensors**: Force/torque, tactile, etc.

### Perception Pipeline

```
Raw Sensor Data → Preprocessing → Feature Extraction → Object Detection → Decision Making
```

## Sensor Setup in Isaac Sim

### Camera Configuration

```python
from omni.isaac.core.utils.prims import get_prim_at_path, define_prim
from omni.isaac.sensor import Camera
from pxr import Gf
import numpy as np

class CameraSetup:
    def __init__(self, prim_path, position, orientation):
        self.prim_path = prim_path
        self.position = position
        self.orientation = orientation

        # Create camera prim
        define_prim(prim_path, "Camera")
        self.camera = Camera(
            prim_path=prim_path,
            frequency=30,  # Hz
            resolution=(640, 480)
        )

        # Set camera properties
        self.camera.set_position(position)
        self.camera.set_orientation(orientation)

    def get_rgb_image(self):
        return self.camera.get_rgb()

    def get_depth_image(self):
        return self.camera.get_depth()

    def get_point_cloud(self):
        return self.camera.get_point_cloud()
```

### LIDAR Configuration

```python
from omni.isaac.range_sensor import LidarRtx
from pxr import Gf

class LidarSetup:
    def __init__(self, prim_path, position, orientation):
        self.lidar = LidarRtx(
            prim_path=prim_path,
            translation=position,
            orientation=orientation,
            config="Example_Rotary",
            rotation_frequency=20,
            horizontal_samples=1080,
            vertical_samples=32,
            horizontal_fov=360,
            vertical_fov=45
        )

        # Enable different types of data
        self.lidar.add_ground_truth_to_frame()
        self.lidar.add_position_channels_to_frame()

    def get_lidar_data(self):
        return self.lidar.get_sensor_reading()

    def get_point_cloud(self):
        return self.lidar.get_point_cloud()
```

## Isaac ROS Perception Integration

### Isaac ROS Sensor Bridge

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, LaserScan
from cv_bridge import CvBridge
import numpy as np

class IsaacPerceptionBridge(Node):
    def __init__(self):
        super().__init__('isaac_perception_bridge')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Publishers for different sensor types
        self.rgb_pub = self.create_publisher(Image, '/camera/rgb/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/rgb/camera_info', 10)
        self.lidar_pub = self.create_publisher(LaserScan, '/scan', 10)

        # Timer for publishing sensor data
        self.timer = self.create_timer(0.033, self.publish_sensor_data)  # ~30Hz

        # Isaac Sim sensor references
        self.isaac_camera = None
        self.isaac_lidar = None

    def publish_sensor_data(self):
        # Publish RGB image
        if self.isaac_camera:
            rgb_image = self.isaac_camera.get_rgb()
            if rgb_image is not None:
                ros_image = self.cv_bridge.cv2_to_imgmsg(rgb_image, encoding="bgr8")
                ros_image.header.stamp = self.get_clock().now().to_msg()
                ros_image.header.frame_id = "camera_rgb_optical_frame"
                self.rgb_pub.publish(ros_image)

        # Publish depth image
        if self.isaac_camera:
            depth_image = self.isaac_camera.get_depth()
            if depth_image is not None:
                ros_depth = self.cv_bridge.cv2_to_imgmsg(depth_image, encoding="32FC1")
                ros_depth.header.stamp = self.get_clock().now().to_msg()
                ros_depth.header.frame_id = "camera_depth_optical_frame"
                self.depth_pub.publish(ros_depth)

        # Publish LIDAR scan
        if self.isaac_lidar:
            lidar_data = self.isaac_lidar.get_lidar_data()
            if lidar_data:
                scan_msg = self.create_laser_scan_msg(lidar_data)
                self.lidar_pub.publish(scan_msg)

    def create_laser_scan_msg(self, lidar_data):
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = "laser_frame"
        scan.angle_min = -np.pi
        scan.angle_max = np.pi
        scan.angle_increment = 2 * np.pi / len(lidar_data)
        scan.time_increment = 0.0
        scan.scan_time = 0.1
        scan.range_min = 0.1
        scan.range_max = 50.0
        scan.ranges = lidar_data
        return scan
```

## Computer Vision Processing

### Object Detection Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        self.cv_bridge = CvBridge()

        # Subscribe to camera image
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10
        )

        # Publish detection results
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/object_detections', 10
        )

        # Initialize detection model (using OpenCV DNN as example)
        # In practice, you'd use a trained model like YOLO or SSD
        self.detector = self.initialize_detector()

    def initialize_detector(self):
        # This would initialize your object detection model
        # For example, using OpenCV's DNN module with a pre-trained model
        return None  # Placeholder

    def image_callback(self, msg):
        # Convert ROS image to OpenCV
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Perform object detection
        detections = self.detect_objects(cv_image)

        # Publish results
        detection_msg = self.create_detection_message(detections, msg.header)
        self.detection_pub.publish(detection_msg)

    def detect_objects(self, image):
        # Perform object detection on the image
        # This is a simplified example - in practice, you'd use a trained model
        detections = []

        # Example: detect colored objects
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for detection
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red, upper_red)

        # Find contours
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'class': 'red_object',
                    'confidence': 0.8,
                    'bbox': (x, y, w, h)
                })

        return detections

    def create_detection_message(self, detections, header):
        detection_array = Detection2DArray()
        detection_array.header = header

        for det in detections:
            detection = Detection2D()
            detection.header = header

            # Set bounding box
            bbox = detection.bbox
            bbox.center.x = det['bbox'][0] + det['bbox'][2] / 2
            bbox.center.y = det['bbox'][1] + det['bbox'][3] / 2
            bbox.size_x = det['bbox'][2]
            bbox.size_y = det['bbox'][3]

            # Set hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = det['class']
            hypothesis.hypothesis.score = det['confidence']
            detection.results.append(hypothesis)

            detection_array.detections.append(detection)

        return detection_array
```

## NVIDIA Isaac Perception Tools

### Isaac ROS Perception Packages

```python
# Example of using Isaac ROS perception packages
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from stereo_msgs.msg import DisparityImage
from message_filters import ApproximateTimeSynchronizer, Subscriber

class IsaacPerceptionPipeline(Node):
    def __init__(self):
        super().__init__('isaac_perception_pipeline')

        # Subscribe to stereo pair
        left_sub = Subscriber(self, Image, '/camera/left/image_rect_color')
        right_sub = Subscriber(self, Image, '/camera/right/image_rect_color')

        # Synchronize stereo images
        ts = ApproximateTimeSynchronizer(
            [left_sub, right_sub],
            queue_size=10,
            slop=0.1
        )
        ts.registerCallback(self.stereo_callback)

        # Publishers
        self.disparity_pub = self.create_publisher(DisparityImage, '/disparity', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/points2', 10)

    def stereo_callback(self, left_msg, right_msg):
        # Process stereo images to generate disparity and point cloud
        # This would use Isaac's optimized stereo processing
        pass
```

## 3D Perception and Point Cloud Processing

### Point Cloud Processing Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2

class PointCloudProcessor(Node):
    def __init__(self):
        super().__init__('point_cloud_processor')

        # Subscribe to point cloud
        self.pc_sub = self.create_subscription(
            PointCloud2, '/points2', self.pc_callback, 10
        )

        # Publisher for processed point cloud
        self.processed_pc_pub = self.create_publisher(
            PointCloud2, '/processed_points', 10
        )

    def pc_callback(self, msg):
        # Convert PointCloud2 to numpy array
        points_list = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points_list.append([point[0], point[1], point[2]])

        points = np.array(points_list)

        # Process point cloud (example: ground plane removal)
        processed_points = self.remove_ground_plane(points)

        # Convert back to PointCloud2 and publish
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = msg.header.frame_id

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]

        processed_msg = pc2.create_cloud(header, fields, processed_points)
        self.processed_pc_pub.publish(processed_msg)

    def remove_ground_plane(self, points):
        # Simple ground plane removal using RANSAC or height threshold
        # This is a simplified example
        ground_height = 0.1  # Adjust based on robot height
        return points[points[:, 2] > ground_height]
```

## Perception Performance Optimization

### GPU-Accelerated Processing

```python
import torch
import torchvision.transforms as T
from PIL import Image as PILImage

class GPUPerceptionProcessor:
    def __init__(self):
        # Check for GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load models to GPU
        self.detection_model = self.load_detection_model().to(self.device)
        self.segmentation_model = self.load_segmentation_model().to(self.device)

        # Set models to evaluation mode
        self.detection_model.eval()
        self.segmentation_model.eval()

    def load_detection_model(self):
        # Load a pre-trained detection model (e.g., YOLO, SSD)
        # This would be your specific model
        return None  # Placeholder

    def load_segmentation_model(self):
        # Load a pre-trained segmentation model
        return None  # Placeholder

    def process_image_gpu(self, image):
        # Convert image to tensor and move to GPU
        transform = T.Compose([T.ToTensor()])
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            detection_results = self.detection_model(image_tensor)
            segmentation_results = self.segmentation_model(image_tensor)

        return detection_results, segmentation_results
```

## Perception Evaluation

### Accuracy Metrics Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Float32
import numpy as np

class PerceptionEvaluator(Node):
    def __init__(self):
        super().__init__('perception_evaluator')

        # Subscribers
        self.detection_sub = self.create_subscription(
            Detection2DArray, '/object_detections', self.detection_callback, 10
        )
        self.ground_truth_sub = self.create_subscription(
            Detection2DArray, '/ground_truth', self.ground_truth_callback, 10
        )

        # Publishers for metrics
        self.precision_pub = self.create_publisher(Float32, '/perception_precision', 10)
        self.recall_pub = self.create_publisher(Float32, '/perception_recall', 10)

        # Storage for evaluation
        self.ground_truth = None
        self.detection_results = None

    def detection_callback(self, msg):
        self.detection_results = msg
        if self.ground_truth is not None:
            self.evaluate_perception()

    def ground_truth_callback(self, msg):
        self.ground_truth = msg

    def evaluate_perception(self):
        # Calculate precision and recall
        # This is a simplified evaluation
        if not self.ground_truth.detections or not self.detection_results.detections:
            return

        # Calculate IoU between detections and ground truth
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        # Simple evaluation logic (in practice, you'd use proper matching algorithms)
        for det in self.detection_results.detections:
            matched = False
            for gt in self.ground_truth.detections:
                if self.calculate_iou(det.bbox, gt.bbox) > 0.5:
                    true_positives += 1
                    matched = True
                    break
            if not matched:
                false_positives += 1

        false_negatives = len(self.ground_truth.detections) - true_positives

        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        # Publish metrics
        precision_msg = Float32()
        precision_msg.data = precision
        self.precision_pub.publish(precision_msg)

        recall_msg = Float32()
        recall_msg.data = recall
        self.recall_pub.publish(recall_msg)

    def calculate_iou(self, bbox1, bbox2):
        # Calculate Intersection over Union
        # Simplified implementation
        return 0.0  # Placeholder
```

## Exercise Tasks

1. Set up camera and LIDAR sensors in Isaac Sim
2. Implement a ROS 2 node to bridge Isaac Sim sensor data
3. Create an object detection node that processes camera images
4. Implement point cloud processing for 3D perception
5. Set up GPU-accelerated perception processing
6. Create an evaluation node to measure perception accuracy

## Troubleshooting

### Common Issues

- **Sensor data not publishing**: Check Isaac Sim extension settings
- **Performance issues**: Monitor GPU and CPU usage
- **Synchronization problems**: Verify time stamps and QoS settings
- **Memory issues**: Process data in batches or reduce resolution

## Summary

In this lab, you learned to set up and implement perception systems for robot AI. You configured sensors in Isaac Sim, created ROS 2 bridges, implemented computer vision pipelines, and evaluated perception performance. These skills are essential for building intelligent robotic systems.