# Lab 3.2: Isaac Sim Perception Systems

## Overview

In this lab, you will learn how to implement perception systems in Isaac Sim for robotics applications. You'll work with various sensors including cameras, LiDAR, IMU, and other perception sensors, and learn how to process and integrate sensor data for robotics applications. This includes understanding sensor models, configuring sensor parameters, and integrating with ROS.

## Objectives

By the end of this lab, you will be able to:
- Configure and use different types of sensors in Isaac Sim
- Process camera images and point cloud data
- Integrate perception data with ROS topics
- Implement basic computer vision algorithms
- Create perception pipelines for robotics
- Validate sensor data quality and accuracy

## Prerequisites

- Completion of Lab 3.1: Isaac Sim Setup and Environment
- Understanding of ROS 2 topics and messages
- Basic knowledge of computer vision concepts
- Experience with Isaac Sim basics

## Duration

4-5 hours

## Exercise 1: Camera Sensor Setup and Configuration

### Step 1: Create a basic camera setup

Create `~/isaac_sim_examples/camera_sensor.py`:

```python
#!/usr/bin/env python3
# camera_sensor.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.semantics import add_semantic_data_to_stage
import numpy as np
import cv2
import carb

# Initialize the world
world = World(stage_units_in_meters=1.0)

# Create a simple environment
ground_plane = create_primitive(
    prim_path="/World/GroundPlane",
    primitive_type="Plane",
    scale=[10, 10, 1],
    position=[0, 0, 0]
)

# Create objects with different materials
red_cube = create_primitive(
    prim_path="/World/RedCube",
    primitive_type="Cube",
    scale=[0.5, 0.5, 0.5],
    position=[1, 0, 0.25],
    color=[1, 0, 0]
)

blue_cube = create_primitive(
    prim_path="/World/BlueCube",
    primitive_type="Cube",
    scale=[0.5, 0.5, 0.5],
    position=[0, 1, 0.25],
    color=[0, 0, 1]
)

green_cube = create_primitive(
    prim_path="/World/GreenCube",
    primitive_type="Cube",
    scale=[0.5, 0.5, 0.5],
    position=[1, 1, 0.25],
    color=[0, 1, 0]
)

# Create a robot with camera
robot = create_primitive(
    prim_path="/World/Robot",
    primitive_type="Cylinder",
    scale=[0.3, 0.3, 0.5],
    position=[0, 0, 0.25]
)

# Create camera sensor
camera = Camera(
    prim_path="/World/Robot/Camera",
    frequency=30,
    resolution=(640, 480)
)

# Set camera position relative to robot
camera.set_local_pose(translation=np.array([0.3, 0, 0.2]))

# Set camera view for visualization
set_camera_view(eye=[3, 3, 3], target=[0, 0, 0])

# Play the simulation
world.reset()

# Initialize image data collection
image_count = 0

for i in range(500):
    world.step(render=True)

    # Capture and process images every 30 steps (1Hz)
    if i % 30 == 0:
        # Get RGB image
        rgb_image = camera.get_rgb()

        if rgb_image is not None:
            print(f"Captured image {image_count}: shape={rgb_image.shape}, dtype={rgb_image.dtype}")

            # Process the image (example: convert to OpenCV format)
            # Note: Isaac Sim uses different coordinate systems
            processed_image = np.flip(rgb_image, axis=0)  # Flip vertically

            # Save image for inspection
            import os
            os.makedirs("~/isaac_sim_outputs", exist_ok=True)
            output_path = f"~/isaac_sim_outputs/camera_image_{image_count:03d}.png"
            # Note: In practice, you'd use cv2.imwrite or similar
            print(f"Image would be saved to: {output_path}")

            image_count += 1

# Stop the simulation
world.stop()
print("Camera sensor simulation completed")
```

### Step 2: Advanced camera configuration

Create `~/isaac_sim_examples/advanced_camera.py`:

```python
#!/usr/bin/env python3
# advanced_camera.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.sensor import Camera
from omni.isaac.range_sensor import create_lidar
import numpy as np
import carb

class AdvancedCameraSystem:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.cameras = []
        self.setup_environment()

    def setup_environment(self):
        """Set up the environment with multiple objects."""
        # Create ground plane
        create_primitive(
            prim_path="/World/GroundPlane",
            primitive_type="Plane",
            scale=[10, 10, 1],
            position=[0, 0, 0]
        )

        # Create various objects for perception testing
        for i in range(10):
            position = [np.random.uniform(-3, 3), np.random.uniform(-3, 3), 0.5]
            create_primitive(
                prim_path=f"/World/Object_{i}",
                primitive_type="Cylinder",
                scale=[0.2, 0.2, 1.0],
                position=position,
                color=[np.random.rand(), np.random.rand(), np.random.rand()]
            )

        # Create robot
        self.robot = create_primitive(
            prim_path="/World/Robot",
            primitive_type="Cylinder",
            scale=[0.4, 0.4, 0.8],
            position=[0, 0, 0.4]
        )

    def add_camera_system(self):
        """Add multiple cameras with different configurations."""
        # Front-facing RGB camera
        front_camera = Camera(
            prim_path="/World/Robot/FrontCamera",
            frequency=30,
            resolution=(1280, 720),
            position=[0.3, 0, 0.2]
        )
        front_camera.set_local_pose(translation=np.array([0.3, 0, 0.2]))
        self.cameras.append(front_camera)

        # Depth camera
        depth_camera = Camera(
            prim_path="/World/Robot/DepthCamera",
            frequency=30,
            resolution=(640, 480),
            position=[0.3, 0, 0.2]
        )
        depth_camera.set_local_pose(translation=np.array([0.3, 0, 0.2]))

        # Enable depth data
        depth_camera.add_depth_to_frame()
        depth_camera.add_instance_segmentation_to_frame()
        self.cameras.append(depth_camera)

        # Segmentation camera
        seg_camera = Camera(
            prim_path="/World/Robot/SegCamera",
            frequency=15,
            resolution=(640, 480),
            position=[0.3, 0, 0.2]
        )
        seg_camera.set_local_pose(translation=np.array([0.3, 0, 0.2]))
        seg_camera.add_instance_segmentation_to_frame()
        self.cameras.append(seg_camera)

    def process_camera_data(self):
        """Process data from all cameras."""
        for i, camera in enumerate(self.cameras):
            # Get different types of data based on camera configuration
            if i == 0:  # RGB camera
                rgb_image = camera.get_rgb()
                if rgb_image is not None:
                    print(f"RGB Camera {i}: Captured image with shape {rgb_image.shape}")

            elif i == 1:  # Depth + RGB camera
                rgb_image = camera.get_rgb()
                depth_image = camera.get_depth()

                if rgb_image is not None:
                    print(f"Depth Camera {i}: RGB shape {rgb_image.shape}")
                if depth_image is not None:
                    print(f"Depth Camera {i}: Depth shape {depth_image.shape}")

            elif i == 2:  # Segmentation camera
                seg_image = camera.get_semantic_segmentation()
                if seg_image is not None:
                    print(f"Segmentation Camera {i}: Segmentation shape {seg_image.shape}")

    def run_simulation(self, steps=1000):
        """Run the simulation with camera data processing."""
        self.world.reset()
        self.add_camera_system()

        # Set camera view
        set_camera_view(eye=[5, 5, 5], target=[0, 0, 0])

        for i in range(steps):
            self.world.step(render=True)

            # Process camera data every 60 steps (0.5Hz)
            if i % 60 == 0:
                self.process_camera_data()

        print(f"Advanced camera system simulation completed with {steps} steps")

# Create and run the advanced camera system
if __name__ == "__main__":
    camera_system = AdvancedCameraSystem()
    camera_system.run_simulation(steps=600)
```

## Exercise 2: LiDAR Sensor Setup and Processing

### Step 1: Create LiDAR sensor configuration

Create `~/isaac_sim_examples/lidar_sensor.py`:

```python
#!/usr/bin/env python3
# lidar_sensor.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.range_sensor import LidarRtx
import numpy as np
import carb

class LidarSensorSystem:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.lidar = None
        self.setup_environment()

    def setup_environment(self):
        """Set up environment with obstacles for LiDAR testing."""
        # Create ground plane
        create_primitive(
            prim_path="/World/GroundPlane",
            primitive_type="Plane",
            scale=[20, 20, 1],
            position=[0, 0, 0]
        )

        # Create walls
        wall_thickness = 0.1
        wall_height = 2.0

        # North wall
        create_primitive(
            prim_path="/World/Wall_North",
            primitive_type="Cuboid",
            scale=[20, wall_thickness, wall_height],
            position=[0, 10, wall_height/2]
        )

        # South wall
        create_primitive(
            prim_path="/World/Wall_South",
            primitive_type="Cuboid",
            scale=[20, wall_thickness, wall_height],
            position=[0, -10, wall_height/2]
        )

        # East wall
        create_primitive(
            prim_path="/World/Wall_East",
            primitive_type="Cuboid",
            scale=[wall_thickness, 20, wall_height],
            position=[10, 0, wall_height/2]
        )

        # West wall
        create_primitive(
            prim_path="/World/Wall_West",
            primitive_type="Cuboid",
            scale=[wall_thickness, 20, wall_height],
            position=[-10, 0, wall_height/2]
        )

        # Create obstacles inside the area
        for i in range(15):
            position = [np.random.uniform(-8, 8), np.random.uniform(-8, 8), 0.5]
            if np.linalg.norm(position[:2]) > 1.0:  # Keep center clear
                create_primitive(
                    prim_path=f"/World/Obstacle_{i}",
                    primitive_type="Cylinder",
                    scale=[0.3, 0.3, 1.0],
                    position=position
                )

        # Create robot with LiDAR
        self.robot = create_primitive(
            prim_path="/World/Robot",
            primitive_type="Cylinder",
            scale=[0.5, 0.5, 1.0],
            position=[0, 0, 0.5]
        )

    def add_lidar(self):
        """Add LiDAR sensor to the robot."""
        self.lidar = LidarRtx(
            prim_path="/World/Robot/Lidar",
            translation=(0.0, 0.0, 0.8),  # Mount on top of robot
            config="Example_Rotary",
            range_resolution=0.005,  # 5mm resolution
            rotation_frequency=10,   # 10 Hz rotation
            horizontal_resolution=0.25,  # 0.25 degree horizontal resolution
            vertical_resolution=0.4,     # 0.4 degree vertical resolution
            horizontal_samples=1080,     # Samples per revolution
            vertical_samples=64,         # Vertical beams
            max_range=25.0,              # Maximum range 25m
            min_range=0.1                # Minimum range 10cm
        )

    def process_lidar_data(self):
        """Process LiDAR scan data."""
        try:
            # Get LiDAR data
            lidar_data = self.lidar.get_linear_depth_data()

            if lidar_data is not None:
                # Calculate statistics
                valid_points = lidar_data[lidar_data > 0]  # Filter out invalid ranges

                if len(valid_points) > 0:
                    avg_distance = np.mean(valid_points)
                    min_distance = np.min(valid_points)
                    max_distance = np.max(valid_points)

                    print(f"LiDAR Scan: Points={len(valid_points)}, "
                          f"Avg={avg_distance:.2f}m, Min={min_distance:.2f}m, Max={max_distance:.2f}m")

                    # Calculate hit rate (percentage of valid measurements)
                    total_points = lidar_data.size
                    hit_rate = len(valid_points) / total_points if total_points > 0 else 0
                    print(f"Hit Rate: {hit_rate:.2%}")
                else:
                    print("LiDAR Scan: No valid points detected")
            else:
                print("LiDAR Scan: No data available")

        except Exception as e:
            print(f"Error processing LiDAR data: {e}")

    def run_simulation(self, steps=1000):
        """Run simulation with LiDAR data processing."""
        self.world.reset()
        self.add_lidar()

        # Set camera view
        set_camera_view(eye=[15, 15, 15], target=[0, 0, 0])

        for i in range(steps):
            self.world.step(render=True)

            # Process LiDAR data every 30 steps (1Hz)
            if i % 30 == 0:
                self.process_lidar_data()

        print(f"LiDAR sensor simulation completed with {steps} steps")

# Create and run the LiDAR system
if __name__ == "__main__":
    lidar_system = LidarSensorSystem()
    lidar_system.run_simulation(steps=600)
```

### Step 2: Multi-sensor fusion system

Create `~/isaac_sim_examples/multi_sensor_fusion.py`:

```python
#!/usr/bin/env python3
# multi_sensor_fusion.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.sensor import Camera, ImuSensor
from omni.isaac.range_sensor import LidarRtx
import numpy as np
import carb

class MultiSensorFusion:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.sensors = {}
        self.setup_environment()

    def setup_environment(self):
        """Set up environment with multiple sensor targets."""
        # Create ground plane
        create_primitive(
            prim_path="/World/GroundPlane",
            primitive_type="Plane",
            scale=[20, 20, 1],
            position=[0, 0, 0]
        )

        # Create landmarks for localization
        landmark_positions = [
            [5, 5, 0.5], [-5, 5, 0.5], [-5, -5, 0.5], [5, -5, 0.5],  # Corners
            [0, 0, 0.5], [0, 5, 0.5], [5, 0, 0.5], [-5, 0, 0.5], [0, -5, 0.5]  # Center and sides
        ]

        for i, pos in enumerate(landmark_positions):
            create_primitive(
                prim_path=f"/World/Landmark_{i}",
                primitive_type="Cylinder",
                scale=[0.2, 0.2, 1.0],
                position=pos,
                color=[1, 0, 0]  # Red landmarks
            )

        # Create robot
        self.robot = create_primitive(
            prim_path="/World/Robot",
            primitive_type="Cylinder",
            scale=[0.5, 0.5, 1.0],
            position=[0, 0, 0.5]
        )

    def add_sensors(self):
        """Add multiple types of sensors to the robot."""
        # Camera sensor
        camera = Camera(
            prim_path="/World/Robot/Camera",
            frequency=30,
            resolution=(640, 480),
            position=[0.3, 0, 0.5]
        )
        camera.set_local_pose(translation=np.array([0.3, 0, 0.5]))
        camera.add_depth_to_frame()
        self.sensors['camera'] = camera

        # LiDAR sensor
        lidar = LidarRtx(
            prim_path="/World/Robot/Lidar",
            translation=(0.0, 0.0, 0.8),
            config="Example_Rotary",
            range_resolution=0.005,
            rotation_frequency=10,
            horizontal_resolution=0.25,
            vertical_resolution=0.4,
            horizontal_samples=1080,
            vertical_samples=64,
            max_range=25.0,
            min_range=0.1
        )
        self.sensors['lidar'] = lidar

        # IMU sensor
        imu = ImuSensor(
            prim_path="/World/Robot/Imu",
            name="robot_imu",
            translation=np.array([0.0, 0.0, 0.5]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )
        self.sensors['imu'] = imu

    def process_sensor_fusion(self):
        """Process and fuse data from multiple sensors."""
        # Get camera data
        if 'camera' in self.sensors:
            rgb_image = self.sensors['camera'].get_rgb()
            depth_image = self.sensors['camera'].get_depth()

            if rgb_image is not None:
                print(f"Camera: RGB image shape {rgb_image.shape}")
            if depth_image is not None:
                print(f"Camera: Depth image shape {depth_image.shape}")

        # Get LiDAR data
        if 'lidar' in self.sensors:
            lidar_data = self.sensors['lidar'].get_linear_depth_data()
            if lidar_data is not None:
                valid_points = lidar_data[lidar_data > 0]
                if len(valid_points) > 0:
                    print(f"LiDAR: {len(valid_points)} valid points, avg range {np.mean(valid_points):.2f}m")

        # Get IMU data
        if 'imu' in self.sensors:
            # IMU data processing would typically happen here
            print("IMU: Data available (processing simulated)")

        # Sensor fusion logic would go here
        # For example: combine camera landmarks with LiDAR points for better localization
        print("Multi-sensor fusion: Processing complete")

    def run_simulation(self, steps=1000):
        """Run simulation with multi-sensor fusion."""
        self.world.reset()
        self.add_sensors()

        # Set camera view
        set_camera_view(eye=[10, 10, 10], target=[0, 0, 0])

        for i in range(steps):
            self.world.step(render=True)

            # Process sensor fusion every 60 steps (0.5Hz)
            if i % 60 == 0:
                self.process_sensor_fusion()

        print(f"Multi-sensor fusion simulation completed with {steps} steps")

# Create and run the multi-sensor system
if __name__ == "__main__":
    fusion_system = MultiSensorFusion()
    fusion_system.run_simulation(steps=600)
```

## Exercise 3: ROS Integration for Perception

### Step 1: Create ROS bridge configuration for sensors

Create `~/isaac_sim_examples/ros_perception_bridge.py`:

```python
#!/usr/bin/env python3
# ros_perception_bridge.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.sensor import Camera
from omni.isaac.range_sensor import LidarRtx
import omni.isaac.ros2_bridge._ros2_bridge as ros2_bridge
import numpy as np
import carb

class ROSPerceptionBridge:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.ros2_bridge = ros2_bridge.acquire_ros2_bridge_interface()
        self.sensors = {}
        self.setup_environment()

    def setup_environment(self):
        """Set up environment for ROS perception testing."""
        # Create ground plane
        create_primitive(
            prim_path="/World/GroundPlane",
            primitive_type="Plane",
            scale=[20, 20, 1],
            position=[0, 0, 0]
        )

        # Create test objects
        for i in range(5):
            position = [np.random.uniform(-8, 8), np.random.uniform(-8, 8), 0.5]
            create_primitive(
                prim_path=f"/World/TestObject_{i}",
                primitive_type="Cylinder",
                scale=[0.3, 0.3, 1.0],
                position=position
            )

        # Create robot
        self.robot = create_primitive(
            prim_path="/World/Robot",
            primitive_type="Cylinder",
            scale=[0.5, 0.5, 1.0],
            position=[0, 0, 0.5]
        )

    def add_ros_sensors(self):
        """Add sensors with ROS bridge integration."""
        # Camera with ROS bridge
        camera = Camera(
            prim_path="/World/Robot/Camera",
            frequency=30,
            resolution=(640, 480),
            position=[0.3, 0, 0.5]
        )
        camera.set_local_pose(translation=np.array([0.3, 0, 0.5]))
        camera.add_depth_to_frame()
        self.sensors['camera'] = camera

        # LiDAR with ROS bridge
        lidar = LidarRtx(
            prim_path="/World/Robot/Lidar",
            translation=(0.0, 0.0, 0.8),
            config="Example_Rotary",
            range_resolution=0.005,
            rotation_frequency=10,
            horizontal_resolution=0.25,
            vertical_resolution=0.4,
            horizontal_samples=1080,
            vertical_samples=64,
            max_range=25.0,
            min_range=0.1
        )
        self.sensors['lidar'] = lidar

        # Enable ROS bridge for sensors
        self.enable_ros_bridge()

    def enable_ros_bridge(self):
        """Enable ROS bridge for all sensors."""
        # Publish camera data to ROS
        if 'camera' in self.sensors:
            self.ros2_bridge.publish_camera(
                self.sensors['camera'],
                topic_name="/humanoid/camera/image_raw",
                sensor_name="camera"
            )

            # Also publish depth
            self.ros2_bridge.publish_depth(
                self.sensors['camera'],
                topic_name="/humanoid/camera/depth",
                sensor_name="depth_camera"
            )

        # Publish LiDAR data to ROS
        if 'lidar' in self.sensors:
            self.ros2_bridge.publish_lidar(
                self.sensors['lidar'],
                topic_name="/humanoid/scan",
                sensor_name="lidar"
            )

    def run_simulation(self, steps=1000):
        """Run simulation with ROS perception bridge."""
        self.world.reset()
        self.add_ros_sensors()

        # Set camera view
        set_camera_view(eye=[10, 10, 10], target=[0, 0, 0])

        for i in range(steps):
            self.world.step(render=True)

            # Periodic status update
            if i % 100 == 0:
                print(f"ROS Perception Bridge: Simulation step {i}/{steps}")

        print(f"ROS perception bridge simulation completed with {steps} steps")

# Create and run the ROS perception bridge
if __name__ == "__main__":
    ros_bridge = ROSPerceptionBridge()
    ros_bridge.run_simulation(steps=600)
```

## Exercise 4: Computer Vision Algorithms Integration

### Step 1: Create a perception pipeline with computer vision

Create `~/isaac_sim_examples/computer_vision_pipeline.py`:

```python
#!/usr/bin/env python3
# computer_vision_pipeline.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.sensor import Camera
import numpy as np
import cv2
import carb

class ComputerVisionPipeline:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.camera = None
        self.setup_environment()

    def setup_environment(self):
        """Set up environment with objects for computer vision."""
        # Create ground plane
        create_primitive(
            prim_path="/World/GroundPlane",
            primitive_type="Plane",
            scale=[15, 15, 1],
            position=[0, 0, 0]
        )

        # Create objects with different shapes and colors
        # Red circles
        for i in range(3):
            create_primitive(
                prim_path=f"/World/RedCircle_{i}",
                primitive_type="Cylinder",
                scale=[0.3, 0.3, 0.5],
                position=[np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0.25],
                color=[1, 0, 0]
            )

        # Blue squares
        for i in range(3):
            create_primitive(
                prim_path=f"/World/BlueSquare_{i}",
                primitive_type="Cube",
                scale=[0.3, 0.3, 0.5],
                position=[np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0.25],
                color=[0, 0, 1]
            )

        # Green triangles (approximated with cones)
        for i in range(3):
            create_primitive(
                prim_path=f"/World/GreenTriangle_{i}",
                primitive_type="Cone",
                scale=[0.3, 0.3, 0.5],
                position=[np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0.25],
                color=[0, 1, 0]
            )

        # Create robot with camera
        self.robot = create_primitive(
            prim_path="/World/Robot",
            primitive_type="Cylinder",
            scale=[0.5, 0.5, 1.0],
            position=[0, 0, 0.5]
        )

    def add_camera(self):
        """Add camera to robot."""
        self.camera = Camera(
            prim_path="/World/Robot/Camera",
            frequency=30,
            resolution=(640, 480),
            position=[0.3, 0, 0.5]
        )
        self.camera.set_local_pose(translation=np.array([0.3, 0, 0.5]))

    def object_detection_pipeline(self, image):
        """Process image for object detection."""
        # Convert Isaac Sim image format to OpenCV format
        # Isaac Sim images are typically in RGB format
        rgb_image = np.flip(image, axis=0)  # Flip vertically
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Convert to HSV for color-based segmentation
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        # Define color ranges for different objects
        color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'red2': ([170, 50, 50], [180, 255, 255]),  # Red wraps around in HSV
            'blue': ([100, 50, 50], [130, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255])
        }

        detected_objects = []

        for color_name, (lower, upper) in color_ranges.items():
            # Create mask for the color
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv_image, lower, upper)

            # If it's red, combine both red ranges
            if color_name == 'red':
                mask2 = cv2.inRange(hsv_image, np.array([170, 50, 50]), np.array([180, 255, 255]))
                mask = cv2.bitwise_or(mask, mask2)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Filter by area to remove noise
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate center
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Determine object type based on aspect ratio
                    aspect_ratio = float(w) / h
                    if 0.8 <= aspect_ratio <= 1.2:
                        obj_type = "circle/square"
                    elif aspect_ratio > 1.2:
                        obj_type = "horizontal"
                    else:
                        obj_type = "vertical"

                    detected_objects.append({
                        'type': obj_type,
                        'color': color_name,
                        'center': (center_x, center_y),
                        'bbox': (x, y, w, h),
                        'area': area
                    })

        return detected_objects

    def process_perception_data(self):
        """Process perception data from camera."""
        if self.camera:
            rgb_image = self.camera.get_rgb()

            if rgb_image is not None:
                # Run object detection
                detected_objects = self.object_detection_pipeline(rgb_image)

                print(f"Detected {len(detected_objects)} objects:")
                for i, obj in enumerate(detected_objects):
                    print(f"  {i+1}. {obj['color']} {obj['type']} at {obj['center']}, area={obj['area']:.0f}")

                return detected_objects
            else:
                print("No camera image available")
                return []
        else:
            print("No camera available")
            return []

    def run_simulation(self, steps=1000):
        """Run simulation with computer vision processing."""
        self.world.reset()
        self.add_camera()

        # Set camera view
        set_camera_view(eye=[10, 10, 10], target=[0, 0, 0])

        for i in range(steps):
            self.world.step(render=True)

            # Process perception data every 30 steps (1Hz)
            if i % 30 == 0:
                detected_objects = self.process_perception_data()

        print(f"Computer vision pipeline simulation completed with {steps} steps")

# Create and run the computer vision pipeline
if __name__ == "__main__":
    cv_pipeline = ComputerVisionPipeline()
    cv_pipeline.run_simulation(steps=600)
```

## Exercise 5: Perception Data Validation and Quality Assurance

### Step 1: Create perception quality metrics

Create `~/isaac_sim_examples/perception_quality.py`:

```python
#!/usr/bin/env python3
# perception_quality.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.sensor import Camera
from omni.isaac.range_sensor import LidarRtx
import numpy as np
import carb

class PerceptionQualityAssessment:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.sensors = {}
        self.quality_metrics = {}
        self.setup_environment()

    def setup_environment(self):
        """Set up controlled environment for quality assessment."""
        # Create ground plane
        create_primitive(
            prim_path="/World/GroundPlane",
            primitive_type="Plane",
            scale=[20, 20, 1],
            position=[0, 0, 0]
        )

        # Create standardized test objects at known positions
        self.test_objects = [
            {'name': 'Target_1', 'position': [2, 0, 0.5], 'size': [0.5, 0.5, 1.0], 'type': 'cube'},
            {'name': 'Target_2', 'position': [0, 2, 0.5], 'size': [0.3, 0.3, 0.8], 'type': 'cylinder'},
            {'name': 'Target_3', 'position': [-2, 0, 0.5], 'size': [0.4, 0.4, 0.6], 'type': 'sphere'},
            {'name': 'Target_4', 'position': [0, -2, 0.5], 'size': [0.6, 0.2, 0.4], 'type': 'box'},
        ]

        for obj in self.test_objects:
            create_primitive(
                prim_path=f"/World/{obj['name']}",
                primitive_type=obj['type'].capitalize() if obj['type'] != 'box' else 'Cuboid',
                scale=obj['size'],
                position=obj['position']
            )

        # Create robot with sensors
        self.robot = create_primitive(
            prim_path="/World/Robot",
            primitive_type="Cylinder",
            scale=[0.5, 0.5, 1.0],
            position=[0, 0, 0.5]
        )

    def add_quality_sensors(self):
        """Add sensors configured for quality assessment."""
        # High-resolution camera for detailed perception
        camera = Camera(
            prim_path="/World/Robot/QualityCamera",
            frequency=60,
            resolution=(1280, 720),  # High resolution
            position=[0.3, 0, 0.5]
        )
        camera.set_local_pose(translation=np.array([0.3, 0, 0.5]))
        camera.add_depth_to_frame()
        camera.add_instance_segmentation_to_frame()
        self.sensors['quality_camera'] = camera

        # Precise LiDAR for ground truth comparison
        lidar = LidarRtx(
            prim_path="/World/Robot/PrecisionLidar",
            translation=(0.0, 0.0, 0.8),
            config="Example_Rotary",
            range_resolution=0.001,  # 1mm resolution
            rotation_frequency=20,
            horizontal_resolution=0.1,  # 0.1 degree
            vertical_resolution=0.2,   # 0.2 degree
            horizontal_samples=2160,   # Higher resolution
            vertical_samples=128,
            max_range=20.0,
            min_range=0.05
        )
        self.sensors['precision_lidar'] = lidar

    def assess_camera_quality(self):
        """Assess camera perception quality."""
        camera = self.sensors['quality_camera']

        # Get various data types
        rgb_image = camera.get_rgb()
        depth_image = camera.get_depth()
        segmentation = camera.get_semantic_segmentation()

        metrics = {}

        if rgb_image is not None:
            # Image quality metrics
            height, width, channels = rgb_image.shape
            metrics['image_resolution'] = f"{width}x{height}"
            metrics['image_channels'] = channels

            # Brightness assessment
            brightness = np.mean(rgb_image)
            metrics['brightness'] = brightness

            # Contrast assessment (simplified)
            contrast = np.std(rgb_image)
            metrics['contrast'] = contrast

        if depth_image is not None:
            # Depth quality metrics
            valid_depths = depth_image[depth_image > 0]
            if len(valid_depths) > 0:
                metrics['depth_coverage'] = len(valid_depths) / depth_image.size
                metrics['avg_depth'] = np.mean(valid_depths)
                metrics['depth_range'] = [np.min(valid_depths), np.max(valid_depths)]

        if segmentation is not None:
            # Segmentation quality
            unique_labels = np.unique(segmentation)
            metrics['segmentation_classes'] = len(unique_labels)

        return metrics

    def assess_lidar_quality(self):
        """Assess LiDAR perception quality."""
        lidar = self.sensors['precision_lidar']

        try:
            lidar_data = lidar.get_linear_depth_data()

            if lidar_data is not None:
                # LiDAR quality metrics
                valid_points = lidar_data[lidar_data > 0]

                metrics = {
                    'total_points': lidar_data.size,
                    'valid_points': len(valid_points),
                    'hit_rate': len(valid_points) / lidar_data.size if lidar_data.size > 0 else 0,
                    'avg_range': np.mean(valid_points) if len(valid_points) > 0 else 0,
                    'range_std': np.std(valid_points) if len(valid_points) > 0 else 0
                }

                # Range accuracy assessment
                if len(valid_points) > 0:
                    # In a real system, you'd compare with ground truth
                    # Here we just assess the data quality
                    metrics['range_accuracy'] = 'high' if metrics['hit_rate'] > 0.8 else 'low'

                return metrics
            else:
                return {'error': 'No LiDAR data available'}

        except Exception as e:
            return {'error': f'LiDAR processing error: {str(e)}'}

    def run_quality_assessment(self, steps=500):
        """Run perception quality assessment."""
        self.world.reset()
        self.add_quality_sensors()

        # Set camera view
        set_camera_view(eye=[10, 10, 10], target=[0, 0, 0])

        for i in range(steps):
            self.world.step(render=True)

            # Assess quality every 100 steps
            if i % 100 == 0:
                print(f"\n--- Quality Assessment at Step {i} ---")

                # Camera quality assessment
                camera_metrics = self.assess_camera_quality()
                print("Camera Quality Metrics:")
                for key, value in camera_metrics.items():
                    print(f"  {key}: {value}")

                # LiDAR quality assessment
                lidar_metrics = self.assess_lidar_quality()
                print("LiDAR Quality Metrics:")
                for key, value in lidar_metrics.items():
                    print(f"  {key}: {value}")

        print(f"Perception quality assessment completed with {steps} steps")

# Create and run the quality assessment
if __name__ == "__main__":
    quality_assessment = PerceptionQualityAssessment()
    quality_assessment.run_quality_assessment(steps=500)
```

## Exercise 6: Troubleshooting and Optimization

### Step 1: Create perception optimization tools

Create `~/isaac_sim_examples/perception_optimization.py`:

```python
#!/usr/bin/env python3
# perception_optimization.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.sensor import Camera
from omni.isaac.range_sensor import LidarRtx
import numpy as np
import time
import carb

class PerceptionOptimization:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.sensors = {}
        self.performance_stats = {}
        self.setup_environment()

    def setup_environment(self):
        """Set up environment for performance testing."""
        # Create ground plane
        create_primitive(
            prim_path="/World/GroundPlane",
            primitive_type="Plane",
            scale=[30, 30, 1],
            position=[0, 0, 0]
        )

        # Create many objects for performance testing
        for i in range(50):
            position = [np.random.uniform(-10, 10), np.random.uniform(-10, 10), 0.5]
            create_primitive(
                prim_path=f"/World/PerformanceObj_{i}",
                primitive_type="Cylinder",
                scale=[0.2, 0.2, 0.8],
                position=position
            )

        # Create robot with multiple sensors
        self.robot = create_primitive(
            prim_path="/World/Robot",
            primitive_type="Cylinder",
            scale=[0.5, 0.5, 1.0],
            position=[0, 0, 0.5]
        )

    def add_optimized_sensors(self):
        """Add sensors with optimized configurations."""
        # Optimized camera (balanced performance and quality)
        optimized_camera = Camera(
            prim_path="/World/Robot/OptimizedCamera",
            frequency=30,  # Balanced frequency
            resolution=(640, 480),  # Balanced resolution
            position=[0.3, 0, 0.5]
        )
        optimized_camera.set_local_pose(translation=np.array([0.3, 0, 0.5]))
        self.sensors['optimized_camera'] = optimized_camera

        # Optimized LiDAR (balanced performance and quality)
        optimized_lidar = LidarRtx(
            prim_path="/World/Robot/OptimizedLidar",
            translation=(0.0, 0.0, 0.8),
            config="Example_Rotary",
            range_resolution=0.005,  # Good resolution
            rotation_frequency=10,   # Balanced frequency
            horizontal_resolution=0.25,  # Good resolution
            vertical_resolution=0.4,
            horizontal_samples=1080,  # Balanced samples
            vertical_samples=64,
            max_range=25.0,
            min_range=0.1
        )
        self.sensors['optimized_lidar'] = optimized_lidar

    def measure_performance(self):
        """Measure performance of perception system."""
        start_time = time.time()

        # Process camera data
        if 'optimized_camera' in self.sensors:
            start_camera = time.time()
            rgb_image = self.sensors['optimized_camera'].get_rgb()
            camera_time = time.time() - start_camera

            if rgb_image is not None:
                camera_fps = 1.0 / camera_time if camera_time > 0 else float('inf')
            else:
                camera_fps = 0

        # Process LiDAR data
        if 'optimized_lidar' in self.sensors:
            start_lidar = time.time()
            lidar_data = self.sensors['optimized_lidar'].get_linear_depth_data()
            lidar_time = time.time() - start_lidar

            if lidar_data is not None:
                lidar_fps = 1.0 / lidar_time if lidar_time > 0 else float('inf')
            else:
                lidar_fps = 0

        total_time = time.time() - start_time

        return {
            'camera_processing_time': camera_time,
            'lidar_processing_time': lidar_time,
            'total_processing_time': total_time,
            'camera_fps': camera_fps,
            'lidar_fps': lidar_fps
        }

    def run_optimization_test(self, steps=300):
        """Run optimization test."""
        self.world.reset()
        self.add_optimized_sensors()

        # Set camera view
        set_camera_view(eye=[15, 15, 15], target=[0, 0, 0])

        total_camera_time = 0
        total_lidar_time = 0
        total_steps = 0

        for i in range(steps):
            step_start = time.time()
            self.world.step(render=True)

            # Measure performance every 50 steps
            if i % 50 == 0 and i > 0:
                perf_metrics = self.measure_performance()

                total_camera_time += perf_metrics['camera_processing_time']
                total_lidar_time += perf_metrics['lidar_processing_time']
                total_steps += 1

                print(f"\n--- Performance Metrics (Step {i}) ---")
                print(f"Camera processing: {perf_metrics['camera_processing_time']:.4f}s ({perf_metrics['camera_fps']:.1f} FPS)")
                print(f"LiDAR processing: {perf_metrics['lidar_processing_time']:.4f}s ({perf_metrics['lidar_fps']:.1f} FPS)")
                print(f"Total processing: {perf_metrics['total_processing_time']:.4f}s")

        # Calculate averages
        if total_steps > 0:
            avg_camera_time = total_camera_time / total_steps
            avg_lidar_time = total_lidar_time / total_steps

            print(f"\n--- Average Performance ---")
            print(f"Average camera processing: {avg_camera_time:.4f}s")
            print(f"Average LiDAR processing: {avg_lidar_time:.4f}s")

        print(f"Perception optimization test completed with {steps} steps")

# Create and run the optimization test
if __name__ == "__main__":
    optimization_test = PerceptionOptimization()
    optimization_test.run_optimization_test(steps=300)
```

## Troubleshooting

### Common Issues and Solutions

1. **Camera images appear distorted or incorrect**:
   - Check camera intrinsic parameters
   - Verify coordinate system transformations
   - Ensure proper lens distortion settings

2. **LiDAR data has artifacts or incorrect ranges**:
   - Verify LiDAR configuration parameters
   - Check for occlusions or reflections
   - Adjust range resolution and sample rates

3. **Performance issues with multiple sensors**:
   - Reduce sensor frequencies
   - Lower resolution settings
   - Optimize rendering quality

4. **ROS bridge connection problems**:
   - Verify ROS 2 network configuration
   - Check topic names and message types
   - Ensure Isaac Sim and ROS are on same network

5. **Segmentation data not available**:
   - Enable segmentation in camera configuration
   - Check semantic annotation on objects
   - Verify Isaac Sim extensions are loaded

## Assessment Questions

1. How do you configure camera intrinsic parameters in Isaac Sim?
2. What factors affect LiDAR performance in simulation?
3. How would you implement sensor fusion in Isaac Sim?
4. What are the key metrics for evaluating perception system quality?
5. How do you optimize perception performance in Isaac Sim?

## Extension Exercises

1. Implement a SLAM system using Isaac Sim sensors
2. Create a 3D object detection pipeline
3. Implement semantic segmentation with neural networks
4. Create a multi-camera stereo vision system
5. Implement sensor calibration procedures

## Summary

In this lab, you successfully:
- Configured and used various perception sensors in Isaac Sim
- Processed camera images, LiDAR point clouds, and other sensor data
- Integrated perception systems with ROS
- Implemented computer vision algorithms for object detection
- Assessed perception quality and optimized performance
- Validated sensor data accuracy and reliability

These skills are essential for developing robust perception systems in robotics applications. The ability to configure, calibrate, and optimize perception sensors is crucial for building reliable autonomous robots that can accurately perceive and understand their environment.