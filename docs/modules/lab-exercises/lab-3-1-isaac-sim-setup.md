# Lab 3.1: Isaac Sim Setup and Environment

## Overview

In this lab, you will install and configure NVIDIA Isaac Sim, set up the development environment, and create your first Isaac Sim scene with a robot. You'll learn about Isaac Sim's architecture, the Omniverse ecosystem, and how to integrate it with ROS for robotics applications.

## Objectives

By the end of this lab, you will be able to:
- Install and configure NVIDIA Isaac Sim
- Set up the Omniverse Launcher and extensions
- Create basic Isaac Sim scenes and environments
- Understand Isaac Sim's physics simulation capabilities
- Integrate Isaac Sim with ROS using Isaac ROS bridge
- Create and configure robot models in Isaac Sim

## Prerequisites

- NVIDIA GPU with CUDA support (RTX 3080/4080 or higher recommended)
- Ubuntu 20.04 or 22.04 LTS
- NVIDIA GPU drivers (535.129.03 or later)
- CUDA Toolkit 11.8 or 12.x
- Omniverse Launcher installed
- Basic understanding of robotics concepts
- Completion of Module 1 (ROS 2 basics)

## Duration

4-5 hours

## Exercise 1: Install Isaac Sim and Prerequisites

### Step 1: Verify System Requirements

First, verify your system meets the requirements:

```bash
# Check GPU and driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Check available disk space (Isaac Sim requires ~100GB)
df -h | grep -E '(/$|/home)'
```

### Step 2: Install Omniverse Launcher

1. Go to https://www.nvidia.com/en-us/omniverse/download/
2. Download and install Omniverse Launcher for Linux
3. Launch Omniverse Launcher and sign in with your NVIDIA Developer account

### Step 3: Install Isaac Sim through Omniverse Launcher

1. In Omniverse Launcher, go to the "Apps" section
2. Find and install "Isaac Sim" (latest version)
3. Wait for the installation to complete (this may take 30+ minutes)

### Step 4: Verify Isaac Sim Installation

```bash
# Navigate to Isaac Sim installation directory (typically)
cd ~/.local/share/ov/pkg/isaac-sim-2023.1.1

# Run Isaac Sim to verify installation
./python.sh -c "import omni; print('Isaac Sim import successful')"
```

## Exercise 2: Set up Isaac Sim Environment

### Step 1: Create Isaac Sim configuration

Create a configuration file for Isaac Sim:

```bash
mkdir -p ~/.config/NVIDIA/Isaac-Sim/
```

Create `~/.config/NVIDIA/Isaac-Sim/config.yaml`:

```yaml
# Isaac Sim Configuration
app:
  window_width: 1920
  window_height: 1080
  enable_audio: false

physics:
  solver_position_iteration_count: 16
  solver_velocity_iteration_count: 8
  gpu_max_rigid_contact_count: 524288
  gpu_max_rigid_patch_count: 32768

renderer:
  resolution_width: 1920
  resolution_height: 1080
  max_render_width: 3840
  max_render_height: 2160

exts:
  omni.isaac.ros2_bridge:
    enabled: true
  omni.isaac.sensor:
    enabled: true
  omni.isaac.range_sensor:
    enabled: true
  omni.isaac.core_nodes:
    enabled: true
  omni.isaac.debug_draw:
    enabled: true
  omni.isaac.dynamic_control:
    enabled: true
  omni.isaac.urdf_importer:
    enabled: true
  omni.isaac.version_checker:
    enabled: true
```

### Step 2: Set up environment variables

Add Isaac Sim paths to your `~/.bashrc`:

```bash
# Isaac Sim Environment Variables
export ISAACSIM_PATH="$HOME/.local/share/ov/pkg/isaac-sim-2023.1.1"
export ISAACSIM_PYTHON_PATH="$ISAACSIM_PATH/python.sh"
export OMNI_RESOURCES="$ISAACSIM_PATH/kit/exts"

# Add to PATH
export PATH="$ISAACSIM_PATH/python/bin:$PATH"
export PYTHONPATH="$ISAACSIM_PATH/python:$PYTHONPATH"
```

Apply the changes:

```bash
source ~/.bashrc
```

### Step 3: Install Isaac ROS packages

```bash
# Create Isaac ROS workspace
mkdir -p ~/isaac_ros_ws/src
cd ~/isaac_ros_ws/src

# Clone Isaac ROS common packages
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark.git

# Clone sensor packages
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_detectnet.git

# Clone navigation packages
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_navigation.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_occupancy_grid_localizer.git

# Build the workspace
cd ~/isaac_ros_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select isaac_ros_common
colcon build --symlink-install
```

## Exercise 3: Launch Isaac Sim and Create Basic Scene

### Step 1: Launch Isaac Sim

```bash
# Launch Isaac Sim
cd ~/.local/share/ov/pkg/isaac-sim-2023.1.1
./isaac-sim.sh
```

### Step 2: Create a basic scene using Python API

Create a Python script to create a basic Isaac Sim scene:

Create `~/isaac_sim_examples/basic_scene.py`:

```python
#!/usr/bin/env python3
# basic_scene.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.viewports import set_camera_view
import carb

# Initialize the world
world = World(stage_units_in_meters=1.0)

# Create ground plane
ground_plane = create_primitive(
    prim_path="/World/GroundPlane",
    primitive_type="Plane",
    scale=[10, 10, 1],
    position=[0, 0, 0],
    orientation=[0, 0, 0, 1]
)

# Create a simple box
box = create_primitive(
    prim_path="/World/Box",
    primitive_type="Cube",
    scale=[0.5, 0.5, 0.5],
    position=[1.0, 0.0, 0.25],
    orientation=[0, 0, 0, 1]
)

# Create a sphere
sphere = create_primitive(
    prim_path="/World/Sphere",
    primitive_type="Sphere",
    scale=[0.3, 0.3, 0.3],
    position=[0.0, 1.0, 0.3],
    orientation=[0, 0, 0, 1]
)

# Set camera view
set_camera_view(eye=[5, 5, 5], target=[0, 0, 0])

# Play the simulation
world.reset()
for i in range(500):
    world.step(render=True)

# Stop the simulation
world.stop()
```

### Step 3: Run the basic scene script

```bash
# Run the basic scene script within Isaac Sim
cd ~/.local/share/ov/pkg/isaac-sim-2023.1.1
./python.sh ~/isaac_sim_examples/basic_scene.py
```

## Exercise 4: Import and Configure Robot Models

### Step 1: Create a simple robot model

Create a basic robot model using Isaac Sim's Python API:

Create `~/isaac_sim_examples/simple_robot.py`:

```python
#!/usr/bin/env python3
# simple_robot.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.semantics import add_semantic_group_to_stage
from omni.isaac.core.utils.viewports import set_camera_view
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, Sdf
import numpy as np

# Initialize the world
world = World(stage_units_in_meters=1.0)

# Create a simple wheeled robot
def create_simple_robot(robot_name="SimpleRobot", position=[0, 0, 0.5]):
    """Create a simple wheeled robot with a box body and two wheels."""

    # Create robot root prim
    robot_prim_path = f"/World/{robot_name}"
    create_prim(
        prim_path=robot_prim_path,
        prim_type="Xform",
        position=position
    )

    # Create robot body (chassis)
    body_path = f"{robot_prim_path}/Chassis"
    create_prim(
        prim_path=body_path,
        prim_type="Cube",
        position=[0, 0, 0],
        scale=[0.5, 0.3, 0.2],
        orientation=[0, 0, 0, 1]
    )

    # Create left wheel
    left_wheel_path = f"{robot_prim_path}/LeftWheel"
    create_prim(
        prim_path=left_wheel_path,
        prim_type="Cylinder",
        position=[0.15, 0.15, -0.1],
        scale=[0.1, 0.1, 0.05],
        orientation=[0.707, 0, 0, 0.707]  # Rotate 90 degrees around X
    )

    # Create right wheel
    right_wheel_path = f"{robot_prim_path}/RightWheel"
    create_prim(
        prim_path=right_wheel_path,
        prim_type="Cylinder",
        position=[0.15, -0.15, -0.1],
        scale=[0.1, 0.1, 0.05],
        orientation=[0.707, 0, 0, 0.707]  # Rotate 90 degrees around X
    )

    # Add physics to the robot parts
    stage = omni.usd.get_context().get_stage()

    # Add rigid body to chassis
    chassis_prim = stage.GetPrimAtPath(body_path)
    UsdPhysics.RigidBodyAPI.Apply(chassis_prim, "physics:rigidBodyAPI")

    # Add rigid body to wheels
    left_wheel_prim = stage.GetPrimAtPath(left_wheel_path)
    right_wheel_prim = stage.GetPrimAtPath(right_wheel_path)
    UsdPhysics.RigidBodyAPI.Apply(left_wheel_prim, "physics:rigidBodyAPI")
    UsdPhysics.RigidBodyAPI.Apply(right_wheel_prim, "physics:rigidBodyAPI")

# Create the robot
create_simple_robot("MyRobot", [0, 0, 0.5])

# Set camera view
set_camera_view(eye=[3, 3, 3], target=[0, 0, 0])

# Play the simulation
world.reset()
for i in range(1000):
    world.step(render=True)

    # Simple control (optional)
    if i > 100:
        # Add forces to wheels for movement
        pass

# Stop the simulation
world.stop()
```

### Step 2: Import a URDF robot model

Create a script to import a URDF model:

Create `~/isaac_sim_examples/urdf_import.py`:

```python
#!/usr/bin/env python3
# urdf_import.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.urdf_importer import _urdf_importer

# Initialize the world
world = World(stage_units_in_meters=1.0)

# Initialize URDF importer
urdf_interface = _urdf_importer.acquire_urdf_interface()

# Import a sample URDF (you can use the one from previous modules)
# For this example, we'll create a simple URDF string
simple_urdf = """
<?xml version="1.0"?>
<robot name="simple_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <link name="wheel_left">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <link name="wheel_right">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_left"/>
    <origin xyz="0.15 0.15 -0.1" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_right"/>
    <origin xyz="0.15 -0.15 -0.1" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
"""

# For now, we'll create the robot manually as the URDF importer can be complex
# In practice, you would use: urdf_interface.import_urdf(...)

# Set camera view
set_camera_view(eye=[3, 3, 3], target=[0, 0, 0])

# Play the simulation
world.reset()
for i in range(500):
    world.step(render=True)

# Stop the simulation
world.stop()
```

## Exercise 5: Set up Isaac ROS Bridge

### Step 1: Create a ROS bridge configuration

Create `~/isaac_sim_examples/ros_bridge_example.py`:

```python
#!/usr/bin/env python3
# ros_bridge_example.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.viewports import set_camera_view
import omni.isaac.ros2_bridge._ros2_bridge as ros2_bridge

# Initialize the world
world = World(stage_units_in_meters=1.0)

# Create a simple robot
robot_position = [0, 0, 0.5]
create_primitive(
    prim_path="/World/Robot",
    primitive_type="Cylinder",
    scale=[0.2, 0.2, 0.5],
    position=robot_position,
    orientation=[0, 0, 0, 1]
)

# Set camera view
set_camera_view(eye=[3, 3, 3], target=[0, 0, 0])

# Initialize ROS 2 bridge
ros2_bridge_instance = ros2_bridge.acquire_ros2_bridge_interface()

# Create some sensors to bridge
from omni.isaac.sensor import Camera
from omni.isaac.range_sensor import LidarRtx

# Create a camera
camera = Camera(
    prim_path="/World/Robot/Camera",
    frequency=30,
    resolution=(640, 480)
)

# Create a LiDAR sensor
lidar = LidarRtx(
    prim_path="/World/Robot/Lidar",
    translation=(0.0, 0.0, 0.2),
    config="Example_Rotary",
    range_resolution=0.005
)

# Play the simulation
world.reset()

# Enable ROS bridge for the sensors
ros2_bridge_instance.publish_camera(camera, topic_name="/isaac_sim/camera/image")
ros2_bridge_instance.publish_lidar(lidar, topic_name="/isaac_sim/lidar/scan")

for i in range(1000):
    world.step(render=True)

# Stop the simulation
world.stop()
```

### Step 2: Test ROS bridge functionality

```bash
# In a separate terminal, source ROS and Isaac ROS workspace
source /opt/ros/humble/setup.bash
source ~/isaac_ros_ws/install/setup.bash

# Check if ROS topics are available
ros2 topic list | grep isaac_sim

# Echo camera data (if available)
ros2 topic echo /isaac_sim/camera/image --field data
```

## Exercise 6: Create a Complete Isaac Sim Scene

### Step 1: Create a complex scene with multiple objects

Create `~/isaac_sim_examples/complex_scene.py`:

```python
#!/usr/bin/env python3
# complex_scene.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.materials import PhysicsMaterial
import numpy as np

# Initialize the world
world = World(stage_units_in_meters=1.0)

# Create ground plane with texture
ground_plane = create_primitive(
    prim_path="/World/GroundPlane",
    primitive_type="Plane",
    scale=[10, 10, 1],
    position=[0, 0, 0],
    orientation=[0, 0, 0, 1]
)

# Create various objects
objects = []

# Create a table
table = create_primitive(
    prim_path="/World/Table",
    primitive_type="Cuboid",
    scale=[1.0, 0.6, 0.8],
    position=[2, 0, 0.4],
    orientation=[0, 0, 0, 1]
)
objects.append(table)

# Create some objects on the table
cup = create_primitive(
    prim_path="/World/Cup",
    primitive_type="Cylinder",
    scale=[0.1, 0.1, 0.15],
    position=[2.2, 0.1, 0.9],
    orientation=[0, 0, 0, 1]
)
objects.append(cup)

box = create_primitive(
    prim_path="/World/Box",
    primitive_type="Cube",
    scale=[0.2, 0.2, 0.2],
    position=[1.8, -0.1, 0.9],
    orientation=[0, 0, 0, 1]
)
objects.append(box)

# Create a simple robot
robot = create_primitive(
    prim_path="/World/Robot",
    primitive_type="Cylinder",
    scale=[0.3, 0.3, 0.5],
    position=[0, 0, 0.25],
    orientation=[0, 0, 0, 1]
)

# Add physics materials for different friction properties
physics_material = PhysicsMaterial(
    prim_path="/World/PhysicsMaterial",
    static_friction=0.5,
    dynamic_friction=0.5,
    restitution=0.1
)

# Set camera view
set_camera_view(eye=[5, 5, 5], target=[0, 0, 0])

# Play the simulation
world.reset()

# Add some dynamics to the scene
for i in range(1000):
    world.step(render=True)

    # Simple dynamics: move the robot periodically
    if i % 100 == 0:
        robot_position = [np.sin(i/100), np.cos(i/100), 0.25]
        # In practice, you would use the robot's articulation controller
        # robot.set_world_pose(position=robot_position)

# Stop the simulation
world.stop()

print("Complex scene simulation completed")
```

### Step 2: Run the complex scene

```bash
cd ~/.local/share/ov/pkg/isaac-sim-2023.1.1
./python.sh ~/isaac_sim_examples/complex_scene.py
```

## Exercise 7: Configure Isaac Sim for Robotics Development

### Step 1: Create a configuration file for robotics development

Create `~/isaac_sim_examples/robot_config.py`:

```python
#!/usr/bin/env python3
# robot_config.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.semantics import add_semantic_data_to_stage
from omni.isaac.sensor import Camera, ImuSensor
from omni.isaac.range_sensor import LidarRtx
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
import numpy as np

class IsaacRobotEnvironment:
    def __init__(self, stage_units_in_meters=1.0):
        self.world = World(stage_units_in_meters=stage_units_in_meters)
        self.stage = omni.usd.get_context().get_stage()

    def setup_environment(self):
        """Set up the basic environment with ground and lighting."""
        # Create ground plane
        self.create_ground_plane()

        # Add lighting
        self.add_lighting()

        # Add basic objects for testing
        self.add_test_objects()

    def create_ground_plane(self):
        """Create a ground plane for the environment."""
        # Create plane prim
        plane_prim = create_prim(
            prim_path="/World/GroundPlane",
            prim_type="Plane",
            scale=[20, 20, 1],
            position=[0, 0, 0],
            orientation=[0, 0, 0, 1]
        )

        # Add physics to ground plane
        UsdPhysics.CollisionAPI.Apply(self.stage.GetPrimAtPath("/World/GroundPlane"))

    def add_lighting(self):
        """Add lighting to the environment."""
        # Add dome light
        dome_light = create_prim(
            prim_path="/World/DomeLight",
            prim_type="DomeLight",
            position=[0, 0, 0],
            orientation=[0, 0, 0, 1]
        )

        # Configure dome light
        dome_light_prim = self.stage.GetPrimAtPath("/World/DomeLight")
        light_api = UsdGeom.LightAPI(dome_light_prim)
        light_api.GetIntensityAttr().Set(3000)

    def add_test_objects(self):
        """Add test objects to the environment."""
        # Create a few obstacles
        for i in range(5):
            position = [np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0.5]
            create_primitive(
                prim_path=f"/World/Obstacle_{i}",
                primitive_type="Cylinder",
                scale=[0.3, 0.3, 1.0],
                position=position,
                orientation=[0, 0, 0, 1]
            )

    def add_robot_with_sensors(self, robot_name="Robot", position=[0, 0, 1.0]):
        """Add a robot with sensors to the environment."""
        # Create robot body
        robot_body = create_primitive(
            prim_path=f"/World/{robot_name}",
            primitive_type="Cylinder",
            scale=[0.4, 0.4, 0.8],
            position=position,
            orientation=[0, 0, 0, 1]
        )

        # Add camera
        camera = Camera(
            prim_path=f"/World/{robot_name}/Camera",
            frequency=30,
            resolution=(640, 480)
        )
        camera.set_local_pose(translation=np.array([0.3, 0, 0.2]))

        # Add IMU
        imu = ImuSensor(
            prim_path=f"/World/{robot_name}/Imu",
            name=f"{robot_name}_imu",
            translation=np.array([0.0, 0.0, 0.3])
        )

        # Add LiDAR
        lidar = LidarRtx(
            prim_path=f"/World/{robot_name}/Lidar",
            translation=(0.0, 0.0, 0.4),
            config="Example_Rotary",
            range_resolution=0.005
        )

        return robot_body, camera, imu, lidar

    def run_simulation(self, steps=1000):
        """Run the simulation for a specified number of steps."""
        self.world.reset()

        for i in range(steps):
            self.world.step(render=True)

            if i % 100 == 0:
                print(f"Simulation step: {i}/{steps}")

        print("Simulation completed successfully")

# Create and run the robot environment
if __name__ == "__main__":
    env = IsaacRobotEnvironment()
    env.setup_environment()
    env.add_robot_with_sensors()

    # Set camera view
    set_camera_view(eye=[8, 8, 8], target=[0, 0, 0])

    # Run simulation
    env.run_simulation(steps=2000)
```

## Exercise 8: Test Isaac Sim Installation

### Step 1: Create a comprehensive test script

Create `~/isaac_sim_examples/installation_test.py`:

```python
#!/usr/bin/env python3
# installation_test.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.sensor import Camera
from omni.isaac.range_sensor import LidarRtx
import numpy as np
import sys

def test_isaac_sim_installation():
    """Test Isaac Sim installation and basic functionality."""
    print("Testing Isaac Sim Installation...")

    try:
        # Test 1: Import core modules
        print("✓ Core modules imported successfully")

        # Test 2: Initialize world
        world = World(stage_units_in_meters=1.0)
        print("✓ World initialized successfully")

        # Test 3: Create basic objects
        ground = create_primitive(
            prim_path="/World/GroundPlane",
            primitive_type="Plane",
            scale=[10, 10, 1],
            position=[0, 0, 0]
        )

        robot = create_primitive(
            prim_path="/World/Robot",
            primitive_type="Cylinder",
            scale=[0.3, 0.3, 0.5],
            position=[0, 0, 0.25]
        )
        print("✓ Basic objects created successfully")

        # Test 4: Add sensors
        camera = Camera(
            prim_path="/World/Robot/Camera",
            frequency=30,
            resolution=(640, 480)
        )

        lidar = LidarRtx(
            prim_path="/World/Robot/Lidar",
            translation=(0.0, 0.0, 0.3),
            config="Example_Rotary",
            range_resolution=0.005
        )
        print("✓ Sensors added successfully")

        # Test 5: Set camera view
        set_camera_view(eye=[3, 3, 3], target=[0, 0, 0])
        print("✓ Camera view set successfully")

        # Test 6: Run simulation
        world.reset()
        for i in range(100):  # Run for 100 steps
            world.step(render=True)

        print("✓ Simulation ran successfully")
        print("✓ All tests passed! Isaac Sim installation is working correctly")

        return True

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_isaac_sim_installation()
    sys.exit(0 if success else 1)
```

### Step 2: Run the installation test

```bash
cd ~/.local/share/ov/pkg/isaac-sim-2023.1.1
./python.sh ~/isaac_sim_examples/installation_test.py
```

## Troubleshooting

### Common Issues and Solutions

1. **Isaac Sim fails to launch**:
   - Verify GPU drivers and CUDA installation
   - Check that your GPU meets minimum requirements
   - Ensure no other applications are using GPU exclusively

2. **Python import errors**:
   - Verify Isaac Sim installation path
   - Check that environment variables are set correctly
   - Ensure you're using the Isaac Sim Python interpreter

3. **Physics simulation instability**:
   - Adjust solver parameters in config
   - Reduce simulation time step
   - Check mass and inertia values for objects

4. **ROS bridge not working**:
   - Verify ROS 2 humble installation
   - Check Isaac ROS packages build
   - Ensure proper network configuration

5. **Performance issues**:
   - Reduce scene complexity
   - Lower rendering resolution
   - Adjust physics parameters for performance

## Assessment Questions

1. What are the minimum system requirements for Isaac Sim?
2. How do you configure Isaac Sim for robotics applications?
3. What is the role of the ROS bridge in Isaac Sim?
4. How do you add sensors to objects in Isaac Sim?
5. What are the key differences between Isaac Sim and Gazebo?

## Extension Exercises

1. Create a more complex robot model with multiple joints and links
2. Implement a simple navigation task in Isaac Sim
3. Add realistic lighting and materials to the environment
4. Create a custom USD asset and import it into Isaac Sim
5. Implement a reinforcement learning environment using Isaac Sim

## Summary

In this lab, you successfully:
- Installed and configured NVIDIA Isaac Sim
- Set up the Omniverse environment and extensions
- Created basic and complex scenes in Isaac Sim
- Added sensors to robot models
- Configured the ROS bridge for Isaac Sim
- Validated the installation with comprehensive tests

These foundational skills are essential for working with Isaac Sim in robotics applications. Isaac Sim provides a high-fidelity simulation environment that is particularly well-suited for AI and robotics research, offering realistic physics, rendering, and sensor simulation capabilities that are crucial for developing and testing advanced robotics algorithms.