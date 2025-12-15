---
sidebar_label: 'Lab 3.1: Isaac Sim Setup'
---

# Lab Exercise 3.1: NVIDIA Isaac Sim Setup

This lab exercise guides you through setting up NVIDIA Isaac Sim for robotics simulation and AI development.

## Objectives

- Install NVIDIA Isaac Sim
- Configure basic simulation environment
- Import robot models into Isaac Sim
- Set up basic robot control

## Prerequisites

- NVIDIA GPU with CUDA compute capability 6.0 or higher
- NVIDIA Driver version 470 or later
- Ubuntu 20.04 or 22.04 (or Windows 10/11 with WSL2)

## System Requirements and Setup

### Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3080 or better recommended)
- **CPU**: Multi-core processor (Intel i7 or AMD Ryzen 7 or better)
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ available space

### Software Prerequisites

1. Install NVIDIA drivers:
   ```bash
   sudo apt update
   sudo apt install nvidia-driver-535
   ```

2. Install CUDA:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
   sudo sh cuda_12.3.0_545.23.06_linux.run
   ```

3. Add CUDA to your PATH in `.bashrc`:
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

## Isaac Sim Installation

### Download Isaac Sim

1. Go to NVIDIA Developer website and download Isaac Sim
2. Extract the package:
   ```bash
   tar -xf isaac-sim-4.2.0-linux-x86_64-release.tar.bz2
   cd isaac-sim-4.2.0
   ```

### Install Dependencies

```bash
# Install system dependencies
sudo apt update
sudo apt install python3-pip python3-dev python3-venv

# Install Isaac Sim dependencies
./install-deps.sh
```

### Initial Setup

```bash
# Create and activate virtual environment
python3 -m venv ~/isaac-sim-env
source ~/isaac-sim-env/bin/activate

# Install Isaac Sim Python packages
pip install -e .
```

## Launching Isaac Sim

### Basic Launch

```bash
# Launch Isaac Sim
./isaac-sim.sh
```

### Alternative Launch Methods

1. **From Python**:
   ```python
   import omni
   from omni.isaac.kit import SimulationApp

   # Start Isaac Sim application
   config = {
       "headless": False,
       "window_width": 1280,
       "window_height": 720,
   }
   simulation_app = SimulationApp(config)
   simulation_app.run()
   ```

2. **Headless Mode** (for training):
   ```bash
   ./isaac-sim.sh --/renderer/disableAll=1 --/headless/rendering=1
   ```

## Basic Isaac Sim Interface

### Key Components

- **Viewport**: Main 3D scene view
- **Stage**: Scene hierarchy and object management
- **Property Panel**: Object properties and settings
- **Timeline**: Animation and simulation controls

### Basic Navigation

- **Orbit**: Right mouse button + drag
- **Pan**: Middle mouse button + drag or Shift + left drag
- **Zoom**: Mouse wheel or Alt + right drag
- **Select**: Left click on objects

## Robot Import and Setup

### Importing Robots

Isaac Sim supports several formats:
- **USD**: Native format (recommended)
- **URDF**: Via URDF Importer extension
- **FBX/OBJ**: For visual assets

### Using URDF Importer

1. Enable URDF Importer extension:
   - Go to Window → Extensions
   - Search for "URDF Importer"
   - Enable the extension

2. Import URDF:
   - Go to File → Import → URDF
   - Select your URDF file
   - Configure import settings

### Example Robot Setup Script

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
import numpy as np

# Create world instance
world = World(stage_units_in_meters=1.0)

# Add robot to the stage
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    print("Could not find Isaac Sim assets folder")
else:
    # Example: Adding a simple robot (replace with your robot path)
    robot_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
    add_reference_to_stage(usd_path=robot_path, prim_path="/World/Robot")

    # Add ground plane
    add_reference_to_stage(
        usd_path=assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd",
        prim_path="/World/Light"
    )

    # Play the simulation
    world.play()

    # Run simulation for a few steps
    for i in range(100):
        world.step(render=True)

    world.stop()

# Close the application
world.clear()
```

## Isaac Sim Extensions

### Key Extensions

1. **URDF Importer**: Import robots from URDF
2. **ROS2 Bridge**: Connect to ROS 2
3. **Isaac Sim Python**: Python API access
4. **Sensors**: Add various sensor types
5. **Robot Apps**: Pre-built robot applications

### Enabling Extensions

```python
import omni
from omni.isaac.core.utils.extensions import enable_extension

# Enable required extensions
extensions_to_enable = [
    "omni.isaac.ros2_bridge",
    "omni.isaac.urdf_importer",
    "omni.isaac.range_sensor",
    "omni.isaac.sensor"
]

for ext in extensions_to_enable:
    enable_extension(ext)
```

## Environment Setup

### Creating Custom Environments

```python
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_prim
from pxr import Gf

# Add ground plane
create_prim("/World/GroundPlane", "Plane", position=Gf.Vec3f(0, 0, 0), size=100)

# Add lighting
create_prim("/World/Light", "DistantLight", position=Gf.Vec3f(0, 0, 5), intensity=3000)

# Add objects to environment
create_prim(
    "/World/Box",
    "Cube",
    position=Gf.Vec3f(1.0, 0.0, 0.5),
    orientation=Gf.Quatf(0, 0, 0, 1),
    scale=Gf.Vec3f(0.2, 0.2, 0.2)
)
```

## Isaac Sim Python API

### Basic World Management

```python
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.articulations import Articulation
import numpy as np

class IsaacSimRobotController:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.robot = None

    def setup_robot(self, robot_path, prim_path):
        # Add robot to stage
        self.world.scene.add(Articulation(prim_path=prim_path, name="my_robot"))

        # Get robot reference
        self.robot = self.world.scene.get_object("my_robot")

    def control_robot(self):
        # Play the simulation
        self.world.play()

        # Control robot for some steps
        for i in range(1000):
            # Get current robot state
            joint_positions = self.robot.get_joint_positions()

            # Apply some control (example: move joints sinusoidally)
            target_positions = np.sin(i * 0.01) * 0.5
            self.robot.set_joint_positions(np.array([target_positions] * len(joint_positions)))

            # Step simulation
            self.world.step(render=True)

        self.world.stop()

# Usage
controller = IsaacSimRobotController()
# controller.setup_robot(robot_path, "/World/Robot")
# controller.control_robot()
```

## ROS 2 Integration

### Setting up ROS 2 Bridge

1. Enable ROS 2 Bridge extension in Isaac Sim
2. Set ROS domain ID:
   ```bash
   export ROS_DOMAIN_ID=1
   ```

### Example ROS 2 Integration

```python
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core import World
import rclpy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

# Enable ROS 2 bridge
enable_extension("omni.isaac.ros2_bridge")

# Initialize ROS 2
rclpy.init()

# Create ROS 2 node
ros_node = rclpy.create_node('isaac_sim_bridge')

# Publishers and subscribers would be created here
# The ROS 2 bridge handles the connection automatically
```

## Performance Optimization

### Graphics Settings

For better performance:
- Adjust rendering quality in Window → Settings → Renderer
- Use lower resolution for training runs
- Disable unnecessary visual effects

### Simulation Settings

```python
# Optimize simulation settings
from omni.isaac.core import World

# Set fixed sub-step size for consistent physics
world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0/60.0,  # Physics step size
    rendering_dt=1.0/30.0  # Rendering step size
)
```

## Isaac Sim Projects

### Creating a Project Structure

```
my_robot_project/
├── configs/
│   ├── robot_config.yaml
│   └── simulation_config.yaml
├── scripts/
│   ├── setup_robot.py
│   └── control_robot.py
├── assets/
│   └── robots/
└── scenes/
    └── training_env.usd
```

### Example Project Setup

```python
# setup_robot.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
import carb

class RobotProject:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.assets_root = get_assets_root_path()

    def setup_scene(self):
        # Add robot
        robot_path = self.assets_root + "/Isaac/Robots/Turtlebot/turtlebot3_differential.usd"
        add_reference_to_stage(usd_path=robot_path, prim_path="/World/Robot")

        # Add environment
        add_reference_to_stage(
            usd_path=self.assets_root + "/Isaac/Environments/Simple_Room/simple_room.usd",
            prim_path="/World/Room"
        )

    def run_simulation(self):
        self.world.reset()
        self.world.play()

        for i in range(1000):
            self.world.step(render=True)

            if i % 100 == 0:
                carb.log_info(f"Simulation step: {i}")

        self.world.stop()

# Run the project
project = RobotProject()
project.setup_scene()
project.run_simulation()
```

## Exercise Tasks

1. Install Isaac Sim on your system
2. Launch Isaac Sim and explore the interface
3. Import a simple robot model (URDF or USD)
4. Create a basic environment with ground plane and lighting
5. Write a Python script to control the robot in simulation
6. Test the simulation by running for multiple steps

## Troubleshooting

### Common Issues

- **Black screen on launch**: Check GPU drivers and OpenGL support
- **Memory errors**: Reduce scene complexity or increase swap space
- **Python import errors**: Verify virtual environment setup
- **Performance issues**: Adjust graphics settings and physics parameters

### Performance Tips

- Use lower rendering resolution for training
- Reduce physics sub-step size for faster simulation
- Simplify collision meshes where possible
- Use fixed time steps for consistency

## Summary

In this lab, you learned to install and set up NVIDIA Isaac Sim for robotics simulation. You explored the interface, imported robot models, and created basic simulation environments. Isaac Sim provides a powerful platform for robot development and AI training with GPU acceleration.