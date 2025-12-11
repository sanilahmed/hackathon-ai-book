# Isaac Sim Setup

## Overview

This guide covers the installation and configuration of NVIDIA Isaac Sim, a powerful robotics simulation platform built on NVIDIA Omniverse. Isaac Sim provides high-fidelity physics simulation, photorealistic rendering, and AI training capabilities essential for developing intelligent humanoid robots.

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA RTX 3080/4080 or higher (RTX 6000 Ada recommended)
- **VRAM**: Minimum 10GB, 24GB+ recommended for complex scenes
- **CPU**: 8+ cores, 16+ threads (Intel i7/AMD Ryzen 7 or higher)
- **RAM**: 32GB+ (64GB recommended for large simulations)
- **Storage**: 100GB+ SSD for Isaac Sim installation and assets
- **OS**: Ubuntu 20.04 LTS or Windows 10/11

### Software Requirements
- NVIDIA GPU drivers: 535.129.03 or later
- CUDA: 11.8 or 12.x
- Docker: 20.10 or later
- Python: 3.8 - 3.10
- Omniverse Launcher: Latest version

## Installation Process

### 1. Install NVIDIA GPU Drivers and CUDA

```bash
# Update system packages
sudo apt update
sudo apt upgrade -y

# Install NVIDIA drivers
sudo apt install nvidia-driver-535 nvidia-utils-535

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda-toolkit-12-0
```

### 2. Install Omniverse Launcher

1. Download the Omniverse Launcher from [NVIDIA Developer Portal](https://developer.nvidia.com/nvidia-omniverse)
2. Install the launcher application
3. Sign in with your NVIDIA Developer account
4. Install Isaac Sim through the launcher

### 3. Install Isaac Sim via Omniverse

1. Launch Omniverse Launcher
2. Browse the "Apps" section
3. Install "Isaac Sim" application
4. Launch Isaac Sim to complete the initial setup

### 4. Install Isaac ROS Packages

```bash
# Clone Isaac ROS workspace
mkdir -p ~/isaac_ros_ws/src
cd ~/isaac_ros_ws/src
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark.git

# Clone perception packages
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

### 5. Install Isaac Lab (IsaacLab)

```bash
# Create Isaac Lab environment
mkdir -p ~/isaac_lab
cd ~/isaac_lab

# Create conda environment
conda create -n isaaclab python=3.10
conda activate isaaclab

# Install Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh -i

# Install additional dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Configuration

### 1. Isaac Sim Configuration

Create a configuration file at `~/.config/NVIDIA/Isaac-Sim/config.yaml`:

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
```

### 2. Environment Setup

Add the following to your `~/.bashrc`:

```bash
# Isaac Sim Environment Variables
export ISAACSIM_PATH="/home/$USER/.local/share/ov/pkg/isaac-sim-2023.1.1"
export ISAACSIM_PYTHON_PATH="$ISAACSIM_PATH/python.sh"
export OMNI_RESOURCES="$ISAACSIM_PATH/kit/exts"

# Isaac ROS Workspace
export ISAAC_ROS_WS="/home/$USER/isaac_ros_ws"
source $ISAAC_ROS_WS/install/setup.bash

# Isaac Lab Environment
export ISAACLAB_PATH="/home/$USER/isaac_lab/IsaacLab"
```

### 3. Verify Installation

```bash
# Test Isaac Sim
$ISAACSIM_PYTHON_PATH -c "import omni; print('Isaac Sim import successful')"

# Test Isaac ROS packages
source $ISAAC_ROS_WS/install/setup.bash
ros2 pkg list | grep isaac
```

## Testing with Humanoid Robot

### 1. Import Humanoid Model

```python
# Example Python script to load humanoid in Isaac Sim
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

# Initialize the world
world = World(stage_units_in_meters=1.0)

# Add humanoid robot from previous modules
add_reference_to_stage(
    usd_path="/path/to/humanoid_robot.usd",
    prim_path="/World/HumanoidRobot"
)

# Play the simulation
world.reset()
for i in range(1000):
    world.step(render=True)
```

### 2. Configure Robot for Isaac Sim

Create a USD file for the humanoid robot with Isaac Sim-specific configurations:

```usd
# humanoid_robot.usd
# This file defines the humanoid robot for Isaac Sim
# Based on the SDF model from Module 2
# Includes physics properties, sensors, and articulation

def Xform HumanoidRobot
{
    def PhysicsScene PhysicsScene
    {
        float physics:gravityMagnitude = 9.81
    }

    def Xform Body
    {
        def RigidBody "torso"
        {
            # Torso configuration
        }

        def RevoluteJoint "left_hip_joint"
        {
            # Hip joint configuration
        }

        # Additional joints and bodies...
    }
}
```

## Troubleshooting

### Common Issues and Solutions

1. **GPU Memory Issues**
   - Reduce simulation complexity
   - Lower rendering resolution
   - Close other GPU-intensive applications

2. **CUDA Compatibility**
   - Verify CUDA version matches Isaac Sim requirements
   - Check NVIDIA driver compatibility
   - Reinstall CUDA toolkit if necessary

3. **ROS Bridge Issues**
   - Verify ROS 2 humble installation
   - Check network configuration
   - Ensure Isaac Sim extension is enabled

## Next Steps

After completing the Isaac Sim setup, proceed to implement perception systems that will utilize Isaac's advanced computer vision capabilities for the humanoid robot.

---
[Next: Perception Systems](./perception-systems.md) | [Previous: Module 3 Index](./index.md)