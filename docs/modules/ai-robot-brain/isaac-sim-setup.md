---
sidebar_label: 'Isaac Sim Setup'
---

# NVIDIA Isaac Sim Setup

This document covers setting up NVIDIA Isaac Sim for robot AI development.

## System Requirements

- NVIDIA GPU with CUDA compute capability 6.0 or higher
- NVIDIA Driver version 470 or later
- CUDA 11.8 or later
- Ubuntu 20.04 or 22.04 (or Windows 10/11)

## Installation

### Prerequisites

1. Install NVIDIA drivers:
   ```bash
   sudo apt update
   sudo apt install nvidia-driver-535
   ```

2. Install CUDA:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
   sudo sh cuda_11.8.0_520.61.05_linux.run
   ```

### Isaac Sim Installation

1. Download Isaac Sim from NVIDIA Developer website
2. Extract the package:
   ```bash
   tar -xf isaac-sim-2023.1.1-linux-x86_64-release.tar.bz2
   cd isaac-sim-2023.1.1
   ```

3. Install dependencies:
   ```bash
   ./install-deps.sh
   ```

4. Run the setup:
   ```bash
   ./isaac-sim.sh
   ```

## Basic Configuration

### Environment Setup

Add to your `.bashrc`:
```bash
export ISAAC_SIM_PATH=/path/to/isaac-sim
export PYTHONPATH=$ISAAC_SIM_PATH/python:$PYTHONPATH
```

### Launch Isaac Sim

```bash
cd $ISAAC_SIM_PATH
./isaac-sim.sh
```

## Integration with ROS 2

### ROS Bridge

Isaac Sim includes a ROS 2 bridge for integration:
1. Enable the ROS 2 Bridge extension
2. Configure ROS domain ID
3. Map Isaac Sim topics to ROS 2 topics

### Example Integration

```python
# Python example connecting to Isaac Sim
import omni
import carb
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

# Create a world instance
world = World(stage_units_in_meters=1.0)

# Add your robot to the stage
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

add_reference_to_stage(
    usd_path=assets_root_path + "/Isaac/Robots/Franka/franka.usd",
    prim_path="/World/Robot"
)

# Play the simulation
world.play()
for i in range(1000):
    world.step(render=True)
world.stop()
```

## Robot Import

### Supported Formats

Isaac Sim supports various robot formats:
- USD (Universal Scene Description)
- URDF with conversion tools
- FBX and OBJ for visual assets

### Importing URDF

Use the URDF import extension:
1. Open Isaac Sim
2. Go to Extensions → Isaac Examples → URDF Importer
3. Select your URDF file
4. Configure import settings

## Best Practices

- Regularly update Isaac Sim for bug fixes and new features
- Use lightweight models for real-time simulation
- Optimize materials and textures for performance
- Test on hardware before deploying to real robots