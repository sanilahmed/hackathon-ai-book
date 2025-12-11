# Quickstart Guide: Physical AI & Humanoid Robotics Technical Book

## Overview
This guide helps you set up the environment and begin working with the Physical AI & Humanoid Robotics technical book. The book contains 4 modules covering ROS 2, Digital Twin simulation, AI-Robot integration, and Vision-Language-Action systems.

## Prerequisites
- Ubuntu 22.04 LTS (as specified in feature requirements)
- At least 16GB RAM (recommended 32GB for full simulation)
- NVIDIA GPU with CUDA support (for Isaac Sim - optional but recommended)
- 50GB+ free disk space
- Internet connection for package downloads

## Environment Setup

### 1. System Dependencies
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install basic development tools
sudo apt install -y build-essential cmake git python3-pip python3-dev
sudo apt install -y python3-colcon-common-extensions python3-rosdep
sudo apt install -y python3-vcstool wget curl gnupg lsb-release
```

### 2. ROS 2 Humble Hawksbill Installation
```bash
# Set up locale
sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8

# Add ROS 2 repository
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update

# Add ROS 2 GPG key
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add ROS 2 repository to sources list
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Humble
sudo apt update
sudo apt install -y ros-humble-desktop
sudo apt install -y ros-dev-tools

# Install ROS 2 Python packages
pip3 install -U argcomplete
```

### 3. Gazebo Installation
```bash
# Install Gazebo Garden (recommended version for ROS 2 Humble)
sudo apt install -y gz-garden
# Or install the older Gazebo Classic if preferred
sudo apt install -y gazebo libgazebo-dev
```

### 4. Unity Hub Installation (for Digital Twin visualization)
```bash
# Download Unity Hub from official website
# Or install via snap (if available)
sudo snap install unity-hub
```

### 5. NVIDIA Isaac Sim Setup (if GPU available)
```bash
# Install NVIDIA drivers (if not already installed)
sudo apt install -y nvidia-driver-535

# Install Isaac Sim from NVIDIA developer website
# Follow the official installation guide for Ubuntu 22.04
```

### 6. Docusaurus Documentation Setup
```bash
# Install Node.js 18+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install npm packages
npm install -g docusaurus
```

## Book Repository Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/physical-ai-book.git
cd physical-ai-book
```

### 2. Set up ROS 2 Workspace
```bash
# Create ROS 2 workspace
mkdir -p book/ros2-workspace/src
cd book/ros2-workspace

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Build the workspace
colcon build
source install/setup.bash
```

### 3. Install Python Dependencies
```bash
cd /path/to/physical-ai-book
pip3 install -r requirements.txt
```

## Module-Specific Setup

### Module 1: ROS 2 Robotic Nervous System
```bash
# Navigate to the ROS 2 workspace
cd book/ros2-workspace

# Build ROS 2 packages for Module 1
colcon build --packages-select ros2_nervous_system_examples
source install/setup.bash

# Run the first example
ros2 run ros2_nervous_system_examples talker
# In another terminal:
ros2 run ros2_nervous_system_examples listener
```

### Module 2: Digital Twin (Gazebo + Unity)
```bash
# Start Gazebo with humanoid robot model
cd book/gazebo-models
gz sim -r -v 4 simple_humanoid.sdf

# Unity scenes are located in book/unity-scenes/
# Open Unity Hub and import the scenes for visualization
```

### Module 3: AI-Robot Brain (NVIDIA Isaac)
```bash
# Launch Isaac Sim environment
cd /path/to/isaac-sim
./isaac-sim.python.sh --enable-gui

# Run perception pipeline examples
cd book/isaac-configs
# Follow the Isaac ROS examples in the documentation
```

### Module 4: Vision-Language-Action (VLA)
```bash
# Navigate to VLA examples
cd book/vla-examples

# Run the complete VLA pipeline
python3 vla_pipeline.py
```

## Running Lab Exercises

### 1. Module 1 Labs
```bash
# Navigate to Module 1 lab directory
cd docs/modules/ros2-nervous-system/labs

# Follow the instructions in lab1_basic_communication.md
# Example:
cd lab1_basic_communication
python3 solution.py
```

### 2. Module 2 Labs
```bash
# Navigate to Module 2 lab directory
cd docs/modules/digital-twin/labs

# Example lab execution:
cd lab2_gazebo_simulation
bash setup.sh
gz sim -r -v 4 humanoid_world.sdf
```

## Validation

### 1. Environment Validation
```bash
# Check ROS 2 installation
ros2 --version

# Check Gazebo installation
gz --version

# Check Python dependencies
python3 -c "import rclpy; print('ROS 2 Python client OK')"
```

### 2. Lab Completion Validation
Each lab includes a validation script:
```bash
# Example validation for Module 1 lab
cd docs/modules/ros2-nervous-system/labs/lab1_basic_communication
python3 validate_solution.py
```

## Troubleshooting

### Common Issues

1. **ROS 2 packages not found**
   - Ensure you've sourced the ROS 2 environment: `source /opt/ros/humble/setup.bash`
   - Ensure you've sourced your workspace: `source install/setup.bash`

2. **Gazebo not starting**
   - Check graphics drivers: `nvidia-smi` (for NVIDIA) or `glxinfo | grep version`
   - Ensure proper X11 forwarding if running remotely

3. **Isaac Sim GPU errors**
   - Verify CUDA installation: `nvidia-smi` and `nvcc --version`
   - Check Isaac Sim compatibility with your GPU

### Getting Help
- Check the troubleshooting sections in each module
- Review the FAQ in the reference section
- Examine the logs in the `logs/` directory

## Next Steps
1. Start with Module 1: ROS 2 Robotic Nervous System
2. Complete the learning objectives for each module
3. Execute the reproducible lab exercises
4. Validate your understanding with the assessment tools
5. Progress through all 4 modules sequentially