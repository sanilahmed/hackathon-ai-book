---
sidebar_label: 'Lab 1.1: ROS 2 Setup'
---

# Lab Exercise 1.1: ROS 2 Setup

This lab exercise guides you through setting up ROS 2 on your system.

## Objectives

- Install ROS 2 Humble Hawksbill
- Verify the installation
- Set up the development environment
- Create your first workspace

## System Requirements

- Ubuntu 22.04 (Jammy Jellyfish) or Windows 10/11 with WSL2
- At least 4GB RAM (8GB recommended)
- At least 10GB free disk space

## Installation on Ubuntu

1. Set locale:
   ```bash
   locale  # check for UTF-8
   sudo apt update && sudo apt install locales
   sudo locale-gen en_US en_US.UTF-8
   sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
   export LANG=en_US.UTF-8
   ```

2. Add the ROS 2 apt repository:
   ```bash
   sudo apt update && sudo apt install curl gnupg lsb-release
   sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
   ```

3. Install ROS 2 packages:
   ```bash
   sudo apt update
   sudo apt install ros-humble-desktop
   ```

4. Install colcon build tools:
   ```bash
   sudo apt install python3-colcon-common-extensions
   ```

## Environment Setup

1. Source the ROS 2 installation:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. Add to your bashrc to automatically source:
   ```bash
   echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
   ```

## Verification

1. Test the installation:
   ```bash
   ros2 run demo_nodes_cpp talker
   ```

2. In another terminal:
   ```bash
   source /opt/ros/humble/setup.bash
   ros2 run demo_nodes_py listener
   ```

You should see the talker publishing messages and the listener receiving them.

## Creating a Workspace

1. Create the workspace directory:
   ```bash
   mkdir -p ~/ros2_ws/src
   ```

2. Navigate to the workspace:
   ```bash
   cd ~/ros2_ws
   ```

3. Build the workspace:
   ```bash
   colcon build
   ```

4. Source the workspace:
   ```bash
   source install/setup.bash
   ```

## Troubleshooting

- If you get permission errors, make sure your user is in the dialout group
- If packages can't be found, verify your apt repository setup
- For WSL2 users, ensure you're using the Linux installation instructions

## Summary

You have successfully installed ROS 2 Humble Hawksbill and created your first workspace. You're now ready to start developing ROS 2 packages.