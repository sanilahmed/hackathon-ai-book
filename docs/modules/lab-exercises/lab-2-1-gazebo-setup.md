# Lab 2.1: Gazebo Simulation Environment Setup

## Overview

In this lab, you will install and configure Gazebo Garden, create your first simulation environment, and understand the fundamentals of physics simulation for robotics. You'll learn to create worlds, spawn robots, and interact with the simulation environment.

## Objectives

By the end of this lab, you will be able to:
- Install and configure Gazebo Garden simulation environment
- Create basic simulation worlds with objects and lighting
- Understand Gazebo's physics engine and parameters
- Spawn and control objects in the simulation
- Use Gazebo GUI tools for environment design
- Configure basic sensors in the simulation

## Prerequisites

- Ubuntu 20.04 or 22.04 LTS
- Basic understanding of Linux command line
- Completion of Module 1 (ROS 2 basics)
- Administrative privileges for package installation
- Internet connection for downloading Gazebo packages

## Duration

3-4 hours

## Exercise 1: Install Gazebo Garden

### Step 1: Add Gazebo repository

```bash
# Add Gazebo repository
sudo apt update && sudo apt install wget lsb-release gnupg
wget https://packages.osrfoundation.org/gazebo.gpg -O - | sudo gpg --dearmor -o /usr/share/keyrings/gazebo-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gazebo-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

sudo apt update
```

### Step 2: Install Gazebo Garden

```bash
# Install Gazebo Garden
sudo apt install gz-garden

# Install additional tools
sudo apt install gz-sim7 gz-tools2 python3-gz-sim7-dev

# Install ROS 2 Gazebo bridge
sudo apt install ros-humble-gazebo-ros-pkgs
```

### Step 3: Verify installation

```bash
# Check Gazebo version
gz --version

# Launch Gazebo GUI to verify installation
gz sim
```

## Exercise 2: Create Your First Gazebo World

### Step 1: Create a workspace for Gazebo worlds

```bash
mkdir -p ~/gazebo_ws/worlds
mkdir -p ~/gazebo_ws/models
```

### Step 2: Create a basic world file

Create `~/gazebo_ws/worlds/basic_world.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="basic_world">
    <!-- Include the default sun and ground plane -->
    <include>
      <uri>model://sun</uri>
    </include>

    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Physics engine configuration -->
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Create a simple box object -->
    <model name="box1">
      <pose>1 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
            <specular>1 1 1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Create a sphere object -->
    <model name="sphere1">
      <pose>-1 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <sphere>
              <radius>0.3</radius>
            </sphere>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <sphere>
              <radius>0.3</radius>
            </sphere>
          </geometry>
          <material>
            <ambient>0 1 0 1</ambient>
            <diffuse>0 1 0 1</diffuse>
            <specular>1 1 1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Create a cylinder object -->
    <model name="cylinder1">
      <pose>0 1 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.2</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.2</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0 0 1 1</ambient>
            <diffuse>0 0 1 1</diffuse>
            <specular>1 1 1 1</specular>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

### Step 3: Launch the world

```bash
gz sim ~/gazebo_ws/worlds/basic_world.sdf
```

## Exercise 3: Create a More Complex World with Obstacles

### Step 1: Create a maze world

Create `~/gazebo_ws/worlds/maze_world.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="maze_world">
    <!-- Include the default sun and ground plane -->
    <include>
      <uri>model://sun</uri>
    </include>

    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Physics engine configuration -->
    <physics name="default_physics" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Create walls for a simple maze -->
    <!-- Outer walls -->
    <model name="wall_1">
      <pose>0 -5 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.1 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.1 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
            <specular>0.5 0.5 0.5 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <model name="wall_2">
      <pose>0 5 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.1 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.1 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
            <specular>0.5 0.5 0.5 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <model name="wall_3">
      <pose>-5 0 1 0 0 1.5708</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.1 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.1 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
            <specular>0.5 0.5 0.5 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <model name="wall_4">
      <pose>5 0 1 0 0 1.5708</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.1 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.1 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
            <specular>0.5 0.5 0.5 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Inner maze walls -->
    <model name="maze_wall_1">
      <pose>-2 0 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 4 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 4 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
            <diffuse>0.3 0.3 0.3 1</diffuse>
            <specular>0.3 0.3 0.3 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <model name="maze_wall_2">
      <pose>2 2 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>4 0.1 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>4 0.1 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
            <diffuse>0.3 0.3 0.3 1</diffuse>
            <specular>0.3 0.3 0.3 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Create a target object -->
    <model name="target">
      <pose>4 4 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.2</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.2</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>1 1 0 1</ambient>
            <diffuse>1 1 0 1</diffuse>
            <specular>1 1 0 1</specular>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

### Step 2: Launch the maze world

```bash
gz sim ~/gazebo_ws/worlds/maze_world.sdf
```

## Exercise 4: Create a Custom Model

### Step 1: Create a simple robot model

Create the model directory structure:

```bash
mkdir -p ~/gazebo_ws/models/simple_robot/meshes
mkdir -p ~/gazebo_ws/models/simple_robot/materials/textures
mkdir -p ~/gazebo_ws/models/simple_robot/materials/scripts
```

### Step 2: Create the model configuration file

Create `~/gazebo_ws/models/simple_robot/model.config`:

```xml
<?xml version="1.0"?>
<model>
  <name>simple_robot</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>
  <description>A simple wheeled robot for Gazebo simulation.</description>
</model>
```

### Step 3: Create the robot model SDF

Create `~/gazebo_ws/models/simple_robot/model.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="simple_robot">
    <link name="chassis">
      <pose>0 0 0.1 0 0 0</pose>
      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.2</iyy>
          <iyz>0.0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>

      <collision name="collision">
        <geometry>
          <box>
            <size>0.5 0.3 0.2</size>
          </box>
        </geometry>
      </collision>

      <visual name="visual">
        <geometry>
          <box>
            <size>0.5 0.3 0.2</size>
          </box>
        </geometry>
        <material>
          <ambient>0.0 0.8 0.0 1.0</ambient>
          <diffuse>0.0 0.8 0.0 1.0</diffuse>
          <specular>0.0 0.8 0.0 1.0</specular>
        </material>
      </visual>
    </link>

    <!-- Left wheel -->
    <link name="left_wheel">
      <pose>0.15 0.15 -0.1 0 0 0</pose>
      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.001</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.001</iyy>
          <iyz>0.0</iyz>
          <izz>0.002</izz>
        </inertia>
      </inertial>

      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
      </collision>

      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.2 1.0</ambient>
          <diffuse>0.2 0.2 0.2 1.0</diffuse>
          <specular>0.2 0.2 0.2 1.0</specular>
        </material>
      </visual>
    </link>

    <!-- Right wheel -->
    <link name="right_wheel">
      <pose>0.15 -0.15 -0.1 0 0 0</pose>
      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.001</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.001</iyy>
          <iyz>0.0</iyz>
          <izz>0.002</izz>
        </inertia>
      </inertial>

      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
      </collision>

      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.2 1.0</ambient>
          <diffuse>0.2 0.2 0.2 1.0</diffuse>
          <specular>0.2 0.2 0.2 1.0</specular>
        </material>
      </visual>
    </link>

    <!-- Camera -->
    <link name="camera_link">
      <pose>0.2 0 0.05 0 0 0</pose>
      <inertial>
        <mass>0.1</mass>
        <inertia>
          <ixx>0.0001</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.0001</iyy>
          <iyz>0.0</iyz>
          <izz>0.0001</izz>
        </inertia>
      </inertial>

      <visual name="visual">
        <geometry>
          <box>
            <size>0.05 0.05 0.05</size>
          </box>
        </geometry>
        <material>
          <ambient>1.0 0.0 0.0 1.0</ambient>
          <diffuse>1.0 0.0 0.0 1.0</diffuse>
          <specular>1.0 0.0 0.0 1.0</specular>
        </material>
      </visual>
    </link>

    <!-- Joints -->
    <joint name="left_wheel_joint" type="revolute">
      <parent>chassis</parent>
      <child>left_wheel</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1e+16</lower>
          <upper>1e+16</upper>
          <effort>-1</effort>
          <velocity>-1</velocity>
        </limit>
      </axis>
      <pose>0.15 0.15 -0.1 0 0 0</pose>
    </joint>

    <joint name="right_wheel_joint" type="revolute">
      <parent>chassis</parent>
      <child>right_wheel</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1e+16</lower>
          <upper>1e+16</upper>
          <effort>-1</effort>
          <velocity>-1</velocity>
        </limit>
      </axis>
      <pose>0.15 -0.15 -0.1 0 0 0</pose>
    </joint>

    <joint name="camera_joint" type="fixed">
      <parent>chassis</parent>
      <child>camera_link</child>
      <pose>0.2 0 0.05 0 0 0</pose>
    </joint>
  </model>
</sdf>
```

## Exercise 5: Test the Custom Model

### Step 1: Set GAZEBO_MODEL_PATH

```bash
export GAZEBO_MODEL_PATH=~/gazebo_ws/models:$GAZEBO_MODEL_PATH
```

### Step 2: Create a world with the custom robot

Create `~/gazebo_ws/worlds/robot_world.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="robot_world">
    <!-- Include the default sun and ground plane -->
    <include>
      <uri>model://sun</uri>
    </include>

    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Physics engine configuration -->
    <physics name="default_physics" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Spawn our custom robot -->
    <include>
      <uri>model://simple_robot</uri>
      <pose>0 0 0.2 0 0 0</pose>
    </include>

    <!-- Add some obstacles for the robot to navigate around -->
    <model name="obstacle_1">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.5 0.5 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.5 0.5 1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 1 1</ambient>
            <diffuse>1 0 1 1</diffuse>
            <specular>1 1 1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <model name="obstacle_2">
      <pose>-2 1 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>1</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>1</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>1 0.5 0 1</ambient>
            <diffuse>1 0.5 0 1</diffuse>
            <specular>1 1 1 1</specular>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

### Step 3: Launch the robot world

```bash
gz sim ~/gazebo_ws/worlds/robot_world.sdf
```

## Exercise 6: Physics Configuration and Parameters

### Step 1: Create a world with different physics parameters

Create `~/gazebo_ws/worlds/physics_test.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="physics_test">
    <!-- Include the default sun and ground plane -->
    <include>
      <uri>model://sun</uri>
    </include>

    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Physics engine with custom parameters -->
    <physics name="custom_physics" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>

      <!-- Solver parameters -->
      <solver>
        <type>quick</type>
        <iters>10</iters>
        <sor>1.3</sor>
      </solver>

      <!-- Surface parameters -->
      <surface>
        <friction>
          <ode>
            <mu>1.0</mu>
            <mu2>1.0</mu2>
          </ode>
        </friction>
        <bounce>
          <restitution_coefficient>0.1</restitution_coefficient>
          <threshold>100000</threshold>
        </bounce>
        <contact>
          <ode>
            <soft_cfm>0</soft_cfm>
            <soft_erp>0.2</soft_erp>
            <kp>1e+13</kp>
            <kd>1</kd>
            <max_vel>100.0</max_vel>
            <min_depth>0.001</min_depth>
          </ode>
        </contact>
      </surface>
    </physics>

    <!-- Create objects with different properties -->
    <model name="bouncy_sphere">
      <pose>0 0 5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <sphere>
              <radius>0.2</radius>
            </sphere>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <sphere>
              <radius>0.2</radius>
            </sphere>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
            <specular>1 1 1 1</specular>
          </material>
        </visual>
        <surface>
          <bounce>
            <restitution_coefficient>0.8</restitution_coefficient>
          </bounce>
        </surface>
      </link>
    </model>

    <model name="friction_sphere">
      <pose>1 0 5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <sphere>
              <radius>0.2</radius>
            </sphere>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <sphere>
              <radius>0.2</radius>
            </sphere>
          </geometry>
          <material>
            <ambient>0 1 0 1</ambient>
            <diffuse>0 1 0 1</diffuse>
            <specular>1 1 1 1</specular>
          </material>
        </visual>
        <surface>
          <friction>
            <ode>
              <mu>5.0</mu>
              <mu2>5.0</mu2>
            </ode>
          </friction>
        </surface>
      </link>
    </model>
  </world>
</sdf>
```

### Step 2: Launch the physics test world

```bash
gz sim ~/gazebo_ws/worlds/physics_test.sdf
```

## Exercise 7: Using Gazebo GUI Tools

### Step 1: Explore the Gazebo GUI

When Gazebo is running, familiarize yourself with:
- **Scene Tree**: Shows all objects in the world
- **Tools**: Various tools for editing and interaction
- **View Controls**: Camera navigation and manipulation
- **Play/Pause**: Control simulation time
- **Step**: Single-step through simulation

### Step 2: Use the Insert Tool

1. Click the "Insert" tool in the toolbar
2. Browse available models (box, sphere, cylinder, etc.)
3. Click to place models in the world
4. Use the move/rotate tools to adjust positions

### Step 3: Use the Entity Tree

1. Open the Entity Tree panel
2. Select different objects to see their properties
3. Modify properties like color, size, or position
4. Add/remove objects as needed

## Exercise 8: Environment Variables and Configuration

### Step 1: Set up environment variables

Create a script to set up Gazebo environment:

Create `~/gazebo_ws/setup_gazebo.sh`:

```bash
#!/bin/bash
# Setup script for Gazebo environment

# Set model path to include our custom models
export GAZEBO_MODEL_PATH=~/gazebo_ws/models:$GAZEBO_MODEL_PATH

# Set resource path to include our worlds
export GAZEBO_RESOURCE_PATH=~/gazebo_ws:$GAZEBO_RESOURCE_PATH

# Set plugin path
export GAZEBO_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gazebo-11/plugins:$GAZEBO_PLUGIN_PATH

# Set media path for textures and materials
export GAZEBO_MEDIA_PATH=/usr/share/gazebo-11/media:$GAZEBO_MEDIA_PATH

echo "Gazebo environment configured:"
echo "GAZEBO_MODEL_PATH: $GAZEBO_MODEL_PATH"
echo "GAZEBO_RESOURCE_PATH: $GAZEBO_RESOURCE_PATH"
```

Make it executable:

```bash
chmod +x ~/gazebo_ws/setup_gazebo.sh
```

### Step 2: Use the setup script

```bash
source ~/gazebo_ws/setup_gazebo.sh
```

## Exercise 9: Command Line Tools

### Step 1: Use gz command line tools

```bash
# List available worlds
gz sdf --list

# Validate an SDF file
gz sdf --check ~/gazebo_ws/worlds/basic_world.sdf

# List running Gazebo instances
gz topic -l

# Echo simulation time
gz topic -e /clock
```

### Step 2: Create a launch script

Create `~/gazebo_ws/launch_basic_world.sh`:

```bash
#!/bin/bash
# Launch script for basic world

# Source Gazebo environment
source ~/gazebo_ws/setup_gazebo.sh

# Launch Gazebo with our world
gz sim ~/gazebo_ws/worlds/basic_world.sdf -g
```

## Troubleshooting

### Common Issues and Solutions

1. **Gazebo fails to start with graphics errors**:
   - Check graphics drivers and OpenGL support
   - Try running with software rendering: `LIBGL_ALWAYS_SOFTWARE=1 gz sim`
   - Ensure proper X11 forwarding if running remotely

2. **Models not appearing in Gazebo**:
   - Verify GAZEBO_MODEL_PATH includes your model directory
   - Check model folder structure (model.sdf, model.config)
   - Ensure model.config has correct name and SDF reference

3. **Physics simulation is unstable**:
   - Adjust max_step_size to smaller values (e.g., 0.001)
   - Modify solver parameters (iterations, SOR)
   - Check mass and inertia values in models

4. **Simulation runs too slowly**:
   - Increase max_step_size (trade accuracy for speed)
   - Reduce real_time_update_rate
   - Simplify collision geometries

5. **SDF validation errors**:
   - Use `gz sdf --check <file.sdf>` to validate
   - Check XML syntax and SDF schema compliance
   - Ensure all required elements are present

## Assessment Questions

1. What is the purpose of the physics element in SDF files?
2. How do you create a custom model in Gazebo?
3. What are the differences between collision and visual elements?
4. How can you adjust simulation parameters for better performance?
5. What is the role of the GAZEBO_MODEL_PATH environment variable?

## Extension Exercises

1. Create a world with multiple robots that can interact with each other
2. Add lighting effects and environment maps to your world
3. Implement a simple sensor (camera, LiDAR) in your robot model
4. Create a complex environment with ramps, stairs, or uneven terrain
5. Experiment with different physics engines and parameters

## Summary

In this lab, you successfully:
- Installed and configured Gazebo Garden simulation environment
- Created basic and complex simulation worlds
- Developed custom robot models in SDF format
- Configured physics parameters for different simulation scenarios
- Used Gazebo GUI tools for environment design
- Set up environment variables for custom models and worlds

These foundational skills are essential for creating realistic simulation environments for robotics development. The ability to create detailed worlds and accurate robot models in simulation is crucial for testing algorithms, validating control systems, and developing AI behaviors before deployment on physical robots.