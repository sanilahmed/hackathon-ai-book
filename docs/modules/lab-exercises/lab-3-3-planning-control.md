# Lab 3.3: Isaac Sim Planning and Control Systems

## Overview

In this lab, you will learn how to implement planning and control systems in Isaac Sim for robotics applications. You'll work with path planning algorithms, trajectory generation, motion control, and integration with Isaac Sim's physics engine. This includes understanding the control architecture, implementing various control strategies, and integrating with ROS for navigation and manipulation tasks.

## Objectives

By the end of this lab, you will be able to:
- Implement path planning algorithms in Isaac Sim
- Create trajectory generators for robot motion
- Implement motion control systems using Isaac Sim's control interface
- Integrate planning and control with ROS navigation stack
- Implement feedback control for robot motion
- Create behavior trees for complex robot behaviors
- Validate control performance and stability

## Prerequisites

- Completion of Lab 3.1: Isaac Sim Setup and Environment
- Completion of Lab 3.2: Isaac Sim Perception Systems
- Understanding of ROS navigation stack
- Basic knowledge of control theory
- Experience with Isaac Sim and sensor integration

## Duration

5-6 hours

## Exercise 1: Path Planning and Navigation

### Step 1: Create a basic navigation environment

Create `~/isaac_sim_examples/navigation_environment.py`:

```python
#!/usr/bin/env python3
# navigation_environment.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.robots import Robot
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.semantics import add_semantic_data_to_stage
import numpy as np
import carb

class NavigationEnvironment:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.robot = None
        self.setup_environment()

    def setup_environment(self):
        """Set up navigation environment with obstacles."""
        # Create ground plane
        create_primitive(
            prim_path="/World/GroundPlane",
            primitive_type="Plane",
            scale=[20, 20, 1],
            position=[0, 0, 0]
        )

        # Create walls to form corridors
        wall_thickness = 0.2
        wall_height = 2.0

        # Outer walls
        create_primitive(
            prim_path="/World/Wall_North",
            primitive_type="Cuboid",
            scale=[20, wall_thickness, wall_height],
            position=[0, 10, wall_height/2],
            color=[0.5, 0.5, 0.5]
        )

        create_primitive(
            prim_path="/World/Wall_South",
            primitive_type="Cuboid",
            scale=[20, wall_thickness, wall_height],
            position=[0, -10, wall_height/2],
            color=[0.5, 0.5, 0.5]
        )

        create_primitive(
            prim_path="/World/Wall_East",
            primitive_type="Cuboid",
            scale=[wall_thickness, 20, wall_height],
            position=[10, 0, wall_height/2],
            color=[0.5, 0.5, 0.5]
        )

        create_primitive(
            prim_path="/World/Wall_West",
            primitive_type="Cuboid",
            scale=[wall_thickness, 20, wall_height],
            position=[-10, 0, wall_height/2],
            color=[0.5, 0.5, 0.5]
        )

        # Create obstacles to form navigation challenges
        # Create a corridor with obstacles
        for i in range(5):
            # Left column of obstacles
            create_primitive(
                prim_path=f"/World/Obstacle_Left_{i}",
                primitive_type="Cylinder",
                scale=[0.4, 0.4, 1.0],
                position=[-6, 2*i - 4, 0.5],
                color=[0.8, 0.2, 0.2]
            )

            # Right column of obstacles
            create_primitive(
                prim_path=f"/World/Obstacle_Right_{i}",
                primitive_type="Cylinder",
                scale=[0.4, 0.4, 1.0],
                position=[6, 2*i - 4, 0.5],
                color=[0.2, 0.8, 0.2]
            )

        # Create start and goal markers
        create_primitive(
            prim_path="/World/Start",
            primitive_type="Sphere",
            scale=[0.3, 0.3, 0.3],
            position=[-8, 0, 0.15],
            color=[0, 1, 0]  # Green for start
        )

        create_primitive(
            prim_path="/World/Goal",
            primitive_type="Sphere",
            scale=[0.3, 0.3, 0.3],
            position=[8, 0, 0.15],
            color=[1, 0, 0]  # Red for goal
        )

        # Create a simple robot
        self.robot = create_primitive(
            prim_path="/World/Robot",
            primitive_type="Cylinder",
            scale=[0.5, 0.5, 1.0],
            position=[-8, 0, 0.5],
            color=[0.1, 0.1, 0.8]  # Blue robot
        )

    def get_robot_position(self):
        """Get current robot position."""
        if self.robot:
            from pxr import UsdGeom
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath("/World/Robot")
            xform = UsdGeom.Xformable(prim)
            transform = xform.ComputeLocalToWorldTransform(omni.usd.get_context().get_time_code())
            pos = transform.ExtractTranslation()
            return np.array([pos[0], pos[1], pos[2]])
        return None

    def move_robot(self, target_position):
        """Move robot to target position (simplified for demo)."""
        # In a real implementation, this would use proper kinematics and control
        if self.robot:
            from pxr import Gf, UsdGeom
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath("/World/Robot")
            xform = UsdGeom.Xformable(prim)

            # Set new position
            new_transform = xform.GetLocalTransformation()
            new_transform.SetTranslate(Gf.Vec3d(target_position[0], target_position[1], target_position[2]))
            xform.SetLocalTransformation(new_transform)

    def run_simulation(self, steps=1000):
        """Run navigation simulation."""
        self.world.reset()

        # Set camera view
        set_camera_view(eye=[15, 15, 15], target=[0, 0, 0])

        for i in range(steps):
            self.world.step(render=True)

            # Simple navigation logic (for demonstration)
            if i % 100 == 0:
                current_pos = self.get_robot_position()
                if current_pos is not None:
                    print(f"Robot position: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}]")

        print("Navigation environment simulation completed")

# Create and run the navigation environment
if __name__ == "__main__":
    nav_env = NavigationEnvironment()
    nav_env.run_simulation(steps=500)
```

### Step 2: Implement A* path planning algorithm

Create `~/isaac_sim_examples/path_planning.py`:

```python
#!/usr/bin/env python3
# path_planning.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.viewports import set_camera_view
import numpy as np
import heapq
from scipy.spatial.distance import euclidean
import carb

class PathPlanner:
    def __init__(self, world_size=20, resolution=0.5):
        self.world_size = world_size
        self.resolution = resolution
        self.grid_size = int(world_size / resolution)
        self.grid = np.zeros((self.grid_size, self.grid_size))

        # Initialize the world for path planning
        self.world = World(stage_units_in_meters=1.0)
        self.setup_environment()

    def setup_environment(self):
        """Set up environment with obstacles for path planning."""
        # Create ground plane
        create_primitive(
            prim_path="/World/GroundPlane",
            primitive_type="Plane",
            scale=[self.world_size, self.world_size, 1],
            position=[0, 0, 0]
        )

        # Create obstacles (these will be marked in the grid)
        obstacle_positions = [
            [5, 5], [5, 6], [5, 7],  # Column obstacle
            [-5, -5], [-5, -6], [-5, -7],  # Column obstacle
            [2, 2], [2, -2], [-2, 2], [-2, -2],  # Square obstacles
            [0, 5], [1, 5], [-1, 5], [0, 6],  # Central obstacle
        ]

        self.obstacles = []
        for i, pos in enumerate(obstacle_positions):
            obstacle = create_primitive(
                prim_path=f"/World/Obstacle_{i}",
                primitive_type="Cylinder",
                scale=[0.4, 0.4, 1.0],
                position=[pos[0], pos[1], 0.5],
                color=[0.8, 0.2, 0.2]
            )
            self.obstacles.append(obstacle)

            # Mark obstacle in grid
            grid_x = int((pos[0] + self.world_size/2) / self.resolution)
            grid_y = int((pos[1] + self.world_size/2) / self.resolution)

            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.grid[grid_y, grid_x] = 1  # Occupied

        # Create start and goal markers
        self.start_marker = create_primitive(
            prim_path="/World/Start",
            primitive_type="Sphere",
            scale=[0.3, 0.3, 0.3],
            position=[-8, -8, 0.15],
            color=[0, 1, 0]
        )

        self.goal_marker = create_primitive(
            prim_path="/World/Goal",
            primitive_type="Sphere",
            scale=[0.3, 0.3, 0.3],
            position=[8, 8, 0.15],
            color=[1, 0, 0]
        )

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates."""
        grid_x = int((x + self.world_size/2) / self.resolution)
        grid_y = int((y + self.world_size/2) / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates."""
        x = grid_x * self.resolution - self.world_size/2 + self.resolution/2
        y = grid_y * self.resolution - self.world_size/2 + self.resolution/2
        return x, y

    def get_neighbors(self, pos):
        """Get valid neighboring cells."""
        x, y = pos
        neighbors = []

        # 8-directional movement (including diagonals)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check bounds
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                # Check if not occupied
                if self.grid[ny, nx] == 0:
                    neighbors.append((nx, ny))

        return neighbors

    def heuristic(self, a, b):
        """Heuristic function for A* (Euclidean distance)."""
        return euclidean(a, b)

    def a_star(self, start, goal):
        """A* pathfinding algorithm."""
        start = self.world_to_grid(start[0], start[1])
        goal = self.world_to_grid(goal[0], goal[1])

        # Convert to grid coordinates
        if start[0] < 0 or start[0] >= self.grid_size or start[1] < 0 or start[1] >= self.grid_size:
            print("Start position outside grid")
            return None

        if goal[0] < 0 or goal[0] >= self.grid_size or goal[1] < 0 or goal[1] >= self.grid_size:
            print("Goal position outside grid")
            return None

        if self.grid[start[1], start[0]] == 1:
            print("Start position is occupied")
            return None

        if self.grid[goal[1], goal[0]] == 1:
            print("Goal position is occupied")
            return None

        # Initialize A*
        frontier = [(0, start)]
        came_from = {}
        cost_so_far = {start: 0}
        priority = {start: 0}

        while frontier:
            current = heapq.heappop(frontier)[1]

            if current == goal:
                break

            for next_pos in self.get_neighbors(current):
                new_cost = cost_so_far[current] + euclidean(current, next_pos)

                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current

        # Reconstruct path
        if goal not in came_from:
            return None  # No path found

        path = []
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()

        # Convert path back to world coordinates
        world_path = []
        for grid_pos in path:
            world_x, world_y = self.grid_to_world(grid_pos[0], grid_pos[1])
            world_path.append([world_x, world_y, 0.5])  # Add height

        return world_path

    def visualize_path(self, path):
        """Create visualization of the path."""
        if path is None:
            print("No path to visualize")
            return

        # Create path markers
        for i, pos in enumerate(path):
            create_primitive(
                prim_path=f"/World/PathMarker_{i}",
                primitive_type="Sphere",
                scale=[0.1, 0.1, 0.1],
                position=[pos[0], pos[1], pos[2]],
                color=[0, 0, 1]  # Blue path markers
            )

    def run_path_planning(self):
        """Run path planning demonstration."""
        self.world.reset()

        # Set camera view
        set_camera_view(eye=[15, 15, 15], target=[0, 0, 0])

        # Plan path
        start_pos = [-8, -8, 0.5]
        goal_pos = [8, 8, 0.5]

        print(f"Planning path from {start_pos[:2]} to {goal_pos[:2]}")
        path = self.a_star(start_pos, goal_pos)

        if path:
            print(f"Found path with {len(path)} waypoints")
            self.visualize_path(path)
        else:
            print("No path found")

        # Run simulation briefly to show results
        for i in range(200):
            self.world.step(render=True)

        print("Path planning demonstration completed")

# Create and run the path planner
if __name__ == "__main__":
    planner = PathPlanner()
    planner.run_path_planning()
```

## Exercise 2: Motion Control Systems

### Step 1: Create a motion control system

Create `~/isaac_sim_examples/motion_control.py`:

```python
#!/usr/bin/env python3
# motion_control.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
import numpy as np
import carb
from scipy.spatial.transform import Rotation as R

class DifferentialDriveController(BaseController):
    """Simple differential drive controller for wheeled robot."""

    def __init__(
        self,
        name: str = "diff_drive_controller",
        wheel_radius: float = 0.1,
        wheel_base: float = 0.4
    ) -> None:
        super().__init__(name=name)
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base

        # Control parameters
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        # PID controller parameters
        self.kp = 1.0
        self.ki = 0.1
        self.kd = 0.05

        # For PID control
        self.prev_error = 0.0
        self.integral = 0.0

    def forward(
        self,
        current_joint_positions,
        current_joint_velocities,
        target_positions,
        target_velocities,
        dt
    ):
        """Calculate joint commands based on target velocities."""
        # Convert linear/angular velocities to wheel velocities
        left_wheel_vel = (self.linear_velocity - self.angular_velocity * self.wheel_base / 2) / self.wheel_radius
        right_wheel_vel = (self.linear_velocity + self.angular_velocity * self.wheel_base / 2) / self.wheel_radius

        # Return joint position and velocity commands
        return {
            "position": target_positions,
            "velocity": [left_wheel_vel, right_wheel_vel]
        }

class MotionControlSystem:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.robot = None
        self.controller = None
        self.setup_environment()

    def setup_environment(self):
        """Set up environment for motion control."""
        # Create ground plane
        create_primitive(
            prim_path="/World/GroundPlane",
            primitive_type="Plane",
            scale=[20, 20, 1],
            position=[0, 0, 0]
        )

        # Create a simple wheeled robot
        self.robot = create_primitive(
            prim_path="/World/Robot",
            primitive_type="Cylinder",
            scale=[0.5, 0.5, 0.8],
            position=[0, 0, 0.4],
            color=[0.2, 0.6, 1.0]
        )

        # Create wheels
        self.left_wheel = create_primitive(
            prim_path="/World/Robot/LeftWheel",
            primitive_type="Cylinder",
            scale=[0.15, 0.15, 0.05],
            position=[0.2, 0.25, 0.1],
            color=[0.3, 0.3, 0.3]
        )

        self.right_wheel = create_primitive(
            prim_path="/World/Robot/RightWheel",
            primitive_type="Cylinder",
            scale=[0.15, 0.15, 0.05],
            position=[0.2, -0.25, 0.1],
            color=[0.3, 0.3, 0.3]
        )

        # Create reference frames for navigation
        for i in range(10):
            create_primitive(
                prim_path=f"/World/Reference_{i}",
                primitive_type="Cylinder",
                scale=[0.2, 0.2, 0.5],
                position=[np.random.uniform(-8, 8), np.random.uniform(-8, 8), 0.25],
                color=[1, 0.5, 0]
            )

        # Create controller
        self.controller = DifferentialDriveController(
            name="diff_drive_controller",
            wheel_radius=0.1,
            wheel_base=0.5
        )

    def simple_navigation(self):
        """Implement simple navigation behavior."""
        # Get robot position
        robot_pos = self.get_robot_position()
        if robot_pos is None:
            return [0, 0]  # Stop if no position

        # Define a simple navigation pattern (square path)
        target_points = [
            [5, 0, 0.4],
            [5, 5, 0.4],
            [0, 5, 0.4],
            [0, 0, 0.4]
        ]

        # Determine current target based on time
        current_target_idx = int(self.world.current_time_step_index / 300) % len(target_points)
        target = target_points[current_target_idx]

        # Calculate desired velocity toward target
        direction = np.array(target[:2]) - robot_pos[:2]
        distance = np.linalg.norm(direction)

        if distance > 0.5:  # If not close to target
            # Normalize direction
            direction = direction / distance

            # Calculate linear and angular velocities
            linear_vel = min(distance * 0.5, 1.0)  # Scale with distance, max 1 m/s
            angular_vel = np.arctan2(direction[1], direction[0]) * 0.5  # Turn toward target

            return [linear_vel, angular_vel]
        else:
            # Close to target, slow down
            return [0.2, 0.0]

    def get_robot_position(self):
        """Get current robot position."""
        if self.robot:
            from pxr import UsdGeom
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath("/World/Robot")
            xform = UsdGeom.Xformable(prim)
            transform = xform.ComputeLocalToWorldTransform(omni.usd.get_context().get_time_code())
            pos = transform.ExtractTranslation()
            return np.array([pos[0], pos[1], pos[2]])
        return None

    def run_motion_control(self, steps=1500):
        """Run motion control simulation."""
        self.world.reset()

        # Set camera view
        set_camera_view(eye=[10, 10, 10], target=[0, 0, 0])

        for i in range(steps):
            self.world.step(render=True)

            # Calculate navigation commands every 10 steps
            if i % 10 == 0:
                commands = self.simple_navigation()

                # Apply commands to controller
                self.controller.linear_velocity = commands[0]
                self.controller.angular_velocity = commands[1]

                # Print status
                pos = self.get_robot_position()
                if pos is not None:
                    print(f"Step {i}: Pos=({pos[0]:.2f}, {pos[1]:.2f}), Cmds=({commands[0]:.2f}, {commands[1]:.2f})")

        print("Motion control simulation completed")

# Create and run the motion control system
if __name__ == "__main__":
    motion_ctrl = MotionControlSystem()
    motion_ctrl.run_motion_control(steps=1000)
```

## Exercise 3: Advanced Control Systems

### Step 1: Implement PID controller with feedback

Create `~/isaac_sim_examples/pid_control.py`:

```python
#!/usr/bin/env python3
# pid_control.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.viewports import set_camera_view
import numpy as np
import carb

class PIDController:
    """PID controller implementation."""

    def __init__(self, kp=1.0, ki=0.1, kd=0.05, setpoint=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint

        self.previous_error = 0.0
        self.integral = 0.0
        self.derivative = 0.0

    def compute(self, current_value, dt=0.01):
        """Compute control output."""
        error = self.setpoint - current_value

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term
        if dt > 0:
            self.derivative = (error - self.previous_error) / dt
        d_term = self.kd * self.derivative

        # Store error for next iteration
        self.previous_error = error

        # Calculate output
        output = p_term + i_term + d_term

        return output

class AdvancedControlSystem:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.robots = []
        self.controllers = []
        self.setup_environment()

    def setup_environment(self):
        """Set up environment with multiple robots for control testing."""
        # Create ground plane
        create_primitive(
            prim_path="/World/GroundPlane",
            primitive_type="Plane",
            scale=[30, 30, 1],
            position=[0, 0, 0]
        )

        # Create multiple robots with different control tasks
        robot_configs = [
            {'name': 'Robot1', 'start_pos': [-8, -8, 0.5], 'target': [8, 8, 0.5]},
            {'name': 'Robot2', 'start_pos': [8, -8, 0.5], 'target': [-8, 8, 0.5]},
            {'name': 'Robot3', 'start_pos': [0, 8, 0.5], 'target': [0, -8, 0.5]},
        ]

        for config in robot_configs:
            robot = create_primitive(
                prim_path=f"/World/{config['name']}",
                primitive_type="Cylinder",
                scale=[0.4, 0.4, 0.8],
                position=config['start_pos'],
                color=[0.2, 0.8, 0.2] if '1' in config['name'] else [0.8, 0.2, 0.2] if '2' in config['name'] else [0.2, 0.2, 0.8]
            )
            self.robots.append({'robot': robot, 'config': config})

            # Create PID controllers for X and Y positions
            x_controller = PIDController(kp=2.0, ki=0.1, kd=0.05, setpoint=config['target'][0])
            y_controller = PIDController(kp=2.0, ki=0.1, kd=0.05, setpoint=config['target'][1])
            self.controllers.append({'x': x_controller, 'y': y_controller, 'robot': robot})

        # Create obstacles
        for i in range(15):
            create_primitive(
                prim_path=f"/World/Obstacle_{i}",
                primitive_type="Cylinder",
                scale=[0.3, 0.3, 1.0],
                position=[np.random.uniform(-10, 10), np.random.uniform(-10, 10), 0.5],
                color=[0.5, 0.5, 0.5]
            )

        # Create target markers
        for i, config in enumerate(robot_configs):
            create_primitive(
                prim_path=f"/World/Target_{i}",
                primitive_type="Sphere",
                scale=[0.3, 0.3, 0.3],
                position=config['target'],
                color=[1, 1, 0]  # Yellow targets
            )

    def get_robot_position(self, robot_prim):
        """Get robot position."""
        if robot_prim:
            from pxr import UsdGeom
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(str(robot_prim.GetPrimPath()))
            xform = UsdGeom.Xformable(prim)
            transform = xform.ComputeLocalToWorldTransform(omni.usd.get_context().get_time_code())
            pos = transform.ExtractTranslation()
            return np.array([pos[0], pos[1], pos[2]])
        return None

    def update_robot_position(self, robot_prim, new_pos):
        """Update robot position."""
        if robot_prim:
            from pxr import Gf, UsdGeom
            stage = omni.usd.get_context().get_stage()
            prim_path = str(robot_prim.GetPrimPath())
            prim = stage.GetPrimAtPath(prim_path)
            xform = UsdGeom.Xformable(prim)

            # Get current transform and update position
            transform = xform.GetLocalTransformation()
            transform.SetTranslate(Gf.Vec3d(new_pos[0], new_pos[1], new_pos[2]))
            xform.SetLocalTransformation(transform)

    def run_pid_control(self, steps=2000):
        """Run PID control simulation."""
        self.world.reset()

        # Set camera view
        set_camera_view(eye=[20, 20, 20], target=[0, 0, 0])

        for i in range(steps):
            self.world.step(render=True)

            # Update robot positions using PID control every 5 steps
            if i % 5 == 0:
                dt = 0.005  # Assuming 200Hz control rate

                for j, (robot_info, controller_info) in enumerate(zip(self.robots, self.controllers)):
                    current_pos = self.get_robot_position(robot_info['robot'])

                    if current_pos is not None:
                        # Get current position
                        current_x, current_y, current_z = current_pos

                        # Update PID setpoints
                        target_x = robot_info['config']['target'][0]
                        target_y = robot_info['config']['target'][1]

                        controller_info['x'].setpoint = target_x
                        controller_info['y'].setpoint = target_y

                        # Compute control outputs
                        x_output = controller_info['x'].compute(current_x, dt)
                        y_output = controller_info['y'].compute(current_y, dt)

                        # Calculate new position (simple integration)
                        new_x = current_x + x_output * dt
                        new_y = current_y + y_output * dt
                        new_z = current_z

                        # Apply position update
                        self.update_robot_position(robot_info['robot'], [new_x, new_y, new_z])

                        # Print status every 100 steps
                        if i % 100 == 0:
                            distance_to_target = np.sqrt((new_x - target_x)**2 + (new_y - target_y)**2)
                            print(f"Robot {j+1}: Pos=({new_x:.2f}, {new_y:.2f}), "
                                  f"Target=({target_x:.2f}, {target_y:.2f}), "
                                  f"Distance={distance_to_target:.2f}m")

        print("PID control simulation completed")

# Create and run the advanced control system
if __name__ == "__main__":
    advanced_ctrl = AdvancedControlSystem()
    advanced_ctrl.run_pid_control(steps=1500)
```

## Exercise 4: ROS Integration for Planning and Control

### Step 1: Create ROS navigation integration

Create `~/isaac_sim_examples/ros_navigation.py`:

```python
#!/usr/bin/env python3
# ros_navigation.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.sensor import Camera
import omni.isaac.ros2_bridge._ros2_bridge as ros2_bridge
import numpy as np
import carb

class ROSNavigationSystem:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.ros2_bridge = ros2_bridge.acquire_ros2_bridge_interface()
        self.robots = []
        self.setup_environment()

    def setup_environment(self):
        """Set up environment for ROS navigation testing."""
        # Create ground plane
        create_primitive(
            prim_path="/World/GroundPlane",
            primitive_type="Plane",
            scale=[25, 25, 1],
            position=[0, 0, 0]
        )

        # Create a navigation map-like environment
        # Create walls to form rooms and corridors
        wall_thickness = 0.3
        wall_height = 2.0

        # Create room walls
        room_centers = [[-6, -6], [6, -6], [-6, 6], [6, 6]]
        for center in room_centers:
            # Room walls
            create_primitive(
                prim_path=f"/World/RoomWall_Horizontal1_{center[0]}_{center[1]}",
                primitive_type="Cuboid",
                scale=[6, wall_thickness, wall_height],
                position=[center[0], center[1] + 3, wall_height/2]
            )
            create_primitive(
                prim_path=f"/World/RoomWall_Horizontal2_{center[0]}_{center[1]}",
                primitive_type="Cuboid",
                scale=[6, wall_thickness, wall_height],
                position=[center[0], center[1] - 3, wall_height/2]
            )
            create_primitive(
                prim_path=f"/World/RoomWall_Vertical1_{center[0]}_{center[1]}",
                primitive_type="Cuboid",
                scale=[wall_thickness, 6, wall_height],
                position=[center[0] + 3, center[1], wall_height/2]
            )
            create_primitive(
                prim_path=f"/World/RoomWall_Vertical2_{center[0]}_{center[1]}",
                primitive_type="Cuboid",
                scale=[wall_thickness, 6, wall_height],
                position=[center[0] - 3, center[1], wall_height/2]
            )

        # Create robot with sensors
        robot = create_primitive(
            prim_path="/World/Robot",
            primitive_type="Cylinder",
            scale=[0.5, 0.5, 1.0],
            position=[-8, -8, 0.5],
            color=[0.2, 0.6, 1.0]
        )
        self.robots.append(robot)

        # Add camera for perception
        camera = Camera(
            prim_path="/World/Robot/Camera",
            frequency=30,
            resolution=(640, 480),
            position=[0.3, 0, 0.5]
        )
        camera.set_local_pose(translation=np.array([0.3, 0, 0.5]))

        # Add LiDAR for navigation
        from omni.isaac.range_sensor import LidarRtx
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

        # Enable ROS bridge for navigation
        self.enable_ros_bridge(camera, lidar)

    def enable_ros_bridge(self, camera, lidar):
        """Enable ROS bridge for navigation sensors."""
        # Publish camera data
        self.ros2_bridge.publish_camera(
            camera,
            topic_name="/humanoid/camera/image_raw",
            sensor_name="camera"
        )

        # Publish LiDAR data
        self.ros2_bridge.publish_lidar(
            lidar,
            topic_name="/humanoid/scan",
            sensor_name="lidar"
        )

        # Publish TF transforms
        self.ros2_bridge.publish_transform_tree(
            prim_path="/World",
            topic_name="/tf"
        )

        # Publish robot state
        self.ros2_bridge.publish_robot_state(
            prim_path="/World/Robot",
            topic_name="/joint_states"
        )

    def run_ros_navigation(self, steps=1000):
        """Run ROS navigation simulation."""
        self.world.reset()

        # Set camera view
        set_camera_view(eye=[20, 20, 20], target=[0, 0, 0])

        for i in range(steps):
            self.world.step(render=True)

            # Print status every 100 steps
            if i % 100 == 0:
                print(f"ROS Navigation simulation step: {i}/{steps}")

        print("ROS navigation simulation completed")

# Create and run the ROS navigation system
if __name__ == "__main__":
    ros_nav = ROSNavigationSystem()
    ros_nav.run_ros_navigation(steps=800)
```

## Exercise 5: Behavior Trees for Complex Navigation

### Step 1: Implement a behavior tree system

Create `~/isaac_sim_examples/behavior_trees.py`:

```python
#!/usr/bin/env python3
# behavior_trees.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.viewports import set_camera_view
import numpy as np
import carb

class BehaviorNode:
    """Base class for behavior tree nodes."""

    def __init__(self, name):
        self.name = name
        self.status = "IDLE"

    def tick(self, blackboard):
        """Execute the behavior. Return 'SUCCESS', 'FAILURE', or 'RUNNING'."""
        raise NotImplementedError

class SelectorNode(BehaviorNode):
    """Selector node - returns SUCCESS if any child succeeds."""

    def __init__(self, name, children):
        super().__init__(name)
        self.children = children

    def tick(self, blackboard):
        for child in self.children:
            status = child.tick(blackboard)
            if status == "SUCCESS":
                return "SUCCESS"
            elif status == "RUNNING":
                return "RUNNING"
        return "FAILURE"

class SequenceNode(BehaviorNode):
    """Sequence node - returns FAILURE if any child fails."""

    def __init__(self, name, children):
        super().__init__(name)
        self.children = children

    def tick(self, blackboard):
        for child in self.children:
            status = child.tick(blackboard)
            if status == "FAILURE":
                return "FAILURE"
            elif status == "RUNNING":
                return "RUNNING"
        return "SUCCESS"

class ConditionNode(BehaviorNode):
    """Condition node - checks a condition."""

    def __init__(self, name, condition_func):
        super().__init__(name)
        self.condition_func = condition_func

    def tick(self, blackboard):
        if self.condition_func(blackboard):
            return "SUCCESS"
        else:
            return "FAILURE"

class ActionNode(BehaviorNode):
    """Action node - performs an action."""

    def __init__(self, name, action_func):
        super().__init__(name)
        self.action_func = action_func

    def tick(self, blackboard):
        return self.action_func(blackboard)

class NavigationBehaviorTree:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.blackboard = {
            'robot_position': np.array([0, 0, 0.5]),
            'goal_position': np.array([8, 8, 0.5]),
            'obstacles': [],
            'current_waypoint': None,
            'navigation_status': 'IDLE'
        }
        self.behavior_tree = self.create_behavior_tree()
        self.setup_environment()

    def setup_environment(self):
        """Set up environment for behavior tree testing."""
        # Create ground plane
        create_primitive(
            prim_path="/World/GroundPlane",
            primitive_type="Plane",
            scale=[20, 20, 1],
            position=[0, 0, 0]
        )

        # Create robot
        self.robot = create_primitive(
            prim_path="/World/Robot",
            primitive_type="Cylinder",
            scale=[0.5, 0.5, 1.0],
            position=[0, 0, 0.5],
            color=[0.2, 0.6, 1.0]
        )

        # Create goal
        create_primitive(
            prim_path="/World/Goal",
            primitive_type="Sphere",
            scale=[0.3, 0.3, 0.3],
            position=[8, 8, 0.15],
            color=[1, 0, 0]
        )

        # Create obstacles
        for i in range(8):
            create_primitive(
                prim_path=f"/World/Obstacle_{i}",
                primitive_type="Cylinder",
                scale=[0.4, 0.4, 1.0],
                position=[np.random.uniform(-6, 6), np.random.uniform(-6, 6), 0.5],
                color=[0.8, 0.2, 0.2]
            )

        # Update blackboard with initial positions
        self.blackboard['robot_position'] = np.array([0, 0, 0.5])
        self.blackboard['goal_position'] = np.array([8, 8, 0.5])

    def create_behavior_tree(self):
        """Create navigation behavior tree."""
        # Define conditions
        def is_at_goal(blackboard):
            robot_pos = blackboard['robot_position']
            goal_pos = blackboard['goal_position']
            distance = np.linalg.norm(robot_pos[:2] - goal_pos[:2])
            return distance < 0.5

        def is_path_clear(blackboard):
            # Simplified path clearance check
            return True  # In a real implementation, this would check for obstacles

        def has_battery_power(blackboard):
            # Simulate battery level
            return blackboard.get('battery_level', 100) > 10

        # Define actions
        def plan_path(blackboard):
            # In a real implementation, this would call path planning
            print("Planning path to goal...")
            blackboard['current_waypoint'] = blackboard['goal_position']
            return "SUCCESS"

        def move_to_waypoint(blackboard):
            # Move robot towards waypoint
            if blackboard['current_waypoint'] is not None:
                robot_pos = blackboard['robot_position']
                waypoint = blackboard['current_waypoint']

                # Simple movement toward waypoint
                direction = waypoint[:2] - robot_pos[:2]
                distance = np.linalg.norm(direction)

                if distance > 0.1:
                    # Move robot
                    move_step = direction / distance * 0.1  # 10cm step
                    new_pos = robot_pos.copy()
                    new_pos[0] += move_step[0]
                    new_pos[1] += move_step[1]

                    # Update robot position in Isaac Sim
                    self.update_robot_position(new_pos)
                    blackboard['robot_position'] = new_pos

                    print(f"Moving to waypoint: ({waypoint[0]:.2f}, {waypoint[1]:.2f})")
                    return "RUNNING"
                else:
                    print("Reached waypoint!")
                    return "SUCCESS"
            return "FAILURE"

        def recharge_battery(blackboard):
            print("Recharging battery...")
            blackboard['battery_level'] = 100
            return "SUCCESS"

        # Create behavior tree nodes
        # Root: Selector that tries navigation or recharging
        root = SelectorNode("Root", [
            # Sequence: Check battery and navigate
            SequenceNode("Navigate", [
                ConditionNode("Has Battery Power", has_battery_power),
                ConditionNode("Not At Goal", lambda bb: not is_at_goal(bb)),
                ActionNode("Plan Path", plan_path),
                ActionNode("Move to Waypoint", move_to_waypoint)
            ]),
            # If navigation fails due to low battery, recharge
            SequenceNode("Recharge", [
                ConditionNode("Needs Charging", lambda bb: bb.get('battery_level', 100) <= 10),
                ActionNode("Recharge Battery", recharge_battery)
            ])
        ])

        return root

    def update_robot_position(self, new_pos):
        """Update robot position in Isaac Sim."""
        if self.robot:
            from pxr import Gf, UsdGeom
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath("/World/Robot")
            xform = UsdGeom.Xformable(prim)

            transform = xform.GetLocalTransformation()
            transform.SetTranslate(Gf.Vec3d(new_pos[0], new_pos[1], new_pos[2]))
            xform.SetLocalTransformation(transform)

    def run_behavior_tree(self, steps=1000):
        """Run behavior tree simulation."""
        self.world.reset()

        # Set camera view
        set_camera_view(eye=[15, 15, 15], target=[0, 0, 0])

        for i in range(steps):
            self.world.step(render=True)

            # Execute behavior tree every 10 steps
            if i % 10 == 0:
                status = self.behavior_tree.tick(self.blackboard)

                # Print status
                robot_pos = self.blackboard['robot_position']
                goal_pos = self.blackboard['goal_position']
                distance = np.linalg.norm(robot_pos[:2] - goal_pos[:2])

                print(f"Step {i}: BT Status='{status}', Distance to goal={distance:.2f}m")

                # Check if goal reached
                if distance < 0.5:
                    print("GOAL REACHED!")
                    break

        print("Behavior tree navigation simulation completed")

# Create and run the behavior tree navigation system
if __name__ == "__main__":
    behavior_nav = NavigationBehaviorTree()
    behavior_nav.run_behavior_tree(steps=1000)
```

## Exercise 6: Control System Validation and Testing

### Step 1: Create control validation tools

Create `~/isaac_sim_examples/control_validation.py`:

```python
#!/usr/bin/env python3
# control_validation.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.viewports import set_camera_view
import numpy as np
import carb

class ControlValidationSystem:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.robots = []
        self.validation_metrics = {}
        self.setup_environment()

    def setup_environment(self):
        """Set up environment for control validation."""
        # Create ground plane
        create_primitive(
            prim_path="/World/GroundPlane",
            primitive_type="Plane",
            scale=[30, 30, 1],
            position=[0, 0, 0]
        )

        # Create test trajectories for validation
        test_configs = [
            {'name': 'Circle', 'type': 'circular', 'center': [0, 0], 'radius': 5},
            {'name': 'Square', 'type': 'square', 'center': [8, 0], 'size': 4},
            {'name': 'Figure8', 'type': 'figure8', 'center': [-8, 0], 'scale': 3},
        ]

        for i, config in enumerate(test_configs):
            robot = create_primitive(
                prim_path=f"/World/ValidationRobot_{i}",
                primitive_type="Cylinder",
                scale=[0.4, 0.4, 0.8],
                position=self.get_start_position(config),
                color=[0.8, 0.2, 0.8] if i == 0 else [0.2, 0.8, 0.8] if i == 1 else [0.8, 0.8, 0.2]
            )
            self.robots.append({'robot': robot, 'config': config, 'trajectory_idx': 0})

        # Create trajectory markers for visualization
        for i, config in enumerate(test_configs):
            trajectory = self.generate_trajectory(config)
            for j, pos in enumerate(trajectory):
                if j % 10 == 0:  # Show every 10th point
                    create_primitive(
                        prim_path=f"/World/TrajMarker_{i}_{j}",
                        primitive_type="Sphere",
                        scale=[0.05, 0.05, 0.05],
                        position=[pos[0], pos[1], 0.1],
                        color=[1, 1, 0] if i == 0 else [0, 1, 1] if i == 1 else [1, 0, 1]
                    )

    def get_start_position(self, config):
        """Get start position for trajectory."""
        if config['type'] == 'circular':
            return [config['center'][0] + config['radius'], config['center'][1], 0.4]
        elif config['type'] == 'square':
            return [config['center'][0] - config['size']/2, config['center'][1] - config['size']/2, 0.4]
        elif config['type'] == 'figure8':
            return [config['center'][0], config['center'][1], 0.4]
        return [0, 0, 0.4]

    def generate_trajectory(self, config, num_points=100):
        """Generate test trajectory."""
        trajectory = []

        if config['type'] == 'circular':
            center = config['center']
            radius = config['radius']
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                trajectory.append([x, y, 0.4])

        elif config['type'] == 'square':
            center = config['center']
            size = config['size']
            half_size = size / 2
            side_length = num_points // 4

            # Bottom side
            for i in range(side_length):
                x = center[0] - half_size + (2 * half_size * i / side_length)
                y = center[1] - half_size
                trajectory.append([x, y, 0.4])

            # Right side
            for i in range(side_length):
                x = center[0] + half_size
                y = center[1] - half_size + (2 * half_size * i / side_length)
                trajectory.append([x, y, 0.4])

            # Top side
            for i in range(side_length):
                x = center[0] + half_size - (2 * half_size * i / side_length)
                y = center[1] + half_size
                trajectory.append([x, y, 0.4])

            # Left side
            for i in range(side_length):
                x = center[0] - half_size
                y = center[1] + half_size - (2 * half_size * i / side_length)
                trajectory.append([x, y, 0.4])

        elif config['type'] == 'figure8':
            center = config['center']
            scale = config['scale']
            for i in range(num_points):
                t = 2 * np.pi * i / num_points
                x = center[0] + scale * np.sin(t)
                y = center[1] + scale * np.sin(t) * np.cos(t)
                trajectory.append([x, y, 0.4])

        return trajectory

    def get_robot_position(self, robot_prim):
        """Get robot position."""
        if robot_prim:
            from pxr import UsdGeom
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(str(robot_prim.GetPrimPath()))
            xform = UsdGeom.Xformable(prim)
            transform = xform.ComputeLocalToWorldTransform(omni.usd.get_context().get_time_code())
            pos = transform.ExtractTranslation()
            return np.array([pos[0], pos[1], pos[2]])
        return None

    def update_robot_position(self, robot_prim, new_pos):
        """Update robot position."""
        if robot_prim:
            from pxr import Gf, UsdGeom
            stage = omni.usd.get_context().get_stage()
            prim_path = str(robot_prim.GetPrimPath())
            prim = stage.GetPrimAtPath(prim_path)
            xform = UsdGeom.Xformable(prim)

            transform = xform.GetLocalTransformation()
            transform.SetTranslate(Gf.Vec3d(new_pos[0], new_pos[1], new_pos[2]))
            xform.SetLocalTransformation(transform)

    def calculate_control_metrics(self, robot_info, target_pos):
        """Calculate control performance metrics."""
        current_pos = self.get_robot_position(robot_info['robot'])
        if current_pos is None:
            return {}

        # Calculate tracking error
        error = np.linalg.norm(current_pos[:2] - target_pos[:2])

        # Calculate other metrics
        metrics = {
            'tracking_error': error,
            'distance_to_target': error,
            'position_accuracy': 1.0 / (1.0 + error) if error > 0 else 1.0,
            'control_effort': np.linalg.norm([current_pos[0], current_pos[1]])  # Simplified
        }

        return metrics

    def run_validation_test(self, steps=2000):
        """Run control validation test."""
        self.world.reset()

        # Set camera view
        set_camera_view(eye=[25, 25, 25], target=[0, 0, 0])

        # Generate trajectories for all robots
        trajectories = []
        for robot_info in self.robots:
            trajectory = self.generate_trajectory(robot_info['config'], num_points=200)
            trajectories.append(trajectory)

        for i in range(steps):
            self.world.step(render=True)

            # Update robots following their trajectories
            for j, (robot_info, trajectory) in enumerate(zip(self.robots, trajectories)):
                # Get current target in trajectory
                target_idx = (i // 5) % len(trajectory)  # Update every 5 steps
                target_pos = trajectory[target_idx]

                # Get current position
                current_pos = self.get_robot_position(robot_info['robot'])

                if current_pos is not None:
                    # Simple proportional control to follow trajectory
                    direction = np.array(target_pos[:2]) - current_pos[:2]
                    distance = np.linalg.norm(direction)

                    if distance > 0.05:  # Threshold for movement
                        move_vector = direction / distance * 0.05  # Small step
                        new_pos = current_pos.copy()
                        new_pos[0] += move_vector[0]
                        new_pos[1] += move_vector[1]

                        # Update robot position
                        self.update_robot_position(robot_info['robot'], new_pos)

                    # Calculate and store metrics
                    metrics = self.calculate_control_metrics(robot_info, target_pos)

                    # Print metrics every 100 steps
                    if i % 100 == 0:
                        print(f"Robot {j} (Type: {robot_info['config']['type']}): "
                              f"Error={metrics.get('tracking_error', 0):.3f}m, "
                              f"Accuracy={metrics.get('position_accuracy', 0):.3f}")

        print("Control validation test completed")

# Create and run the control validation system
if __name__ == "__main__":
    validation_sys = ControlValidationSystem()
    validation_sys.run_validation_test(steps=1500)
```

## Troubleshooting

### Common Issues and Solutions

1. **Control instability or oscillation**:
   - Adjust PID parameters (reduce kp, increase ki/d)
   - Check time step consistency
   - Verify sensor feedback accuracy

2. **Path planning failures**:
   - Verify grid resolution is appropriate
   - Check obstacle detection
   - Ensure start/goal positions are valid

3. **Behavior tree execution issues**:
   - Verify node connections
   - Check condition functions return proper values
   - Ensure actions update state correctly

4. **ROS integration problems**:
   - Verify network configuration
   - Check topic/service availability
   - Ensure proper message types

5. **Performance issues**:
   - Reduce control frequency for complex systems
   - Optimize path planning algorithms
   - Use appropriate simulation settings

## Assessment Questions

1. How do you tune PID controller parameters for optimal performance?
2. What are the key differences between A* and Dijkstra path planning algorithms?
3. How would you implement obstacle avoidance in real-time navigation?
4. What metrics would you use to evaluate control system performance?
5. How do you handle dynamic obstacles in path planning?

## Extension Exercises

1. Implement a full navigation stack with local and global planners
2. Create a manipulation control system for robotic arms
3. Implement reinforcement learning for navigation policy
4. Create a multi-robot coordination system
5. Implement advanced control techniques like MPC or LQR

## Summary

In this lab, you successfully:
- Implemented path planning algorithms (A*) in Isaac Sim
- Created motion control systems with PID controllers
- Integrated planning and control with ROS navigation stack
- Developed behavior trees for complex navigation tasks
- Validated control system performance and stability
- Tested various control strategies in simulation

These skills are essential for developing autonomous robotic systems capable of navigating and performing tasks in complex environments. The combination of path planning, motion control, and behavioral decision-making forms the foundation of autonomous robotics systems.