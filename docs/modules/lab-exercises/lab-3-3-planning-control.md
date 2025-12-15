---
sidebar_label: 'Lab 3.3: Planning and Control'
---

# Lab Exercise 3.3: Planning and Control in AI-Robot Brain

This lab exercise covers implementing planning and control systems for robot AI using NVIDIA Isaac.

## Objectives

- Implement path planning algorithms
- Create motion control systems
- Integrate planning with perception
- Test control performance in simulation

## Prerequisites

- ROS 2 Humble with navigation packages
- Isaac Sim environment
- Basic knowledge of control theory
- Completed perception systems lab

## Planning System Overview

### Path Planning vs. Motion Planning

- **Path Planning**: Find collision-free path from start to goal
- **Motion Planning**: Find dynamically feasible trajectory considering robot dynamics

### Planning Architecture

```
Goal → Global Planner → Local Planner → Controller → Robot
```

## Global Path Planning

### A* Path Planner Implementation

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
import numpy as np
import heapq
from math import sqrt

class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')

        # Publishers and subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )
        self.path_pub = self.create_publisher(Path, '/global_plan', 10)
        self.marker_pub = self.create_publisher(Marker, '/path_marker', 10)

        # Storage
        self.map_data = None
        self.map_resolution = 0.0
        self.map_origin = [0.0, 0.0]

    def map_callback(self, msg):
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_resolution = msg.info.resolution
        self.map_origin = [msg.info.origin.position.x, msg.info.origin.position.y]

    def plan_path(self, start, goal):
        if self.map_data is None:
            return None

        # Convert world coordinates to map indices
        start_idx = self.world_to_map(start)
        goal_idx = self.world_to_map(goal)

        # Run A* algorithm
        path_indices = self.astar(start_idx, goal_idx)

        if path_indices:
            # Convert back to world coordinates
            path = []
            for idx in path_indices:
                world_pos = self.map_to_world(idx)
                pose = PoseStamped()
                pose.pose.position.x = world_pos[0]
                pose.pose.position.y = world_pos[1]
                pose.pose.position.z = 0.0
                path.append(pose)

            return path

        return None

    def astar(self, start, goal):
        # A* algorithm implementation
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + self.distance(current, neighbor)

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def heuristic(self, a, b):
        # Euclidean distance heuristic
        return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def get_neighbors(self, pos):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_pos = (pos[0] + dx, pos[1] + dy)

                # Check bounds
                if (0 <= new_pos[0] < self.map_data.shape[1] and
                    0 <= new_pos[1] < self.map_data.shape[0]):

                    # Check if cell is free (value < 50 in occupancy grid)
                    if self.map_data[new_pos[1], new_pos[0]] < 50:
                        neighbors.append(new_pos)

        return neighbors

    def distance(self, a, b):
        return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def world_to_map(self, world_pos):
        map_x = int((world_pos[0] - self.map_origin[0]) / self.map_resolution)
        map_y = int((world_pos[1] - self.map_origin[1]) / self.map_resolution)
        return (map_x, map_y)

    def map_to_world(self, map_idx):
        world_x = map_idx[0] * self.map_resolution + self.map_origin[0]
        world_y = map_idx[1] * self.map_resolution + self.map_origin[1]
        return [world_x, world_y]
```

## Local Path Planning and Trajectory Generation

### Dynamic Window Approach (DWA)

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformListener, Buffer
from nav_msgs.msg import Path
import numpy as np

class DWALocalPlanner(Node):
    def __init__(self):
        super().__init__('dwa_local_planner')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.path_sub = self.create_subscription(Path, '/global_plan', self.path_callback, 10)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10
        )

        # TF listener for robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Robot parameters
        self.max_speed = 0.5  # m/s
        self.max_yaw_rate = 1.0  # rad/s
        self.max_accel = 0.5  # m/s^2
        self.max_delta_yaw_rate = 3.2  # rad/s^2

        # DWA parameters
        self.predict_time = 3.0  # s
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0

        # Storage
        self.laser_data = None
        self.global_path = None
        self.current_pose = None
        self.current_velocity = [0.0, 0.0]  # [linear, angular]

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

    def scan_callback(self, msg):
        self.laser_data = msg

    def path_callback(self, msg):
        self.global_path = msg.poses

    def pose_callback(self, msg):
        self.current_pose = msg.pose.pose

    def control_loop(self):
        if (self.current_pose is None or
            self.global_path is None or
            self.laser_data is None):
            return

        # Get robot state
        robot_pos = [self.current_pose.position.x, self.current_pose.position.y]
        robot_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)

        # Get goal from global path
        goal = self.get_next_waypoint(robot_pos)

        if goal is None:
            # Stop robot if no goal
            cmd_vel = Twist()
            self.cmd_vel_pub.publish(cmd_vel)
            return

        # Run DWA
        best_u = self.dwa_control(robot_pos, robot_yaw, goal)

        if best_u is not None:
            cmd_vel = Twist()
            cmd_vel.linear.x = best_u[0]
            cmd_vel.angular.z = best_u[1]
            self.cmd_vel_pub.publish(cmd_vel)

    def dwa_control(self, robot_pos, robot_yaw, goal):
        # Calculate dynamic window
        vs = self.calc_dynamic_window()

        # Evaluate trajectories
        best_u = None
        min_score = float('inf')

        for v in np.arange(vs[0], vs[1], 0.1):  # Linear velocity
            for yaw_rate in np.arange(vs[2], vs[3], 0.1):  # Angular velocity
                trajectory = self.predict_trajectory(v, yaw_rate, robot_pos, robot_yaw)

                # Calculate costs
                to_goal_cost = self.calc_to_goal_cost(trajectory, goal)
                speed_cost = self.calc_speed_cost(trajectory)
                obstacle_cost = self.calc_obstacle_cost(trajectory)

                # Weighted sum of costs
                final_cost = (self.to_goal_cost_gain * to_goal_cost +
                             self.speed_cost_gain * speed_cost +
                             self.obstacle_cost_gain * obstacle_cost)

                if final_cost < min_score:
                    min_score = final_cost
                    best_u = [v, yaw_rate]

        return best_u

    def calc_dynamic_window(self):
        # Calculate dynamic window based on current velocity and limits
        vs = [0, self.max_speed, -self.max_yaw_rate, self.max_yaw_rate]

        vd = [self.current_velocity[0] - self.max_accel * 0.1,
              self.current_velocity[0] + self.max_accel * 0.1,
              self.current_velocity[1] - self.max_delta_yaw_rate * 0.1,
              self.current_velocity[1] + self.max_delta_yaw_rate * 0.1]

        # Clamp to dynamic window
        dw = [max(vs[0], vd[0]), min(vs[1], vd[1]),
              max(vs[2], vd[2]), min(vs[3], vd[3])]

        return dw

    def predict_trajectory(self, v, yaw_rate, robot_pos, robot_yaw):
        trajectory = []
        time = 0

        while time <= self.predict_time:
            new_x = robot_pos[0] + v * np.cos(robot_yaw) * time
            new_y = robot_pos[1] + v * np.sin(robot_yaw) * time
            new_yaw = robot_yaw + yaw_rate * time

            trajectory.append([new_x, new_y, new_yaw])
            time += 0.1

        return trajectory

    def calc_to_goal_cost(self, trajectory, goal):
        # Calculate distance to goal from end of trajectory
        dx = goal[0] - trajectory[-1][0]
        dy = goal[1] - trajectory[-1][1]
        error_angle = np.arctan2(dy, dx)
        cost = abs(error_angle - trajectory[-1][2])
        return cost

    def calc_speed_cost(self, trajectory):
        # Calculate cost based on speed (prefer higher speeds)
        speed = abs(trajectory[0][0])  # Simplified
        max_speed_cost = abs(self.max_speed)
        return max_speed_cost - speed

    def calc_obstacle_cost(self, trajectory):
        # Calculate cost based on obstacle proximity
        min_dist = float('inf')

        for point in trajectory:
            for i, range_val in enumerate(self.laser_data.ranges):
                if not np.isnan(range_val) and range_val < min_dist:
                    min_dist = range_val

        return 1.0 / min_dist if min_dist != 0 else float('inf')

    def get_yaw_from_quaternion(self, quat):
        # Convert quaternion to yaw angle
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def get_next_waypoint(self, robot_pos):
        # Find the next waypoint in the global path
        if not self.global_path:
            return None

        # Simple approach: find closest point and return the next one
        min_dist = float('inf')
        closest_idx = 0

        for i, pose in enumerate(self.global_path):
            dist = np.sqrt((pose.pose.position.x - robot_pos[0])**2 +
                          (pose.pose.position.y - robot_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Return next waypoint
        next_idx = min(closest_idx + 5, len(self.global_path) - 1)  # Look ahead
        next_waypoint = self.global_path[next_idx]

        return [next_waypoint.pose.position.x, next_waypoint.pose.position.y]
```

## Control Systems

### PID Controller Implementation

```python
class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain

        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = None

    def compute(self, setpoint, measured_value, dt=None):
        current_time = time.time()

        if self.prev_time is None:
            self.prev_time = current_time

        if dt is None:
            dt = current_time - self.prev_time

        if dt <= 0:
            return 0.0

        # Calculate error
        error = setpoint - measured_value

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        # Calculate output
        output = p_term + i_term + d_term

        # Store values for next iteration
        self.prev_error = error
        self.prev_time = current_time

        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = None
```

### Pure Pursuit Controller

```python
import math

class PurePursuitController:
    def __init__(self, lookahead_distance=1.0):
        self.lookahead_distance = lookahead_distance

    def calculate_control(self, robot_pos, robot_yaw, path):
        # Find the look-ahead point on the path
        look_ahead_point = self.find_look_ahead_point(robot_pos, path)

        if look_ahead_point is None:
            return 0.0, 0.0  # No control if no point found

        # Calculate the angle to the look-ahead point
        alpha = math.atan2(
            look_ahead_point[1] - robot_pos[1],
            look_ahead_point[0] - robot_pos[0]
        ) - robot_yaw

        # Calculate curvature
        curvature = 2 * math.sin(alpha) / self.lookahead_distance

        # Calculate angular velocity (assuming constant linear velocity)
        linear_vel = 0.3  # m/s
        angular_vel = linear_vel * curvature

        return linear_vel, angular_vel

    def find_look_ahead_point(self, robot_pos, path):
        for point in path:
            dist = math.sqrt(
                (point[0] - robot_pos[0])**2 +
                (point[1] - robot_pos[1])**2
            )
            if dist >= self.lookahead_distance:
                return [point[0], point[1]]

        # If no point is far enough, return the last point
        if path:
            return [path[-1][0], path[-1][1]]

        return None
```

## Isaac-Specific Planning and Control

### Isaac Manipulation Planning

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid
import numpy as np

class IsaacManipulationController:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.robot = None
        self.object = None

    def setup_environment(self):
        # Add Franka robot
        self.robot = self.world.scene.add(
            Franka(prim_path="/World/Franka", name="franka", position=np.array([0, 0, 0]))
        )

        # Add object to manipulate
        self.object = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Object",
                name="object",
                position=np.array([0.5, 0, 0.1]),
                size=0.1,
                color=np.array([0.8, 0.1, 0.1])
            )
        )

    def plan_manipulation(self, target_position):
        # Simple manipulation planning (in practice, use MoveIt or similar)
        # This would involve inverse kinematics and trajectory planning

        # Get current end-effector pose
        ee_position = self.robot.get_end_effector_frame()[:3, 3]

        # Calculate path to target
        path = self.interpolate_path(ee_position, target_position)

        return path

    def interpolate_path(self, start, end, steps=10):
        path = []
        for i in range(steps + 1):
            t = i / steps
            pos = start + t * (end - start)
            path.append(pos)
        return path

    def execute_manipulation(self, target_position):
        self.world.play()

        # Plan path
        path = self.plan_manipulation(target_position)

        # Execute path following
        for i in range(1000):  # Simulation steps
            if i < len(path):
                target_pos = path[i] if i < len(path) else path[-1]

                # Simple position control (in practice, use proper IK)
                joint_positions = self.calculate_ik(target_pos)
                self.robot.set_joint_positions(joint_positions)

            self.world.step(render=True)

        self.world.stop()

    def calculate_ik(self, target_position):
        # Simplified inverse kinematics
        # In practice, use Isaac's IK solvers or MoveIt
        return np.zeros(7)  # Placeholder
```

## Model Predictive Control (MPC)

### Linear MPC Implementation

```python
import numpy as np
from scipy.optimize import minimize

class LinearMPC:
    def __init__(self, A, B, Q, R, horizon=10):
        self.A = A  # State transition matrix
        self.B = B  # Control input matrix
        self.Q = Q  # State cost matrix
        self.R = R  # Control cost matrix
        self.horizon = horizon

    def solve(self, current_state, reference_trajectory):
        def cost_function(u_flat):
            # Reshape control sequence
            U = u_flat.reshape((self.horizon, -1))

            total_cost = 0
            state = current_state.copy()

            for k in range(self.horizon):
                # Predict next state
                state = self.A @ state + self.B @ U[k]

                # Calculate cost
                if k < len(reference_trajectory):
                    error = state - reference_trajectory[k]
                else:
                    error = state - reference_trajectory[-1]

                stage_cost = error.T @ self.Q @ error + U[k].T @ self.R @ U[k]
                total_cost += stage_cost

            return total_cost

        # Initial guess
        u_init = np.zeros(self.horizon * self.B.shape[1])

        # Solve optimization problem
        result = minimize(cost_function, u_init, method='SLSQP')

        if result.success:
            U_opt = result.x.reshape((self.horizon, -1))
            return U_opt[0]  # Return first control input
        else:
            return np.zeros(self.B.shape[1])  # Return zero if optimization fails
```

## Planning and Control Integration Node

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
import numpy as np

class PlanningControlNode(Node):
    def __init__(self):
        super().__init__('planning_control_node')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_sub = self.create_subscription(Path, '/global_plan', self.path_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Control parameters
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.current_path = None
        self.current_pose = None
        self.safe_to_move = True

        # Control timer
        self.control_timer = self.create_timer(0.05, self.control_callback)

        # PID controllers
        self.linear_pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
        self.angular_pid = PIDController(kp=2.0, ki=0.1, kd=0.1)

    def path_callback(self, msg):
        self.current_path = msg.poses

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def scan_callback(self, msg):
        # Simple obstacle detection
        min_range = min(msg.ranges) if msg.ranges else float('inf')
        self.safe_to_move = min_range > 0.5  # Safe if obstacle > 0.5m away

    def control_callback(self):
        if not self.safe_to_move:
            # Emergency stop
            cmd_vel = Twist()
            self.cmd_vel_pub.publish(cmd_vel)
            return

        if self.current_path is None or self.current_pose is None:
            return

        # Get next waypoint
        target = self.get_next_waypoint()
        if target is None:
            return

        # Calculate control commands
        linear_cmd = self.linear_pid.compute(target.linear_distance, 0)
        angular_cmd = self.angular_pid.compute(target.angular_error, 0)

        # Apply velocity limits
        linear_cmd = max(-0.5, min(0.5, linear_cmd))  # ±0.5 m/s
        angular_cmd = max(-1.0, min(1.0, angular_cmd))  # ±1.0 rad/s

        # Publish command
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_cmd
        cmd_vel.angular.z = angular_cmd
        self.cmd_vel_pub.publish(cmd_vel)

    def get_next_waypoint(self):
        # Simplified waypoint following
        if not self.current_path:
            return None

        # Find closest point on path
        current_pos = [
            self.current_pose.position.x,
            self.current_pose.position.y
        ]

        min_dist = float('inf')
        closest_point = None

        for pose in self.current_path:
            dist = np.sqrt(
                (pose.pose.position.x - current_pos[0])**2 +
                (pose.pose.position.y - current_pos[1])**2
            )
            if dist < min_dist:
                min_dist = dist
                closest_point = pose

        if closest_point:
            # Calculate required heading
            dx = closest_point.pose.position.x - current_pos[0]
            dy = closest_point.pose.position.y - current_pos[1]
            target_angle = np.arctan2(dy, dx)

            # Current robot angle
            current_angle = self.get_yaw_from_quaternion(self.current_pose.orientation)

            # Angular error
            angular_error = target_angle - current_angle

            return type('Waypoint', (), {
                'linear_distance': min_dist,
                'angular_error': angular_error
            })()

        return None

    def get_yaw_from_quaternion(self, quat):
        # Convert quaternion to yaw angle
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return np.arctan2(siny_cosp, cosy_cosp)
```

## Exercise Tasks

1. Implement the A* path planner and test it with different maps
2. Create a DWA local planner for obstacle avoidance
3. Implement PID controllers for motion control
4. Test the complete planning-control pipeline in Isaac Sim
5. Evaluate planning and control performance metrics
6. Optimize controller parameters for better performance

## Performance Evaluation

### Control Performance Metrics

```python
class ControlPerformanceEvaluator:
    def __init__(self):
        self.path_errors = []
        self.control_efforts = []
        self.execution_times = []

    def evaluate_tracking(self, desired_trajectory, actual_trajectory):
        # Calculate tracking error
        errors = []
        for i in range(min(len(desired_trajectory), len(actual_trajectory))):
            error = np.linalg.norm(
                np.array(desired_trajectory[i][:2]) -
                np.array(actual_trajectory[i][:2])
            )
            errors.append(error)

        rmse = np.sqrt(np.mean(np.square(errors)))
        max_error = np.max(errors)

        return rmse, max_error

    def evaluate_efficiency(self, control_commands):
        # Calculate control effort
        effort = np.sum(np.abs(control_commands))
        return effort
```

## Troubleshooting

### Common Issues

- **Path planning failures**: Check map resolution and obstacle inflation
- **Control instability**: Tune PID parameters appropriately
- **Collision issues**: Increase safety margins in local planner
- **Performance problems**: Optimize algorithm complexity

## Summary

In this lab, you learned to implement planning and control systems for robot AI. You created global and local planners, implemented various control algorithms, and integrated them with perception systems. These components form the core of autonomous robot navigation and manipulation.