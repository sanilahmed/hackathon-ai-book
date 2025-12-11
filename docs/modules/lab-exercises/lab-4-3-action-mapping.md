# Lab 4.3: Action Mapping in Vision-Language-Action Systems

## Overview

This lab exercise focuses on implementing action mapping systems that convert high-level vision-language understanding into low-level robot control commands. Students will learn to bridge the gap between perceptual understanding and executable actions for humanoid robots.

## Learning Objectives

By the end of this lab, students will be able to:
1. Implement action mapping networks that convert multimodal representations to motor commands
2. Design inverse kinematics systems for humanoid robot control
3. Create hierarchical action planning and execution frameworks
4. Evaluate action mapping performance using trajectory accuracy metrics
5. Integrate action mapping with perception and language understanding systems

## Prerequisites

- Completion of Lab 4.1: VLA Fundamentals and Lab 4.2: Multimodal Perception
- Understanding of robotics kinematics and control theory
- Knowledge of deep reinforcement learning concepts
- Familiarity with ROS 2 and robot control interfaces

## Theory Background

Action mapping in VLA systems involves translating abstract goals derived from vision and language into concrete motor commands. Key components include:

- **Motor Primitives**: Low-level movement patterns that can be combined for complex behaviors
- **Inverse Kinematics**: Mathematical methods to compute joint angles for desired end-effector positions
- **Action Spaces**: Representation of possible robot actions (discrete or continuous)
- **Policy Networks**: Neural networks that map states to actions
- **Trajectory Optimization**: Methods to generate smooth, feasible robot trajectories

## Lab Exercise

### Part 1: Action Space Representation

First, let's define different action space representations for humanoid robots:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActionSpaceConverter:
    def __init__(self, robot_dof=14):
        """
        Convert between different action space representations

        Args:
            robot_dof: Number of degrees of freedom in the robot
        """
        self.robot_dof = robot_dof

    def joint_space_to_cartesian(self, joint_positions, link_names):
        """
        Convert joint space positions to Cartesian coordinates (simplified)

        Args:
            joint_positions: [batch_size, robot_dof] - Joint angle positions
            link_names: List of link names to compute positions for

        Returns:
            cartesian_positions: [batch_size, len(link_names), 3] - Cartesian coordinates
        """
        # This is a simplified representation - in practice, use FK solvers like KDL or Pinocchio
        batch_size = joint_positions.shape[0]

        # For demonstration, we'll simulate a simple transformation
        # In reality, this would involve complex forward kinematics
        cartesian_positions = torch.zeros(batch_size, len(link_names), 3)

        # Simulated FK computation
        for i in range(len(link_names)):
            # Simplified transformation based on joint angles
            scale_factor = 0.1 * (i + 1)
            cartesian_positions[:, i, 0] = torch.sin(joint_positions[:, 0]) * scale_factor
            cartesian_positions[:, i, 1] = torch.cos(joint_positions[:, 1]) * scale_factor
            cartesian_positions[:, i, 2] = joint_positions[:, 2] * scale_factor

        return cartesian_positions

    def cartesian_to_joint_space(self, target_positions, current_joints):
        """
        Convert Cartesian target positions to joint space (inverse kinematics - simplified)

        Args:
            target_positions: [batch_size, num_targets, 3] - Desired Cartesian positions
            current_joints: [batch_size, robot_dof] - Current joint configuration

        Returns:
            joint_commands: [batch_size, robot_dof] - Joint angle commands
        """
        # Simplified IK - in practice, use numerical IK solvers
        batch_size = target_positions.shape[0]

        # Initialize joint commands with current joints
        joint_commands = current_joints.clone()

        # Apply simplified inverse kinematics
        for i in range(target_positions.shape[1]):
            # Map Cartesian error to joint adjustments
            pos_error = target_positions[:, i, :] - self.joint_space_to_cartesian(
                current_joints, ['target']
            ).squeeze(1)

            # Simple Jacobian-based update (simplified)
            joint_delta = torch.zeros_like(current_joints)
            joint_delta[:, :3] = pos_error * 0.1  # Map position error to first 3 joints

            joint_commands = current_joints + joint_delta

        return joint_commands

# Test the action space converter
def test_action_space_converter():
    batch_size, robot_dof = 4, 14
    link_names = ['left_hand', 'right_hand', 'head']

    converter = ActionSpaceConverter(robot_dof)

    joint_positions = torch.randn(batch_size, robot_dof)
    cartesian_pos = converter.joint_space_to_cartesian(joint_positions, link_names)

    print(f"Joint positions shape: {joint_positions.shape}")
    print(f"Cartesian positions shape: {cartesian_pos.shape}")

    target_positions = torch.randn(batch_size, 2, 3)
    joint_commands = converter.cartesian_to_joint_space(target_positions, joint_positions)

    print(f"Target positions shape: {target_positions.shape}")
    print(f"Joint commands shape: {joint_commands.shape}")

    return joint_commands

if __name__ == "__main__":
    test_action_space_converter()
```

### Part 2: Action Mapping Network

Now let's create a neural network that maps multimodal representations to action commands:

```python
class ActionMappingNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dims=[512, 256, 128]):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        # Build the network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        # Output layer for action prediction
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # Action normalization parameters
        self.register_buffer('action_mean', torch.zeros(action_dim))
        self.register_buffer('action_std', torch.ones(action_dim))

    def forward(self, multimodal_features):
        """
        Args:
            multimodal_features: [batch_size, input_dim] - Fused vision-language features
        Returns:
            action_pred: [batch_size, action_dim] - Predicted action commands
        """
        raw_action = self.network(multimodal_features)

        # Normalize action outputs
        normalized_action = (raw_action - self.action_mean) / (self.action_std + 1e-8)

        return normalized_action

    def set_action_normalization(self, mean, std):
        """Set action normalization parameters"""
        self.action_mean.copy_(torch.tensor(mean))
        self.action_std.copy_(torch.tensor(std))

# Test the action mapping network
def test_action_mapping_network():
    batch_size, input_dim, action_dim = 8, 256, 14  # 14 DOF humanoid

    multimodal_features = torch.randn(batch_size, input_dim)

    action_mapper = ActionMappingNetwork(input_dim, action_dim)
    action_pred = action_mapper(multimodal_features)

    print(f"Input features shape: {multimodal_features.shape}")
    print(f"Action prediction shape: {action_pred.shape}")
    print(f"Action prediction range: [{action_pred.min():.3f}, {action_pred.max():.3f}]")

    return action_pred

if __name__ == "__main__":
    test_action_mapping_network()
```

### Part 3: Hierarchical Action Planner

Let's implement a hierarchical action planner that breaks down high-level goals into executable steps:

```python
class HierarchicalActionPlanner:
    def __init__(self, max_steps=100):
        self.max_steps = max_steps

    def plan_from_command(self, command_str, current_state, goal_state):
        """
        Plan a sequence of actions to achieve a goal from a natural language command

        Args:
            command_str: Natural language command
            current_state: Current robot state
            goal_state: Desired goal state

        Returns:
            action_sequence: List of action tensors representing the plan
        """
        # Parse the command to identify high-level goal
        parsed_goal = self.parse_command(command_str)

        # Generate intermediate waypoints
        waypoints = self.generate_waypoints(current_state, goal_state)

        # Create action sequence
        action_sequence = self.create_action_sequence(waypoints, parsed_goal)

        return action_sequence

    def parse_command(self, command_str):
        """Parse natural language command into structured goal"""
        command_lower = command_str.lower()

        # Simple keyword-based parsing (in practice, use NLP models)
        if 'move' in command_lower or 'go' in command_lower:
            return {'type': 'navigation', 'direction': self.extract_direction(command_str)}
        elif 'pick' in command_lower or 'grasp' in command_lower:
            return {'type': 'manipulation', 'object': self.extract_object(command_str)}
        elif 'turn' in command_lower or 'rotate' in command_lower:
            return {'type': 'orientation', 'angle': self.extract_angle(command_str)}
        else:
            return {'type': 'unknown', 'command': command_str}

    def extract_direction(self, command_str):
        """Extract direction from command"""
        if 'forward' in command_str.lower():
            return 'forward'
        elif 'backward' in command_str.lower():
            return 'backward'
        elif 'left' in command_str.lower():
            return 'left'
        elif 'right' in command_str.lower():
            return 'right'
        else:
            return 'forward'  # default

    def extract_object(self, command_str):
        """Extract object from command"""
        # Simple extraction - in practice, use NER models
        words = command_str.lower().split()
        objects = ['box', 'ball', 'cup', 'book', 'object']

        for word in words:
            if word in objects:
                return word

        return 'object'  # default

    def extract_angle(self, command_str):
        """Extract angle from command"""
        import re
        # Look for angle specifications
        angle_match = re.search(r'(\d+)\s*(degrees|deg)', command_str.lower())
        if angle_match:
            return int(angle_match.group(1))
        return 90  # default angle

    def generate_waypoints(self, current_state, goal_state):
        """Generate intermediate waypoints between current and goal states"""
        # Simple linear interpolation (in practice, use motion planning algorithms)
        num_waypoints = 10
        waypoints = []

        for i in range(num_waypoints + 1):
            ratio = i / num_waypoints
            waypoint = current_state + ratio * (goal_state - current_state)
            waypoints.append(waypoint)

        return waypoints

    def create_action_sequence(self, waypoints, parsed_goal):
        """Convert waypoints to action sequence"""
        action_sequence = []

        for i in range(len(waypoints) - 1):
            # Calculate difference between consecutive waypoints
            delta = waypoints[i + 1] - waypoints[i]
            action_tensor = torch.tensor(delta, dtype=torch.float32)
            action_sequence.append(action_tensor)

        return action_sequence

# Test the hierarchical planner
def test_hierarchical_planner():
    planner = HierarchicalActionPlanner()

    current_state = torch.randn(14)  # 14 DOF humanoid state
    goal_state = torch.randn(14)     # Target state
    command = "Move the robot forward to pick up the red box"

    action_sequence = planner.plan_from_command(command, current_state, goal_state)

    print(f"Command: {command}")
    print(f"Parsed goal: {planner.parse_command(command)}")
    print(f"Number of actions in sequence: {len(action_sequence)}")
    if action_sequence:
        print(f"First action shape: {action_sequence[0].shape}")
        print(f"First action: {action_sequence[0][:5]}...")  # Show first 5 elements

    return action_sequence

if __name__ == "__main__":
    test_hierarchical_planner()
```

### Part 4: Policy Gradient Action Learner

Now let's implement a reinforcement learning approach for learning action mappings:

```python
class PolicyGradientActionLearner(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Action distribution parameters
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        """Forward pass through actor and critic"""
        action_mean = self.actor(state)
        state_value = self.critic(state)
        return action_mean, state_value

    def get_action(self, state, deterministic=False):
        """Sample action from the policy"""
        action_mean, _ = self.forward(state)

        if deterministic:
            return action_mean, None

        # Sample from Gaussian distribution
        action_std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    def evaluate_actions(self, states, actions):
        """Evaluate actions for policy gradient computation"""
        action_means, state_values = self.forward(states)

        action_std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_means, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, state_values.squeeze(), entropy

# Test the policy gradient learner
def test_policy_gradient_learner():
    batch_size, state_dim, action_dim = 16, 256, 14  # State from VLA system, 14 DOF

    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)

    learner = PolicyGradientActionLearner(state_dim, action_dim)

    # Test sampling actions
    sampled_actions, log_probs = learner.get_action(states[0:1], deterministic=False)
    print(f"Sampled action shape: {sampled_actions.shape}")
    print(f"Log probability shape: {log_probs.shape}")

    # Test action evaluation
    log_probs, state_values, entropy = learner.evaluate_actions(states, actions)
    print(f"Log probs shape: {log_probs.shape}")
    print(f"State values shape: {state_values.shape}")
    print(f"Entropy shape: {entropy.shape}")

    return learner

if __name__ == "__main__":
    test_policy_gradient_learner()
```

### Part 5: Action Execution and Control

Let's create a system for executing actions on the robot:

```python
class ActionExecutionController:
    def __init__(self, robot_interface):
        self.robot_interface = robot_interface
        self.action_history = []

    def execute_action_sequence(self, action_sequence, max_duration=10.0):
        """
        Execute a sequence of actions on the robot

        Args:
            action_sequence: List of action tensors
            max_duration: Maximum time to execute the sequence

        Returns:
            execution_result: Dictionary with execution metrics
        """
        start_time = time.time()
        success_count = 0
        total_actions = len(action_sequence)

        for i, action in enumerate(action_sequence):
            if time.time() - start_time > max_duration:
                break

            # Execute individual action
            success = self.execute_single_action(action)
            if success:
                success_count += 1

            # Store in history
            self.action_history.append({
                'step': i,
                'action': action.numpy(),
                'success': success,
                'timestamp': time.time()
            })

        execution_result = {
            'success_rate': success_count / total_actions if total_actions > 0 else 0,
            'total_executed': success_count,
            'total_requested': total_actions,
            'execution_time': time.time() - start_time
        }

        return execution_result

    def execute_single_action(self, action):
        """Execute a single action on the robot"""
        try:
            # Convert action to robot command format
            robot_cmd = self.convert_action_to_command(action)

            # Send command to robot
            self.robot_interface.send_command(robot_cmd)

            # Wait for execution
            success = self.wait_for_execution_completion()

            return success
        except Exception as e:
            print(f"Error executing action: {e}")
            return False

    def convert_action_to_command(self, action):
        """Convert action tensor to robot command format"""
        # This would typically convert to joint positions, velocities, or torques
        # depending on the robot interface
        command = {
            'joint_positions': action.tolist()[:7],  # Example: first 7 joints
            'velocities': [0.0] * 7,  # Zero velocity for position control
            'effort': [0.0] * 7       # Zero effort (torque) control
        }
        return command

    def wait_for_execution_completion(self, timeout=2.0):
        """Wait for action execution to complete"""
        # In practice, this would check robot feedback
        import time
        time.sleep(0.1)  # Simulate waiting
        return True  # Simulate success

# Robot interface mock for testing
class MockRobotInterface:
    def __init__(self):
        self.current_joint_positions = torch.zeros(14)

    def send_command(self, command):
        """Send command to robot (mock implementation)"""
        print(f"Sending command: {command}")
        # Update simulated joint positions
        if 'joint_positions' in command:
            self.current_joint_positions[:len(command['joint_positions'])] = \
                torch.tensor(command['joint_positions'], dtype=torch.float32)

    def get_current_state(self):
        """Get current robot state"""
        return self.current_joint_positions

# Test the action execution controller
def test_action_execution_controller():
    import time

    robot_interface = MockRobotInterface()
    controller = ActionExecutionController(robot_interface)

    # Create a simple action sequence
    action_sequence = [
        torch.randn(14) * 0.1,  # Small random movements
        torch.randn(14) * 0.1,
        torch.randn(14) * 0.1
    ]

    result = controller.execute_action_sequence(action_sequence, max_duration=5.0)

    print("Execution Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    print(f"Action history length: {len(controller.action_history)}")

    return result

if __name__ == "__main__":
    test_action_execution_controller()
```

### Part 6: Integration with ROS 2

Finally, let's create a ROS 2 node that integrates the action mapping system:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import torch
import numpy as np
import time

class ActionMappingNode(Node):
    def __init__(self):
        super().__init__('action_mapping_node')

        # Initialize action mapping network
        self.initialize_action_mapper()

        # Publishers and subscribers
        self.perception_sub = self.create_subscription(
            Float32MultiArray,
            '/multimodal_features',
            self.perception_callback,
            10
        )

        self.action_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Timer for processing loop
        self.timer = self.create_timer(0.05, self.process_loop)  # 20 Hz

        # Internal state
        self.current_features = None
        self.current_joint_state = None
        self.action_history = []

    def initialize_action_mapper(self):
        """Initialize the action mapping network"""
        # Create action mapper (using the classes defined above)
        self.action_mapper = ActionMappingNetwork(input_dim=256, action_dim=14)
        self.planner = HierarchicalActionPlanner()

        # Set to evaluation mode
        self.action_mapper.eval()
        self.get_logger().info('Action mapping network initialized')

    def perception_callback(self, msg):
        """Process incoming multimodal features"""
        try:
            # Convert Float32MultiArray to tensor
            features = torch.tensor(list(msg.data), dtype=torch.float32).unsqueeze(0)

            if features.shape[1] != 256:  # Expected feature dimension
                self.get_logger().warn(f'Unexpected feature dimension: {features.shape[1]}')
                return

            self.current_features = features
        except Exception as e:
            self.get_logger().error(f'Error processing perception: {str(e)}')

    def joint_state_callback(self, msg):
        """Process incoming joint states"""
        try:
            self.current_joint_state = {
                'position': msg.position,
                'velocity': msg.velocity,
                'effort': msg.effort
            }
        except Exception as e:
            self.get_logger().error(f'Error processing joint state: {str(e)}')

    def process_loop(self):
        """Main processing loop"""
        if self.current_features is not None and self.current_joint_state is not None:
            try:
                # Generate action using the action mapping network
                with torch.no_grad():
                    predicted_action = self.action_mapper(self.current_features)

                # Convert action to joint trajectory
                joint_trajectory = self.create_joint_trajectory(predicted_action)

                # Publish the trajectory
                self.action_pub.publish(joint_trajectory)

                # Store in history
                self.action_history.append({
                    'timestamp': time.time(),
                    'action': predicted_action.numpy()[0],
                    'published': True
                })

                # Log action
                self.get_logger().info(f'Published action with norm: {torch.norm(predicted_action).item():.3f}')

            except Exception as e:
                self.get_logger().error(f'Error in processing loop: {str(e)}')

    def create_joint_trajectory(self, action_tensor):
        """Create a joint trajectory message from action tensor"""
        traj_msg = JointTrajectory()

        # Define joint names (these should match your robot's joint names)
        joint_names = [
            'torso_joint', 'head_joint',
            'left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint',
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
        ]

        # Ensure we have enough joint names
        if len(joint_names) > action_tensor.shape[1]:
            # Pad with zeros for missing joints
            padded_action = torch.zeros(len(joint_names))
            padded_action[:action_tensor.shape[1]] = action_tensor[0]
            action_values = padded_action.numpy()
        else:
            # Truncate to match joint names
            action_values = action_tensor[0, :len(joint_names)].numpy()

        traj_msg.joint_names = joint_names

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = action_values.tolist()
        point.velocities = [0.0] * len(action_values)  # Zero velocity
        point.accelerations = [0.0] * len(action_values)  # Zero acceleration

        # Set timing (execute immediately)
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 50000000  # 50 ms

        traj_msg.points = [point]

        return traj_msg

    def destroy_node(self):
        """Cleanup when node is destroyed"""
        # Save action history to file
        import json
        with open('/tmp/action_history.json', 'w') as f:
            json.dump(self.action_history, f, indent=2)

        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    action_mapping_node = ActionMappingNode()

    try:
        rclpy.spin(action_mapping_node)
    except KeyboardInterrupt:
        pass
    finally:
        action_mapping_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Implementation Steps

1. Create the action mapping files in your workspace
2. Implement the action space converter for different representations
3. Build the action mapping neural network
4. Create the hierarchical action planner
5. Implement the policy gradient learner
6. Develop the action execution controller
7. Deploy the ROS 2 integration node

## Expected Outcomes

After completing this lab, you should have:
1. A working action space converter that can transform between joint and Cartesian spaces
2. An action mapping network that converts multimodal representations to motor commands
3. A hierarchical planner that breaks down high-level commands into executable actions
4. A reinforcement learning component for adaptive action selection
5. An execution controller that manages action sequencing and monitoring
6. A ROS 2 node that integrates action mapping with the robot control system

## Troubleshooting Tips

- If actions are unstable, check action space bounds and normalization
- If planning fails, verify waypoint generation and collision checking
- If RL training is unstable, adjust learning rates and reward shaping
- Monitor robot safety limits during execution
- Verify joint limits and singularity avoidance

## Further Exploration

- Implement advanced motion planning algorithms (RRT*, CHOMP, TrajOpt)
- Add tactile feedback integration for manipulation tasks
- Create learned action priors using human demonstrations
- Implement model predictive control for dynamic action adjustment