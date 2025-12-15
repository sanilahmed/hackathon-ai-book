---
sidebar_label: 'Lab 3.4: Reinforcement Learning'
---

# Lab Exercise 3.4: Reinforcement Learning in AI-Robot Brain

This lab exercise covers implementing reinforcement learning for robot control using NVIDIA Isaac.

## Objectives

- Set up RL environment in Isaac Sim
- Implement DQN for discrete action spaces
- Implement DDPG for continuous control
- Train agents for navigation and manipulation tasks

## Prerequisites

- Isaac Sim installed and configured
- ROS 2 Humble with necessary packages
- Python with PyTorch or TensorFlow
- Completed perception and control labs

## RL Environment Setup

### Isaac RL Environment Class

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import DynamicCuboid
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gym

class IsaacRLEnvironment:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.robot = None
        self.target = None
        self.obstacles = []

        # RL parameters
        self.state_dim = 8  # Example: [robot_x, robot_y, robot_yaw, target_x, target_y, vel_x, vel_y, ang_vel]
        self.action_dim = 4  # Example: [forward, backward, left, right] for discrete or [lin_vel, ang_vel] for continuous
        self.max_steps = 500

        self.setup_environment()

    def setup_environment(self):
        # Add ground plane
        add_reference_to_stage(
            usd_path="/Isaac/Environments/Simple_Room/simple_room.usd",
            prim_path="/World/Room"
        )

        # Add robot (example with TurtleBot)
        self.robot = self.world.scene.add(
            Robot(
                prim_path="/World/Robot",
                name="turtlebot",
                usd_path="/Isaac/Robots/Turtlebot/turtlebot3_differential.usd",
                position=np.array([0.0, 0.0, 0.1])
            )
        )

        # Add target object
        self.target = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Target",
                name="target",
                position=np.array([2.0, 2.0, 0.1]),
                size=0.2,
                color=np.array([0.0, 1.0, 0.0])
            )
        )

        # Add obstacles
        for i in range(3):
            obstacle = self.world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Obstacle_{i}",
                    name=f"obstacle_{i}",
                    position=np.array([1.0 + i*0.5, 0.5 + i*0.3, 0.1]),
                    size=0.3,
                    color=np.array([1.0, 0.0, 0.0])
                )
            )
            self.obstacles.append(obstacle)

    def reset(self):
        """Reset the environment to initial state"""
        self.world.reset()

        # Reset robot position
        self.robot.set_world_pose(position=np.array([0.0, 0.0, 0.1]))
        self.robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        self.robot.set_angular_velocity(np.array([0.0, 0.0, 0.0]))

        # Reset target position
        self.target.set_world_pose(position=np.array([2.0, 2.0, 0.1]))

        # Get initial state
        state = self.get_state()
        return state

    def get_state(self):
        """Get current state of the environment"""
        # Get robot position and orientation
        robot_pos, robot_orn = self.robot.get_world_pose()
        robot_lin_vel, robot_ang_vel = self.robot.get_linear_velocity(), self.robot.get_angular_velocity()

        # Get target position
        target_pos, _ = self.target.get_world_pose()

        # Create state vector
        state = np.array([
            robot_pos[0],  # x
            robot_pos[1],  # y
            robot_orn[2],  # yaw (simplified)
            target_pos[0], # target x
            target_pos[1], # target y
            robot_lin_vel[0], # linear vel x
            robot_lin_vel[1], # linear vel y
            robot_ang_vel[2]  # angular vel z
        ])

        return state

    def step(self, action):
        """Execute action and return (next_state, reward, done, info)"""
        # Apply action to robot
        self.apply_action(action)

        # Step simulation
        self.world.step(render=True)

        # Get next state
        next_state = self.get_state()

        # Calculate reward
        reward = self.calculate_reward()

        # Check if episode is done
        done = self.is_done()

        # Additional info
        info = {}

        return next_state, reward, done, info

    def apply_action(self, action):
        """Apply action to the robot"""
        # Example: discrete actions [forward, backward, left, right]
        linear_vel = 0.0
        angular_vel = 0.0

        if action == 0:  # Forward
            linear_vel = 0.5
        elif action == 1:  # Backward
            linear_vel = -0.5
        elif action == 2:  # Left
            angular_vel = 0.5
        elif action == 3:  # Right
            angular_vel = -0.5

        # Apply velocities to robot (this is simplified - actual implementation depends on robot type)
        # For differential drive robots, you would set wheel velocities
        pass

    def calculate_reward(self):
        """Calculate reward based on current state"""
        robot_pos, _ = self.robot.get_world_pose()
        target_pos, _ = self.target.get_world_pose()

        # Distance to target
        dist_to_target = np.linalg.norm(robot_pos[:2] - target_pos[:2])

        # Reward based on distance (closer = higher reward)
        reward = -dist_to_target  # Negative because we want to minimize distance

        # Bonus for reaching target
        if dist_to_target < 0.3:
            reward += 100
            return reward

        # Penalty for collisions (simplified check)
        if self.check_collision():
            reward -= 10

        return reward

    def check_collision(self):
        """Check if robot is colliding with obstacles"""
        # Simplified collision detection
        robot_pos, _ = self.robot.get_world_pose()

        for obstacle in self.obstacles:
            obs_pos, _ = obstacle.get_world_pose()
            dist = np.linalg.norm(robot_pos[:2] - obs_pos[:2])
            if dist < 0.4:  # Collision threshold
                return True

        return False

    def is_done(self):
        """Check if episode is done"""
        robot_pos, _ = self.robot.get_world_pose()
        target_pos, _ = self.target.get_world_pose()

        # Done if reached target or exceeded max steps
        dist_to_target = np.linalg.norm(robot_pos[:2] - target_pos[:2])

        if dist_to_target < 0.3:
            return True

        # Check if out of bounds
        if abs(robot_pos[0]) > 10 or abs(robot_pos[1]) > 10:
            return True

        return False
```

## Deep Q-Network (DQN) Implementation

### DQN Neural Network

```python
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)
```

### DQN Agent

```python
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Neural networks
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.memory = deque(maxlen=10000)
        self.batch_size = 32

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

## Deep Deterministic Policy Gradient (DDPG)

### Actor and Critic Networks

```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.network(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=1))
```

### DDPG Agent

```python
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        self.memory = deque(maxlen=100000)
        self.batch_size = 100

        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def update_target_networks(self):
        """Soft update target networks"""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, add_noise=True):
        """Choose action with optional noise for exploration"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).cpu().data.numpy().flatten()

        if add_noise:
            noise = np.random.normal(0, 0.1, size=self.action_dim)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def replay(self):
        """Train the networks on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.FloatTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch]).unsqueeze(1)

        # Critic update
        next_actions = self.actor_target(next_states)
        next_q_values = self.critic_target(next_states, next_actions)
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        current_q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.update_target_networks()
```

## Isaac RL Training Loop

### Training Script

```python
def train_dqn_agent():
    # Create environment
    env = IsaacRLEnvironment()

    # Initialize agent
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    # Training parameters
    episodes = 1000
    max_steps = 500
    target_update_freq = 100

    scores = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # Choose action
            action = agent.act(state)

            # Take action
            next_state, reward, done, _ = env.step(action)

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Train agent
            agent.replay()

            state = next_state
            total_reward += reward

            if done:
                break

        scores.append(total_reward)

        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_network()

        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")

    return agent, scores

def train_ddpg_agent():
    # Create continuous environment (modify IsaacRLEnvironment for continuous actions)
    env = IsaacRLEnvironment()

    # Initialize DDPG agent
    agent = DDPGAgent(
        state_dim=env.state_dim,
        action_dim=2,  # [linear_vel, angular_vel]
        max_action=1.0
    )

    # Training parameters
    episodes = 1000
    max_steps = 500

    scores = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # Choose action with noise for exploration
            action = agent.act(state, add_noise=True)

            # Take action
            next_state, reward, done, _ = env.step(action)

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Train agent
            agent.replay()

            state = next_state
            total_reward += reward

            if done:
                break

        scores.append(total_reward)

        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}")

    return agent, scores
```

## Isaac Sim RL Integration

### Isaac RL Training Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Twist
import numpy as np

class IsaacRLTrainingNode(Node):
    def __init__(self):
        super().__init__('isaac_rl_training')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.reward_pub = self.create_publisher(Float32, '/rl/reward', 10)
        self.done_pub = self.create_publisher(Bool, '/rl/done', 10)

        # RL agent
        self.rl_agent = None
        self.current_state = None
        self.episode_step = 0
        self.max_episode_steps = 500

        # Timer for RL control loop
        self.control_timer = self.create_timer(0.1, self.rl_control_loop)

        # Initialize RL agent
        self.initialize_agent()

    def initialize_agent(self):
        """Initialize the RL agent"""
        # Initialize your RL agent here
        # This would typically load a pre-trained model or create a new one
        pass

    def rl_control_loop(self):
        """Main RL control loop"""
        if self.rl_agent is None:
            return

        # Get current state from sensors (this would come from perception system)
        current_state = self.get_current_state()

        if current_state is not None:
            # Get action from RL agent
            action = self.rl_agent.act(current_state)

            # Convert action to robot command
            cmd_vel = self.convert_action_to_cmd_vel(action)

            # Publish command
            self.cmd_vel_pub.publish(cmd_vel)

            # Store state for next iteration
            self.current_state = current_state
            self.episode_step += 1

    def get_current_state(self):
        """Get current state from perception system"""
        # This would integrate with perception system to get robot state
        # For example: [robot_x, robot_y, robot_yaw, target_x, target_y, obstacles_distances, ...]
        return None  # Placeholder

    def convert_action_to_cmd_vel(self, action):
        """Convert RL action to Twist message"""
        cmd_vel = Twist()

        # Example: action is [linear_vel, angular_vel] for continuous control
        if len(action) >= 2:
            cmd_vel.linear.x = float(action[0])
            cmd_vel.angular.z = float(action[1])

        return cmd_vel

    def calculate_reward(self):
        """Calculate reward based on current state and goal"""
        # Implement reward calculation based on task
        # For navigation: reward based on distance to goal
        # For manipulation: reward based on object grasp success
        return 0.0  # Placeholder
```

## NVIDIA Isaac Gym Integration

### Isaac Gym Environment Wrapper

```python
from omni.isaac.gym.vec_env import VecEnvBase
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
import torch
import numpy as np

class IsaacGymEnv(VecEnvBase):
    def __init__(self, headless):
        super().__init__(headless=headless)

        self._world = World(stage_units_in_meters=1.0)
        self._setup_scene()

        self.set_defaults()

    def _setup_scene(self):
        """Setup the scene with robots and objects"""
        # Add your specific scene setup here
        pass

    def set_defaults(self):
        """Set default values for observations, actions, etc."""
        # Define action and observation spaces
        self.num_actions = 2  # Example: linear and angular velocity
        self.num_observations = 8  # Example: various state parameters

    def reset(self):
        """Reset the environment"""
        # Reset robot positions, targets, etc.
        self._world.reset()

        # Return observations
        obs = torch.zeros((self.num_envs, self.num_observations), device=self._device)
        return obs

    def step(self, actions):
        """Execute actions and return results"""
        # Apply actions to robots
        # Step simulation
        # Calculate rewards
        # Check if done
        # Return (observations, rewards, dones, info)

        obs = torch.zeros((self.num_envs, self.num_observations), device=self._device)
        rewards = torch.zeros(self.num_envs, device=self._device)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        info = {}

        return obs, rewards, dones, info
```

## Exercise Tasks

1. Set up the Isaac RL environment with proper state and action spaces
2. Implement the DQN algorithm and train it for a simple navigation task
3. Implement the DDPG algorithm for continuous control
4. Create a training loop that runs in Isaac Sim
5. Evaluate the trained agent's performance
6. Compare discrete vs. continuous action spaces for your specific task

## Evaluation Metrics

### RL Performance Evaluation

```python
class RLEvaluation:
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []

    def evaluate_agent(self, agent, env, num_episodes=100):
        """Evaluate trained agent"""
        total_rewards = []
        episode_lengths = []
        successes = 0

        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            step_count = 0
            done = False

            while not done and step_count < env.max_steps:
                action = agent.act(state, add_noise=False)  # No exploration noise during evaluation
                state, reward, done, _ = env.step(action)
                total_reward += reward
                step_count += 1

            total_rewards.append(total_reward)
            episode_lengths.append(step_count)

            # Check if task was completed successfully
            if self.check_success(state):
                successes += 1

        success_rate = successes / num_episodes

        return {
            'avg_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'avg_length': np.mean(episode_lengths),
            'success_rate': success_rate
        }

    def check_success(self, state):
        """Check if the task was completed successfully"""
        # Define success criteria based on your task
        return False  # Placeholder
```

## Troubleshooting

### Common Issues

- **Training instability**: Adjust learning rates and add experience replay
- **Slow convergence**: Increase network capacity or adjust hyperparameters
- **Exploration issues**: Tune epsilon decay or noise parameters
- **Isaac Sim integration**: Check GPU memory and simulation parameters

## Summary

In this lab, you learned to implement reinforcement learning for robot control using NVIDIA Isaac. You created DQN and DDPG agents, set up RL environments in Isaac Sim, and trained agents for navigation tasks. These techniques are fundamental for developing adaptive robotic systems.