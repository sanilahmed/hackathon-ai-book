# Lab 3.4: Isaac Sim Reinforcement Learning Systems

## Overview

In this lab, you will learn how to implement reinforcement learning systems in Isaac Sim for robotics applications. You'll work with Isaac Lab's reinforcement learning framework, create environments for training, implement various RL algorithms, and integrate with ROS for real-world transfer. This includes understanding the RL environment architecture, reward shaping, and training policies for complex robotic behaviors.

## Objectives

By the end of this lab, you will be able to:
- Set up Isaac Lab for reinforcement learning
- Create custom RL environments in Isaac Sim
- Implement and train various RL algorithms (PPO, SAC, etc.)
- Design effective reward functions for robotic tasks
- Integrate RL policies with ROS control systems
- Implement domain randomization for sim-to-real transfer
- Validate and evaluate trained RL policies

## Prerequisites

- Completion of Lab 3.1-3.3: Isaac Sim Setup, Perception, and Planning/Control
- Understanding of reinforcement learning concepts
- Experience with Isaac Sim and ROS integration
- Basic knowledge of neural networks and deep learning

## Duration

6-8 hours

## Exercise 1: Isaac Lab Setup and Environment Creation

### Step 1: Install Isaac Lab

First, let's set up Isaac Lab:

```bash
# Create Isaac Lab directory
mkdir -p ~/isaac_lab
cd ~/isaac_lab

# Clone Isaac Lab repository
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Install Isaac Lab
./isaaclab.sh -i

# Install additional dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Create a basic RL environment

Create `~/isaac_lab_examples/basic_rl_env.py`:

```python
#!/usr/bin/env python3
# basic_rl_env.py
"""Basic reinforcement learning environment in Isaac Lab."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import carb

from omni.isaac.orbit.assets import AssetBase
from omni.isaac.orbit.envs import RLTaskEnv
from omni.isaac.orbit.assets import RigidObjectCfg, AssetBaseCfg
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils import configclass

@configclass
class BasicRLEnvCfg:
    # Environment settings
    episode_length = 1000
    decimation = 4  # Control decimation
    action_scale = 1.0
    obs_type = "proprioceptive"  # "proprioceptive" or "cameras"

    # Scene settings
    scene = SceneEntityCfg(
        num_envs=1024,
        env_spacing=2.5,
    )

    # Robot settings
    robot = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.5),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn_config=RigidObjectCfg.SpawnConfiguration(
                mass=1.0,
                rigid_body_enabled=True,
                collision_mesh_visible=True,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

class BasicRLEnv(RLTaskEnv):
    """Basic reinforcement learning environment."""

    def __init__(self, cfg: BasicRLEnvCfg):
        super().__init__(cfg)

        # Action space: [x_force, y_force, z_force]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Observation space: [position, velocity, goal_position]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),  # 3 position + 3 velocity + 3 goal
            dtype=np.float32
        )

        # Goal position
        self.goal_position = np.array([5.0, 5.0, 0.5])

    def _reset_idx(self, env_ids):
        """Reset environment with given IDs."""
        # Reset robot position to origin
        self.robot.data.default_root_state[env_ids] = torch.zeros_like(
            self.robot.data.default_root_state[env_ids]
        )
        self.robot.data.default_root_state[env_ids, :3] = torch.tensor(
            [0.0, 0.0, 0.5], device=self.device
        )

        # Reset robot velocities
        self.robot.data.vel_b[:, :] = 0.0
        self.robot.data.ang_vel_b[:, :] = 0.0

    def _compute_observations(self, env_ids=None):
        """Compute observations."""
        if env_ids is None:
            env_ids = slice(None)

        # Get current state
        positions = self.robot.data.root_pos_w[env_ids].cpu().numpy()
        velocities = self.robot.data.root_vel_w[env_ids].cpu().numpy()

        # Create observation vector
        obs = np.zeros((len(positions), 9))
        obs[:, :3] = positions  # Robot position
        obs[:, 3:6] = velocities  # Robot velocity
        obs[:, 6:9] = self.goal_position  # Goal position

        return obs

    def _compute_rewards(self, env_ids=None):
        """Compute rewards."""
        if env_ids is None:
            env_ids = slice(None)

        # Get current positions
        current_pos = self.robot.data.root_pos_w[env_ids, :2]
        goal_pos = torch.tensor(self.goal_position[:2], device=self.device).unsqueeze(0).expand_as(current_pos)

        # Calculate distance to goal
        distance = torch.norm(current_pos - goal_pos, dim=1)

        # Reward based on distance to goal (closer = higher reward)
        rewards = -distance

        # Bonus for reaching goal
        goal_reached = distance < 0.5
        rewards[goal_reached] += 100.0

        return rewards

    def _compute_terminals(self, env_ids=None):
        """Compute terminals (done flags)."""
        if env_ids is None:
            env_ids = slice(None)

        # Get current positions
        current_pos = self.robot.data.root_pos_w[env_ids, :2]
        goal_pos = torch.tensor(self.goal_position[:2], device=self.device).unsqueeze(0).expand_as(current_pos)

        # Calculate distance to goal
        distance = torch.norm(current_pos - goal_pos, dim=1)

        # Terminate if goal is reached or episode length exceeded
        goal_reached = distance < 0.5
        max_episode_length = self.episode_length
        time_out = self.episode_length_buf[env_ids] >= max_episode_length

        dones = goal_reached | time_out
        resets = goal_reached | time_out

        return dones, resets

    def step(self, action):
        """Execute one step in the environment."""
        # Apply action (forces to the robot)
        if self.cfg.action_scale != 1.0:
            action = action * self.cfg.action_scale

        # Convert action to forces
        forces = torch.tensor(action, device=self.device, dtype=torch.float32).unsqueeze(1).repeat(1, 3)

        # Apply forces to the robot
        self.robot.set_external_force(forces, indices=slice(None))

        # Step physics
        self.world.step(render=True)

        # Compute observations, rewards, and terminals
        obs = self._compute_observations()
        rewards = self._compute_rewards()
        dones, resets = self._compute_terminals()

        # Update episode lengths
        self.episode_length_buf += 1

        # Reset environments that are done
        reset_env_ids = resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self.episode_length_buf[reset_env_ids] = 0

        return obs, rewards.cpu().numpy(), dones.cpu().numpy(), {}

    def reset(self):
        """Reset the environment."""
        self._reset_idx(slice(None))
        self.episode_length_buf[:] = 0
        return self._compute_observations()

# Create and test the environment
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Basic RL Environment")
    parser.add_argument("--num-envs", type=int, default=1024, help="Number of environments")
    args = parser.parse_args()

    # Create environment configuration
    cfg = BasicRLEnvCfg()
    cfg.scene.num_envs = args.num_envs

    # Create environment
    env = BasicRLEnv(cfg)

    print(f"Environment created with {args.num_envs} environments")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Test environment
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    # Take random steps
    for i in range(100):
        # Random action
        action = np.random.uniform(-1, 1, size=(args.num_envs, 3))

        # Step environment
        obs, rewards, dones, info = env.step(action)

        if i % 20 == 0:
            print(f"Step {i}: Average reward = {np.mean(rewards):.3f}")

    print("Environment test completed successfully")
```

### Step 3: Create a more complex environment for navigation

Create `~/isaac_lab_examples/navigation_env.py`:

```python
#!/usr/bin/env python3
# navigation_env.py
"""Navigation reinforcement learning environment."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import carb

from omni.isaac.orbit.assets import RigidObjectCfg, AssetBaseCfg
from omni.isaac.orbit.envs import RLTaskEnv
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.assets import ArticulationCfg

@configclass
class NavigationEnvCfg:
    # Environment settings
    episode_length = 500
    decimation = 4
    action_scale = 1.0
    obs_type = "proprioceptive"

    # Scene settings
    scene = SceneEntityCfg(
        num_envs=2048,
        env_spacing=3.0,
    )

    # Robot settings (simple wheeled robot)
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=ArticulationCfg.SPAWN_CFG(
            asset_path="{ISAAC_ORBIT_NUCLEUS_DIR}/Robots/Ant/ant.usd",
            activate_contact_sensors=True,
            rigid_contact_offset=0.03,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            joint_pos={
                ".*": 0.0,
            },
        ),
    )

class NavigationEnv(RLTaskEnv):
    """Navigation environment for reinforcement learning."""

    def __init__(self, cfg: NavigationEnvCfg):
        super().__init__(cfg)

        # Action space: [left_wheel_vel, right_wheel_vel]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # Observation space: [robot_pos, robot_vel, goal_pos, obstacle_distances]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),  # 3 pos + 3 vel + 3 goal + 6 obstacle distances
            dtype=np.float32
        )

        # Generate random goals for each environment
        self.goals = torch.zeros((self.num_envs, 3), device=self.device)
        self.generate_goals()

        # Generate obstacles
        self.obstacles = torch.zeros((10, 3), device=self.device)
        self.generate_obstacles()

    def generate_goals(self):
        """Generate random goals."""
        # Goals within a certain area
        x_range = [-8.0, 8.0]
        y_range = [-8.0, 8.0]

        self.goals[:, 0] = torch.FloatTensor(self.num_envs).uniform_(x_range[0], x_range[1]).to(self.device)
        self.goals[:, 1] = torch.FloatTensor(self.num_envs).uniform_(y_range[0], y_range[1]).to(self.device)
        self.goals[:, 2] = 0.5  # Fixed height

    def generate_obstacles(self):
        """Generate obstacles."""
        # Static obstacles in the environment
        obstacle_positions = [
            [-6, 0, 0.5], [6, 0, 0.5], [0, 6, 0.5], [0, -6, 0.5],
            [-4, 4, 0.5], [4, 4, 0.5], [-4, -4, 0.5], [4, -4, 0.5]
        ]

        for i, pos in enumerate(obstacle_positions):
            if i < len(self.obstacles):
                self.obstacles[i] = torch.tensor(pos, device=self.device)

    def _reset_idx(self, env_ids):
        """Reset environment with given IDs."""
        # Reset robot to origin
        self.robot.data.default_root_state[env_ids, :3] = torch.tensor(
            [0.0, 0.0, 0.5], device=self.device
        ).unsqueeze(0).expand(len(env_ids), -1)

        # Reset robot velocities
        self.robot.data.vel_b[env_ids, :] = 0.0
        self.robot.data.ang_vel_b[env_ids, :] = 0.0

        # Reset robot joint positions and velocities
        self.robot.data.default_joint_pos[env_ids] = self.robot.data.default_joint_pos[env_ids]
        self.robot.data.joint_vel[env_ids] = 0.0

        # Generate new goals for reset environments
        x_range = [-8.0, 8.0]
        y_range = [-8.0, 8.0]

        self.goals[env_ids, 0] = torch.FloatTensor(len(env_ids)).uniform_(x_range[0], x_range[1]).to(self.device)
        self.goals[env_ids, 1] = torch.FloatTensor(len(env_ids)).uniform_(y_range[0], y_range[1]).to(self.device)

    def _compute_observations(self, env_ids=None):
        """Compute observations."""
        if env_ids is None:
            env_ids = slice(None)

        # Get current state
        positions = self.robot.data.root_pos_w[env_ids].clone()
        velocities = self.robot.data.root_vel_w[env_ids].clone()
        goals = self.goals[env_ids].clone()

        # Calculate obstacle distances
        robot_pos_2d = positions[:, :2]  # Only x, y
        obstacle_pos_2d = self.obstacles[:, :2].unsqueeze(0).expand(len(env_ids), -1, -1)

        # Calculate distances to obstacles
        dist_to_obstacles = torch.norm(
            robot_pos_2d.unsqueeze(1) - obstacle_pos_2d,
            dim=2
        )  # [num_envs, num_obstacles]

        # Get 6 closest obstacles (pad if needed)
        if dist_to_obstacles.shape[1] >= 6:
            closest_obstacles, _ = torch.topk(dist_to_obstacles, 6, dim=1, largest=False)
        else:
            closest_obstacles = torch.cat([
                torch.topk(dist_to_obstacles, dist_to_obstacles.shape[1], dim=1, largest=False)[0],
                torch.full((dist_to_obstacles.shape[0], 6 - dist_to_obstacles.shape[1]),
                          float('inf'), device=self.device)
            ], dim=1)

        # Create observation vector
        obs = torch.cat([
            positions,  # 3: position
            velocities,  # 3: velocity
            goals,  # 3: goal position
            closest_obstacles  # 6: obstacle distances
        ], dim=1)  # Total: 15

        return obs.cpu().numpy()

    def _compute_rewards(self, env_ids=None):
        """Compute rewards."""
        if env_ids is None:
            env_ids = slice(None)

        # Get current positions
        current_pos = self.robot.data.root_pos_w[env_ids, :2]
        goals_2d = self.goals[env_ids, :2]

        # Calculate distance to goal
        distance_to_goal = torch.norm(current_pos - goals_2d, dim=1)

        # Calculate progress (decrease in distance compared to previous step)
        prev_distance = getattr(self, '_prev_distance', distance_to_goal.clone())
        progress = prev_distance - distance_to_goal
        self._prev_distance = distance_to_goal.clone()

        # Calculate distance to obstacles
        robot_pos_2d = current_pos
        obstacle_pos_2d = self.obstacles[:, :2].unsqueeze(0).expand(len(env_ids), -1, -1)
        dist_to_obstacles = torch.norm(
            robot_pos_2d.unsqueeze(1) - obstacle_pos_2d,
            dim=2
        )
        min_obstacle_dist = torch.min(dist_to_obstacles, dim=1)[0]

        # Reward components
        goal_reward = -distance_to_goal * 0.1  # Negative distance penalty
        progress_reward = progress * 10.0  # Positive progress reward
        obstacle_penalty = torch.where(
            min_obstacle_dist < 1.0,
            -10.0 * (1.0 - min_obstacle_dist),
            torch.zeros_like(min_obstacle_dist)
        )  # Penalty for being too close to obstacles

        # Bonus for reaching goal
        goal_reached = distance_to_goal < 0.5
        goal_bonus = torch.where(goal_reached, torch.ones_like(goal_reached) * 100.0, torch.zeros_like(goal_reached))

        # Total reward
        rewards = goal_reward + progress_reward + obstacle_penalty + goal_bonus

        return rewards

    def _compute_terminals(self, env_ids=None):
        """Compute terminals."""
        if env_ids is None:
            env_ids = slice(None)

        # Get current positions
        current_pos = self.robot.data.root_pos_w[env_ids, :2]
        goals_2d = self.goals[env_ids, :2]

        # Calculate distance to goal
        distance_to_goal = torch.norm(current_pos - goals_2d, dim=1)

        # Terminate if goal is reached
        goal_reached = distance_to_goal < 0.5

        # Terminate if max episode length reached
        max_episode_length = self.episode_length
        time_out = self.episode_length_buf[env_ids] >= max_episode_length

        # Terminate if robot goes out of bounds
        out_of_bounds = (torch.abs(current_pos[:, 0]) > 10.0) | (torch.abs(current_pos[:, 1]) > 10.0)

        dones = goal_reached | time_out | out_of_bounds
        resets = goal_reached | time_out | out_of_bounds

        return dones, resets

    def step(self, action):
        """Execute one step in the environment."""
        # Apply action (velocity commands to joints)
        if self.cfg.action_scale != 1.0:
            action = action * self.cfg.action_scale

        # Convert action to joint velocities
        joint_velocities = torch.tensor(action, device=self.device, dtype=torch.float32)

        # Apply actions to the robot (simplified - in reality, you'd need to map to specific joints)
        # For Ant robot, the action would control the legs
        self.robot.set_joint_velocity_target(joint_velocities, joint_ids=slice(None))

        # Step physics
        self.world.step(render=True)

        # Compute observations, rewards, and terminals
        obs = self._compute_observations()
        rewards = self._compute_rewards()
        dones, resets = self._compute_terminals()

        # Update episode lengths
        self.episode_length_buf += 1

        # Reset environments that are done
        reset_env_ids = resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self.episode_length_buf[reset_env_ids] = 0

        return obs, rewards.cpu().numpy(), dones.cpu().numpy(), {}

# Create and test the navigation environment
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Navigation RL Environment")
    parser.add_argument("--num-envs", type=int, default=2048, help="Number of environments")
    args = parser.parse_args()

    # Create environment configuration
    cfg = NavigationEnvCfg()
    cfg.scene.num_envs = args.num_envs

    # Create environment
    env = NavigationEnv(cfg)

    print(f"Navigation environment created with {args.num_envs} environments")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Test environment
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    # Take random steps
    for i in range(100):
        # Random action
        action = np.random.uniform(-1, 1, size=(args.num_envs, 2))

        # Step environment
        obs, rewards, dones, info = env.step(action)

        if i % 20 == 0:
            print(f"Step {i}: Average reward = {np.mean(rewards):.3f}, "
                  f"Success rate = {np.mean(dones):.3f}")

    print("Navigation environment test completed successfully")
```

## Exercise 2: Implement PPO Algorithm

### Step 1: Create a PPO implementation

Create `~/isaac_lab_examples/ppo_agent.py`:

```python
#!/usr/bin/env python3
# ppo_agent.py
"""Proximal Policy Optimization (PPO) implementation for Isaac Lab."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor (policy network)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic (value network)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """Forward pass for both actor and critic."""
        features = self.feature_extractor(state)

        # Actor: mean and std for action distribution
        action_mean = torch.tanh(self.actor_mean(features))
        action_std = torch.exp(self.actor_logstd)

        # Critic: value estimation
        value = self.critic(features)

        return action_mean, action_std, value

    def get_action(self, state):
        """Sample action from the policy."""
        action_mean, action_std, value = self.forward(state)

        # Create distribution
        dist = torch.distributions.Normal(action_mean, action_std)

        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value

    def evaluate(self, state, action):
        """Evaluate action under current policy."""
        action_mean, action_std, value = self.forward(state)

        # Create distribution
        dist = torch.distributions.Normal(action_mean, action_std)

        # Calculate log probability
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Calculate entropy
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, value

class PPOAgent:
    """PPO Agent implementation."""

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2,
                 k_epochs=4, hidden_dim=256):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        # Initialize networks
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Memory for storing experiences
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []

    def select_action(self, state):
        """Select action using current policy."""
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, _ = self.policy.get_action(state)

        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]

    def store_transition(self, state, action, log_prob, reward, is_terminal):
        """Store transition in memory."""
        self.states.append(torch.FloatTensor(state))
        self.actions.append(torch.FloatTensor(action))
        self.log_probs.append(torch.FloatTensor([log_prob]))
        self.rewards.append(torch.FloatTensor([reward]))
        self.is_terminals.append(torch.FloatTensor([is_terminal]))

    def compute_returns(self):
        """Compute discounted returns."""
        returns = []
        Gt = 0
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                Gt = 0
            Gt = reward[0] + self.gamma * Gt
            returns.insert(0, Gt)

        # Normalize returns
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update(self):
        """Update policy using PPO."""
        # Convert to tensors
        old_states = torch.stack(self.states).detach()
        old_actions = torch.stack(self.actions).detach()
        old_log_probs = torch.stack(self.log_probs).detach()

        # Compute returns
        returns = self.compute_returns()

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluate old actions and values
            log_probs, entropy, state_values = self.policy.evaluate(old_states, old_actions)

            # Find the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(log_probs - old_log_probs.detach())

            # Compute advantage
            advantages = returns - state_values.detach()

            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Compute actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values, returns.unsqueeze(1))
            entropy_loss = entropy.mean()

            # Total loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

            # Perform backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear memory
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []

# Training function
def train_ppo(env, agent, num_episodes=1000):
    """Train PPO agent."""
    score_history = deque(maxlen=100)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0

        while True:
            # Select action
            action, log_prob = agent.select_action(state)

            # Take action in environment
            next_state, reward, done, _ = env.step(action)

            # Store transition
            agent.store_transition(state, action, log_prob, reward, done)

            state = next_state
            episode_reward += reward
            episode_steps += 1

            if done:
                break

        # Update policy after episode
        agent.update()

        # Track scores
        score_history.append(episode_reward)
        avg_score = np.mean(score_history)

        if episode % 50 == 0:
            print(f'Episode {episode}, Average Score: {avg_score:.2f}, Episode Reward: {episode_reward:.2f}')

    print("Training completed!")

# Example usage
if __name__ == "__main__":
    print("PPO Agent implementation created")
    print("This is a template that would be used with actual RL environments")
```

### Step 2: Create a complete training script

Create `~/isaac_lab_examples/train_navigation_ppo.py`:

```python
#!/usr/bin/env python3
# train_navigation_ppo.py
"""Train navigation policy using PPO in Isaac Lab."""

import torch
import numpy as np
from collections import deque
import argparse

# Import your PPO agent
from ppo_agent import PPOAgent

def train_navigation_ppo(env, num_episodes=2000, max_steps_per_episode=500):
    """Train navigation policy using PPO."""
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize PPO agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4
    )

    # Training loop
    score_history = deque(maxlen=100)
    avg_score_history = []

    print(f"Starting PPO training with {num_episodes} episodes...")

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0

        for step in range(max_steps_per_episode):
            # Select action using current policy
            action, log_prob = agent.select_action(state)

            # Take action in environment
            next_state, reward, done, info = env.step(action)

            # Store transition in agent's memory
            agent.store_transition(state, action, log_prob, reward, done)

            state = next_state
            episode_reward += reward
            episode_steps += 1

            if done:
                break

        # Update policy after collecting experiences
        agent.update()

        # Track scores
        score_history.append(episode_reward)
        avg_score = np.mean(score_history)
        avg_score_history.append(avg_score)

        # Print progress
        if episode % 50 == 0:
            print(f'Episode {episode:4d}, Average Score: {avg_score:8.2f}, '
                  f'Episode Reward: {episode_reward:8.2f}, Steps: {episode_steps}')

        # Early stopping if performance plateaus
        if episode > 100 and avg_score > 50:  # Adjust threshold as needed
            print(f"Early stopping at episode {episode}, average score: {avg_score:.2f}")
            break

    print(f"Training completed after {episode+1} episodes")
    print(f"Final average score: {np.mean(avg_score_history[-100:]):.2f}")

    return agent, avg_score_history

def test_trained_policy(env, agent, num_episodes=10):
    """Test the trained policy."""
    print("\nTesting trained policy...")

    test_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0

        for step in range(500):  # Max steps for testing
            # Select action (without exploration)
            with torch.no_grad():
                action, _ = agent.select_action(state)

            # Take action in environment
            state, reward, done, info = env.step(action)

            episode_reward += reward
            episode_steps += 1

            if done:
                break

        test_rewards.append(episode_reward)
        print(f'Test Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {episode_steps}')

    avg_test_reward = np.mean(test_rewards)
    print(f'Average test reward: {avg_test_reward:.2f}')

    return avg_test_reward

# Main training script
if __name__ == "__main__":
    import sys
    sys.path.append('.')  # Add current directory to path

    # Note: In a real implementation, you would import your actual navigation environment
    # For now, this serves as a template

    parser = argparse.ArgumentParser("PPO Navigation Training")
    parser.add_argument("--num-envs", type=int, default=2048, help="Number of parallel environments")
    parser.add_argument("--num-episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    args = parser.parse_args()

    print("PPO Navigation Training Script")
    print(f"Parameters: num_envs={args.num_envs}, num_episodes={args.num_episodes}")

    # In a real implementation, you would:
    # 1. Create the navigation environment
    # 2. Train the PPO agent
    # 3. Test the trained policy

    print("\nThis is a template. In a real implementation, you would:")
    print("1. Create your Isaac Lab navigation environment")
    print("2. Initialize the PPO agent with environment dimensions")
    print("3. Train the agent using the train_navigation_ppo function")
    print("4. Test the trained policy")

    # Example of how you would use it:
    # env = NavigationEnv(cfg)  # Your actual environment
    # agent, scores = train_navigation_ppo(env, num_episodes=args.num_episodes)
    # test_reward = test_trained_policy(env, agent)

    print("\nTemplate execution completed successfully")
```

## Exercise 3: Domain Randomization for Sim-to-Real Transfer

### Step 1: Implement domain randomization

Create `~/isaac_lab_examples/domain_randomization.py`:

```python
#!/usr/bin/env python3
# domain_randomization.py
"""Domain randomization for sim-to-real transfer in Isaac Lab."""

import torch
import numpy as np
from omni.isaac.orbit.assets import RigidObjectCfg
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils import configclass

@configclass
class DomainRandomizationConfig:
    """Configuration for domain randomization."""

    # Physics randomization
    physics_randomization = {
        'mass_ratio_range': [0.8, 1.2],
        'friction_range': [0.5, 1.5],
        'restitution_range': [0.0, 0.2],
        'dof_damping_range': [0.8, 1.2],
        'actuator_strength_range': [0.8, 1.2],
    }

    # Visual randomization
    visual_randomization = {
        'lighting_intensity_range': [100, 1000],
        'lighting_color_temperature_range': [3000, 8000],
        'material_albedo_range': [0.1, 1.0],
        'material_roughness_range': [0.0, 1.0],
        'material_metallic_range': [0.0, 1.0],
    }

    # Sensor randomization
    sensor_randomization = {
        'noise_range': [0.0, 0.05],
        'latency_range': [0.0, 0.02],
        'delay_range': [0.0, 0.01],
    }

    # Randomization intervals
    randomization_intervals = {
        'physics': 100,  # Randomize every 100 episodes
        'visual': 10,    # Randomize every 10 episodes
        'sensor': 5,     # Randomize every 5 episodes
    }

class DomainRandomizer:
    """Domain randomization manager."""

    def __init__(self, config: DomainRandomizationConfig):
        self.config = config
        self.current_physics_params = {}
        self.current_visual_params = {}
        self.current_sensor_params = {}

        # Initialize with default parameters
        self.randomize_physics()
        self.randomize_visual()
        self.randomize_sensor()

    def randomize_physics(self):
        """Randomize physics parameters."""
        physics_params = self.config.physics_randomization

        # Randomize mass ratios
        self.current_physics_params['mass_ratio'] = np.random.uniform(
            physics_params['mass_ratio_range'][0],
            physics_params['mass_ratio_range'][1]
        )

        # Randomize friction
        self.current_physics_params['friction'] = np.random.uniform(
            physics_params['friction_range'][0],
            physics_params['friction_range'][1]
        )

        # Randomize restitution
        self.current_physics_params['restitution'] = np.random.uniform(
            physics_params['restitution_range'][0],
            physics_params['restitution_range'][1]
        )

        # Randomize damping
        self.current_physics_params['damping_ratio'] = np.random.uniform(
            physics_params['dof_damping_range'][0],
            physics_params['dof_damping_range'][1]
        )

        # Randomize actuator strength
        self.current_physics_params['actuator_strength'] = np.random.uniform(
            physics_params['actuator_strength_range'][0],
            physics_params['actuator_strength_range'][1]
        )

        print(f"Physics randomized: mass_ratio={self.current_physics_params['mass_ratio']:.2f}, "
              f"friction={self.current_physics_params['friction']:.2f}")

    def randomize_visual(self):
        """Randomize visual parameters."""
        visual_params = self.config.visual_randomization

        # Randomize lighting
        self.current_visual_params['lighting_intensity'] = np.random.uniform(
            visual_params['lighting_intensity_range'][0],
            visual_params['lighting_intensity_range'][1]
        )

        self.current_visual_params['lighting_color_temp'] = np.random.uniform(
            visual_params['lighting_color_temperature_range'][0],
            visual_params['lighting_color_temperature_range'][1]
        )

        # Randomize materials
        self.current_visual_params['material_albedo'] = np.random.uniform(
            visual_params['material_albedo_range'][0],
            visual_params['material_albedo_range'][1]
        )

        self.current_visual_params['material_roughness'] = np.random.uniform(
            visual_params['material_roughness_range'][0],
            visual_params['material_roughness_range'][1]
        )

        self.current_visual_params['material_metallic'] = np.random.uniform(
            visual_params['material_metallic_range'][0],
            visual_params['material_metallic_range'][1]
        )

        print(f"Visual randomized: intensity={self.current_visual_params['lighting_intensity']:.0f}, "
              f"albedo={self.current_visual_params['material_albedo']:.2f}")

    def randomize_sensor(self):
        """Randomize sensor parameters."""
        sensor_params = self.config.sensor_randomization

        # Randomize noise
        self.current_sensor_params['noise'] = np.random.uniform(
            sensor_params['noise_range'][0],
            sensor_params['noise_range'][1]
        )

        # Randomize latency
        self.current_sensor_params['latency'] = np.random.uniform(
            sensor_params['latency_range'][0],
            sensor_params['latency_range'][1]
        )

        # Randomize delay
        self.current_sensor_params['delay'] = np.random.uniform(
            sensor_params['delay_range'][0],
            sensor_params['delay_range'][1]
        )

        print(f"Sensor randomized: noise={self.current_sensor_params['noise']:.3f}, "
              f"latency={self.current_sensor_params['latency']:.3f}s")

    def apply_randomization(self, env):
        """Apply current randomization parameters to environment."""
        # This would typically involve updating the physics engine parameters
        # For Isaac Sim, this involves updating material properties, lighting, etc.

        # Example: Update robot mass
        if hasattr(env, 'robot') and hasattr(env.robot, 'root_physx_view'):
            # Scale masses based on randomization
            current_masses = env.robot.root_physx_view.get_masses()
            scaled_masses = current_masses * self.current_physics_params['mass_ratio']
            env.robot.root_physx_view.set_masses(scaled_masses)

        # Example: Apply friction randomization
        # This would involve updating material properties in the scene

        print("Domain randomization applied to environment")

    def update_randomization(self, episode_count):
        """Update randomization based on episode count."""
        intervals = self.config.randomization_intervals

        # Randomize physics
        if episode_count % intervals['physics'] == 0:
            self.randomize_physics()

        # Randomize visual
        if episode_count % intervals['visual'] == 0:
            self.randomize_visual()

        # Randomize sensor
        if episode_count % intervals['sensor'] == 0:
            self.randomize_sensor()

class DomainRandomizedEnv:
    """Environment wrapper that applies domain randomization."""

    def __init__(self, base_env, domain_randomizer):
        self.base_env = base_env
        self.domain_randomizer = domain_randomizer
        self.episode_count = 0

    def reset(self):
        """Reset environment with possible domain randomization."""
        # Update randomization based on episode count
        self.domain_randomizer.update_randomization(self.episode_count)

        # Apply randomization to environment
        self.domain_randomizer.apply_randomization(self.base_env)

        # Increment episode count
        self.episode_count += 1

        # Reset base environment
        return self.base_env.reset()

    def step(self, action):
        """Take step in environment."""
        # Apply sensor noise if needed
        noisy_action = self.add_sensor_noise(action)

        # Step base environment
        obs, reward, done, info = self.base_env.step(noisy_action)

        return obs, reward, done, info

    def add_sensor_noise(self, action):
        """Add noise to actions based on randomization."""
        noise_level = self.domain_randomizer.current_sensor_params.get('noise', 0.0)
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, size=action.shape)
            return action + noise
        return action

    def __getattr__(self, name):
        """Delegate attribute access to base environment."""
        return getattr(self.base_env, name)

# Example usage
if __name__ == "__main__":
    print("Domain Randomization System Created")

    # Example of how to use domain randomization
    # config = DomainRandomizationConfig()
    # randomizer = DomainRandomizer(config)
    #
    # # Wrap your environment with domain randomization
    # # env = DomainRandomizedEnv(your_navigation_env, randomizer)
    #
    # # Training would then use the wrapped environment
    # # This helps with sim-to-real transfer

    print("Domain randomization system template created")
    print("This would be integrated with your actual RL training environment")
```

## Exercise 4: ROS Integration for RL Policies

### Step 1: Create ROS bridge for RL policies

Create `~/isaac_lab_examples/rl_ros_integration.py`:

```python
#!/usr/bin/env python3
# rl_ros_integration.py
"""ROS integration for reinforcement learning policies."""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
import numpy as np
import torch

class RLPolicyNode(Node):
    """ROS node for reinforcement learning policy."""

    def __init__(self):
        super().__init__('rl_policy_node')

        # Declare parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('action_frequency', 20.0)  # Hz
        self.declare_parameter('observation_window', 10)  # frames

        # Get parameters
        self.model_path = self.get_parameter('model_path').value
        self.action_frequency = self.get_parameter('action_frequency').value
        self.observation_window = self.get_parameter('observation_window').value

        # Initialize RL policy
        self.rl_policy = None
        self.load_policy()

        # Observation buffers
        self.joint_state_buffer = []
        self.imu_buffer = []
        self.laser_scan_buffer = []
        self.odometry_buffer = []

        # Publishers
        self.action_pub = self.create_publisher(Float32MultiArray, '/rl_action', 10)
        self.status_pub = self.create_publisher(Bool, '/rl_running', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.laser_scan_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_scan_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        # Timer for policy execution
        self.action_timer = self.create_timer(1.0/self.action_frequency, self.execute_policy)

        # Status variables
        self.is_initialized = False
        self.has_received_data = False

        self.get_logger().info('RL Policy Node initialized')

    def load_policy(self):
        """Load trained RL policy."""
        if self.model_path and self.model_path != '':
            try:
                # Load PyTorch model
                self.rl_policy = torch.load(self.model_path)
                self.is_initialized = True
                self.get_logger().info(f'Loaded RL policy from: {self.model_path}')
            except Exception as e:
                self.get_logger().error(f'Failed to load RL policy: {e}')
        else:
            self.get_logger().warn('No model path provided, using dummy policy')
            # Create a dummy policy for testing
            self.rl_policy = self.create_dummy_policy()
            self.is_initialized = True

    def create_dummy_policy(self):
        """Create a dummy policy for testing."""
        class DummyPolicy:
            def get_action(self, obs):
                # Return random action for testing
                return torch.randn(2)  # 2D action space

        return DummyPolicy()

    def joint_state_callback(self, msg):
        """Handle joint state messages."""
        self.joint_state_buffer.append({
            'position': list(msg.position),
            'velocity': list(msg.velocity),
            'effort': list(msg.effort),
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        })

        # Keep only recent observations
        if len(self.joint_state_buffer) > self.observation_window:
            self.joint_state_buffer = self.joint_state_buffer[-self.observation_window:]

        self.has_received_data = True

    def imu_callback(self, msg):
        """Handle IMU messages."""
        self.imu_buffer.append({
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        })

        # Keep only recent observations
        if len(self.imu_buffer) > self.observation_window:
            self.imu_buffer = self.imu_buffer[-self.observation_window:]

    def laser_scan_callback(self, msg):
        """Handle laser scan messages."""
        self.laser_scan_buffer.append({
            'ranges': list(msg.ranges),
            'intensities': list(msg.intensities),
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment,
            'time_increment': msg.time_increment,
            'scan_time': msg.scan_time,
            'range_min': msg.range_min,
            'range_max': msg.range_max,
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        })

        # Keep only recent observations
        if len(self.laser_scan_buffer) > self.observation_window:
            self.laser_scan_buffer = self.laser_scan_buffer[-self.observation_window:]

    def odom_callback(self, msg):
        """Handle odometry messages."""
        self.odometry_buffer.append({
            'position': [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
            'orientation': [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                           msg.pose.pose.orientation.z, msg.pose.pose.orientation.w],
            'linear_velocity': [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
            'angular_velocity': [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z],
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        })

        # Keep only recent observations
        if len(self.odometry_buffer) > self.observation_window:
            self.odometry_buffer = self.odometry_buffer[-self.observation_window:]

    def construct_observation(self):
        """Construct observation vector from buffered sensor data."""
        if not self.has_received_data:
            return None

        # Get most recent data
        joint_data = self.joint_state_buffer[-1] if self.joint_state_buffer else None
        imu_data = self.imu_buffer[-1] if self.imu_buffer else None
        laser_data = self.laser_scan_buffer[-1] if self.laser_scan_buffer else None
        odom_data = self.odometry_buffer[-1] if self.odometry_buffer else None

        # Construct observation vector
        obs = []

        # Add position and velocity from odometry
        if odom_data:
            obs.extend(odom_data['position'][:2])  # x, y position
            obs.extend(odom_data['linear_velocity'][:2])  # x, y velocity
            obs.extend(odom_data['angular_velocity'][:1])  # angular velocity z
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # Default values

        # Add IMU data
        if imu_data:
            obs.extend(imu_data['orientation'][:3])  # Only x, y, z of orientation
            obs.extend(imu_data['angular_velocity'])
            obs.extend(imu_data['linear_acceleration'])
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Add processed laser scan data (first 10 ranges for simplicity)
        if laser_data:
            ranges = laser_data['ranges']
            if len(ranges) >= 10:
                obs.extend(ranges[:10])  # First 10 range readings
            else:
                obs.extend(ranges + [30.0] * (10 - len(ranges)))  # Pad with max range
        else:
            obs.extend([30.0] * 10)  # Default max range values

        # Add joint positions if available
        if joint_data:
            positions = joint_data['position']
            if len(positions) >= 6:
                obs.extend(positions[:6])  # First 6 joint positions
            else:
                obs.extend(positions + [0.0] * (6 - len(positions)))
        else:
            obs.extend([0.0] * 6)

        return np.array(obs, dtype=np.float32)

    def execute_policy(self):
        """Execute RL policy and publish action."""
        if not self.is_initialized:
            return

        # Construct observation
        obs = self.construct_observation()
        if obs is None:
            return

        try:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            # Get action from policy
            with torch.no_grad():
                action = self.rl_policy.get_action(obs_tensor)

            # Convert action to numpy array
            action_np = action.cpu().numpy().flatten()

            # Publish action
            action_msg = Float32MultiArray()
            action_msg.data = action_np.tolist()
            self.action_pub.publish(action_msg)

            # Publish status
            status_msg = Bool()
            status_msg.data = True
            self.status_pub.publish(status_msg)

            self.get_logger().debug(f'Published action: {action_np}')

        except Exception as e:
            self.get_logger().error(f'Error executing policy: {e}')

    def destroy_node(self):
        """Cleanup before node destruction."""
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    rl_node = RLPolicyNode()

    try:
        rclpy.spin(rl_node)
    except KeyboardInterrupt:
        pass
    finally:
        rl_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercise 5: Policy Evaluation and Validation

### Step 1: Create policy evaluation tools

Create `~/isaac_lab_examples/policy_evaluation.py`:

```python
#!/usr/bin/env python3
# policy_evaluation.py
"""Policy evaluation and validation tools."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import csv
import os

class PolicyEvaluator:
    """Policy evaluation system."""

    def __init__(self, env, policy, num_episodes=100):
        self.env = env
        self.policy = policy
        self.num_episodes = num_episodes

        # Metrics tracking
        self.episode_rewards = deque(maxlen=num_episodes)
        self.episode_lengths = deque(maxlen=num_episodes)
        self.success_rates = deque(maxlen=num_episodes)
        self.collision_rates = deque(maxlen=num_episodes)
        self.energy_consumption = deque(maxlen=num_episodes)

    def evaluate_policy(self):
        """Evaluate policy performance."""
        print(f"Evaluating policy over {self.num_episodes} episodes...")

        total_successes = 0
        total_collisions = 0
        total_energy = 0

        for episode in range(self.num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_energy = 0
            has_collided = False
            has_succeeded = False

            while True:
                # Get action from policy
                with torch.no_grad():
                    if isinstance(state, np.ndarray):
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    else:
                        state_tensor = state.unsqueeze(0)

                    action = self.policy.get_action(state_tensor)

                # Take action in environment
                next_state, reward, done, info = self.env.step(action.cpu().numpy().flatten())

                # Calculate energy consumption (simplified)
                action_magnitude = np.linalg.norm(action.cpu().numpy())
                episode_energy += action_magnitude

                # Track collisions and success (implementation depends on environment)
                if 'collision' in info and info['collision']:
                    has_collided = True
                    total_collisions += 1

                if 'success' in info and info['success']:
                    has_succeeded = True
                    total_successes += 1
                    break

                state = next_state
                episode_reward += reward
                episode_length += 1

                if done:
                    break

            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.success_rates.append(1.0 if has_succeeded else 0.0)
            self.collision_rates.append(1.0 if has_collided else 0.0)
            self.energy_consumption.append(episode_energy)

            # Print progress
            if episode % 20 == 0:
                avg_reward = np.mean(list(self.episode_rewards)[-20:])
                print(f"Episode {episode:3d}: Reward={episode_reward:8.2f}, "
                      f"Length={episode_length:3d}, Avg Reward={avg_reward:8.2f}")

        # Calculate final metrics
        final_metrics = {
            'avg_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'avg_length': np.mean(self.episode_lengths),
            'success_rate': np.mean(self.success_rates),
            'collision_rate': np.mean(self.collision_rates),
            'avg_energy': np.mean(self.energy_consumption),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards)
        }

        return final_metrics

    def plot_evaluation_results(self, metrics):
        """Plot evaluation results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Plot reward over episodes
        ax1.plot(list(self.episode_rewards))
        ax1.set_title(f'Reward per Episode\nAvg: {metrics["avg_reward"]:.2f}')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')

        # Plot episode lengths
        ax2.plot(list(self.episode_lengths))
        ax2.set_title(f'Episode Length\nAvg: {metrics["avg_length"]:.1f}')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')

        # Plot success and collision rates over time
        success_window = 20
        if len(self.success_rates) >= success_window:
            success_smooth = np.convolve(list(self.success_rates),
                                       np.ones(success_window)/success_window, mode='valid')
            ax3.plot(success_smooth)
        ax3.set_title(f'Success Rate\nOverall: {metrics["success_rate"]:.2%}')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Success Rate')

        collision_smooth = np.convolve(list(self.collision_rates),
                                     np.ones(success_window)/success_window, mode='valid')
        ax4.plot(collision_smooth, color='red')
        ax4.set_title(f'Collision Rate\nOverall: {metrics["collision_rate"]:.2%}')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Collision Rate')

        plt.tight_layout()
        plt.savefig('policy_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_evaluation_results(self, metrics, filename='policy_evaluation.csv'):
        """Save evaluation results to CSV."""
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['metric', 'value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for metric, value in metrics.items():
                writer.writerow({'metric': metric, 'value': value})

        print(f"Evaluation results saved to {filename}")

class PolicyComparison:
    """Compare multiple policies."""

    def __init__(self):
        self.policy_results = {}

    def add_policy_results(self, policy_name, results):
        """Add results for a policy."""
        self.policy_results[policy_name] = results

    def compare_policies(self):
        """Compare all policies."""
        print("\nPolicy Comparison Results:")
        print("=" * 60)

        # Metrics to compare
        metrics = ['avg_reward', 'success_rate', 'collision_rate', 'avg_length']

        # Print header
        print(f"{'Policy':<20}", end="")
        for metric in metrics:
            print(f"{metric:<15}", end="")
        print()
        print("-" * 60)

        # Print results for each policy
        for policy_name, results in self.policy_results.items():
            print(f"{policy_name:<20}", end="")
            for metric in metrics:
                if metric in results:
                    if metric in ['success_rate', 'collision_rate']:
                        print(f"{results[metric]:<15.2%}", end="")
                    else:
                        print(f"{results[metric]:<15.2f}", end="")
                else:
                    print(f"{'N/A':<15}", end="")
            print()

        print("=" * 60)

# Example usage
if __name__ == "__main__":
    print("Policy Evaluation System Created")

    # Example of how to use the evaluation system
    # evaluator = PolicyEvaluator(your_env, your_policy, num_episodes=100)
    # metrics = evaluator.evaluate_policy()
    # evaluator.plot_evaluation_results(metrics)
    # evaluator.save_evaluation_results(metrics)

    print("Policy evaluation system template created")
    print("This would be used to evaluate trained RL policies")
```

## Troubleshooting

### Common Issues and Solutions

1. **Training instability or divergence**:
   - Reduce learning rate
   - Increase batch size
   - Use learning rate scheduling
   - Implement gradient clipping

2. **Slow convergence**:
   - Check reward function design
   - Verify observation space normalization
   - Consider curriculum learning
   - Tune hyperparameters

3. **Domain randomization issues**:
   - Start with narrow randomization ranges
   - Gradually increase randomization
   - Monitor training stability
   - Use validation environments

4. **ROS integration problems**:
   - Verify message type compatibility
   - Check network configuration
   - Ensure proper timing synchronization
   - Monitor bandwidth usage

5. **Memory and performance issues**:
   - Use experience replay efficiently
   - Optimize neural network architecture
   - Consider distributed training
   - Profile and optimize critical paths

## Assessment Questions

1. How do you design effective reward functions for continuous control tasks?
2. What are the key differences between PPO, SAC, and DDPG algorithms?
3. How does domain randomization help with sim-to-real transfer?
4. What metrics would you use to evaluate RL policy performance?
5. How do you handle sparse rewards in navigation tasks?

## Extension Exercises

1. Implement curriculum learning for complex tasks
2. Create a hierarchical RL system for long-horizon tasks
3. Implement meta-learning for fast adaptation
4. Create a multi-agent RL environment
5. Implement advanced exploration strategies

## Summary

In this lab, you successfully:
- Set up Isaac Lab for reinforcement learning
- Created custom RL environments for robotics tasks
- Implemented PPO algorithm for navigation
- Applied domain randomization for sim-to-real transfer
- Integrated RL policies with ROS for real-world deployment
- Evaluated and validated trained policies

These skills are essential for developing autonomous robotic systems capable of learning complex behaviors through interaction with the environment. The combination of reinforcement learning, domain randomization, and ROS integration enables the creation of adaptable robotic systems that can operate effectively in real-world environments.