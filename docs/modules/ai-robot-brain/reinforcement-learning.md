# Reinforcement Learning

## Overview

This section covers the implementation of reinforcement learning (RL) algorithms for humanoid robot behavior using NVIDIA Isaac. Reinforcement learning enables robots to learn complex behaviors through interaction with the environment, making it essential for autonomous humanoid capabilities.

## Isaac RL Framework

NVIDIA Isaac provides an optimized RL framework that includes:

- **Isaac Lab**: Framework for robot learning and deployment
- **GPU-accelerated simulation**: Parallel environments for faster training
- **Pre-built RL algorithms**: PPO, SAC, TD3, and more
- **Domain randomization**: Transfer learning from simulation to reality

### Isaac Lab Setup

```python
# Setting up Isaac Lab for RL training
import omni
from omni.isaac.kit import SimulationApp

# Launch Isaac Sim with RL configuration
config = {
    "headless": False,
    "enable_cameras": True,
    "physics_dt": 1.0/60.0,
    "rendering_dt": 1.0/60.0,
    "stage_units_in_meters": 1.0
}

simulation_app = SimulationApp(config)

# Import Isaac Lab components
from omni.isaac.orbit_tasks.utils import parse_env_cfg
from omni.isaac.orbit_tasks.locomotion.velocity.velocity_env_cfg import AnymalDFlatEnvCfg
```

### RL Environment Configuration

```python
# Example RL environment configuration for humanoid
from omni.isaac.orbit_assets import HUMANOID_ASSETS
from omni.isaac.orbit.envs import RLTaskEnv
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import ContactSensorCfg
from omni.isaac.orbit actuators import DCMotorCfg

class HumanoidEnvCfg:
    # Scene
    scene = SceneEntityCfg(
        num_envs=4096,  # Number of parallel environments
        env_spacing=2.5,
    )

    # Robot
    robot = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=HUMANOID_ASSETS.HUMANOID_PX,
        init_state={
            "joint_pos": {
                ".*": 0.0,
            },
            "joint_vel": {
                ".*": 0.0,
            },
        },
    )

    # Actuators
    actuators = {
        "legs": DCMotorCfg(
            joint_names=["hip_.*", "knee_.*", "ankle_.*"],
            effort_limit=80.0,
            velocity_limit=100.0,
            stiffness=10.0,
            damping=1.0,
        ),
        "arms": DCMotorCfg(
            joint_names=["shoulder_.*", "elbow_.*"],
            effort_limit=40.0,
            velocity_limit=100.0,
            stiffness=5.0,
            damping=0.5,
        ),
    }

    # Sensors
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        update_period=0.005,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/.*"],
    )

    # Rewards
    class rewards:
        termination_penalty = -200.0
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5
        lin_vel_z = -2.0
        ang_vel_xy = -0.05
        dof_vel = -0.01
        action_rate = -0.01
        stand_still = -0.5
```

## Deep Reinforcement Learning Algorithms

### Proximal Policy Optimization (PPO)

```python
# PPO implementation for humanoid locomotion
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Actor (policy network)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic (value network)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.feature_extractor(state)

        # Actor
        action_mean = torch.tanh(self.actor_mean(features))
        action_std = torch.exp(self.actor_logstd)

        # Critic
        value = self.critic(features)

        return action_mean, action_std, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action_mean, action_std, _ = self.actor_critic(state)

        # Sample action from normal distribution
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=1)

        return action.cpu().data.numpy().flatten(), action_logprob.cpu().data.numpy()

    def update(self, states, actions, rewards, logprobs, next_states, dones):
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        old_logprobs = torch.FloatTensor(logprobs).to(self.device)

        # Get current policy values
        action_means, action_stds, values = self.actor_critic(states)
        dist = Normal(action_means, action_stds)
        logprobs = dist.log_prob(actions).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        new_values = self.actor_critic(states)[2]

        # Calculate advantages
        advantages = rewards + self.gamma * new_values.detach() * (1 - dones) - values.detach()

        # Calculate ratios
        ratios = torch.exp(logprobs - old_logprobs)

        # PPO loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratings, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = entropy.mean()

        # Total loss
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

### Soft Actor-Critic (SAC) Implementation

```python
# SAC implementation for continuous humanoid control
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, alpha=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # Copy weights to target
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=3e-4)

        self.alpha = alpha
        self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=3e-4)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=256, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # Action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
```

## Isaac-Specific RL Training

### Parallel Environment Training

```python
# Parallel environment setup for faster RL training
import gymnasium as gym
from omni.isaac.orbit.envs import RLTaskEnv
from omni.isaac.orbit_tasks.utils import parse_env_cfg
import torch

def train_humanoid_rl():
    # Parse environment configuration
    env_cfg = parse_env_cfg("Isaac-Velocity-Flat-Humanoid-v0", use_gpu=True)

    # Create environment
    env = gym.make("Isaac-Velocity-Flat-Humanoid-v0", cfg=env_cfg)

    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize agent
    agent = PPOAgent(obs_dim, action_dim)

    # Training loop
    for episode in range(10000):
        obs, _ = env.reset()
        episode_reward = 0
        episode_steps = 0

        while True:
            # Select action
            action, logprob = agent.select_action(obs)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Store transition
            agent.store_transition(obs, action, reward, logprob, next_obs, terminated)

            obs = next_obs
            episode_reward += reward
            episode_steps += 1

            if terminated or truncated:
                break

        # Update agent
        if episode % 10 == 0:
            agent.update_networks()

        print(f"Episode {episode}: Reward = {episode_reward:.2f}")

    env.close()
```

### Domain Randomization

```python
# Domain randomization for sim-to-real transfer
from omni.isaac.orbit.assets import AssetBaseCfg
import numpy as np

class DomainRandomization:
    def __init__(self):
        self.randomization_params = {
            'mass_ratio_range': [0.8, 1.2],
            'friction_range': [0.5, 1.5],
            'restitution_range': [0.0, 0.2],
            'dof_damping_range': [0.5, 1.5],
            'actuator_strength_range': [0.8, 1.2],
        }

    def randomize_mass(self, asset_cfg):
        # Randomize mass properties
        mass_ratio = np.random.uniform(
            self.randomization_params['mass_ratio_range'][0],
            self.randomization_params['mass_ratio_range'][1]
        )
        # Apply mass scaling to asset
        return asset_cfg

    def randomize_dynamics(self, asset_cfg):
        # Randomize dynamic properties
        friction = np.random.uniform(
            self.randomization_params['friction_range'][0],
            self.randomization_params['friction_range'][1]
        )
        # Apply friction scaling
        return asset_cfg

    def apply_randomization(self, env_cfg):
        # Apply domain randomization to environment
        env_cfg.scene = self.randomize_mass(env_cfg.scene)
        env_cfg.scene = self.randomize_dynamics(env_cfg.scene)
        return env_cfg
```

## Locomotion Learning

### Walking Gait Learning

```python
# Learning humanoid walking gaits with RL
import torch
import numpy as np

class GaitLearning:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.gait_patterns = {
            'walk': self.walk_gait,
            'trot': self.trot_gait,
            'pace': self.pace_gait,
        }

    def walk_gait(self, phase, speed=1.0):
        # Generate walking gait pattern
        # phase: 0 to 2*pi, represents gait cycle
        left_leg_phase = phase
        right_leg_phase = (phase + np.pi) % (2 * np.pi)

        # Calculate joint angles for walking
        left_hip = 0.2 * np.sin(left_leg_phase)
        left_knee = 0.3 * np.sin(2 * left_leg_phase)
        left_ankle = -0.1 * np.sin(left_leg_phase)

        right_hip = 0.2 * np.sin(right_leg_phase)
        right_knee = 0.3 * np.sin(2 * right_leg_phase)
        right_ankle = -0.1 * np.sin(right_leg_phase)

        return np.array([left_hip, left_knee, left_ankle,
                        right_hip, right_knee, right_ankle])

    def train_gait_network(self):
        # Train neural network to generate gait patterns
        gait_network = nn.Sequential(
            nn.Linear(3, 64),  # phase, speed, terrain
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # joint commands
            nn.Tanh()
        )
        return gait_network
```

### Balance Recovery Learning

```python
# Learning balance recovery behaviors
class BalanceRecovery:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.com_estimator = CenterOfMassEstimator()
        self.balance_policy = BalancePolicyNetwork()

    def estimate_balance_state(self):
        # Estimate current balance state
        com_pos = self.com_estimator.get_com_position()
        com_vel = self.com_estimator.get_com_velocity()
        robot_orientation = self.robot.get_orientation()
        robot_angular_vel = self.robot.get_angular_velocity()

        # Balance state vector
        balance_state = np.concatenate([
            com_pos, com_vel,
            robot_orientation,
            robot_angular_vel
        ])
        return balance_state

    def compute_balance_action(self, balance_state):
        # Use trained policy to compute balance recovery action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(balance_state).unsqueeze(0)
            action = self.balance_policy(state_tensor)
        return action.numpy().flatten()

    def train_balance_policy(self):
        # Train balance recovery policy using RL
        # Reward function encourages staying upright
        def balance_reward(state, action, next_state):
            com_height = next_state[2]  # z-component of COM
            orientation = next_state[6:10]  # quaternion
            angular_vel = next_state[10:13]

            # Reward for maintaining upright position
            upright_reward = 10.0 * (1 - abs(orientation[2]))  # z component of quaternion
            height_reward = 5.0 * max(0, com_height - 0.5)  # minimum height
            stability_reward = -2.0 * np.linalg.norm(angular_vel)  # penalize angular velocity

            return upright_reward + height_reward + stability_reward
```

## Manipulation Learning

### Object Manipulation with RL

```python
# Learning manipulation tasks with RL
class ManipulationLearning:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.ik_solver = InverseKinematicsSolver()
        self.manipulation_agent = ManipulationRLAgent()

    def grasp_object(self, object_pose):
        # Learn to grasp objects using RL
        grasp_policy = self.manipulation_agent.get_grasp_policy()

        # Calculate grasp approach
        approach_poses = self.calculate_grasp_approach(object_pose)

        for approach_pose in approach_poses:
            # Use RL policy to determine grasp parameters
            grasp_params = grasp_policy(approach_pose)

            # Execute grasp
            success = self.execute_grasp(approach_pose, grasp_params)

            if success:
                return True

        return False

    def calculate_grasp_approach(self, object_pose):
        # Calculate multiple grasp approach poses
        approaches = []
        for angle in np.linspace(0, 2*np.pi, 8):
            approach_pose = object_pose.copy()
            approach_pose[0] += 0.1 * np.cos(angle)  # offset from object
            approach_pose[1] += 0.1 * np.sin(angle)
            approaches.append(approach_pose)
        return approaches
```

## Multi-Task Learning

### Hierarchical RL for Complex Tasks

```python
# Hierarchical reinforcement learning for complex humanoid tasks
class HierarchicalRL:
    def __init__(self):
        self.high_level_policy = HighLevelPolicy()
        self.low_level_skills = {
            'walk': SkillPolicy('walk'),
            'grasp': SkillPolicy('grasp'),
            'balance': SkillPolicy('balance'),
            'manipulate': SkillPolicy('manipulate')
        }

    def execute_task(self, high_level_goal):
        # Use high-level policy to determine sequence of skills
        skill_sequence = self.high_level_policy.plan(high_level_goal)

        for skill in skill_sequence:
            if skill in self.low_level_skills:
                success = self.low_level_skills[skill].execute()
                if not success:
                    return False

        return True

class SkillPolicy:
    def __init__(self, skill_name):
        self.skill_name = skill_name
        self.skill_network = self.build_skill_network()

    def build_skill_network(self):
        # Build skill-specific neural network
        if self.skill_name == 'walk':
            return nn.Sequential(
                nn.Linear(24, 128),  # state: 24-dim
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 12)  # 6 joints per leg
            )
        # Add other skill networks...
        return nn.Sequential()

    def execute(self):
        # Execute the skill using trained policy
        pass
```

## Isaac RL Training Pipeline

### Training Script

```python
#!/usr/bin/env python3
# Isaac RL training script for humanoid
import hydra
from omegaconf import DictConfig
from omni.isaac.orbit_tasks.utils import parse_env_cfg
from omni.isaac.orbit_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import torch

@hydra.main(config_path="cfg", config_name="train_cfg", version_base="1.2")
def main(cfg: DictConfig):
    """Main function for training humanoid RL agent."""
    # Parse environment configuration
    env_cfg = parse_env_cfg(cfg.env_id, use_gpu=True)

    # Create environment
    env = hydra.utils.instantiate(cfg.task, cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)

    # Create agent
    agent = PPO(
        policy=cfg.policy,
        env=env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Training
    agent.learn(total_timesteps=cfg.total_timesteps)

    # Save model
    agent.save(cfg.output_dir + "/humanoid_ppo_model")

    # Close environment
    env.close()

if __name__ == "__main__":
    main()
```

### Configuration File

```yaml
# train_cfg.yaml
defaults:
  - task: isaac-humanoid-env
  - _self_

env_id: "Isaac-Velocity-Flat-Humanoid-v0"

task:
  _target_: omni.isaac.orbit.envs.RLTaskEnv
  cfg: ${...}

policy:
  _target_: stable_baselines3.common.policies.ActorCriticPolicy

learning_rate: 1e-3
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2

total_timesteps: 10000000
output_dir: "outputs/humanoid_ppo"

# Training parameters
seed: 42
capture_video: True
capture_video_freq: 2000
capture_video_length: 100
```

## Transfer Learning and Deployment

### Sim-to-Real Transfer

```python
# Sim-to-real transfer techniques
class SimToRealTransfer:
    def __init__(self):
        self.domain_randomization = DomainRandomization()
        self.system_identification = SystemIdentification()
        self.adaptation_network = AdaptationNetwork()

    def prepare_for_real_world(self, sim_policy):
        # Adapt simulation policy for real world
        real_params = self.system_identification.identify_real_robot()
        adapted_policy = self.adaptation_network.adapt(
            sim_policy,
            real_params
        )
        return adapted_policy

    def online_adaptation(self, policy, real_data):
        # Online adaptation during real-world deployment
        adaptation_loss = self.compute_adaptation_loss(policy, real_data)
        policy.update(adaptation_loss)
        return policy

class SystemIdentification:
    def identify_real_robot(self):
        # Identify real robot parameters
        # Mass, friction, actuator dynamics, etc.
        return {
            'mass': self.measure_mass(),
            'friction': self.measure_friction(),
            'actuator_dynamics': self.measure_actuator_dynamics()
        }
```

## Performance Optimization

### Isaac-Specific Optimizations

```python
# Optimized RL training for Isaac
class OptimizedRLTraining:
    def __init__(self):
        # Enable GPU acceleration
        torch.backends.cudnn.benchmark = True
        self.use_mixed_precision = True
        self.use_tensor_cores = True

    def optimize_training_loop(self):
        # Optimize training with gradient accumulation
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_mixed_precision)

        for batch in training_batches:
            with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                loss = self.compute_loss(batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients
            optimizer.zero_grad()

    def parallel_environment_optimization(self):
        # Optimize parallel environments
        env_cfg = parse_env_cfg("Isaac-Humanoid-v0")
        env_cfg.scene.num_envs = 4096  # Max parallel environments

        # Memory optimization
        env_cfg.simulation.device = "cuda:0"
        env_cfg.simulation.use_gpu_pipeline = True

        return env_cfg
```

## Evaluation and Testing

### RL Performance Metrics

```python
# RL evaluation metrics
class RLEvaluation:
    def __init__(self):
        self.metrics = {
            'episode_reward': [],
            'episode_length': [],
            'success_rate': [],
            'stability_score': [],
            'energy_efficiency': []
        }

    def evaluate_policy(self, policy, num_episodes=100):
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            success = False

            while True:
                action = policy.select_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                episode_length += 1

                if terminated or truncated:
                    success = info.get('success', False)
                    break

            # Record metrics
            self.metrics['episode_reward'].append(episode_reward)
            self.metrics['episode_length'].append(episode_length)
            self.metrics['success_rate'].append(success)
            self.metrics['stability_score'].append(self.calculate_stability(obs))

    def calculate_stability(self, state):
        # Calculate stability based on COM position and orientation
        com_pos = state[:3]
        orientation = state[3:7]  # quaternion

        # Stability is higher when COM is within support polygon
        # and robot is upright
        stability = 1.0 - abs(orientation[2])  # upright measure
        return stability
```

## Troubleshooting

### Common RL Issues and Solutions

1. **Training Instability**
   - Reduce learning rate
   - Increase batch size
   - Use gradient clipping

2. **Poor Convergence**
   - Check reward function design
   - Verify action space bounds
   - Adjust network architecture

3. **Sim-to-Real Gap**
   - Increase domain randomization
   - Add noise to sensors
   - Use system identification

---
[Next: Simulation-to-Reality](./sim2real.md) | [Previous: Planning and Control](./planning-control.md)