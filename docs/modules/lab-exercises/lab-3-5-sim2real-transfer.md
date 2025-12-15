---
sidebar_label: 'Lab 3.5: Sim-to-Real Transfer'
---

# Lab Exercise 3.5: Sim-to-Real Transfer in AI-Robot Brain

This lab exercise covers techniques for transferring AI models trained in simulation to real robot systems.

## Objectives

- Understand the reality gap problem
- Implement domain randomization techniques
- Apply domain adaptation methods
- Test sim-to-real transfer with real robots

## Prerequisites

- Isaac Sim environment
- Real robot (or simulated with realistic physics)
- Completed RL and perception labs
- Basic understanding of transfer learning

## Reality Gap Overview

### The Simulation-to-Reality Problem

The reality gap includes differences in:
- **Visual appearance**: Lighting, textures, colors
- **Physics simulation**: Friction, compliance, dynamics
- **Sensor characteristics**: Noise, latency, resolution
- **Actuator behavior**: Delays, inaccuracies, limitations

### Transfer Learning Approaches

1. **Domain Randomization**: Randomize simulation parameters
2. **Domain Adaptation**: Adapt models to new domains
3. **System Identification**: Learn real-world dynamics
4. **Progressive Transfer**: Gradually reduce randomization

## Domain Randomization Implementation

### Visual Domain Randomization

```python
import random
import numpy as np
import cv2
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.materials import OmniPBR

class VisualDomainRandomizer:
    def __init__(self):
        self.light_properties = {
            'intensity_range': (500, 2000),
            'color_range': [(0.8, 0.8, 1.0), (1.0, 0.9, 0.8)],
            'position_jitter': 0.5
        }

        self.material_properties = {
            'roughness_range': (0.1, 0.9),
            'metallic_range': (0.0, 0.2),
            'specular_range': (0.1, 0.9)
        }

    def randomize_lighting(self):
        """Randomize lighting conditions in simulation"""
        # Get all lights in the scene
        lights = self.get_all_lights()

        for light in lights:
            # Randomize intensity
            intensity = random.uniform(
                self.light_properties['intensity_range'][0],
                self.light_properties['intensity_range'][1]
            )
            light.set_attribute('intensity', intensity)

            # Randomize color
            color = random.choice(self.light_properties['color_range'])
            light.set_attribute('color', color)

            # Randomize position slightly
            current_pos = light.get_world_pose()[0]
            jitter = np.random.uniform(
                -self.light_properties['position_jitter'],
                self.light_properties['position_jitter'],
                size=3
            )
            new_pos = current_pos + jitter
            light.set_world_pose(position=new_pos)

    def randomize_materials(self):
        """Randomize material properties"""
        # Get all objects in the scene
        objects = self.get_all_objects()

        for obj in objects:
            # Get current material
            material = obj.get_material()

            # Randomize material properties
            if material:
                roughness = random.uniform(
                    self.material_properties['roughness_range'][0],
                    self.material_properties['roughness_range'][1]
                )
                material.set_attribute('roughness', roughness)

                metallic = random.uniform(
                    self.material_properties['metallic_range'][0],
                    self.material_properties['metallic_range'][1]
                )
                material.set_attribute('metallic', metallic)

    def randomize_textures(self):
        """Randomize object textures"""
        # Apply random textures to objects
        objects = self.get_all_objects()

        for obj in objects:
            # Randomly select from a set of textures
            texture_options = [
                '/Isaac/Textures/rough.png',
                '/Isaac/Textures/smooth.png',
                '/Isaac/Textures/metal.png',
                '/Isaac/Textures/plastic.png'
            ]

            random_texture = random.choice(texture_options)
            obj.set_texture(random_texture)

    def get_all_lights(self):
        """Get all light objects in the scene"""
        # Implementation depends on your specific scene setup
        return []

    def get_all_objects(self):
        """Get all objects in the scene"""
        # Implementation depends on your specific scene setup
        return []
```

### Physics Domain Randomization

```python
class PhysicsDomainRandomizer:
    def __init__(self):
        self.physics_params = {
            'friction_range': (0.1, 1.0),
            'restitution_range': (0.0, 0.5),
            'mass_multiplier_range': (0.8, 1.2),
            'damping_range': (0.01, 0.1)
        }

    def randomize_physics_properties(self, robot):
        """Randomize physics properties of robot and environment"""
        # Randomize robot joint friction
        joint_names = robot.get_joint_names()
        for joint_name in joint_names:
            friction = random.uniform(
                self.physics_params['friction_range'][0],
                self.physics_params['friction_range'][1]
            )
            robot.set_joint_friction(joint_name, friction)

        # Randomize link masses
        link_names = robot.get_link_names()
        for link_name in link_names:
            current_mass = robot.get_link_mass(link_name)
            mass_multiplier = random.uniform(
                self.physics_params['mass_multiplier_range'][0],
                self.physics_params['mass_multiplier_range'][1]
            )
            new_mass = current_mass * mass_multiplier
            robot.set_link_mass(link_name, new_mass)

        # Randomize damping
        for joint_name in joint_names:
            damping = random.uniform(
                self.physics_params['damping_range'][0],
                self.physics_params['damping_range'][1]
            )
            robot.set_joint_damping(joint_name, damping)

    def randomize_environment_physics(self):
        """Randomize environment physics properties"""
        # Randomize ground friction
        ground_friction = random.uniform(
            self.physics_params['friction_range'][0],
            self.physics_params['friction_range'][1]
        )

        # Randomize gravity slightly
        gravity_jitter = np.random.uniform(-0.5, 0.5, size=3)
        # Apply gravity jitter to simulation
```

## Domain Adaptation Techniques

### Unsupervised Domain Adaptation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class DomainAdaptationNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim=256, num_classes=10):
        super(DomainAdaptationNetwork, self).__init__()

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU()
        )

        # Label classifier
        self.label_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_classes)
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2)  # 2 domains: source and target
        )

    def forward(self, x, alpha=0):
        # Extract features
        features = self.feature_extractor(x)

        # Reverse gradients for domain adaptation
        reverse_features = self.gradient_reverse_layer(features, alpha)

        # Get predictions
        class_pred = self.label_classifier(features)
        domain_pred = self.domain_classifier(reverse_features)

        return class_pred, domain_pred

    def gradient_reverse_layer(self, x, alpha):
        """Gradient reversal layer for domain adaptation"""
        return GradientReverseFunction.apply(x, alpha)

class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha
        return input

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DomainAdaptationTrainer:
    def __init__(self, model, learning_rate=1e-3):
        self.model = model
        self.classifier_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train_step(self, source_data, target_data, labels_s, alpha=0.1):
        """Single training step for domain adaptation"""
        # Zero gradients
        self.optimizer.zero_grad()

        # Source domain
        class_pred_s, domain_pred_s = self.model(source_data, alpha=0)
        source_domain_labels = torch.zeros(source_data.size(0), dtype=torch.long)

        # Target domain
        class_pred_t, domain_pred_t = self.model(target_data, alpha=alpha)
        target_domain_labels = torch.ones(target_data.size(0), dtype=torch.long)

        # Compute losses
        class_loss = self.classifier_criterion(class_pred_s, labels_s)
        domain_loss = (self.domain_criterion(domain_pred_s, source_domain_labels) +
                      self.domain_criterion(domain_pred_t, target_domain_labels))

        total_loss = class_loss + domain_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), class_loss.item(), domain_loss.item()
```

## Sim-to-Real Transfer Pipeline

### Transfer Learning Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import torch
import numpy as np

class Sim2RealTransferNode(Node):
    def __init__(self):
        super().__init__('sim2real_transfer')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.transfer_quality_pub = self.create_publisher(Float32, '/transfer_quality', 10)

        # Load pre-trained simulation model
        self.sim_model = self.load_pretrained_model('sim_model.pth')

        # Initialize real-world adaptation model
        self.real_model = self.create_adapted_model(self.sim_model)

        # Adaptation parameters
        self.adaptation_active = True
        self.transfer_quality = 0.0

        # Data buffers for adaptation
        self.real_data_buffer = []
        self.buffer_size = 100

    def load_pretrained_model(self, model_path):
        """Load pre-trained model from simulation"""
        model = torch.load(model_path)
        model.eval()
        return model

    def create_adapted_model(self, sim_model):
        """Create adapted model for real-world"""
        # Fine-tune or adapt the simulation model for real-world
        # This could involve:
        # - Fine-tuning on real data
        # - Adding domain adaptation layers
        # - Adjusting for real sensor characteristics
        return sim_model  # Simplified for this example

    def image_callback(self, msg):
        """Process camera image and run inference"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Preprocess image (adjust for real-world sensor characteristics)
            processed_image = self.preprocess_real_image(cv_image)

            # Run inference
            action = self.infer_action(processed_image)

            # Publish command
            cmd_vel = self.convert_action_to_cmd_vel(action)
            self.cmd_vel_pub.publish(cmd_vel)

            # Collect data for adaptation
            if self.adaptation_active:
                self.collect_real_data(processed_image, action)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def laser_callback(self, msg):
        """Process laser scan data"""
        # Process laser data for obstacle avoidance
        ranges = np.array(msg.ranges)
        # Filter out invalid ranges
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) > 0:
            min_range = np.min(valid_ranges)
            if min_range < 0.5:  # Emergency stop threshold
                # Emergency stop
                cmd_vel = Twist()
                self.cmd_vel_pub.publish(cmd_vel)

    def preprocess_real_image(self, image):
        """Preprocess real-world image to match simulation characteristics"""
        # Adjust for real-world camera characteristics
        # This might include:
        # - Color space adjustments
        # - Noise addition/removal
        # - Resolution matching
        # - Distortion correction

        # Example: adjust brightness/contrast to match simulation range
        adjusted_image = cv2.convertScaleAbs(image, alpha=1.1, beta=10)

        return adjusted_image

    def infer_action(self, processed_image):
        """Run inference to determine action"""
        # Convert image to tensor
        image_tensor = torch.FloatTensor(processed_image).permute(2, 0, 1).unsqueeze(0)

        # Run model inference
        with torch.no_grad():
            action_tensor = self.real_model(image_tensor)
            action = action_tensor.cpu().numpy()

        return action

    def convert_action_to_cmd_vel(self, action):
        """Convert neural network output to Twist command"""
        cmd_vel = Twist()

        # Example: action is [linear_vel, angular_vel]
        if len(action) >= 2:
            cmd_vel.linear.x = float(action[0])
            cmd_vel.angular.z = float(action[1])

        return cmd_vel

    def collect_real_data(self, image, action):
        """Collect real-world data for adaptation"""
        data_point = {
            'image': image,
            'action': action,
            'timestamp': self.get_clock().now().nanoseconds
        }

        self.real_data_buffer.append(data_point)

        # Keep buffer size manageable
        if len(self.real_data_buffer) > self.buffer_size:
            self.real_data_buffer.pop(0)

    def evaluate_transfer_quality(self):
        """Evaluate how well the simulation model works in reality"""
        # This would involve comparing expected vs. actual behavior
        # For now, return a simple metric based on consistency
        if len(self.real_data_buffer) < 10:
            return 0.0

        # Calculate consistency of actions over time
        recent_actions = [data['action'] for data in self.real_data_buffer[-10:]]
        action_variance = np.var(recent_actions)

        # Lower variance indicates more consistent behavior (potentially better transfer)
        quality = max(0.0, 1.0 - action_variance)
        return min(quality, 1.0)
```

## Progressive Domain Randomization

### Adaptive Randomization

```python
class AdaptiveDomainRandomizer:
    def __init__(self):
        self.randomization_strength = 1.0  # Start with high randomization
        self.performance_threshold = 0.8   # Performance threshold
        self.min_randomization = 0.1       # Minimum randomization level
        self.randomization_decay = 0.99    # Decay rate

        self.performance_history = []
        self.max_history = 50

    def update_randomization(self, current_performance):
        """Adaptively adjust domain randomization based on performance"""
        # Add current performance to history
        self.performance_history.append(current_performance)

        if len(self.performance_history) > self.max_history:
            self.performance_history.pop(0)

        # Calculate recent average performance
        if len(self.performance_history) > 10:
            avg_performance = np.mean(self.performance_history[-10:])

            # If performance is good, reduce randomization
            if avg_performance > self.performance_threshold:
                self.randomization_strength *= self.randomization_decay
                self.randomization_strength = max(
                    self.randomization_strength,
                    self.min_randomization
                )
            else:
                # If performance drops, increase randomization
                self.randomization_strength = min(
                    self.randomization_strength * 1.01,
                    1.0
                )

    def get_randomization_params(self):
        """Get current randomization parameters"""
        return {
            'lighting_variation': self.randomization_strength,
            'texture_variation': self.randomization_strength,
            'physics_variation': self.randomization_strength * 0.5,  # Less physics variation
            'sensor_noise': self.randomization_strength
        }
```

## System Identification for Sim-to-Real

### Dynamics Model Learning

```python
class DynamicsLearner:
    def __init__(self):
        self.sim_dynamics = None  # Simulation dynamics model
        self.residual_model = None  # Learned residual between sim and real
        self.data_buffer = []
        self.buffer_size = 1000

    def collect_data_pair(self, sim_state, real_state, action):
        """Collect paired simulation and real-world data"""
        data_point = {
            'sim_state': sim_state,
            'real_state': real_state,
            'action': action,
            'residual': real_state - sim_state  # Simple residual
        }

        self.data_buffer.append(data_point)

        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)

    def train_residual_model(self):
        """Train model to predict simulation-to-reality residual"""
        if len(self.data_buffer) < 100:
            return  # Not enough data yet

        # Prepare training data
        X = []  # State-action pairs
        y = []  # Residuals

        for data_point in self.data_buffer:
            state_action = np.concatenate([
                data_point['sim_state'],
                data_point['action']
            ])
            X.append(state_action)
            y.append(data_point['residual'])

        X = np.array(X)
        y = np.array(y)

        # Train residual model (using simple neural network)
        self.residual_model = self.train_neural_network(X, y)

    def train_neural_network(self, X, y):
        """Train a neural network to predict residuals"""
        import torch.nn as nn
        import torch.optim as optim

        class ResidualNet(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, output_dim)
                )

            def forward(self, x):
                return self.network(x)

        model = ResidualNet(X.shape[1], y.shape[1])
        optimizer = optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        for epoch in range(100):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()

        return model

    def predict_real_state(self, sim_state, action):
        """Predict real-world state given simulation state and action"""
        if self.residual_model is None:
            return sim_state  # If no residual model, return sim state

        state_action = np.concatenate([sim_state, action])
        state_action_tensor = torch.FloatTensor(state_action).unsqueeze(0)

        with torch.no_grad():
            residual = self.residual_model(state_action_tensor).numpy().flatten()

        return sim_state + residual
```

## Real Robot Testing

### Hardware Interface Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import numpy as np

class RealRobotInterface(Node):
    def __init__(self):
        super().__init__('real_robot_interface')

        # Robot-specific publishers and subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)

        # Robot state storage
        self.current_joint_positions = {}
        self.current_pose = None
        self.current_twist = None
        self.current_imu = None

        # Robot parameters (adjust for your specific robot)
        self.wheel_radius = 0.05  # meters
        self.wheel_separation = 0.3  # meters

    def joint_state_callback(self, msg):
        """Update joint state information"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]

    def odom_callback(self, msg):
        """Update odometry information"""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def imu_callback(self, msg):
        """Update IMU information"""
        self.current_imu = msg

    def send_velocity_command(self, linear_vel, angular_vel):
        """Send velocity command to robot"""
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_vel
        cmd_vel.angular.z = angular_vel
        self.cmd_vel_pub.publish(cmd_vel)

    def send_joint_commands(self, joint_positions):
        """Send joint position commands"""
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        joint_cmd.name = list(joint_positions.keys())
        joint_cmd.position = list(joint_positions.values())
        self.joint_cmd_pub.publish(joint_cmd)

    def get_robot_state(self):
        """Get current robot state as a feature vector"""
        if (self.current_pose is None or
            self.current_twist is None or
            self.current_imu is None):
            return None

        # Create state vector [x, y, theta, linear_vel, angular_vel, imu_x, imu_y, imu_z]
        state = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y,
            self.get_yaw_from_quaternion(self.current_pose.orientation),
            self.current_twist.linear.x,
            self.current_twist.angular.z,
            self.current_imu.linear_acceleration.x,
            self.current_imu.linear_acceleration.y,
            self.current_imu.linear_acceleration.z
        ])

        return state

    def get_yaw_from_quaternion(self, quat):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return np.arctan2(siny_cosp, cosy_cosp)
```

## Transfer Evaluation

### Performance Comparison

```python
class TransferEvaluator:
    def __init__(self):
        self.sim_performance = []
        self.real_performance = []
        self.transfer_gap = []

    def evaluate_task_performance(self, agent, sim_env, real_env, num_episodes=20):
        """Compare performance in simulation vs. real world"""
        # Evaluate in simulation
        sim_rewards = []
        for episode in range(num_episodes):
            state = sim_env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.act(state)
                state, reward, done, _ = sim_env.step(action)
                total_reward += reward

            sim_rewards.append(total_reward)

        # Evaluate in real world (or realistic simulation)
        real_rewards = []
        for episode in range(num_episodes):
            state = real_env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.act(state)
                state, reward, done, _ = real_env.step(action)
                total_reward += reward

            real_rewards.append(total_reward)

        # Calculate transfer metrics
        sim_avg = np.mean(sim_rewards)
        real_avg = np.mean(real_rewards)
        gap = (sim_avg - real_avg) / sim_avg if sim_avg != 0 else 0

        self.sim_performance.append(sim_avg)
        self.real_performance.append(real_avg)
        self.transfer_gap.append(gap)

        return {
            'sim_avg_reward': sim_avg,
            'real_avg_reward': real_avg,
            'transfer_gap': gap,
            'transfer_efficiency': (real_avg / sim_avg) if sim_avg != 0 else 0
        }

    def plot_transfer_results(self):
        """Plot transfer learning results"""
        import matplotlib.pyplot as plt

        episodes = range(len(self.transfer_gap))

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(episodes, self.sim_performance, label='Simulation')
        plt.plot(episodes, self.real_performance, label='Real World')
        plt.title('Performance Comparison')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(episodes, self.transfer_gap)
        plt.title('Transfer Gap')
        plt.xlabel('Episode')
        plt.ylabel('Gap')

        plt.subplot(1, 3, 3)
        transfer_efficiency = [(r/s) if s != 0 else 0 for s, r in zip(self.sim_performance, self.real_performance)]
        plt.plot(episodes, transfer_efficiency)
        plt.title('Transfer Efficiency')
        plt.xlabel('Episode')
        plt.ylabel('Efficiency')

        plt.tight_layout()
        plt.show()
```

## Exercise Tasks

1. Implement domain randomization in your Isaac Sim environment
2. Create a domain adaptation network to bridge sim and real domains
3. Set up a transfer learning pipeline with progressive randomization
4. Implement system identification to learn sim-to-real dynamics
5. Test your trained simulation model on real hardware (or realistic simulation)
6. Evaluate and compare performance between simulation and reality

## Troubleshooting

### Common Issues

- **Poor transfer performance**: Increase domain randomization range
- **Training instability**: Reduce learning rate during fine-tuning
- **Sensor differences**: Implement sensor calibration and preprocessing
- **Actuator delays**: Account for real-world timing differences

## Summary

In this lab, you learned techniques for transferring AI models from simulation to real robots. You implemented domain randomization, domain adaptation, and system identification methods to bridge the reality gap. These techniques are crucial for deploying simulation-trained models in real-world robotic applications.