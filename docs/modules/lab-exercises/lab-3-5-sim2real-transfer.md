# Lab 3.5: Sim-to-Real Transfer

## Overview

In this lab, you will learn about sim-to-real transfer techniques for robotics applications. You'll explore domain randomization, system identification, model adaptation, and validation methods to bridge the gap between simulation and real-world robotics. This includes understanding the reality gap, implementing transfer learning techniques, and validating performance on physical robots.

## Objectives

By the end of this lab, you will be able to:
- Understand the challenges of sim-to-real transfer
- Implement domain randomization techniques
- Perform system identification for robot modeling
- Apply model adaptation and fine-tuning methods
- Validate performance in both simulation and reality
- Implement safety measures for real-world deployment
- Design experiments to evaluate transfer effectiveness

## Prerequisites

- Completion of Lab 3.1-3.4: Isaac Sim Setup, Perception, Planning/Control, and RL
- Understanding of system identification concepts
- Experience with reinforcement learning and domain randomization
- Basic knowledge of control theory and robot dynamics

## Duration

5-6 hours

## Exercise 1: Understanding the Reality Gap

### Step 1: Create a reality gap analysis environment

Create `~/isaac_sim_examples/reality_gap_analysis.py`:

```python
#!/usr/bin/env python3
# reality_gap_analysis.py
"""Reality gap analysis for sim-to-real transfer."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class RealityGapAnalyzer:
    """Analyze and quantify the reality gap between simulation and reality."""

    def __init__(self):
        self.simulation_data = []
        self.real_data = []
        self.gap_metrics = {}
        self.transfer_functions = {}

    def collect_simulation_data(self, num_samples=1000):
        """Collect data from simulation environment."""
        print("Collecting simulation data...")

        # Simulate various robot behaviors in simulation
        for i in range(num_samples):
            # Generate random control inputs
            control_input = np.random.uniform(-1, 1, size=6)  # 6 DoF control

            # Simulate robot response in simulation (with idealized physics)
            sim_state = self.simulate_robot_response(control_input)

            self.simulation_data.append({
                'control_input': control_input,
                'state': sim_state,
                'timestamp': i
            })

        print(f"Collected {len(self.simulation_data)} simulation samples")

    def collect_real_data(self, num_samples=500):
        """Collect data from real robot (simulated for this example)."""
        print("Collecting real robot data...")

        # In a real scenario, this would interface with the actual robot
        # For simulation, we'll add realistic noise and dynamics differences
        for i in range(num_samples):
            # Use same control inputs as simulation for comparison
            if i < len(self.simulation_data):
                control_input = self.simulation_data[i]['control_input']
            else:
                control_input = np.random.uniform(-1, 1, size=6)

            # Simulate real robot response (with realistic noise and dynamics)
            real_state = self.simulate_real_robot_response(control_input)

            self.real_data.append({
                'control_input': control_input,
                'state': real_state,
                'timestamp': i
            })

        print(f"Collected {len(self.real_data)} real robot samples")

    def simulate_robot_response(self, control_input):
        """Simulate idealized robot response in simulation."""
        # Simplified robot dynamics model
        # In reality, this would be the simulation environment
        dt = 0.01  # Time step

        # Apply control input with idealized dynamics
        position = control_input[:3] * dt * 0.5  # Position change
        velocity = control_input[:3]  # Velocity
        orientation = control_input[3:6] * dt * 0.1  # Orientation change

        state = np.concatenate([position, velocity, orientation])
        return state

    def simulate_real_robot_response(self, control_input):
        """Simulate realistic robot response with noise and dynamics differences."""
        # Simulate real-world effects:
        # 1. Sensor noise
        # 2. Actuator delays
        # 3. Unmodeled dynamics
        # 4. Environmental factors

        dt = 0.01  # Time step

        # Start with ideal response
        ideal_position = control_input[:3] * dt * 0.4  # Slightly different gain
        ideal_velocity = control_input[:3] * 0.9  # Different velocity scaling
        ideal_orientation = control_input[3:6] * dt * 0.08  # Different orientation scaling

        # Add realistic noise
        noise_level = 0.05
        position_noise = np.random.normal(0, noise_level * 0.1, size=3)
        velocity_noise = np.random.normal(0, noise_level * 0.05, size=3)
        orientation_noise = np.random.normal(0, noise_level * 0.02, size=3)

        # Add systematic differences
        systematic_error = np.array([
            control_input[0] * 0.02,  # Position bias
            control_input[1] * 0.01,
            control_input[2] * 0.015,
            0, 0, 0  # No systematic error in velocity/orientation for simplicity
        ])

        state = np.concatenate([
            ideal_position + position_noise,
            ideal_velocity + velocity_noise,
            ideal_orientation + orientation_noise
        ]) + systematic_error

        return state

    def analyze_reality_gap(self):
        """Analyze the reality gap between simulation and real data."""
        print("Analyzing reality gap...")

        if len(self.simulation_data) == 0 or len(self.real_data) == 0:
            print("Need both simulation and real data to analyze reality gap")
            return

        # Align data by control input (assuming same control inputs were used)
        min_samples = min(len(self.simulation_data), len(self.real_data))

        sim_states = np.array([d['state'] for d in self.simulation_data[:min_samples]])
        real_states = np.array([d['state'] for d in self.real_data[:min_samples]])

        # Calculate various gap metrics
        mse = mean_squared_error(sim_states, real_states)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(sim_states - real_states))
        r2 = r2_score(sim_states, real_states)

        # Calculate state-wise differences
        state_differences = np.abs(sim_states - real_states)
        avg_state_diff = np.mean(state_differences, axis=0)
        std_state_diff = np.std(state_differences, axis=0)

        # Calculate correlation between sim and real states
        correlations = []
        for i in range(sim_states.shape[1]):
            corr = np.corrcoef(sim_states[:, i], real_states[:, i])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)

        self.gap_metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'avg_state_difference': avg_state_diff,
            'std_state_difference': std_state_diff,
            'correlations': correlations,
            'sample_size': min_samples
        }

        print(f"Reality Gap Analysis Results:")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R² Score: {r2:.6f}")
        print(f"  Average State Difference: {avg_state_diff}")
        print(f"  State Correlations: {correlations}")

        return self.gap_metrics

    def visualize_reality_gap(self):
        """Visualize the reality gap analysis."""
        if not self.gap_metrics:
            print("Run analyze_reality_gap() first")
            return

        # Prepare data for visualization
        min_samples = self.gap_metrics['sample_size']
        sim_states = np.array([d['state'] for d in self.simulation_data[:min_samples]])
        real_states = np.array([d['state'] for d in self.real_data[:min_samples]])

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        state_names = ['Pos X', 'Pos Y', 'Pos Z', 'Vel X', 'Vel Y', 'Vel Z']

        for i in range(6):
            axes[i].scatter(sim_states[:, i], real_states[:, i], alpha=0.6, s=1)
            axes[i].plot([sim_states[:, i].min(), sim_states[:, i].max()],
                        [sim_states[:, i].min(), sim_states[:, i].max()], 'r--', lw=2)
            axes[i].set_xlabel(f'Simulated {state_names[i]}')
            axes[i].set_ylabel(f'Real {state_names[i]}')
            axes[i].set_title(f'{state_names[i]}: R² = {self.gap_metrics["correlations"][i]:.3f}')

        plt.tight_layout()
        plt.savefig('reality_gap_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Plot state differences over time
        fig, ax = plt.subplots(figsize=(12, 6))
        time_steps = range(min_samples)
        state_diff = np.abs(sim_states - real_states)

        for i in range(6):
            ax.plot(time_steps, state_diff[:, i], label=state_names[i], alpha=0.7)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Absolute Difference')
        ax.set_title('State Differences Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('state_differences_over_time.png', dpi=300, bbox_inches='tight')
        plt.show()

    def calculate_transfer_efficiency(self):
        """Calculate transfer efficiency metrics."""
        if not self.gap_metrics:
            print("Run analyze_reality_gap() first")
            return

        # Transfer efficiency based on correlation
        avg_correlation = np.mean(self.gap_metrics['correlations'])

        # Transfer efficiency based on prediction accuracy
        efficiency_score = max(0, min(1, avg_correlation))  # Clamp between 0 and 1

        # Additional metrics
        prediction_accuracy = 1.0 / (1.0 + self.gap_metrics['rmse'])  # Higher RMSE = lower accuracy

        transfer_metrics = {
            'correlation_based_efficiency': avg_correlation,
            'prediction_accuracy': prediction_accuracy,
            'overall_efficiency': (avg_correlation + prediction_accuracy) / 2
        }

        print(f"Transfer Efficiency Metrics:")
        for metric, value in transfer_metrics.items():
            print(f"  {metric}: {value:.4f}")

        return transfer_metrics

# Example usage
if __name__ == "__main__":
    analyzer = RealityGapAnalyzer()
    analyzer.collect_simulation_data(num_samples=1000)
    analyzer.collect_real_data(num_samples=800)
    gap_metrics = analyzer.analyze_reality_gap()
    transfer_metrics = analyzer.calculate_transfer_efficiency()
    analyzer.visualize_reality_gap()

    print("\nReality gap analysis completed successfully")
```

## Exercise 2: Domain Randomization Implementation

### Step 1: Create domain randomization system

Create `~/isaac_sim_examples/domain_randomization_system.py`:

```python
#!/usr/bin/env python3
# domain_randomization_system.py
"""Domain randomization system for sim-to-real transfer."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random

class DomainRandomizationSystem:
    """System for implementing domain randomization."""

    def __init__(self):
        self.randomization_ranges = {
            'physics': {
                'mass_ratio': [0.8, 1.2],
                'friction': [0.5, 1.5],
                'restitution': [0.0, 0.2],
                'damping_ratio': [0.8, 1.2],
                'actuator_strength': [0.8, 1.2],
                'gravity': [0.8, 1.2]  # Scaling factor for gravity
            },
            'visual': {
                'lighting_intensity': [0.5, 2.0],
                'lighting_color_temp': [0.8, 1.2],
                'material_albedo': [0.5, 1.0],
                'material_roughness': [0.0, 1.0],
                'material_metallic': [0.0, 1.0]
            },
            'sensor': {
                'noise_std': [0.0, 0.05],
                'bias': [-0.02, 0.02],
                'delay_ms': [0, 20],
                'dropout_rate': [0.0, 0.1]
            }
        }

        self.current_parameters = {}
        self.parameter_history = defaultdict(list)
        self.episode_count = 0

    def randomize_physics_parameters(self):
        """Randomize physics parameters."""
        for param, range_vals in self.randomization_ranges['physics'].items():
            value = np.random.uniform(range_vals[0], range_vals[1])
            self.current_parameters[param] = value
            self.parameter_history[param].append(value)

        print(f"Physics parameters randomized: {self.current_parameters}")

    def randomize_visual_parameters(self):
        """Randomize visual parameters."""
        for param, range_vals in self.randomization_ranges['visual'].items():
            value = np.random.uniform(range_vals[0], range_vals[1])
            self.current_parameters[param] = value
            self.parameter_history[param].append(value)

        print(f"Visual parameters randomized: {self.current_parameters}")

    def randomize_sensor_parameters(self):
        """Randomize sensor parameters."""
        for param, range_vals in self.randomization_ranges['sensor'].items():
            value = np.random.uniform(range_vals[0], range_vals[1])
            self.current_parameters[param] = value
            self.parameter_history[param].append(value)

        print(f"Sensor parameters randomized: {self.current_parameters}")

    def update_randomization(self, episode_interval=10):
        """Update randomization parameters based on episode count."""
        if self.episode_count % episode_interval == 0:
            self.randomize_physics_parameters()
            self.randomize_visual_parameters()
            self.randomize_sensor_parameters()

        self.episode_count += 1

    def get_randomized_parameters(self):
        """Get current randomized parameters."""
        return self.current_parameters.copy()

    def apply_to_simulation(self, sim_env):
        """Apply current randomization parameters to simulation environment."""
        # This would typically involve updating the physics engine parameters
        # For demonstration, we'll just print what would be applied

        print("Applying randomization to simulation:")
        for param, value in self.current_parameters.items():
            print(f"  {param}: {value}")

        # In a real implementation, you would:
        # - Update physics properties in the simulator
        # - Change material properties
        # - Adjust lighting conditions
        # - Add sensor noise/models

    def calculate_diversity_score(self):
        """Calculate diversity score of randomization."""
        if not self.parameter_history:
            return 0.0

        diversity_scores = []
        for param, values in self.parameter_history.items():
            if len(values) > 1:
                # Calculate standard deviation as a measure of diversity
                std_dev = np.std(values)
                diversity_scores.append(std_dev)

        return np.mean(diversity_scores) if diversity_scores else 0.0

    def get_randomization_statistics(self):
        """Get statistics about randomization."""
        stats = {}
        for param, values in self.parameter_history.items():
            if values:
                stats[param] = {
                    'min': np.min(values),
                    'max': np.max(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values)
                }

        return stats

class AdaptiveDomainRandomization(DomainRandomizationSystem):
    """Adaptive domain randomization that adjusts based on training progress."""

    def __init__(self):
        super().__init__()
        self.performance_history = []
        self.diversity_threshold = 0.1
        self.adaptation_enabled = True

    def update_with_performance(self, episode_reward, episode_success):
        """Update randomization based on performance."""
        self.performance_history.append({
            'reward': episode_reward,
            'success': episode_success,
            'episode': self.episode_count
        })

        # If performance is plateauing, increase diversity
        if len(self.performance_history) >= 10:
            recent_performance = self.performance_history[-10:]
            avg_recent = np.mean([p['reward'] for p in recent_performance])

            # Calculate trend
            if len(self.performance_history) >= 20:
                older_performance = self.performance_history[-20:-10]
                avg_older = np.mean([p['reward'] for p in older_performance])

                # If no improvement, increase randomization diversity
                if avg_recent <= avg_older:
                    self.increase_diversity()

        self.episode_count += 1

    def increase_diversity(self):
        """Increase diversity of randomization ranges."""
        print("Increasing randomization diversity due to plateauing performance")

        # Expand randomization ranges
        for category in self.randomization_ranges.values():
            for param, range_vals in category.items():
                center = (range_vals[0] + range_vals[1]) / 2
                width = range_vals[1] - range_vals[0]

                # Increase range by 10%
                new_width = width * 1.1
                new_range = [center - new_width/2, center + new_width/2]

                # Ensure reasonable bounds
                new_range[0] = max(0, new_range[0])  # Lower bound
                new_range[1] = min(2.0, new_range[1])  # Upper bound

                category[param] = new_range

    def decrease_diversity(self):
        """Decrease diversity of randomization ranges."""
        print("Decreasing randomization diversity due to good performance")

        # Shrink randomization ranges
        for category in self.randomization_ranges.values():
            for param, range_vals in category.items():
                center = (range_vals[0] + range_vals[1]) / 2
                width = range_vals[1] - range_vals[0]

                # Decrease range by 5%
                new_width = max(0.1, width * 0.95)  # Minimum width of 0.1
                new_range = [center - new_width/2, center + new_width/2]

                category[param] = new_range

# Example usage
if __name__ == "__main__":
    # Basic domain randomization
    dr_system = DomainRandomizationSystem()

    print("Basic Domain Randomization System:")
    for i in range(5):
        dr_system.update_randomization(episode_interval=1)  # Update every episode
        params = dr_system.get_randomized_parameters()
        print(f"Episode {i+1} parameters: {params}")

    diversity = dr_system.calculate_diversity_score()
    print(f"Diversity score: {diversity:.4f}")

    stats = dr_system.get_randomization_statistics()
    print(f"Statistics calculated for {len(stats)} parameters")

    # Adaptive domain randomization
    adaptive_dr = AdaptiveDomainRandomization()

    print("\nAdaptive Domain Randomization:")
    for i in range(10):
        # Simulate some performance metrics
        reward = np.random.uniform(0, 100)
        success = random.random() > 0.5

        adaptive_dr.update_with_performance(reward, success)
        params = adaptive_dr.get_randomized_parameters()
        print(f"Episode {i+1}: Reward={reward:.2f}, Success={success}")

    print(f"Final diversity score: {adaptive_dr.calculate_diversity_score():.4f}")

    print("\nDomain randomization system completed successfully")
```

## Exercise 3: System Identification

### Step 1: Create system identification tools

Create `~/isaac_sim_examples/system_identification.py`:

```python
#!/usr/bin/env python3
# system_identification.py
"""System identification for robot modeling and sim-to-real transfer."""

import numpy as np
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim

class SystemIdentifier:
    """System identification for robot dynamics modeling."""

    def __init__(self, robot_type='wheeled'):
        self.robot_type = robot_type
        self.model = None
        self.input_data = []
        self.output_data = []
        self.identification_method = 'linear'
        self.model_order = 2

    def collect_data(self, inputs, outputs, timestamps=None):
        """Collect input-output data for system identification."""
        if timestamps is None:
            timestamps = np.arange(len(inputs))

        self.input_data.extend(inputs)
        self.output_data.extend(outputs)

        print(f"Collected {len(inputs)} data points. Total: {len(self.input_data)}")

    def identify_model(self, method='linear', model_order=2):
        """Identify system model using collected data."""
        if len(self.input_data) < 10:
            print("Need at least 10 data points for system identification")
            return False

        self.identification_method = method
        self.model_order = model_order

        X = np.array(self.input_data)
        Y = np.array(self.output_data)

        if method == 'linear':
            self.model = self.identify_linear_model(X, Y)
        elif method == 'polynomial':
            self.model = self.identify_polynomial_model(X, Y, model_order)
        elif method == 'neural':
            self.model = self.identify_neural_model(X, Y)
        else:
            print(f"Unknown identification method: {method}")
            return False

        print(f"System model identified using {method} method")
        return True

    def identify_linear_model(self, X, Y):
        """Identify linear system model."""
        # Use ridge regression for regularization
        model = Ridge(alpha=1.0)
        model.fit(X, Y)
        return model

    def identify_polynomial_model(self, X, Y, order=2):
        """Identify polynomial system model."""
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=order)
        X_poly = poly_features.fit_transform(X)

        # Fit linear model on polynomial features
        model = Ridge(alpha=1.0)
        model.fit(X_poly, Y)

        # Store polynomial features for later use
        model.poly_features = poly_features
        return model

    def identify_neural_model(self, X, Y, hidden_size=64, epochs=1000):
        """Identify neural network model."""
        class NeuralDynamicsModel(nn.Module):
            def __init__(self, input_size, output_size, hidden_size):
                super(NeuralDynamicsModel, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, output_size)
                )

            def forward(self, x):
                return self.network(x)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        Y_tensor = torch.FloatTensor(Y)

        # Create model
        model = NeuralDynamicsModel(X.shape[1], Y.shape[1], hidden_size)

        # Train model
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, Y_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 200 == 0:
                print(f'Neural model training: Epoch {epoch}, Loss: {loss.item():.6f}')

        return model

    def predict(self, inputs):
        """Predict system output using identified model."""
        if self.model is None:
            print("Model not identified yet")
            return None

        inputs = np.array(inputs)

        if self.identification_method == 'neural':
            with torch.no_grad():
                input_tensor = torch.FloatTensor(inputs)
                if len(input_tensor.shape) == 1:
                    input_tensor = input_tensor.unsqueeze(0)
                output_tensor = self.model(input_tensor)
                return output_tensor.numpy()
        else:
            if self.identification_method == 'polynomial':
                inputs = self.model.poly_features.transform(inputs.reshape(1, -1))
            return self.model.predict(inputs)

    def evaluate_model(self, test_inputs, test_outputs):
        """Evaluate identified model performance."""
        if self.model is None:
            print("Model not identified yet")
            return None

        predictions = self.predict(test_inputs)
        if predictions is None:
            return None

        mse = np.mean((predictions - test_outputs) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - test_outputs))

        # Calculate R² score
        ss_res = np.sum((test_outputs - predictions) ** 2)
        ss_tot = np.sum((test_outputs - np.mean(test_outputs)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2
        }

        print(f"Model Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")

        return metrics

    def visualize_identification(self):
        """Visualize system identification results."""
        if self.model is None or len(self.input_data) == 0:
            print("No model or data available for visualization")
            return

        X = np.array(self.input_data)
        Y = np.array(self.output_data)

        predictions = self.predict(X)

        if predictions is None:
            return

        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Actual vs Predicted
        axes[0, 0].scatter(Y.flatten(), predictions.flatten(), alpha=0.6)
        min_val = min(Y.min(), predictions.min())
        max_val = max(Y.max(), predictions.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title('Actual vs Predicted')

        # Plot 2: Prediction errors
        errors = Y.flatten() - predictions.flatten()
        axes[0, 1].hist(errors, bins=50, alpha=0.7)
        axes[0, 1].set_xlabel('Prediction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Prediction Errors')

        # Plot 3: Time series comparison
        time_steps = range(len(Y))
        axes[1, 0].plot(time_steps, Y.flatten(), label='Actual', alpha=0.7)
        axes[1, 0].plot(time_steps, predictions.flatten(), label='Predicted', alpha=0.7)
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Time Series Comparison')
        axes[1, 0].legend()

        # Plot 4: Error over time
        axes[1, 1].plot(time_steps, errors, alpha=0.7)
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].set_title('Prediction Error Over Time')

        plt.tight_layout()
        plt.savefig('system_identification_results.png', dpi=300, bbox_inches='tight')
        plt.show()

class PhysicsParameterEstimator:
    """Estimate physics parameters from system identification results."""

    def __init__(self):
        self.estimated_parameters = {}
        self.uncertainty_bounds = {}

    def estimate_mass_properties(self, acceleration_data, force_data):
        """Estimate mass properties using F = ma."""
        # F = ma => m = F/a
        # Use least squares to estimate mass from multiple measurements
        valid_indices = acceleration_data != 0  # Avoid division by zero

        if np.sum(valid_indices) < 1:
            return None

        accelerations = acceleration_data[valid_indices]
        forces = force_data[valid_indices]

        # Estimate mass using least squares
        masses = forces / accelerations
        estimated_mass = np.mean(masses)
        uncertainty = np.std(masses)

        self.estimated_parameters['mass'] = estimated_mass
        self.uncertainty_bounds['mass'] = uncertainty

        print(f"Estimated mass: {estimated_mass:.4f} ± {uncertainty:.4f} kg")

    def estimate_friction_coefficients(self, velocity_data, force_data):
        """Estimate friction coefficients."""
        # Model: F_friction = mu * N (normal force)
        # For horizontal motion: F_applied - F_friction = ma
        # So F_friction = F_applied - ma

        # This is a simplified model - in reality, you'd need more complex analysis
        friction_estimates = force_data  # Simplified for this example

        avg_friction = np.mean(friction_estimates)
        friction_uncertainty = np.std(friction_estimates)

        self.estimated_parameters['friction'] = avg_friction
        self.uncertainty_bounds['friction'] = friction_uncertainty

        print(f"Estimated friction: {avg_friction:.4f} ± {friction_uncertainty:.4f} N")

    def estimate_damping_coefficients(self, velocity_data, force_data):
        """Estimate damping coefficients."""
        # Model: F_damping = b * v (linear damping)
        # Use least squares to estimate damping coefficient b

        # Only use data where velocity is not zero
        valid_indices = velocity_data != 0

        if np.sum(valid_indices) < 2:
            return None

        velocities = velocity_data[valid_indices]
        forces = force_data[valid_indices]

        # Estimate damping coefficient using least squares
        # F = b*v => b = F/v (when v != 0)
        damping_coeffs = forces / velocities
        estimated_damping = np.mean(damping_coeffs)
        uncertainty = np.std(damping_coeffs)

        self.estimated_parameters['damping'] = estimated_damping
        self.uncertainty_bounds['damping'] = uncertainty

        print(f"Estimated damping: {estimated_damping:.4f} ± {uncertainty:.4f} Ns/m")

    def get_estimated_parameters(self):
        """Get estimated physics parameters."""
        return self.estimated_parameters.copy()

    def get_uncertainty_bounds(self):
        """Get uncertainty bounds for estimated parameters."""
        return self.uncertainty_bounds.copy()

# Example usage
if __name__ == "__main__":
    print("System Identification Example")

    # Create system identifier
    sys_id = SystemIdentifier(robot_type='wheeled')

    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000

    # Inputs: control commands (forces, torques)
    inputs = np.random.uniform(-10, 10, size=(n_samples, 3))  # 3 DoF control

    # Outputs: resulting states (position, velocity changes)
    # Simulate some dynamics: y = 0.5*x + noise
    outputs = 0.5 * inputs + 0.1 * np.random.randn(n_samples, 3)

    # Add some non-linearities
    outputs += 0.05 * inputs**2

    # Collect data
    sys_id.collect_data(inputs, outputs)

    # Identify models using different methods
    methods = ['linear', 'polynomial', 'neural']

    for method in methods:
        print(f"\nIdentifying model using {method} method...")
        success = sys_id.identify_model(method=method, model_order=2)

        if success:
            # Evaluate model
            test_inputs = inputs[-100:]  # Use last 100 samples as test
            test_outputs = outputs[-100:]
            metrics = sys_id.evaluate_model(test_inputs, test_outputs)

            if metrics:
                print(f"{method.upper()} model - R² Score: {metrics['r2_score']:.4f}")

    # Visualize results for the best model (in practice, you'd select based on metrics)
    sys_id.visualize_identification()

    # Physics parameter estimation
    print("\nPhysics Parameter Estimation:")
    physics_estimator = PhysicsParameterEstimator()

    # Example: estimate parameters from collected data
    velocity_data = outputs[:, 0]  # Use first output as velocity
    force_data = inputs[:, 0]      # Use first input as force

    physics_estimator.estimate_mass_properties(velocity_data, force_data)
    physics_estimator.estimate_friction_coefficients(velocity_data, force_data)
    physics_estimator.estimate_damping_coefficients(velocity_data, force_data)

    estimated_params = physics_estimator.get_estimated_parameters()
    print(f"Final estimated parameters: {estimated_params}")

    print("\nSystem identification completed successfully")
```

## Exercise 4: Model Adaptation and Fine-Tuning

### Step 1: Create model adaptation system

Create `~/isaac_sim_examples/model_adaptation.py`:

```python
#!/usr/bin/env python3
# model_adaptation.py
"""Model adaptation and fine-tuning for sim-to-real transfer."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import copy

class ModelAdapter:
    """Adapt simulation models for real-world deployment."""

    def __init__(self, simulation_model):
        self.simulation_model = simulation_model
        self.adapted_model = None
        self.scaler = StandardScaler()
        self.adaptation_history = []

    def adapt_with_real_data(self, real_inputs, real_outputs, adaptation_method='fine_tune'):
        """Adapt model using real-world data."""
        print(f"Adapting model using {adaptation_method} method...")

        # Standardize the data
        real_inputs_scaled = self.scaler.fit_transform(real_inputs)
        real_outputs_scaled = self.scaler.fit_transform(real_outputs)

        if adaptation_method == 'fine_tune':
            self.adapted_model = self.fine_tune_model(real_inputs_scaled, real_outputs_scaled)
        elif adaptation_method == 'domain_adaptation':
            self.adapted_model = self.domain_adaptation(real_inputs_scaled, real_outputs_scaled)
        elif adaptation_method == 'ensemble':
            self.adapted_model = self.ensemble_adaptation(real_inputs_scaled, real_outputs_scaled)
        else:
            print(f"Unknown adaptation method: {adaptation_method}")
            return False

        print(f"Model adaptation completed using {adaptation_method} method")
        return True

    def fine_tune_model(self, real_inputs, real_outputs, learning_rate=1e-4, epochs=100):
        """Fine-tune simulation model with real data."""
        # Clone the simulation model
        adapted_model = copy.deepcopy(self.simulation_model)
        adapted_model.train()

        # Convert to tensors
        inputs_tensor = torch.FloatTensor(real_inputs)
        outputs_tensor = torch.FloatTensor(real_outputs)

        # Create dataset and dataloader
        dataset = TensorDataset(inputs_tensor, outputs_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Use the same optimizer as the original model (if available)
        # For this example, we'll create a new optimizer
        optimizer = optim.Adam(adapted_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Fine-tuning loop
        for epoch in range(epochs):
            total_loss = 0
            for batch_inputs, batch_outputs in dataloader:
                optimizer.zero_grad()
                predictions = adapted_model(batch_inputs)
                loss = criterion(predictions, batch_outputs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f'Fine-tuning Epoch {epoch}, Average Loss: {avg_loss:.6f}')

        return adapted_model

    def domain_adaptation(self, real_inputs, real_outputs):
        """Implement domain adaptation using adversarial training."""
        # Create domain adaptation model
        class DomainAdaptationModel(nn.Module):
            def __init__(self, base_model):
                super(DomainAdaptationModel, self).__init__()
                self.base_model = base_model
                # Add domain classifier
                self.domain_classifier = nn.Sequential(
                    nn.Linear(base_model.network[-1].out_features, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 2)  # Binary classification: sim vs real
                )

            def forward(self, x, domain_label=None):
                features = self.base_model(x)
                if domain_label is not None:
                    domain_pred = self.domain_classifier(features)
                    return features, domain_pred
                return features

        # This is a simplified implementation
        # In practice, you'd implement adversarial training
        adapted_model = self.fine_tune_model(real_inputs, real_outputs)
        return adapted_model

    def ensemble_adaptation(self, real_inputs, real_outputs):
        """Create ensemble of adapted models."""
        models = []

        # Create multiple models with different adaptations
        for i in range(3):  # Create 3 models
            # Add noise to real data to create variations
            noisy_inputs = real_inputs + np.random.normal(0, 0.01, real_inputs.shape)
            noisy_outputs = real_outputs + np.random.normal(0, 0.01, real_outputs.shape)

            model = self.fine_tune_model(noisy_inputs, noisy_outputs, epochs=50)
            models.append(model)

        # Create ensemble wrapper
        class EnsembleModel(nn.Module):
            def __init__(self, models):
                super(EnsembleModel, self).__init__()
                self.models = nn.ModuleList(models)

            def forward(self, x):
                outputs = []
                for model in self.models:
                    outputs.append(model(x))
                # Average predictions
                return torch.stack(outputs).mean(dim=0)

        ensemble_model = EnsembleModel(models)
        return ensemble_model

    def predict(self, inputs):
        """Make predictions using adapted model."""
        if self.adapted_model is None:
            print("Model not adapted yet")
            return None

        inputs_scaled = self.scaler.transform(inputs.reshape(1, -1))
        inputs_tensor = torch.FloatTensor(inputs_scaled)

        with torch.no_grad():
            prediction = self.adapted_model(inputs_tensor)

        # Inverse transform the prediction
        prediction_unscaled = self.scaler.inverse_transform(prediction.numpy())
        return prediction_unscaled

    def evaluate_adaptation(self, test_inputs, test_outputs):
        """Evaluate the adapted model."""
        if self.adapted_model is None:
            print("Model not adapted yet")
            return None

        predictions = []
        for input_vec in test_inputs:
            pred = self.predict(input_vec)
            if pred is not None:
                predictions.append(pred.flatten())

        if not predictions:
            return None

        predictions = np.array(predictions)

        # Calculate metrics
        mse = np.mean((predictions - test_outputs) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - test_outputs))

        # R² score
        ss_res = np.sum((test_outputs - predictions) ** 2)
        ss_tot = np.sum((test_outputs - np.mean(test_outputs)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2
        }

        print("Adapted Model Evaluation:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")

        return metrics

class OnlineAdaptationSystem:
    """Online adaptation system for continuous learning."""

    def __init__(self, base_model, adaptation_rate=0.01):
        self.base_model = base_model
        self.current_model = copy.deepcopy(base_model)
        self.adaptation_rate = adaptation_rate
        self.performance_threshold = 0.05  # Threshold for triggering adaptation
        self.performance_history = []

    def update_model_online(self, input_data, target_data, current_performance):
        """Update model online based on performance."""
        self.performance_history.append(current_performance)

        # Check if adaptation is needed
        if len(self.performance_history) >= 10:
            recent_avg = np.mean(self.performance_history[-10:])
            overall_avg = np.mean(self.performance_history)

            # If performance degrades significantly, adapt
            if recent_avg < overall_avg - self.performance_threshold:
                print("Performance degradation detected, adapting model...")
                self.adapt_online(input_data, target_data)

    def adapt_online(self, input_data, target_data):
        """Perform online adaptation."""
        # Convert to tensors
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
        target_tensor = torch.FloatTensor(target_data).unsqueeze(0)

        # Simple gradient step adaptation
        self.current_model.train()
        optimizer = optim.SGD(self.current_model.parameters(), lr=self.adaptation_rate)

        optimizer.zero_grad()
        prediction = self.current_model(input_tensor)
        loss = nn.MSELoss()(prediction, target_tensor)
        loss.backward()
        optimizer.step()

        print(f"Online adaptation completed. Loss: {loss.item():.6f}")

    def predict(self, input_data):
        """Make prediction with current model."""
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0)

        with torch.no_grad():
            self.current_model.eval()
            prediction = self.current_model(input_tensor)
            return prediction.numpy().flatten()

# Example usage
if __name__ == "__main__":
    print("Model Adaptation System Example")

    # Create a simple simulation model (for demonstration)
    class SimpleSimulationModel(nn.Module):
        def __init__(self, input_size=3, output_size=3):
            super(SimpleSimulationModel, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_size)
            )

        def forward(self, x):
            return self.network(x)

    # Create simulation model
    sim_model = SimpleSimulationModel()

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500

    # Real data (slightly different from simulation)
    real_inputs = np.random.uniform(-1, 1, size=(n_samples, 3))
    real_outputs = 0.8 * real_inputs + 0.05 * np.random.randn(n_samples, 3)  # Different scaling

    # Test data
    test_inputs = np.random.uniform(-1, 1, size=(100, 3))
    test_outputs = 0.8 * test_inputs + 0.05 * np.random.randn(100, 3)

    print("Original simulation model performance:")
    with torch.no_grad():
        sim_predictions = sim_model(torch.FloatTensor(test_inputs))
        sim_mse = nn.MSELoss()(sim_predictions, torch.FloatTensor(test_outputs)).item()
        print(f"  MSE: {sim_mse:.6f}")

    # Test model adaptation
    methods = ['fine_tune', 'ensemble']

    for method in methods:
        print(f"\nTesting {method} adaptation...")
        adapter = ModelAdapter(sim_model)
        success = adapter.adapt_with_real_data(real_inputs, real_outputs, method)

        if success:
            metrics = adapter.evaluate_adaptation(test_inputs, test_outputs)
            if metrics:
                print(f"  Adapted model MSE: {metrics['mse']:.6f}")

    # Test online adaptation
    print(f"\nTesting online adaptation...")
    online_adapter = OnlineAdaptationSystem(sim_model)

    # Simulate online learning scenario
    for i in range(20):
        # Generate new data point
        new_input = np.random.uniform(-1, 1, size=3)
        new_target = 0.8 * new_input + 0.05 * np.random.randn(3)

        # Get prediction and calculate performance
        prediction = online_adapter.predict(new_input)
        performance = -np.mean((prediction - new_target) ** 2)  # Negative MSE as performance

        # Update model if needed
        online_adapter.update_model_online(new_input, new_target, performance)

        if i % 5 == 0:
            print(f"  Step {i}: Performance = {performance:.4f}")

    print("\nModel adaptation system completed successfully")
```

## Exercise 5: Validation and Deployment

### Step 1: Create validation and deployment tools

Create `~/isaac_sim_examples/validation_deployment.py`:

```python
#!/usr/bin/env python3
# validation_deployment.py
"""Validation and deployment system for sim-to-real transfer."""

import numpy as np
import torch
import pickle
import json
import os
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class TransferValidationSystem:
    """System for validating sim-to-real transfer effectiveness."""

    def __init__(self):
        self.simulation_performance = {}
        self.real_world_performance = {}
        self.transfer_metrics = {}
        self.safety_checks = []
        self.validation_results = {}

    def run_comparative_validation(self, sim_model, real_model, test_scenarios):
        """Run comparative validation between simulation and real models."""
        print("Running comparative validation...")

        sim_results = []
        real_results = []

        for scenario in test_scenarios:
            # Test in simulation
            sim_result = self.test_model_in_environment(sim_model, scenario, 'simulation')
            sim_results.append(sim_result)

            # Test in real environment (simulated here)
            real_result = self.test_model_in_environment(real_model, scenario, 'real')
            real_results.append(real_result)

        # Calculate transfer metrics
        self.calculate_transfer_metrics(sim_results, real_results)

        return sim_results, real_results

    def test_model_in_environment(self, model, scenario, environment_type):
        """Test model in specific environment."""
        # Simulate testing the model in the given environment
        # This would involve running the model through various scenarios

        results = {
            'scenario': scenario['name'],
            'environment': environment_type,
            'success_rate': np.random.uniform(0.7, 0.95),  # Simulated results
            'completion_time': np.random.uniform(10, 30),  # seconds
            'energy_consumption': np.random.uniform(50, 100),  # arbitrary units
            'safety_violations': np.random.randint(0, 3),
            'accuracy': np.random.uniform(0.8, 0.98)
        }

        return results

    def calculate_transfer_metrics(self, sim_results, real_results):
        """Calculate transfer effectiveness metrics."""
        if len(sim_results) != len(real_results):
            print("Mismatch in simulation and real results length")
            return

        metrics = defaultdict(list)

        for sim_res, real_res in zip(sim_results, real_results):
            # Calculate transfer effectiveness for each metric
            metrics['success_rate_transfer'].append(
                (real_res['success_rate'] - sim_res['success_rate']) / sim_res['success_rate']
            )
            metrics['time_transfer'].append(
                (real_res['completion_time'] - sim_res['completion_time']) / sim_res['completion_time']
            )
            metrics['accuracy_transfer'].append(
                (real_res['accuracy'] - sim_res['accuracy']) / sim_res['accuracy']
            )

        # Calculate average metrics
        avg_metrics = {}
        for key, values in metrics.items():
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)

        self.transfer_metrics = avg_metrics
        print("Transfer metrics calculated:")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.4f}")

    def run_safety_validation(self, model, safety_constraints):
        """Run safety validation tests."""
        print("Running safety validation...")

        safety_results = []

        for constraint in safety_constraints:
            # Test each safety constraint
            constraint_result = {
                'constraint': constraint['name'],
                'passed': True,
                'violation_count': 0,
                'max_violation': 0.0,
                'details': []
            }

            # Simulate constraint testing
            for i in range(100):  # Run 100 test cases
                # Generate random inputs that might violate constraints
                test_input = np.random.uniform(-2, 2, size=6)

                # Check if constraint is violated
                violation = self.check_constraint_violation(test_input, constraint)

                if violation > 0:
                    constraint_result['violation_count'] += 1
                    constraint_result['max_violation'] = max(constraint_result['max_violation'], violation)
                    constraint_result['details'].append({
                        'input': test_input.tolist(),
                        'violation': violation
                    })

                    if constraint['critical']:
                        constraint_result['passed'] = False

            safety_results.append(constraint_result)

        self.safety_checks = safety_results
        print("Safety validation completed:")
        for result in safety_results:
            status = "PASS" if result['passed'] else "FAIL"
            print(f"  {result['constraint']}: {status} ({result['violation_count']} violations)")

        return safety_results

    def check_constraint_violation(self, input_data, constraint):
        """Check if input violates safety constraint."""
        # This is a simplified example
        # In reality, this would check specific safety constraints
        if constraint['type'] == 'position_limit':
            # Check if position exceeds limits
            pos_limits = constraint['limits']
            for i, (min_val, max_val) in enumerate(pos_limits):
                if i < len(input_data) and (input_data[i] < min_val or input_data[i] > max_val):
                    return abs(input_data[i] - (min_val if input_data[i] < min_val else max_val))
        elif constraint['type'] == 'velocity_limit':
            # Check if velocity exceeds limits
            vel_limit = constraint['limit']
            if abs(input_data[0]) > vel_limit:  # Assume first element is velocity
                return abs(input_data[0]) - vel_limit

        return 0.0

    def generate_validation_report(self, output_dir='validation_reports'):
        """Generate comprehensive validation report."""
        print(f"Generating validation report in {output_dir}...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create report data
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'transfer_metrics': self.transfer_metrics,
            'safety_results': self.safety_checks,
            'summary': {
                'transfer_success': self.is_transfer_successful(),
                'safety_pass': self.are_safety_checks_passed(),
                'overall_risk': self.calculate_overall_risk()
            }
        }

        # Save report as JSON
        report_path = os.path.join(output_dir, f'validation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        # Create visualization
        self.create_validation_visualization(report_data, output_dir)

        print(f"Validation report saved to: {report_path}")
        return report_path

    def is_transfer_successful(self):
        """Check if transfer was successful based on metrics."""
        if not self.transfer_metrics:
            return False

        # Check if success rate transfer is within acceptable range
        success_transfer = self.transfer_metrics.get('success_rate_transfer', 0)
        return success_transfer >= -0.1  # Allow up to 10% drop

    def are_safety_checks_passed(self):
        """Check if all safety checks passed."""
        if not self.safety_checks:
            return True

        return all(result['passed'] for result in self.safety_checks)

    def calculate_overall_risk(self):
        """Calculate overall risk score."""
        if not self.transfer_metrics:
            return 1.0  # High risk if no metrics

        # Combine transfer effectiveness and safety
        transfer_score = 1.0 - abs(self.transfer_metrics.get('success_rate_transfer', 0))
        safety_score = 1.0 if self.are_safety_checks_passed() else 0.0

        # Weighted average
        overall_risk = 0.7 * (1.0 - transfer_score) + 0.3 * (1.0 - safety_score)
        return min(1.0, max(0.0, overall_risk))  # Clamp between 0 and 1

    def create_validation_visualization(self, report_data, output_dir):
        """Create visualization of validation results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Transfer metrics
        transfer_keys = [k for k in report_data['transfer_metrics'].keys() if not k.endswith('_std')]
        transfer_values = [report_data['transfer_metrics'][k] for k in transfer_keys]

        axes[0, 0].bar(range(len(transfer_keys)), transfer_values)
        axes[0, 0].set_xticks(range(len(transfer_keys)))
        axes[0, 0].set_xticklabels([k.replace('_transfer', '') for k in transfer_keys], rotation=45)
        axes[0, 0].set_title('Transfer Effectiveness Metrics')
        axes[0, 0].set_ylabel('Change Ratio')

        # Plot 2: Safety results
        safety_names = [result['constraint'] for result in report_data['safety_results']]
        safety_passed = [1 if result['passed'] else 0 for result in report_data['safety_results']]

        colors = ['green' if passed else 'red' for passed in safety_passed]
        axes[0, 1].bar(range(len(safety_names)), safety_passed, color=colors)
        axes[0, 1].set_xticks(range(len(safety_names)))
        axes[0, 1].set_xticklabels(safety_names, rotation=45)
        axes[0, 1].set_title('Safety Check Results')
        axes[0, 1].set_ylabel('Pass (1) / Fail (0)')
        axes[0, 1].set_ylim(-0.1, 1.1)

        # Plot 3: Risk assessment
        risk_categories = ['Transfer Risk', 'Safety Risk', 'Overall Risk']
        risk_values = [
            1.0 - report_data['summary'].get('transfer_success', 0.5),
            1.0 - report_data['summary'].get('safety_pass', 0.5),
            report_data['summary'].get('overall_risk', 0.5)
        ]

        bars = axes[1, 0].bar(risk_categories, risk_values)
        axes[1, 0].set_title('Risk Assessment')
        axes[1, 0].set_ylabel('Risk Level')
        axes[1, 0].set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, risk_values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.2f}',
                           ha='center', va='bottom')

        # Plot 4: Safety violations
        violation_counts = [result['violation_count'] for result in report_data['safety_results']]
        axes[1, 1].bar(range(len(safety_names)), violation_counts)
        axes[1, 1].set_xticks(range(len(safety_names)))
        axes[1, 1].set_xticklabels(safety_names, rotation=45)
        axes[1, 1].set_title('Safety Violation Counts')
        axes[1, 1].set_ylabel('Violation Count')

        plt.tight_layout()
        viz_path = os.path.join(output_dir, f'validation_visualization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Validation visualization saved to: {viz_path}")

class DeploymentManager:
    """Manage deployment of models to real robots."""

    def __init__(self, model_path, robot_interface):
        self.model_path = model_path
        self.robot_interface = robot_interface
        self.deployment_history = []
        self.active_model = None

    def deploy_model(self, model, validation_report):
        """Deploy model to robot with validation check."""
        print("Deploying model to robot...")

        # Check if validation passed
        if not self.is_deployment_approved(validation_report):
            print("Deployment rejected: Validation requirements not met")
            return False

        # Serialize model
        model_bytes = self.serialize_model(model)

        # Deploy to robot
        deployment_success = self.robot_interface.deploy_model(model_bytes)

        if deployment_success:
            deployment_record = {
                'timestamp': datetime.now().isoformat(),
                'model_path': self.model_path,
                'validation_report': validation_report['timestamp'],
                'deployment_success': True,
                'risk_level': validation_report['summary']['overall_risk']
            }
            self.deployment_history.append(deployment_record)
            self.active_model = model

            print("Model deployed successfully")
            return True
        else:
            print("Model deployment failed")
            return False

    def is_deployment_approved(self, validation_report):
        """Check if deployment is approved based on validation."""
        summary = validation_report['summary']

        # Check that transfer was successful and safety checks passed
        return (summary.get('transfer_success', False) and
                summary.get('safety_pass', False) and
                summary.get('overall_risk', 1.0) < 0.5)  # Risk must be below 0.5

    def serialize_model(self, model):
        """Serialize model for deployment."""
        import io
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return buf.getvalue()

    def rollback_model(self):
        """Rollback to previous model if current model fails."""
        print("Rolling back to previous model...")
        # Implementation would restore previous model
        pass

    def monitor_deployment(self):
        """Monitor deployed model performance."""
        print("Monitoring deployed model...")
        # Implementation would monitor real-world performance
        pass

# Example usage
if __name__ == "__main__":
    print("Transfer Validation and Deployment System")

    # Create validation system
    validator = TransferValidationSystem()

    # Define test scenarios
    test_scenarios = [
        {'name': 'navigation_easy', 'difficulty': 'easy'},
        {'name': 'navigation_medium', 'difficulty': 'medium'},
        {'name': 'navigation_hard', 'difficulty': 'hard'},
        {'name': 'manipulation_simple', 'difficulty': 'easy'},
        {'name': 'manipulation_complex', 'difficulty': 'hard'}
    ]

    # Define safety constraints
    safety_constraints = [
        {'name': 'position_limits', 'type': 'position_limit', 'limits': [(-10, 10), (-10, 10), (-1, 2)], 'critical': True},
        {'name': 'velocity_limits', 'type': 'velocity_limit', 'limit': 2.0, 'critical': True},
        {'name': 'acceleration_limits', 'type': 'velocity_limit', 'limit': 5.0, 'critical': False}
    ]

    # Run validation (simulated)
    print("Running validation tests...")
    sim_results, real_results = validator.run_comparative_validation(
        sim_model="dummy_sim_model",
        real_model="dummy_real_model",
        test_scenarios=test_scenarios
    )

    # Run safety validation
    safety_results = validator.run_safety_validation(
        model="dummy_model",
        safety_constraints=safety_constraints
    )

    # Generate validation report
    report_path = validator.generate_validation_report()

    print(f"\nValidation completed. Report saved to: {report_path}")

    # Example deployment (simulated)
    print("\nSimulating model deployment...")

    class DummyRobotInterface:
        def deploy_model(self, model_bytes):
            print("Deploying model to dummy robot interface...")
            return True  # Simulate successful deployment

    robot_interface = DummyRobotInterface()
    deployment_manager = DeploymentManager("model_path", robot_interface)

    # Simulate deploying a dummy model
    dummy_model = torch.nn.Linear(3, 3)  # Dummy model
    deployment_success = deployment_manager.deploy_model(dummy_model, validator.validation_results)

    if deployment_success:
        print("Model deployment successful!")
    else:
        print("Model deployment failed.")

    print("\nTransfer validation and deployment system completed successfully")
```

## Troubleshooting

### Common Issues and Solutions

1. **Large reality gap after domain randomization**:
   - Start with narrower randomization ranges
   - Use curriculum-based randomization
   - Add more diverse simulation scenarios
   - Validate intermediate models

2. **Poor transfer performance**:
   - Collect more real-world data for adaptation
   - Use ensemble methods for robustness
   - Implement online adaptation
   - Revisit system identification

3. **Safety violations in real deployment**:
   - Implement comprehensive safety checks
   - Use safety filters and barriers
   - Start with conservative parameters
   - Gradually increase capabilities

4. **Model overfitting to simulation**:
   - Increase domain randomization diversity
   - Use regularization techniques
   - Implement early stopping
   - Add more real-world data

5. **Computational resource limitations**:
   - Use model compression techniques
   - Implement efficient inference
   - Use quantization for deployment
   - Optimize neural network architecture

## Assessment Questions

1. How do you quantify the reality gap between simulation and reality?
2. What are the key differences between domain randomization and system identification?
3. How would you design safety validation tests for robot deployment?
4. What metrics would you use to evaluate transfer effectiveness?
5. How do you handle model adaptation with limited real-world data?

## Extension Exercises

1. Implement meta-learning for rapid adaptation
2. Create a curriculum learning system for complex tasks
3. Implement adversarial domain adaptation techniques
4. Create a safety-critical deployment pipeline
5. Implement lifelong learning for continuous adaptation

## Summary

In this lab, you successfully:
- Analyzed and quantified the reality gap between simulation and reality
- Implemented domain randomization techniques for robust simulation
- Performed system identification to model real robot dynamics
- Created model adaptation and fine-tuning systems
- Developed validation and deployment procedures for safe transfer
- Validated transfer effectiveness and safety measures

These skills are essential for bridging the gap between simulation and real-world robotics. The combination of domain randomization, system identification, model adaptation, and rigorous validation enables the development of robust robotic systems that can effectively transfer from simulation to real-world deployment while maintaining safety and performance guarantees.