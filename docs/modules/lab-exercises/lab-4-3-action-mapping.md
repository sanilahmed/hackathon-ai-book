---
sidebar_label: 'Lab 4.3: Action Mapping'
---

# Lab Exercise 4.3: Action Mapping in VLA Systems

This lab exercise covers implementing action mapping systems that translate multimodal perceptions into executable robot actions.

## Objectives

- Implement language-to-action mapping
- Create task planning from multimodal inputs
- Integrate action execution with perception
- Test closed-loop VLA system performance

## Prerequisites

- Completed multimodal perception lab
- Understanding of robot control and planning
- ROS 2 Humble with control packages
- PyTorch/TensorFlow knowledge

## Action Mapping Architecture

### Language-Action Translation Module

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import re

class LanguageActionMapper(nn.Module):
    def __init__(self, vocab_size=10000, hidden_dim=512, action_space_dim=10):
        super().__init__()

        # Language processing
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.language_encoder = nn.Linear(hidden_dim, hidden_dim)

        # Action space mapping
        self.action_projector = nn.Linear(hidden_dim, action_space_dim)
        self.action_classifier = nn.Linear(hidden_dim, action_space_dim)

        # Action type classifier (navigation, manipulation, etc.)
        self.action_type_classifier = nn.Linear(hidden_dim, 5)  # 5 action types

        self.hidden_dim = hidden_dim
        self.action_space_dim = action_space_dim

    def forward(self, language_input, vision_features=None):
        """
        language_input: tokenized language input
        vision_features: optional visual context
        """
        # Process language
        embedded = self.embedding(language_input)
        lstm_out, (hidden, _) = self.lstm(embedded)

        # Use final hidden state as language representation
        lang_repr = hidden[-1]  # (batch, hidden_dim)

        # If vision features provided, fuse with language
        if vision_features is not None:
            combined_repr = torch.cat([lang_repr, vision_features], dim=-1)
            combined_repr = nn.Linear(lang_repr.size(-1) + vision_features.size(-1), self.hidden_dim)(combined_repr)
        else:
            combined_repr = lang_repr

        # Project to action space
        action_logits = self.action_classifier(combined_repr)
        action_probs = torch.softmax(action_logits, dim=-1)

        # Classify action type
        action_type_logits = self.action_type_classifier(combined_repr)
        action_type_probs = torch.softmax(action_type_logits, dim=-1)

        return {
            'action_probs': action_probs,
            'action_type_probs': action_type_probs,
            'action_logits': action_logits,
            'action_type_logits': action_type_logits
        }

    def extract_action_primitives(self, command: str) -> List[Dict]:
        """Extract action primitives from natural language command"""
        action_primitives = []

        # Navigation actions
        nav_patterns = [
            (r'go to|move to|navigate to|go (.+)', 'navigation'),
            (r'forward|backward|left|right', 'navigation'),
            (r'kitchen|bedroom|living room|office', 'navigation')
        ]

        # Manipulation actions
        manipulation_patterns = [
            (r'pick up|grasp|take|lift (.+)', 'manipulation'),
            (r'put down|place|drop (.+)', 'manipulation'),
            (r'open|close (.+)', 'manipulation')
        ]

        # Interaction actions
        interaction_patterns = [
            (r'wave|greet|hello|hi (.+)', 'interaction'),
            (r'follow|accompany (.+)', 'interaction'),
            (r'stop|wait|pause', 'interaction')
        ]

        all_patterns = nav_patterns + manipulation_patterns + interaction_patterns

        for pattern, action_type in all_patterns:
            matches = re.findall(pattern, command, re.IGNORECASE)
            for match in matches:
                action_primitives.append({
                    'type': action_type,
                    'command': command,
                    'target': match if match else None,
                    'confidence': 0.8  # Placeholder confidence
                })

        return action_primitives
```

## Task Planning Module

### Hierarchical Task Planner

```python
class TaskPlanner:
    def __init__(self):
        self.action_library = self._initialize_action_library()
        self.task_graph = {}

    def _initialize_action_library(self):
        """Initialize library of primitive actions"""
        return {
            # Navigation actions
            'move_forward': {
                'primitive': 'base_controller',
                'params': {'linear_vel': 0.3, 'angular_vel': 0.0},
                'preconditions': ['robot_enabled'],
                'effects': ['robot_moved_forward']
            },
            'turn_left': {
                'primitive': 'base_controller',
                'params': {'linear_vel': 0.0, 'angular_vel': 0.5},
                'preconditions': ['robot_enabled'],
                'effects': ['robot_turned_left']
            },
            'navigate_to': {
                'primitive': 'navigation_stack',
                'params': {'goal_tolerance': 0.2},
                'preconditions': ['map_available', 'goal_valid'],
                'effects': ['reached_goal']
            },

            # Manipulation actions
            'grasp_object': {
                'primitive': 'manipulation_controller',
                'params': {'gripper_force': 50},
                'preconditions': ['object_detected', 'arm_free'],
                'effects': ['object_grasped']
            },
            'release_object': {
                'primitive': 'manipulation_controller',
                'params': {'gripper_position': 100},
                'preconditions': ['object_grasped'],
                'effects': ['object_released']
            },

            # Perception actions
            'detect_object': {
                'primitive': 'perception_system',
                'params': {'detection_threshold': 0.7},
                'preconditions': ['camera_enabled'],
                'effects': ['object_location_known']
            }
        }

    def plan_task(self, command: str, world_state: Dict) -> List[Dict]:
        """Generate task plan from command and world state"""
        # Extract action primitives
        action_primitives = self.language_action_mapper.extract_action_primitives(command)

        if not action_primitives:
            return []

        # Create task plan
        task_plan = []
        for primitive in action_primitives:
            action_sequence = self._decompose_action(primitive, world_state)
            task_plan.extend(action_sequence)

        return task_plan

    def _decompose_action(self, primitive: Dict, world_state: Dict) -> List[Dict]:
        """Decompose high-level action into primitive actions"""
        action_type = primitive['type']
        target = primitive['target']

        if action_type == 'navigation':
            return self._create_navigation_plan(target, world_state)
        elif action_type == 'manipulation':
            return self._create_manipulation_plan(target, world_state)
        elif action_type == 'interaction':
            return self._create_interaction_plan(target, world_state)
        else:
            return []

    def _create_navigation_plan(self, target: str, world_state: Dict) -> List[Dict]:
        """Create navigation plan to target location"""
        if target in world_state.get('known_locations', {}):
            goal_pose = world_state['known_locations'][target]
        else:
            # Use perception to find target location
            return [
                {
                    'action': 'detect_object',
                    'params': {'object_name': target},
                    'description': f'Detect {target} location'
                },
                {
                    'action': 'navigate_to',
                    'params': {'goal_pose': 'detected_pose'},
                    'description': f'Navigate to {target}'
                }
            ]

        return [
            {
                'action': 'navigate_to',
                'params': {'goal_pose': goal_pose},
                'description': f'Navigate to {target}'
            }
        ]

    def _create_manipulation_plan(self, target: str, world_state: Dict) -> List[Dict]:
        """Create manipulation plan for target object"""
        plan = []

        # Detect object if not already known
        if target not in world_state.get('object_locations', {}):
            plan.append({
                'action': 'detect_object',
                'params': {'object_name': target},
                'description': f'Detect {target} object'
            })

        # Navigate to object
        plan.append({
            'action': 'navigate_to',
            'params': {'goal_pose': 'object_pose'},
            'description': f'Navigate to {target}'
        })

        # Grasp object
        plan.append({
            'action': 'grasp_object',
            'params': {'object_name': target},
            'description': f'Grasp {target}'
        })

        return plan

    def _create_interaction_plan(self, target: str, world_state: Dict) -> List[Dict]:
        """Create interaction plan"""
        plan = []

        # Navigate to target if it's a person/location
        if target in world_state.get('known_people', {}) or target in world_state.get('known_locations', {}):
            plan.append({
                'action': 'navigate_to',
                'params': {'goal_pose': 'target_pose'},
                'description': f'Navigate to {target}'
            })

        # Perform interaction
        plan.append({
            'action': 'perform_interaction',
            'params': {'interaction_type': 'wave'},
            'description': f'Interact with {target}'
        })

        return plan
```

## Action Execution System

### Robot Action Executor

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose, Point
from std_msgs.msg import String, Bool
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class ActionExecutor(Node):
    def __init__(self):
        super().__init__('action_executor')

        # Publishers for different action types
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/action_status', 10)

        # Action clients
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.traj_client = ActionClient(self, FollowJointTrajectory, 'follow_joint_trajectory')

        # Current task execution
        self.current_task = None
        self.task_queue = []
        self.is_executing = False

        # Timer for action execution
        self.action_timer = self.create_timer(0.1, self.execute_next_action)

    def execute_task_plan(self, task_plan: List[Dict]):
        """Execute a sequence of tasks"""
        self.task_queue.extend(task_plan)
        self.get_logger().info(f'Added {len(task_plan)} tasks to execution queue')

    def execute_next_action(self):
        """Execute the next action in the queue"""
        if not self.task_queue or self.is_executing:
            return

        task = self.task_queue.pop(0)
        self.is_executing = True

        action_type = task['action']
        params = task.get('params', {})

        self.get_logger().info(f'Executing action: {action_type}')

        # Execute action based on type
        if action_type == 'move_forward':
            self.execute_move_forward(params)
        elif action_type == 'turn_left':
            self.execute_turn_left(params)
        elif action_type == 'navigate_to':
            self.execute_navigate_to(params)
        elif action_type == 'grasp_object':
            self.execute_grasp_object(params)
        elif action_type == 'release_object':
            self.execute_release_object(params)
        else:
            self.get_logger().warn(f'Unknown action type: {action_type}')
            self.is_executing = False

    def execute_move_forward(self, params):
        """Execute move forward action"""
        cmd_vel = Twist()
        cmd_vel.linear.x = params.get('linear_vel', 0.3)
        cmd_vel.angular.z = params.get('angular_vel', 0.0)

        self.cmd_vel_pub.publish(cmd_vel)

        # Simple timeout-based completion
        timer = self.create_timer(
            params.get('duration', 2.0),
            lambda: self.action_completed('move_forward')
        )

    def execute_turn_left(self, params):
        """Execute turn left action"""
        cmd_vel = Twist()
        cmd_vel.linear.x = params.get('linear_vel', 0.0)
        cmd_vel.angular.z = params.get('angular_vel', 0.5)

        self.cmd_vel_pub.publish(cmd_vel)

        timer = self.create_timer(
            params.get('duration', 1.0),
            lambda: self.action_completed('turn_left')
        )

    def execute_navigate_to(self, params):
        """Execute navigation action using Nav2"""
        goal_msg = NavigateToPose.Goal()

        # Set goal pose
        pose = params.get('goal_pose', Pose())
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose = pose

        # Send goal
        self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.navigation_feedback_callback
        ).add_done_callback(self.navigation_goal_response_callback)

    def execute_grasp_object(self, params):
        """Execute grasp action"""
        # This would interface with manipulation stack
        object_name = params.get('object_name', 'unknown')

        self.get_logger().info(f'Attempting to grasp {object_name}')

        # Simulate grasp completion
        timer = self.create_timer(
            3.0,  # Grasp takes 3 seconds
            lambda: self.action_completed('grasp_object')
        )

    def execute_release_object(self, params):
        """Execute release action"""
        self.get_logger().info('Releasing object')

        timer = self.create_timer(
            2.0,
            lambda: self.action_completed('release_object')
        )

    def navigation_feedback_callback(self, feedback_msg):
        """Handle navigation feedback"""
        remaining_distance = feedback_msg.feedback.distance_remaining
        self.get_logger().debug(f'Distance remaining: {remaining_distance:.2f}m')

    def navigation_goal_response_callback(self, future):
        """Handle navigation goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Navigation goal rejected')
            self.is_executing = False
            return

        self.get_logger().info('Navigation goal accepted')
        goal_handle.get_result_async().add_done_callback(self.navigation_result_callback)

    def navigation_result_callback(self, future):
        """Handle navigation result"""
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Navigation succeeded')
            self.action_completed('navigate_to')
        else:
            self.get_logger().error(f'Navigation failed with status: {status}')
            self.is_executing = False

    def action_completed(self, action_name):
        """Mark action as completed"""
        self.get_logger().info(f'Action completed: {action_name}')
        self.is_executing = False

        # Publish completion status
        status_msg = String()
        status_msg.data = f'completed_{action_name}'
        self.status_pub.publish(status_msg)
```

## Closed-Loop VLA System

### Complete VLA Integration

```python
class ClosedLoopVLA:
    def __init__(self):
        # Initialize perception system
        self.perception_system = MultimodalPerceptionSystem()

        # Initialize language-action mapper
        self.language_mapper = LanguageActionMapper()

        # Initialize task planner
        self.task_planner = TaskPlanner()

        # Initialize action executor
        self.action_executor = ActionExecutor()

        # World state representation
        self.world_state = {
            'robot_pose': None,
            'object_locations': {},
            'known_locations': {},
            'known_people': {},
            'robot_status': 'idle'
        }

    def execute_command(self, image, command, depth_map=None):
        """Execute a complete VLA command"""
        # Step 1: Multimodal perception
        perception_results = self.perception_system(
            image=image,
            command=command,
            depth_map=depth_map
        )

        # Update world state with perception results
        self.update_world_state(perception_results)

        # Step 2: Language to action mapping
        action_mapping = self.language_mapper(
            language_input=command,
            vision_features=perception_results.get('visual_features')
        )

        # Extract action primitives
        action_primitives = self.language_mapper.extract_action_primitives(command)

        # Step 3: Task planning
        task_plan = self.task_planner.plan_task(command, self.world_state)

        # Step 4: Action execution
        self.action_executor.execute_task_plan(task_plan)

        return {
            'perception_results': perception_results,
            'action_mapping': action_mapping,
            'task_plan': task_plan,
            'execution_status': 'started'
        }

    def update_world_state(self, perception_results):
        """Update world state based on perception results"""
        # Update object locations
        if 'detections' in perception_results:
            for detection in perception_results['detections']:
                obj_name = detection.get('class', 'unknown')
                obj_pose = detection.get('pose', None)
                if obj_pose:
                    self.world_state['object_locations'][obj_name] = obj_pose

        # Update robot pose if available
        if 'robot_pose' in perception_results:
            self.world_state['robot_pose'] = perception_results['robot_pose']

    def get_execution_feedback(self):
        """Get feedback on current execution status"""
        return {
            'world_state': self.world_state,
            'execution_queue_size': len(self.action_executor.task_queue),
            'is_executing': self.action_executor.is_executing
        }
```

## VLA Integration Node

### ROS 2 VLA Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np

class VLARosNode(Node):
    def __init__(self):
        super().__init__('vla_system_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10
        )
        self.command_sub = self.create_subscription(
            String, '/vla/command', self.command_callback, 10
        )
        self.status_pub = self.create_publisher(String, '/vla/status', 10)

        # Initialize complete VLA system
        self.vla_system = ClosedLoopVLA()

        # Store latest image and command
        self.latest_image = None
        self.pending_command = None

    def image_callback(self, msg):
        """Process incoming image"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image

            # If we have a pending command, execute now
            if self.pending_command:
                self.execute_vla_command()

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def command_callback(self, msg):
        """Process incoming command"""
        self.pending_command = msg.data

        # If we have a recent image, execute now
        if self.latest_image:
            self.execute_vla_command()

    def execute_vla_command(self):
        """Execute the VLA command with current image"""
        if self.latest_image is None or self.pending_command is None:
            return

        try:
            # Convert image to tensor
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(self.latest_image)

            # Execute VLA command
            results = self.vla_system.execute_command(
                image=pil_image,
                command=self.pending_command
            )

            # Publish status
            status_msg = String()
            status_msg.data = f"VLA command executed: {self.pending_command}"
            self.status_pub.publish(status_msg)

            self.get_logger().info(f'Executed VLA command: {self.pending_command}')

            # Clear pending command
            self.pending_command = None

        except Exception as e:
            self.get_logger().error(f'Error executing VLA command: {e}')

    def get_system_status(self):
        """Get current system status"""
        feedback = self.vla_system.get_execution_feedback()
        return feedback
```

## Action Learning and Adaptation

### Online Action Learning

```python
class OnlineActionLearner:
    def __init__(self, action_space_dim=10):
        self.action_space_dim = action_space_dim

        # Experience replay buffer
        self.experience_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }

        # Action refinement network
        self.refinement_network = nn.Sequential(
            nn.Linear(action_space_dim * 2, 128),  # current + desired actions
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space_dim),
            nn.Tanh()
        )

        self.optimizer = torch.optim.Adam(self.refinement_network.parameters())
        self.criterion = nn.MSELoss()

    def record_experience(self, state, action, reward, next_state, done):
        """Record experience for learning"""
        self.experience_buffer['states'].append(state)
        self.experience_buffer['actions'].append(action)
        self.experience_buffer['rewards'].append(reward)
        self.experience_buffer['next_states'].append(next_state)
        self.experience_buffer['dones'].append(done)

        # Keep buffer size manageable
        if len(self.experience_buffer['states']) > 10000:
            for key in self.experience_buffer:
                self.experience_buffer[key] = self.experience_buffer[key][-5000:]

    def refine_action(self, current_action, desired_action):
        """Refine action based on experience"""
        # Combine current and desired actions
        combined_input = torch.cat([current_action, desired_action], dim=-1)

        # Get refined action
        refined_action = self.refinement_network(combined_input)

        return refined_action

    def update_model(self, batch_size=32):
        """Update the refinement model"""
        if len(self.experience_buffer['states']) < batch_size:
            return

        # Sample batch
        indices = np.random.choice(len(self.experience_buffer['states']), batch_size)

        states = torch.stack([self.experience_buffer['states'][i] for i in indices])
        actions = torch.stack([self.experience_buffer['actions'][i] for i in indices])
        rewards = torch.tensor([self.experience_buffer['rewards'][i] for i in indices])

        # Use rewards to guide action refinement
        # Higher reward actions should be reinforced
        target_actions = actions + rewards.unsqueeze(1) * 0.1  # Simple reward shaping

        # Train refinement network
        self.optimizer.zero_grad()
        predicted_actions = self.refinement_network(torch.cat([states, actions], dim=1))
        loss = self.criterion(predicted_actions, target_actions)
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

## Performance Evaluation

### VLA System Evaluation

```python
class VLAEvaluator:
    def __init__(self):
        self.metrics = {
            'command_success_rate': [],
            'action_accuracy': [],
            'perception_accuracy': [],
            'execution_time': [],
            'task_completion_rate': []
        }

    def evaluate_command_execution(self, command, expected_outcome, actual_outcome):
        """Evaluate command execution performance"""
        # Calculate success based on expected vs actual outcome
        success = self.compare_outcomes(expected_outcome, actual_outcome)
        self.metrics['command_success_rate'].append(1.0 if success else 0.0)

        return success

    def evaluate_action_accuracy(self, predicted_actions, ground_truth_actions):
        """Evaluate action prediction accuracy"""
        correct = 0
        total = len(ground_truth_actions)

        for pred, gt in zip(predicted_actions, ground_truth_actions):
            if self.actions_match(pred, gt):
                correct += 1

        accuracy = correct / total if total > 0 else 0
        self.metrics['action_accuracy'].append(accuracy)
        return accuracy

    def evaluate_perception_grounding(self, perception_results, ground_truth_objects):
        """Evaluate perception grounding accuracy"""
        correct = 0
        total = len(ground_truth_objects)

        for gt_obj in ground_truth_objects:
            for pred_obj in perception_results.get('detections', []):
                if self.objects_match(gt_obj, pred_obj):
                    correct += 1
                    break

        accuracy = correct / total if total > 0 else 0
        self.metrics['perception_accuracy'].append(accuracy)
        return accuracy

    def objects_match(self, obj1, obj2):
        """Check if two objects match"""
        # Check spatial overlap and semantic similarity
        iou = self.calculate_iou(obj1.get('bbox', [0,0,0,0]), obj2.get('bbox', [0,0,0,0]))
        semantic_sim = self.calculate_semantic_similarity(
            obj1.get('description', ''),
            obj2.get('description', '')
        )
        return iou > 0.5 and semantic_sim > 0.7

    def actions_match(self, action1, action2):
        """Check if two actions match"""
        return (action1.get('type') == action2.get('type') and
                action1.get('target') == action2.get('target'))

    def compare_outcomes(self, expected, actual):
        """Compare expected vs actual outcomes"""
        # This would depend on your specific outcome representation
        # For now, return simple comparison
        return expected == actual

    def get_summary_metrics(self):
        """Get summary of all metrics"""
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values)
                }
        return summary
```

## Exercise Tasks

1. Implement the language-to-action mapping system
2. Create a hierarchical task planning module
3. Build the action execution system with ROS 2 integration
4. Implement the complete closed-loop VLA system
5. Add online learning capabilities for action refinement
6. Evaluate the system performance with various commands

## Troubleshooting

### Common Issues

- **Action execution failures**: Check robot state and preconditions
- **Perception-Action mismatch**: Improve grounding between modalities
- **Task planning errors**: Verify world state representation
- **Timing issues**: Ensure proper synchronization between components

## Summary

In this lab, you implemented the action mapping component of VLA systems. You created language-to-action translation, task planning, action execution, and closed-loop integration. You also implemented online learning for action refinement and evaluation metrics. This completes the full VLA pipeline from vision-language input to robot action execution.