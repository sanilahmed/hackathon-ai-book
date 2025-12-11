# Language-Action Mapping

## Overview

Language-action mapping is the critical component of Vision-Language-Action (VLA) systems that translates natural language instructions into executable robotic actions. This section covers techniques for parsing language, understanding semantic meaning, and generating appropriate motor commands for humanoid robots.

## Architecture of Language-Action Mapping

### High-Level Architecture

```
Natural Language Input
        ↓
[Language Parser] → [Semantic Interpreter] → [Action Generator] → [Robot Actions]
        ↓                   ↓                      ↓
  Syntactic Tree      Semantic Graph        Action Sequence    Motor Commands
  Dependency Parse    Concept Grounding     Temporal Planning   Joint Positions
```

### Language Parser Component

```python
import spacy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class LanguageParser(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        # Load pre-trained language model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.language_model = AutoModel.from_pretrained(model_name)

        # Part-of-speech and dependency parsing
        self.nlp = spacy.load('en_core_web_sm')

        # Intent classification head
        self.intent_classifier = nn.Linear(
            self.language_model.config.hidden_size, 128
        )  # 128 different intents

    def forward(self, text):
        # Tokenize and encode text
        inputs = self.tokenizer(
            text, return_tensors='pt', padding=True, truncation=True
        )

        # Get language embeddings
        outputs = self.language_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Global average pooling

        # Classify intent
        intent_logits = self.intent_classifier(embeddings)

        return {
            'embeddings': embeddings,
            'intent_logits': intent_logits,
            'intent_probs': torch.softmax(intent_logits, dim=-1)
        }

    def parse_dependencies(self, text):
        """Parse syntactic dependencies using spaCy."""
        doc = self.nlp(text)

        dependencies = []
        for token in doc:
            dependencies.append({
                'text': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'dep': token.dep_,
                'head': token.head.text if token.head != token else None
            })

        return dependencies

    def extract_entities(self, text):
        """Extract named entities from text."""
        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })

        return entities
```

### Semantic Interpreter

```python
class SemanticInterpreter(nn.Module):
    def __init__(self):
        super().__init__()
        # Action type classification
        self.action_classifier = nn.Linear(512, 64)  # 64 different action types

        # Object recognition in language
        self.object_classifier = nn.Linear(512, 128)  # 128 object categories

        # Spatial relation understanding
        self.spatial_classifier = nn.Linear(512, 32)   # 32 spatial relations

        # Attribute recognition
        self.attribute_classifier = nn.Linear(512, 64) # 64 attributes

    def forward(self, language_features, vision_features=None):
        """Interpret semantic meaning from language (with optional vision)."""
        # If vision features provided, combine with language
        if vision_features is not None:
            combined_features = torch.cat([language_features, vision_features], dim=-1)
        else:
            combined_features = language_features

        # Classify different semantic components
        action_type = torch.softmax(
            self.action_classifier(combined_features), dim=-1
        )
        object_type = torch.softmax(
            self.object_classifier(combined_features), dim=-1
        )
        spatial_rel = torch.softmax(
            self.spatial_classifier(combined_features), dim=-1
        )
        attributes = torch.softmax(
            self.attribute_classifier(combined_features), dim=-1
        )

        return {
            'action_type': action_type,
            'object_type': object_type,
            'spatial_relation': spatial_rel,
            'attributes': attributes
        }

    def interpret_instruction(self, instruction, vision_features=None):
        """Interpret complete instruction."""
        # Use language parser to get embeddings
        parser_output = self.parse_instruction(instruction)

        # Interpret semantics
        semantic_output = self.forward(
            parser_output['embeddings'],
            vision_features
        )

        return {
            'instruction': instruction,
            'intent': parser_output['intent_probs'],
            'semantic_interpretation': semantic_output,
            'action_candidates': self.generate_action_candidates(semantic_output)
        }

    def generate_action_candidates(self, semantic_output):
        """Generate potential actions based on semantic interpretation."""
        # Map semantic concepts to action space
        action_candidates = []

        # Example mapping logic
        action_type_idx = torch.argmax(semantic_output['action_type'], dim=-1)
        object_type_idx = torch.argmax(semantic_output['object_type'], dim=-1)

        # Generate action based on action type and object
        action = self.map_semantic_to_action(
            action_type_idx.item(),
            object_type_idx.item(),
            semantic_output['spatial_relation'],
            semantic_output['attributes']
        )

        action_candidates.append(action)

        return action_candidates

    def map_semantic_to_action(self, action_type, object_type, spatial_rel, attributes):
        """Map semantic interpretation to specific action."""
        # This is a simplified example - in practice, this would be more complex
        action_mapping = {
            (0, 1): 'pick_up_object',  # action_type=0 (grasp), object_type=1 (cup)
            (1, 2): 'place_object',   # action_type=1 (place), object_type=2 (box)
            (2, 3): 'move_to_object', # action_type=2 (move), object_type=3 (table)
        }

        # Look up action based on semantic types
        action_key = (action_type, object_type)
        action = action_mapping.get(action_key, 'default_action')

        return {
            'action_type': action,
            'target_object': object_type,
            'spatial_constraints': spatial_rel,
            'object_attributes': attributes
        }
```

## Action Space Representation

### Continuous Action Space

```python
import numpy as np

class ContinuousActionSpace:
    def __init__(self, robot_config):
        self.joint_limits = robot_config['joint_limits']
        self.action_dim = len(robot_config['joint_names'])
        self.spatial_dim = 3  # x, y, z position
        self.orientation_dim = 4  # quaternion
        self.gripper_dim = 1    # gripper position

    def decode_language_to_continuous(self, semantic_interpretation):
        """Decode language instruction to continuous action space."""
        # Extract semantic components
        action_type = torch.argmax(semantic_interpretation['action_type'], dim=-1).item()
        target_object = torch.argmax(semantic_interpretation['object_type'], dim=-1).item()
        spatial_constraints = semantic_interpretation['spatial_relation']

        # Map to continuous space based on action type
        if action_type == 0:  # Movement action
            return self.decode_movement_action(spatial_constraints)
        elif action_type == 1:  # Manipulation action
            return self.decode_manipulation_action(target_object, spatial_constraints)
        else:  # Default action
            return self.decode_default_action()

    def decode_movement_action(self, spatial_constraints):
        """Decode movement-related language to continuous movement."""
        # Example: "Go to the left of the table"
        target_pos = np.zeros(3)  # x, y, z

        # Interpret spatial constraints
        spatial_idx = torch.argmax(spatial_constraints, dim=-1).item()
        if spatial_idx == 0:  # left_of
            target_pos[0] -= 0.5  # Move left by 0.5m
        elif spatial_idx == 1:  # right_of
            target_pos[0] += 0.5  # Move right by 0.5m
        elif spatial_idx == 2:  # in_front_of
            target_pos[1] += 0.5  # Move forward by 0.5m
        elif spatial_idx == 3:  # behind
            target_pos[1] -= 0.5  # Move backward by 0.5m

        # Create continuous action vector
        action = np.zeros(self.action_dim + self.spatial_dim + self.orientation_dim + self.gripper_dim)
        action[self.action_dim:self.action_dim + self.spatial_dim] = target_pos

        return action

    def decode_manipulation_action(self, target_object, spatial_constraints):
        """Decode manipulation-related language to joint positions."""
        # Calculate target pose for manipulation
        target_pose = self.calculate_manipulation_pose(target_object, spatial_constraints)

        # Convert to joint space (using inverse kinematics)
        joint_positions = self.inverse_kinematics(target_pose)

        # Create action vector
        action = np.zeros(self.action_dim + self.spatial_dim + self.orientation_dim + self.gripper_dim)
        action[:self.action_dim] = joint_positions

        return action

    def calculate_manipulation_pose(self, target_object, spatial_constraints):
        """Calculate target pose for manipulation based on object and constraints."""
        # In practice, this would use object detection and spatial reasoning
        # For now, return a placeholder
        return {
            'position': np.array([0.5, 0.0, 0.8]),  # Default position
            'orientation': np.array([0, 0, 0, 1])   # Default orientation (w, x, y, z)
        }

    def inverse_kinematics(self, target_pose):
        """Calculate joint positions for target pose."""
        # Placeholder - in practice, use robot-specific IK solver
        return np.zeros(self.action_dim)
```

### Discrete Action Space

```python
class DiscreteActionSpace:
    def __init__(self):
        # Define discrete action vocabulary
        self.action_vocabulary = [
            'move_forward',
            'move_backward',
            'turn_left',
            'turn_right',
            'move_up',
            'move_down',
            'grasp',
            'release',
            'open_gripper',
            'close_gripper',
            'pick_object',
            'place_object',
            'push_object',
            'pull_object',
            'point_at_object',
            'wave',
            'nod',
            'shake_head'
        ]

        self.action_to_id = {action: i for i, action in enumerate(self.action_vocabulary)}
        self.id_to_action = {i: action for i, action in enumerate(self.action_vocabulary)}

    def map_language_to_discrete_action(self, instruction):
        """Map natural language instruction to discrete action."""
        # Simple keyword-based mapping (in practice, use more sophisticated NLP)
        instruction_lower = instruction.lower()

        # Extract action keywords
        for action in self.action_vocabulary:
            if action.replace('_', ' ') in instruction_lower:
                return self.action_to_id[action]

        # If no direct match, use semantic similarity
        return self.find_semantically_similar_action(instruction)

    def find_semantically_similar_action(self, instruction):
        """Find most semantically similar action using embeddings."""
        # Use sentence transformers or similar approach
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('all-MiniLM-L6-v2')
        instruction_embedding = model.encode([instruction])
        action_embeddings = model.encode(self.action_vocabulary)

        # Calculate similarities
        similarities = np.dot(instruction_embedding, action_embeddings.T)[0]
        most_similar_idx = np.argmax(similarities)

        return most_similar_idx
```

## Hierarchical Action Planning

### Task Decomposition

```python
class HierarchicalActionPlanner:
    def __init__(self):
        self.action_parser = LanguageParser()
        self.semantic_interpreter = SemanticInterpreter()
        self.continuous_action_space = ContinuousActionSpace({
            'joint_names': ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'],
            'joint_limits': [(-3.14, 3.14)] * 6
        })

    def decompose_task(self, instruction):
        """Decompose high-level instruction into subtasks."""
        # Parse the instruction
        parsed = self.action_parser(instruction)

        # Interpret semantics
        semantic = self.semantic_interpreter(
            parsed['embeddings']
        )

        # Decompose into subtasks based on semantic interpretation
        subtasks = self.generate_subtasks(instruction, semantic)

        return {
            'original_instruction': instruction,
            'semantic_interpretation': semantic,
            'subtasks': subtasks,
            'execution_plan': self.create_execution_plan(subtasks)
        }

    def generate_subtasks(self, instruction, semantic_interpretation):
        """Generate subtasks for complex instructions."""
        subtasks = []

        # Example: "Pick up the red cup and place it on the table"
        if 'pick' in instruction.lower() and 'place' in instruction.lower():
            # Decompose into pick and place
            subtasks.extend([
                {
                    'type': 'pick_object',
                    'object': self.extract_object(instruction, 'pick'),
                    'attributes': self.extract_attributes(instruction, 'red')
                },
                {
                    'type': 'navigate',
                    'target': self.extract_target(instruction, 'table')
                },
                {
                    'type': 'place_object',
                    'target': self.extract_target(instruction, 'table'),
                    'object': self.extract_object(instruction, 'pick')
                }
            ])
        else:
            # Single task
            subtasks.append({
                'type': self.infer_task_type(instruction, semantic_interpretation),
                'object': self.extract_object(instruction),
                'target': self.extract_target(instruction),
                'attributes': self.extract_attributes(instruction)
            })

        return subtasks

    def extract_object(self, instruction, context=None):
        """Extract object from instruction."""
        # Use NER to extract objects
        entities = self.action_parser.extract_entities(instruction)
        objects = [ent for ent in entities if ent['label'] in ['OBJECT', 'PRODUCT']]
        return objects[0] if objects else None

    def extract_target(self, instruction):
        """Extract target location from instruction."""
        entities = self.action_parser.extract_entities(instruction)
        locations = [ent for ent in entities if ent['label'] in ['LOC', 'FACILITY']]
        return locations[0] if locations else None

    def extract_attributes(self, instruction, target_attribute=None):
        """Extract attributes like color, size, etc."""
        # Extract adjectives that describe objects
        dependencies = self.action_parser.parse_dependencies(instruction)
        attributes = [dep for dep in dependencies if dep['pos'] == 'ADJ']
        return attributes

    def create_execution_plan(self, subtasks):
        """Create executable plan from subtasks."""
        plan = []
        for i, subtask in enumerate(subtasks):
            action = self.map_subtask_to_action(subtask)
            plan.append({
                'step': i,
                'subtask': subtask,
                'action': action,
                'preconditions': self.get_preconditions(subtask),
                'postconditions': self.get_postconditions(subtask)
            })

        return plan

    def map_subtask_to_action(self, subtask):
        """Map subtask to executable action."""
        # Map subtask type to action
        action_mapping = {
            'pick_object': self.create_pick_action,
            'place_object': self.create_place_action,
            'navigate': self.create_navigation_action,
            'grasp': self.create_grasp_action,
            'release': self.create_release_action
        }

        action_func = action_mapping.get(subtask['type'], self.create_default_action)
        return action_func(subtask)

    def create_pick_action(self, subtask):
        """Create pick action."""
        return {
            'type': 'manipulation',
            'action': 'pick',
            'target_object': subtask.get('object'),
            'approach_vector': [0, 0, -1],  # Approach from above
            'gripper_position': 0.0  # Open gripper
        }

    def create_place_action(self, subtask):
        """Create place action."""
        return {
            'type': 'manipulation',
            'action': 'place',
            'target_location': subtask.get('target'),
            'release_height': 0.1,  # Release 10cm above surface
            'gripper_position': 1.0  # Close gripper
        }

    def create_navigation_action(self, subtask):
        """Create navigation action."""
        return {
            'type': 'navigation',
            'action': 'navigate',
            'target_pose': self.calculate_target_pose(subtask),
            'path_planning': True
        }
```

## Language-to-Action Transformers

### End-to-End Transformer Model

```python
import torch.nn.functional as F

class LanguageToActionTransformer(nn.Module):
    def __init__(self, vocab_size=30522, action_dim=12, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(512, d_model)  # Max sequence length

        # Transformer encoder for language processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim)
        )

        # Optional: Vision integration
        self.vision_projection = nn.Linear(512, d_model)  # Vision feature dimension
        self.vision_transformer = nn.TransformerEncoder(encoder_layer, 2)

    def forward(self, input_ids, attention_mask=None, vision_features=None):
        """Forward pass from language tokens to actions."""
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_embeds = self.token_embedding(input_ids)

        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)

        # Combined embeddings
        x = token_embeds + pos_embeds

        # Apply attention mask
        if attention_mask is not None:
            # Convert attention mask for transformer (0 for padded, -inf for attention)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Language transformer
        lang_features = self.transformer_encoder(x)

        # Global representation (mean pooling)
        if attention_mask is not None:
            # Masked mean pooling
            masked_lang_features = lang_features * attention_mask.unsqueeze(-1)
            global_features = masked_lang_features.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            global_features = lang_features.mean(dim=1)

        # Optional: Integrate vision features
        if vision_features is not None:
            vision_embeds = self.vision_projection(vision_features)
            # Concatenate or add vision features
            combined_features = global_features + vision_embeds
        else:
            combined_features = global_features

        # Decode to actions
        actions = self.action_decoder(combined_features)

        return actions

    def generate_action_from_text(self, text, tokenizer, vision_features=None):
        """Generate action from text using the model."""
        # Tokenize input
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

        # Forward pass
        with torch.no_grad():
            actions = self.forward(
                inputs['input_ids'],
                inputs['attention_mask'],
                vision_features
            )

        return actions
```

## Vision-Guided Language-Action Mapping

### Grounded Language Understanding

```python
class GroundedLanguageActionMapper:
    def __init__(self):
        self.clip_model = CLIPBasedPerception()  # From multimodal perception
        self.language_parser = LanguageParser()
        self.action_generator = ContinuousActionSpace({
            'joint_names': ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'],
            'joint_limits': [(-3.14, 3.14)] * 6
        })

    def grounded_action_mapping(self, image, instruction):
        """Map language instruction to action with visual grounding."""
        # Get CLIP features
        image_features, text_features = self.clip_model.encode_image_text_pair(
            image, instruction
        )

        # Parse language
        language_output = self.language_parser(instruction)

        # Combine vision and language features
        combined_features = torch.cat([image_features, text_features], dim=-1)

        # Generate grounded action
        action = self.generate_grounded_action(
            combined_features,
            language_output,
            image_features
        )

        return action

    def generate_grounded_action(self, combined_features, language_output, image_features):
        """Generate action based on grounded understanding."""
        # Use combined features to generate action
        # This could be a neural network or rule-based system
        action_type = torch.argmax(language_output['intent_probs'], dim=-1)

        # Generate action based on action type and visual context
        if action_type == 0:  # Navigation
            return self.generate_navigation_action(combined_features, image_features)
        elif action_type == 1:  # Manipulation
            return self.generate_manipulation_action(combined_features, image_features)
        else:  # Default action
            return self.generate_default_action()

    def generate_navigation_action(self, combined_features, image_features):
        """Generate navigation action based on visual scene."""
        # Analyze scene to find navigable locations
        # Use image features to identify free space, obstacles, targets
        target_direction = self.identify_navigation_target(image_features)
        return self.create_navigation_action(target_direction)

    def generate_manipulation_action(self, combined_features, image_features):
        """Generate manipulation action based on visual objects."""
        # Identify manipulable objects in scene
        target_object = self.identify_target_object(image_features)
        return self.create_manipulation_action(target_object)

    def identify_navigation_target(self, image_features):
        """Identify navigation target from image features."""
        # In practice, this would use object detection and spatial reasoning
        # For now, return a placeholder
        return [0.5, 0.0, 0.0]  # Move forward

    def identify_target_object(self, image_features):
        """Identify target object for manipulation."""
        # Use object detection or segmentation
        # For now, return a placeholder
        return {'position': [0.5, 0.0, 0.8], 'type': 'unknown_object'}
```

## Safety and Validation

### Safe Action Generation

```python
class SafeLanguageActionMapper:
    def __init__(self, base_mapper):
        self.base_mapper = base_mapper
        self.safety_checker = SafetyChecker()

    def safe_map_language_to_action(self, image, instruction):
        """Map language to action with safety validation."""
        # Generate action using base mapper
        raw_action = self.base_mapper.grounded_action_mapping(image, instruction)

        # Validate action safety
        if self.safety_checker.is_safe(raw_action, image, instruction):
            return raw_action
        else:
            # Generate safe fallback action
            safe_action = self.safety_checker.generate_safe_fallback(
                raw_action, image, instruction
            )
            return safe_action

class SafetyChecker:
    def __init__(self):
        # Safety constraint models
        self.collision_detector = CollisionDetector()
        self.stability_checker = StabilityChecker()
        self.ethics_checker = EthicsChecker()

    def is_safe(self, action, image, instruction):
        """Check if action is safe to execute."""
        checks = [
            self.check_collision(action),
            self.check_stability(action),
            self.check_ethics(instruction)
        ]
        return all(checks)

    def check_collision(self, action):
        """Check if action would cause collision."""
        # Use robot model and environment to check collision
        return self.collision_detector.would_collide(action)

    def check_stability(self, action):
        """Check if action maintains robot stability."""
        return self.stability_checker.maintains_stability(action)

    def check_ethics(self, instruction):
        """Check if instruction is ethical to follow."""
        # Use ethical AI models to check instruction
        return self.ethics_checker.is_ethical(instruction)

    def generate_safe_fallback(self, unsafe_action, image, instruction):
        """Generate safe fallback action."""
        # Return neutral or safe default action
        return np.zeros_like(unsafe_action)  # Zero action (no movement)
```

## Training Language-Action Models

### Imitation Learning Approach

```python
class LanguageActionTrainer:
    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, language_input, vision_input, expert_actions):
        """Single training step."""
        # Forward pass
        predicted_actions = self.model(
            language_input['input_ids'],
            language_input['attention_mask'],
            vision_input
        )

        # Compute loss
        loss = self.criterion(predicted_actions, expert_actions)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, dataloader):
        """Train for one epoch."""
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            language_input, vision_input, expert_actions = batch

            loss = self.train_step(language_input, vision_input, expert_actions)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def collect_demonstration_data(self, robot, environment, instructions):
        """Collect demonstration data for training."""
        demonstrations = []

        for instruction in instructions:
            # Reset environment
            state = environment.reset()

            # Execute instruction with human demonstration
            expert_trajectory = self.execute_with_human_demo(
                robot, environment, instruction
            )

            # Store demonstration
            demonstrations.append({
                'instruction': instruction,
                'states': expert_trajectory['states'],
                'actions': expert_trajectory['actions'],
                'language_features': self.encode_language(instruction)
            })

        return demonstrations
```

## Evaluation Metrics

### Language-Action Mapping Evaluation

```python
def evaluate_language_action_mapping(model, test_dataset):
    """Evaluate language-action mapping performance."""
    metrics = {
        'action_accuracy': [],
        'semantic_alignment': [],
        'task_success_rate': [],
        'execution_time': [],
        'safety_violations': []
    }

    for sample in test_dataset:
        image, instruction, expected_action = sample

        # Generate action
        start_time = time.time()
        generated_action = model(image, instruction)
        execution_time = time.time() - start_time

        # Evaluate action accuracy
        action_acc = calculate_action_accuracy(generated_action, expected_action)
        metrics['action_accuracy'].append(action_acc)

        # Evaluate semantic alignment
        semantic_align = evaluate_semantic_alignment(instruction, generated_action)
        metrics['semantic_alignment'].append(semantic_align)

        # Evaluate task success (in simulation/real robot)
        task_success = execute_and_evaluate_task(model, image, instruction)
        metrics['task_success_rate'].append(task_success)

        # Record execution time
        metrics['execution_time'].append(execution_time)

        # Check for safety violations
        safety_violations = check_safety_violations(generated_action)
        metrics['safety_violations'].append(safety_violations)

    # Calculate average metrics
    avg_metrics = {k: sum(v)/len(v) for k, v in metrics.items()}
    return avg_metrics

def calculate_action_accuracy(predicted_action, expected_action):
    """Calculate accuracy of predicted action."""
    # Use appropriate distance metric based on action space
    if isinstance(predicted_action, torch.Tensor):
        predicted_action = predicted_action.cpu().numpy()
    if isinstance(expected_action, torch.Tensor):
        expected_action = expected_action.cpu().numpy()

    # Calculate normalized distance
    distance = np.linalg.norm(predicted_action - expected_action)
    max_distance = np.linalg.norm(expected_action) + 1e-8  # Avoid division by zero
    accuracy = 1.0 / (1.0 + distance / max_distance)

    return accuracy
```

## Integration with ROS 2

### ROS 2 Action Server for Language Commands

```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from humanoid_robot_msgs.action import ExecuteLanguageCommand

class LanguageActionServer(Node):
    def __init__(self):
        super().__init__('language_action_server')

        # Initialize language-action mapper
        self.mapper = SafeLanguageActionMapper(GroundedLanguageActionMapper())

        # Create action server
        self._action_server = ActionServer(
            self,
            ExecuteLanguageCommand,
            'execute_language_command',
            self.execute_callback
        )

        # Subscribe to camera and other sensors
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10
        )

        self.current_image = None

    def camera_callback(self, msg):
        """Receive camera image."""
        self.current_image = msg

    def execute_callback(self, goal_handle):
        """Execute language command."""
        self.get_logger().info(f'Executing command: {goal_handle.request.instruction}')

        feedback_msg = ExecuteLanguageCommand.Feedback()
        result = ExecuteLanguageCommand.Result()

        try:
            # Process current image and instruction
            if self.current_image is None:
                raise RuntimeError('No camera image available')

            # Map language to action with current visual context
            action = self.mapper.safe_map_language_to_action(
                self.current_image,
                goal_handle.request.instruction
            )

            # Execute action
            execution_result = self.execute_action(action)

            if execution_result.success:
                result.success = True
                result.message = 'Command executed successfully'
                goal_handle.succeed()
            else:
                result.success = False
                result.message = f'Execution failed: {execution_result.error}'
                goal_handle.abort()

        except Exception as e:
            self.get_logger().error(f'Command execution error: {e}')
            result.success = False
            result.message = f'Execution failed with error: {str(e)}'
            goal_handle.abort()

        return result

    def execute_action(self, action):
        """Execute the generated action."""
        # Publish action to robot controller
        # This would involve sending joint commands, navigation goals, etc.
        pass
```

## Troubleshooting Common Issues

### 1. Ambiguous Instructions
- Implement disambiguation dialogues
- Use context to resolve ambiguities
- Provide feedback when clarification is needed

### 2. Grounding Failures
- Improve object detection and recognition
- Use multiple modalities for robust grounding
- Implement uncertainty estimation

### 3. Action Space Mismatch
- Ensure consistent action space definitions
- Use appropriate action representations
- Implement action space conversion utilities

---
[Next: VLA Architecture](./vla-architecture.md) | [Previous: Multimodal Perception](./multimodal-perception.md)