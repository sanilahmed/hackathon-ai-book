# Lab 4.1: Vision-Language-Action (VLA) Fundamentals

## Overview

In this lab, you will learn the fundamentals of Vision-Language-Action (VLA) systems for robotics. You'll explore the core concepts of multimodal learning, implement basic vision-language models, and understand how to connect perception, language understanding, and action execution in robotic systems. This includes working with pre-trained models like CLIP, implementing multimodal fusion, and creating simple VLA systems.

## Objectives

By the end of this lab, you will be able to:
- Understand the core concepts of Vision-Language-Action systems
- Implement basic vision and language encoders
- Create multimodal fusion mechanisms
- Integrate VLA systems with robot control
- Work with pre-trained vision-language models
- Implement simple action mapping from language to robot actions
- Evaluate VLA system performance

## Prerequisites

- Completion of Module 1-3 (ROS 2, Digital Twin, AI-Robot Brain)
- Understanding of deep learning fundamentals
- Experience with PyTorch and neural networks
- Basic knowledge of computer vision and NLP
- Experience with Isaac Sim or similar simulation environments

## Duration

4-5 hours

## Exercise 1: Understanding VLA Architecture

### Step 1: Create VLA architecture overview

Create `~/vla_examples/vla_architecture.py`:

```python
#!/usr/bin/env python3
# vla_architecture.py
"""Vision-Language-Action system architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VisionEncoder(nn.Module):
    """Vision encoder for VLA systems."""

    def __init__(self, backbone='resnet50', pretrained=True):
        super(VisionEncoder, self).__init__()

        if backbone == 'resnet50':
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=pretrained)
            # Remove the final classification layer
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'clip_vision':
            # Use CLIP's vision encoder
            import clip
            model, preprocess = clip.load("ViT-B/32", device="cpu")
            self.backbone = model.visual
            self.feature_dim = model.visual.output_dim
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Projection layer to common space
        self.projection = nn.Linear(self.feature_dim, 512)

    def forward(self, images):
        """Forward pass through vision encoder."""
        features = self.backbone(images)
        projected_features = self.projection(features)
        return projected_features


class LanguageEncoder(nn.Module):
    """Language encoder for VLA systems."""

    def __init__(self, model_name='bert-base-uncased'):
        super(LanguageEncoder, self).__init__()

        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Projection layer to common space
        self.projection = nn.Linear(self.model.config.hidden_size, 512)

    def forward(self, texts):
        """Forward pass through language encoder."""
        # Tokenize input texts
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )

        # Get language embeddings
        outputs = self.model(**inputs)
        # Use [CLS] token representation
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Project to common space
        projected_embeddings = self.projection(embeddings)
        return projected_embeddings


class MultimodalFusion(nn.Module):
    """Multimodal fusion for vision-language integration."""

    def __init__(self, feature_dim=512):
        super(MultimodalFusion, self).__init__()
        self.feature_dim = feature_dim

        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )

        # Fusion layers
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )

        # Normalization
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, vision_features, language_features):
        """Fuse vision and language features."""
        # Ensure proper dimensions
        if len(vision_features.shape) == 2:
            vision_features = vision_features.unsqueeze(1)  # [B, 1, D]
        if len(language_features.shape) == 2:
            language_features = language_features.unsqueeze(1)  # [B, 1, D]

        # Cross-attention: vision attends to language and vice versa
        attended_vision, _ = self.cross_attention(
            vision_features, language_features, language_features
        )
        attended_language, _ = self.cross_attention(
            language_features, vision_features, vision_features
        )

        # Concatenate attended features
        combined_features = torch.cat([
            attended_vision.squeeze(1),
            attended_language.squeeze(1)
        ], dim=-1)

        # Apply fusion MLP
        fused_features = self.fusion_mlp(combined_features)
        fused_features = self.norm(fused_features)

        return fused_features


class ActionDecoder(nn.Module):
    """Action decoder for converting fused features to robot actions."""

    def __init__(self, action_space_dim=12):
        super(ActionDecoder, self).__init__()
        self.action_space_dim = action_space_dim

        # Decode fused features to action space
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, action_space_dim)
        )

        # Separate heads for different action types
        self.navigation_head = nn.Linear(512, 3)  # x, y, theta
        self.manipulation_head = nn.Linear(512, 6)  # joint positions
        self.gripper_head = nn.Linear(512, 1)      # gripper position

    def forward(self, fused_features):
        """Decode fused features to actions."""
        # Single action prediction
        actions = self.decoder(fused_features)

        # Alternative: structured action prediction
        structured_actions = {
            'navigation': self.navigation_head(fused_features),
            'manipulation': self.manipulation_head(fused_features),
            'gripper': torch.sigmoid(self.gripper_head(fused_features)),
            'full_action': actions
        }

        return structured_actions


class VLAModel(nn.Module):
    """Complete VLA model combining vision, language, and action components."""

    def __init__(self, action_space_dim=12):
        super(VLAModel, self).__init__()

        # Initialize components
        self.vision_encoder = VisionEncoder()
        self.language_encoder = LanguageEncoder()
        self.multimodal_fusion = MultimodalFusion()
        self.action_decoder = ActionDecoder(action_space_dim)

    def forward(self, images, texts):
        """Forward pass through complete VLA model."""
        # Encode vision
        vision_features = self.vision_encoder(images)

        # Encode language
        language_features = self.language_encoder(texts)

        # Fuse modalities
        fused_features = self.multimodal_fusion(vision_features, language_features)

        # Decode to actions
        actions = self.action_decoder(fused_features)

        return actions

    def process_instruction(self, image, instruction):
        """Process single instruction with image."""
        with torch.no_grad():
            actions = self.forward(image.unsqueeze(0), [instruction])
            return actions


# Example usage and testing
if __name__ == "__main__":
    print("VLA Architecture Example")

    # Create VLA model
    vla_model = VLAModel(action_space_dim=12)
    print(f"VLA model created with {sum(p.numel() for p in vla_model.parameters()):,} parameters")

    # Test with dummy inputs
    dummy_image = torch.randn(1, 3, 224, 224)  # Batch of 1, 3-channel, 224x224 image
    dummy_text = ["Move forward"]

    print("Testing VLA model with dummy inputs...")
    with torch.no_grad():
        output = vla_model(dummy_image, dummy_text)

    print(f"Model output keys: {output.keys()}")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}")

    print("VLA architecture test completed successfully")
```

## Exercise 2: Working with Pre-trained Vision-Language Models

### Step 1: Implement CLIP integration

Create `~/vla_examples/clip_integration.py`:

```python
#!/usr/bin/env python3
# clip_integration.py
"""CLIP integration for VLA systems."""

import torch
import clip
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class CLIPBasedVLA:
    """VLA system based on CLIP model."""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Loading CLIP model on {device}...")

        # Load CLIP model
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()  # Set to evaluation mode

        # Action space definition
        self.action_space = {
            'navigation': ['move forward', 'move backward', 'turn left', 'turn right', 'stop'],
            'manipulation': ['pick up', 'put down', 'grasp', 'release', 'move to'],
            'general': ['go to', 'bring', 'fetch', 'avoid', 'follow']
        }

        print("CLIP model loaded successfully")

    def encode_image(self, image_path):
        """Encode image using CLIP."""
        if isinstance(image_path, str):
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        else:
            # Assume image is already a PIL Image
            image = self.preprocess(image_path).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize

        return image_features

    def encode_text(self, text):
        """Encode text using CLIP."""
        if isinstance(text, list):
            text_tokens = clip.tokenize(text).to(self.device)
        else:
            text_tokens = clip.tokenize([text]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize

        return text_features

    def compute_similarity(self, image_features, text_features):
        """Compute similarity between image and text."""
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarity

    def rank_candidates(self, image_path, candidate_texts):
        """Rank candidate texts by similarity to image."""
        image_features = self.encode_image(image_path)
        text_features = self.encode_text(candidate_texts)

        similarity = self.compute_similarity(image_features, text_features)
        sorted_indices = torch.argsort(similarity[0], descending=True)

        results = []
        for idx in sorted_indices:
            results.append({
                'text': candidate_texts[idx.item()],
                'similarity': similarity[0][idx].item()
            })

        return results

    def infer_action_from_instruction(self, image_path, instruction):
        """Infer action from image and instruction using CLIP."""
        # Combine navigation and manipulation actions
        all_actions = self.action_space['navigation'] + self.action_space['manipulation'] + self.action_space['general']

        # Add the instruction to the list of candidates
        candidates = all_actions + [instruction]

        # Rank all candidates
        rankings = self.rank_candidates(image_path, candidates)

        # Return top-ranked action
        top_action = rankings[0]
        return top_action

    def process_vla_task(self, image_path, instruction):
        """Process complete VLA task: image + instruction -> action."""
        print(f"Processing VLA task: '{instruction}' with image {image_path}")

        # Get action recommendation
        action_recommendation = self.infer_action_from_instruction(image_path, instruction)

        # Convert to robot command
        robot_command = self.convert_to_robot_command(action_recommendation['text'])

        result = {
            'instruction': instruction,
            'recommended_action': action_recommendation,
            'robot_command': robot_command,
            'confidence': action_recommendation['similarity']
        }

        return result

    def convert_to_robot_command(self, action_text):
        """Convert natural language action to robot command."""
        # This is a simplified mapping - in practice, this would be more complex
        action_mapping = {
            'move forward': [1.0, 0.0, 0.0],  # [x_vel, y_vel, angular_vel]
            'move backward': [-1.0, 0.0, 0.0],
            'turn left': [0.0, 0.0, 1.0],
            'turn right': [0.0, 0.0, -1.0],
            'stop': [0.0, 0.0, 0.0],
            'pick up': [0.0, 0.0, 0.0],  # Placeholder for manipulation
            'put down': [0.0, 0.0, 0.0],
            'grasp': [0.0, 0.0, 0.0],
            'release': [0.0, 0.0, 0.0],
            'go to': [0.0, 0.0, 0.0],
            'bring': [0.0, 0.0, 0.0],
            'fetch': [0.0, 0.0, 0.0],
            'avoid': [0.0, 0.0, 0.0],
            'follow': [0.0, 0.0, 0.0]
        }

        return action_mapping.get(action_text.lower(), [0.0, 0.0, 0.0])


class CLIPEnhancedVLA(CLIPBasedVLA):
    """Enhanced VLA system with improved capabilities."""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)

        # Enhanced action space with more specific actions
        self.enhanced_action_space = {
            'navigation': [
                'move forward 1 meter',
                'move backward 0.5 meters',
                'turn left 90 degrees',
                'turn right 90 degrees',
                'move forward slowly',
                'move forward quickly',
                'navigate around obstacle',
                'approach red object',
                'move toward blue object',
                'stop moving'
            ],
            'manipulation': [
                'pick up the red cup',
                'put the cup on the table',
                'grasp the blue ball',
                'release the object',
                'move object to left',
                'move object to right',
                'lift object gently',
                'place object carefully'
            ],
            'complex': [
                'go to the kitchen and bring the cup',
                'navigate to the table and place the object',
                'find the red cube and move it to the box',
                'avoid obstacles while moving forward'
            ]
        }

    def semantic_search(self, image_path, query, top_k=5):
        """Perform semantic search using CLIP."""
        # Create a more comprehensive set of candidates
        all_candidates = []
        for category, actions in self.enhanced_action_space.items():
            all_candidates.extend(actions)

        # Add the query itself
        all_candidates.append(query)

        # Rank candidates
        rankings = self.rank_candidates(image_path, all_candidates)

        # Return top-k results
        return rankings[:top_k]

    def generate_action_sequence(self, image_path, high_level_task):
        """Generate sequence of actions for complex tasks."""
        print(f"Generating action sequence for: '{high_level_task}'")

        # For complex tasks, break them down into subtasks
        if 'and' in high_level_task.lower():
            subtasks = high_level_task.lower().split(' and ')
        else:
            subtasks = [high_level_task]

        action_sequence = []
        for subtask in subtasks:
            # Get relevant actions for this subtask
            top_actions = self.semantic_search(image_path, subtask, top_k=3)
            action_sequence.append({
                'subtask': subtask,
                'recommended_actions': top_actions[:2]  # Top 2 recommendations
            })

        return action_sequence


# Example usage
if __name__ == "__main__":
    print("CLIP Integration Example")

    # Create CLIP-based VLA system
    vla_system = CLIPBasedVLA()

    # Example: Simulate processing an image with instruction
    # Note: In practice, you would use real image paths
    print("\nTesting with dummy image and instruction...")

    # Since we don't have a real image, we'll simulate the process
    dummy_instruction = "Move forward"
    print(f"Instruction: {dummy_instruction}")

    # This would normally process a real image
    # result = vla_system.process_vla_task("dummy_image.jpg", dummy_instruction)
    # print(f"Result: {result}")

    print("\nTesting enhanced VLA system...")
    enhanced_vla = CLIPEnhancedVLA()

    complex_task = "go to the kitchen and bring the cup"
    sequence = enhanced_vla.generate_action_sequence("dummy_image.jpg", complex_task)
    print(f"Action sequence for '{complex_task}':")
    for i, step in enumerate(sequence):
        print(f"  Step {i+1}: {step['subtask']}")
        for action in step['recommended_actions']:
            print(f"    - {action['text']}: {action['similarity']:.3f}")

    print("\nCLIP integration example completed successfully")
```

## Exercise 3: Multimodal Fusion Techniques

### Step 1: Implement advanced fusion methods

Create `~/vla_examples/multimodal_fusion.py`:

```python
#!/usr/bin/env python3
# multimodal_fusion.py
"""Advanced multimodal fusion techniques for VLA systems."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CrossModalAttention(nn.Module):
    """Cross-modal attention for vision-language fusion."""

    def __init__(self, feature_dim=512, num_heads=8):
        super(CrossModalAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        # Linear projections
        self.vision_proj = nn.Linear(feature_dim, feature_dim)
        self.language_proj = nn.Linear(feature_dim, feature_dim)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Linear(feature_dim, feature_dim)

        # Layer normalization
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, vision_features, language_features):
        """Apply cross-modal attention."""
        # Project features
        vision_proj = self.vision_proj(vision_features.unsqueeze(1))
        language_proj = self.language_proj(language_features.unsqueeze(1))

        # Cross-attention: vision attends to language
        attended_vision, attention_weights_vision = self.attention(
            vision_proj, language_proj, language_proj
        )

        # Cross-attention: language attends to vision
        attended_language, attention_weights_language = self.attention(
            language_proj, vision_proj, vision_proj
        )

        # Concatenate and project
        combined = torch.cat([
            attended_vision.squeeze(1),
            attended_language.squeeze(1)
        ], dim=-1)

        output = self.output_proj(combined)
        output = self.norm(output)

        return output, {
            'vision_attention': attention_weights_vision,
            'language_attention': attention_weights_language
        }


class CoAttentionFusion(nn.Module):
    """Co-attention fusion mechanism."""

    def __init__(self, feature_dim=512):
        super(CoAttentionFusion, self).__init__()
        self.feature_dim = feature_dim

        # Attention mechanisms
        self.vision_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Softmax(dim=-1)
        )

        self.language_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Softmax(dim=-1)
        )

        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, vision_features, language_features):
        """Apply co-attention fusion."""
        # Compute attention weights
        vision_att_weights = self.vision_attention(vision_features)
        language_att_weights = self.language_attention(language_features)

        # Apply attention
        attended_vision = vision_features * vision_att_weights
        attended_language = language_features * language_att_weights

        # Concatenate and fuse
        combined = torch.cat([attended_vision, attended_language], dim=-1)
        fused_features = self.fusion_net(combined)

        return fused_features


class HierarchicalFusion(nn.Module):
    """Hierarchical fusion for multi-level feature integration."""

    def __init__(self, feature_dims=[256, 512, 1024]):
        super(HierarchicalFusion, self).__init__()
        self.feature_dims = feature_dims

        # Level-specific fusion modules
        self.level_fusions = nn.ModuleList()
        for i, dim in enumerate(feature_dims):
            if i == 0:
                # First level: simple fusion
                self.level_fusions.append(
                    nn.Sequential(
                        nn.Linear(dim * 2, dim * 2),
                        nn.ReLU(),
                        nn.Linear(dim * 2, dim)
                    )
                )
            else:
                # Subsequent levels: incorporate previous level
                self.level_fusions.append(
                    nn.Sequential(
                        nn.Linear(feature_dims[i-1] + dim * 2, dim * 2),
                        nn.ReLU(),
                        nn.Linear(dim * 2, dim)
                    )
                )

        # Final fusion layer
        self.final_fusion = nn.Sequential(
            nn.Linear(sum(feature_dims), 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

    def forward(self, vision_features_list, language_features):
        """Apply hierarchical fusion across multiple levels."""
        fused_outputs = []

        for i, (vision_features, fusion_module) in enumerate(zip(vision_features_list, self.level_fusions)):
            if i == 0:
                # First level: just fuse vision and language
                level_input = torch.cat([vision_features, language_features], dim=-1)
            else:
                # Subsequent levels: include previous fused output
                prev_fused = fused_outputs[-1]
                level_input = torch.cat([prev_fused, vision_features, language_features], dim=-1)

            level_output = fusion_module(level_input)
            fused_outputs.append(level_output)

        # Final fusion of all levels
        final_input = torch.cat(fused_outputs, dim=-1)
        final_output = self.final_fusion(final_input)

        return final_output


class AdaptiveFusion(nn.Module):
    """Adaptive fusion that learns to weight different modalities."""

    def __init__(self, feature_dim=512):
        super(AdaptiveFusion, self).__init__()
        self.feature_dim = feature_dim

        # Gate networks to compute modality weights
        self.vision_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )

        self.language_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, vision_features, language_features):
        """Apply adaptive fusion with learned weights."""
        # Compute adaptive weights
        vision_weight = self.vision_gate(vision_features)
        language_weight = self.language_gate(language_features)

        # Apply weights
        weighted_vision = vision_features * vision_weight
        weighted_language = language_features * language_weight

        # Concatenate and fuse
        combined = torch.cat([weighted_vision, weighted_language], dim=-1)
        fused_output = self.fusion(combined)

        return fused_output, {
            'vision_weight': vision_weight.mean().item(),
            'language_weight': language_weight.mean().item()
        }


class FusionComparison:
    """Compare different fusion techniques."""

    def __init__(self):
        self.fusion_methods = {
            'cross_attention': CrossModalAttention(),
            'co_attention': CoAttentionFusion(),
            'adaptive': AdaptiveFusion(),
        }

    def test_fusion_methods(self, vision_features, language_features):
        """Test different fusion methods."""
        results = {}

        for name, fusion_module in self.fusion_methods.items():
            if name == 'adaptive':
                output, weights = fusion_module(vision_features, language_features)
                results[name] = {
                    'output': output,
                    'weights': weights
                }
            else:
                output = fusion_module(vision_features, language_features)
                results[name] = {
                    'output': output
                }

        return results

    def analyze_fusion_performance(self, results):
        """Analyze fusion method performance."""
        analysis = {}

        for method_name, result in results.items():
            output = result['output']

            # Compute various metrics
            analysis[method_name] = {
                'output_norm': torch.norm(output).item(),
                'output_variance': torch.var(output).item(),
                'output_mean': torch.mean(output).item(),
                'output_std': torch.std(output).item()
            }

            # Add method-specific metrics
            if 'weights' in result:
                analysis[method_name].update(result['weights'])

        return analysis


# Example usage and testing
if __name__ == "__main__":
    print("Multimodal Fusion Techniques Example")

    # Create fusion comparison system
    fusion_comparison = FusionComparison()

    # Generate dummy features for testing
    batch_size = 4
    feature_dim = 512

    vision_features = torch.randn(batch_size, feature_dim)
    language_features = torch.randn(batch_size, feature_dim)

    print(f"Testing fusion methods with features of shape: {vision_features.shape}")

    # Test all fusion methods
    results = fusion_comparison.test_fusion_methods(vision_features, language_features)

    # Analyze results
    analysis = fusion_comparison.analyze_fusion_performance(results)

    print("\nFusion Method Analysis:")
    for method, metrics in analysis.items():
        print(f"\n{method.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    # Test hierarchical fusion
    print("\nTesting hierarchical fusion...")
    hier_fusion = HierarchicalFusion(feature_dims=[256, 512, 1024])

    # Create features for different levels
    level1_vision = torch.randn(batch_size, 256)
    level2_vision = torch.randn(batch_size, 512)
    level3_vision = torch.randn(batch_size, 1024)
    language_features_large = torch.randn(batch_size, 1024)

    hier_output = hier_fusion([level1_vision, level2_vision, level3_vision], language_features_large)
    print(f"Hierarchical fusion output shape: {hier_output.shape}")

    print("\nMultimodal fusion techniques example completed successfully")
```

## Exercise 4: Action Mapping and Execution

### Step 1: Create action mapping system

Create `~/vla_examples/action_mapping.py`:

```python
#!/usr/bin/env python3
# action_mapping.py
"""Action mapping and execution system for VLA."""

import torch
import torch.nn as nn
import numpy as np
import re
from typing import Dict, List, Tuple, Any

class ActionMapper:
    """Map language instructions to robot actions."""

    def __init__(self):
        # Define action vocabulary
        self.action_types = {
            'navigation': ['move', 'go', 'navigate', 'walk', 'drive', 'travel', 'proceed'],
            'manipulation': ['pick', 'grasp', 'hold', 'carry', 'lift', 'raise', 'catch', 'catch', 'take'],
            'placement': ['place', 'put', 'set', 'drop', 'release', 'position', 'locate', 'deposit'],
            'rotation': ['turn', 'rotate', 'spin', 'pivot', 'face', 'orient', 'aim'],
            'interaction': ['push', 'pull', 'press', 'touch', 'activate', 'operate', 'use'],
            'locomotion': ['step', 'stride', 'crawl', 'jump', 'hop', 'stomp', 'tiptoe']
        }

        # Define spatial relations
        self.spatial_relations = [
            'to', 'toward', 'into', 'onto', 'over', 'under', 'behind', 'in_front_of',
            'left', 'right', 'above', 'below', 'near', 'far', 'close', 'away',
            'beside', 'next_to', 'between', 'among', 'through', 'across'
        ]

        # Define object properties
        self.object_properties = [
            'red', 'blue', 'green', 'yellow', 'big', 'small', 'large', 'tiny',
            'heavy', 'light', 'round', 'square', 'long', 'short', 'tall', 'wide'
        ]

        # Robot action space
        self.robot_action_space = {
            'navigation': {
                'linear_velocity': (-1.0, 1.0),  # m/s
                'angular_velocity': (-1.0, 1.0),  # rad/s
                'duration': (0.1, 5.0)  # seconds
            },
            'manipulation': {
                'joint_positions': (-2.0, 2.0),  # radians
                'gripper_position': (0.0, 1.0),  # normalized
                'force_limit': (0.0, 100.0)  # Newtons
            }
        }

    def parse_instruction(self, instruction: str) -> Dict[str, Any]:
        """Parse natural language instruction into structured representation."""
        instruction_lower = instruction.lower()

        # Extract action type
        action_type = self.identify_action_type(instruction_lower)

        # Extract spatial relations
        spatial_info = self.extract_spatial_relations(instruction_lower)

        # Extract object information
        objects = self.extract_objects(instruction_lower)

        # Extract numerical information
        numbers = self.extract_numbers(instruction_lower)

        # Extract directional information
        directions = self.extract_directions(instruction_lower)

        parsed_instruction = {
            'original': instruction,
            'action_type': action_type,
            'spatial_relations': spatial_info,
            'objects': objects,
            'numbers': numbers,
            'directions': directions,
            'confidence': 1.0  # This would come from NLP model in practice
        }

        return parsed_instruction

    def identify_action_type(self, instruction: str) -> str:
        """Identify the primary action type from instruction."""
        for action_type, keywords in self.action_types.items():
            for keyword in keywords:
                if keyword in instruction:
                    return action_type
        return 'unknown'

    def extract_spatial_relations(self, instruction: str) -> List[str]:
        """Extract spatial relations from instruction."""
        relations = []
        for relation in self.spatial_relations:
            if relation in instruction:
                relations.append(relation)
        return relations

    def extract_objects(self, instruction: str) -> List[str]:
        """Extract object mentions from instruction."""
        # Simple extraction - in practice, use NER
        words = instruction.split()
        objects = []
        properties = []

        for i, word in enumerate(words):
            if word in self.object_properties:
                properties.append(word)
            elif i > 0 and words[i-1] in self.object_properties:
                # Property followed by object
                objects.append(f"{words[i-1]} {word}")
            elif word not in [item for sublist in self.action_types.values() for item in sublist]:
                # Not an action keyword, might be an object
                if word not in self.spatial_relations and word not in self.object_properties:
                    objects.append(word)

        return objects

    def extract_numbers(self, instruction: str) -> List[float]:
        """Extract numerical values from instruction."""
        # Find numbers in the instruction
        numbers = re.findall(r'\d+\.?\d*', instruction)
        return [float(num) for num in numbers if num]

    def extract_directions(self, instruction: str) -> List[str]:
        """Extract directional information."""
        directions = []
        direction_keywords = ['forward', 'backward', 'left', 'right', 'up', 'down', 'north', 'south', 'east', 'west']
        for direction in direction_keywords:
            if direction in instruction:
                directions.append(direction)
        return directions

    def map_to_action_space(self, parsed_instruction: Dict[str, Any]) -> Dict[str, Any]:
        """Map parsed instruction to robot action space."""
        action_type = parsed_instruction['action_type']
        numbers = parsed_instruction['numbers']
        directions = parsed_instruction['directions']

        if action_type == 'navigation':
            action = self.map_navigation_action(parsed_instruction)
        elif action_type == 'manipulation':
            action = self.map_manipulation_action(parsed_instruction)
        elif action_type == 'placement':
            action = self.map_placement_action(parsed_instruction)
        elif action_type == 'rotation':
            action = self.map_rotation_action(parsed_instruction)
        else:
            action = self.map_generic_action(parsed_instruction)

        return action

    def map_navigation_action(self, parsed_instruction: Dict[str, Any]) -> Dict[str, Any]:
        """Map navigation instruction to navigation action."""
        action = {
            'type': 'navigation',
            'linear_velocity': 0.0,
            'angular_velocity': 0.0,
            'duration': 1.0,
            'target_position': None
        }

        # Determine direction and speed based on instruction
        directions = parsed_instruction['directions']
        numbers = parsed_instruction['numbers']

        if 'forward' in directions or 'ahead' in directions:
            action['linear_velocity'] = 0.5
        elif 'backward' in directions or 'back' in directions:
            action['linear_velocity'] = -0.5
        elif 'left' in directions:
            action['angular_velocity'] = 0.5
        elif 'right' in directions:
            action['angular_velocity'] = -0.5

        # Use numbers to determine duration or distance
        if numbers:
            # Assume first number is distance or duration
            distance_or_duration = numbers[0]
            if distance_or_duration < 5:  # Likely duration in seconds
                action['duration'] = distance_or_duration
            else:  # Likely distance in meters
                action['duration'] = distance_or_duration / 0.5  # distance / speed

        return action

    def map_manipulation_action(self, parsed_instruction: Dict[str, Any]) -> Dict[str, Any]:
        """Map manipulation instruction to manipulation action."""
        action = {
            'type': 'manipulation',
            'gripper_position': 0.0,  # Open
            'joint_positions': [0.0] * 6,  # Default joint positions
            'force_limit': 50.0
        }

        instruction = parsed_instruction['original'].lower()

        if 'grasp' in instruction or 'pick' in instruction or 'hold' in instruction:
            action['gripper_position'] = 1.0  # Close gripper
        elif 'release' in instruction or 'drop' in instruction:
            action['gripper_position'] = 0.0  # Open gripper

        # Adjust joint positions based on object properties
        objects = parsed_instruction['objects']
        for obj in objects:
            if 'cup' in obj or 'glass' in obj:
                # Special grip for cup-like objects
                action['joint_positions'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        return action

    def map_placement_action(self, parsed_instruction: Dict[str, Any]) -> Dict[str, Any]:
        """Map placement instruction to placement action."""
        action = {
            'type': 'placement',
            'target_surface': 'table',
            'position_offset': [0.0, 0.0, 0.0],
            'orientation': [0.0, 0.0, 0.0, 1.0]  # quaternion
        }

        spatial_relations = parsed_instruction['spatial_relations']
        objects = parsed_instruction['objects']

        if 'table' in objects or 'desk' in objects:
            action['target_surface'] = 'table'
        elif 'shelf' in objects or 'cabinet' in objects:
            action['target_surface'] = 'shelf'

        # Spatial relations affect placement
        if 'on' in spatial_relations or 'onto' in spatial_relations:
            action['position_offset'][2] = 0.1  # Lift slightly above surface
        elif 'under' in spatial_relations:
            action['position_offset'][2] = -0.1  # Below surface

        return action

    def map_rotation_action(self, parsed_instruction: Dict[str, Any]) -> Dict[str, Any]:
        """Map rotation instruction to rotation action."""
        action = {
            'type': 'rotation',
            'axis': 'z',  # Default to yaw rotation
            'angle': 0.0,
            'angular_velocity': 0.1
        }

        instruction = parsed_instruction['original'].lower()
        numbers = parsed_instruction['numbers']

        if 'degrees' in instruction or '°' in instruction:
            if numbers:
                action['angle'] = np.radians(numbers[0])  # Convert degrees to radians
        elif 'radians' in instruction:
            if numbers:
                action['angle'] = numbers[0]
        else:
            # Default to 90 degrees if not specified
            action['angle'] = np.radians(90)

        # Determine rotation axis
        if 'yaw' in instruction or 'turn' in instruction:
            action['axis'] = 'z'
        elif 'pitch' in instruction:
            action['axis'] = 'y'
        elif 'roll' in instruction:
            action['axis'] = 'x'

        return action

    def map_generic_action(self, parsed_instruction: Dict[str, Any]) -> Dict[str, Any]:
        """Map generic instruction to default action."""
        return {
            'type': 'generic',
            'parameters': {},
            'confidence': 0.5
        }

    def execute_action(self, action: Dict[str, Any], robot_interface):
        """Execute action using robot interface."""
        action_type = action['type']

        if action_type == 'navigation':
            return self.execute_navigation_action(action, robot_interface)
        elif action_type == 'manipulation':
            return self.execute_manipulation_action(action, robot_interface)
        elif action_type == 'placement':
            return self.execute_placement_action(action, robot_interface)
        elif action_type == 'rotation':
            return self.execute_rotation_action(action, robot_interface)
        else:
            print(f"Unknown action type: {action_type}")
            return False

    def execute_navigation_action(self, action: Dict[str, Any], robot_interface):
        """Execute navigation action."""
        print(f"Executing navigation: linear_vel={action['linear_velocity']}, "
              f"angular_vel={action['angular_velocity']}, duration={action['duration']}")
        # In practice, send commands to robot
        return True

    def execute_manipulation_action(self, action: Dict[str, Any], robot_interface):
        """Execute manipulation action."""
        print(f"Executing manipulation: gripper_pos={action['gripper_position']}")
        # In practice, control robot manipulator
        return True

    def execute_placement_action(self, action: Dict[str, Any], robot_interface):
        """Execute placement action."""
        print(f"Executing placement: surface={action['target_surface']}")
        # In practice, execute placement sequence
        return True

    def execute_rotation_action(self, action: Dict[str, Any], robot_interface):
        """Execute rotation action."""
        print(f"Executing rotation: axis={action['axis']}, angle={action['angle']}")
        # In practice, rotate robot/base
        return True


class VLAActionSystem:
    """Complete VLA action system combining parsing and execution."""

    def __init__(self):
        self.action_mapper = ActionMapper()
        self.execution_history = []

    def process_instruction(self, instruction: str, robot_interface=None):
        """Process natural language instruction and execute action."""
        print(f"Processing instruction: '{instruction}'")

        # Parse instruction
        parsed = self.action_mapper.parse_instruction(instruction)
        print(f"Parsed instruction: {parsed}")

        # Map to action space
        action = self.action_mapper.map_to_action_space(parsed)
        print(f"Mapped action: {action}")

        # Execute action
        if robot_interface:
            success = self.action_mapper.execute_action(action, robot_interface)
        else:
            # Simulate execution
            success = True
            print("(Simulated execution)")

        # Record in history
        history_entry = {
            'instruction': instruction,
            'parsed': parsed,
            'action': action,
            'success': success,
            'timestamp': np.datetime64('now')
        }
        self.execution_history.append(history_entry)

        return {
            'success': success,
            'action': action,
            'parsed_instruction': parsed
        }

    def get_execution_history(self):
        """Get execution history."""
        return self.execution_history

    def evaluate_action_quality(self, instruction: str, expected_outcome: str) -> float:
        """Evaluate the quality of action execution."""
        # This would involve comparing expected vs actual outcomes
        # For now, return a dummy score
        return 0.85  # 85% quality score


# Example usage
if __name__ == "__main__":
    print("Action Mapping System Example")

    # Create action system
    action_system = VLAActionSystem()

    # Test instructions
    test_instructions = [
        "Move forward 2 meters",
        "Turn left 90 degrees",
        "Pick up the red cup",
        "Place the object on the table",
        "Navigate to the kitchen and bring the cup"
    ]

    print("\nTesting action mapping system:")
    for instruction in test_instructions:
        result = action_system.process_instruction(instruction)
        print(f"Instruction: '{instruction}' -> Success: {result['success']}")
        print(f"  Action: {result['action']['type']}")
        print()

    # Show execution history
    history = action_system.get_execution_history()
    print(f"Execution history contains {len(history)} entries")

    # Evaluate some actions
    print("\nEvaluating action quality:")
    for instruction in test_instructions[:3]:  # Test first 3
        quality = action_system.evaluate_action_quality(instruction, "successful")
        print(f"  '{instruction}': Quality = {quality:.2f}")

    print("\nAction mapping system example completed successfully")
```

## Exercise 5: VLA Evaluation and Validation

### Step 1: Create evaluation metrics

Create `~/vla_examples/vla_evaluation.py`:

```python
#!/usr/bin/env python3
# vla_evaluation.py
"""Evaluation and validation system for VLA systems."""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

class VLAEvaluator:
    """Evaluation system for Vision-Language-Action models."""

    def __init__(self):
        self.metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'similarity_score': [],
            'language_alignment': [],
            'action_success_rate': [],
            'response_time': [],
            'safety_violations': []
        }
        self.evaluation_history = []

    def evaluate_action_prediction(self, predicted_actions, ground_truth_actions):
        """Evaluate action prediction accuracy."""
        if len(predicted_actions) != len(ground_truth_actions):
            raise ValueError("Predicted and ground truth actions must have same length")

        # Calculate accuracy
        correct = 0
        for pred, gt in zip(predicted_actions, ground_truth_actions):
            if self.action_equal(pred, gt):
                correct += 1

        accuracy = correct / len(predicted_actions) if predicted_actions else 0.0

        # Calculate precision, recall, F1 (for categorical actions)
        if ground_truth_actions and all(isinstance(gt, (int, str)) for gt in ground_truth_actions):
            y_true = ground_truth_actions
            y_pred = predicted_actions[:len(ground_truth_actions)]  # Ensure same length

            try:
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            except:
                precision = recall = f1 = 0.0
        else:
            precision = recall = f1 = 0.0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def action_equal(self, action1, action2):
        """Check if two actions are equal."""
        if isinstance(action1, dict) and isinstance(action2, dict):
            # Compare dictionary actions
            return self.dict_equal(action1, action2)
        elif isinstance(action1, (list, tuple)) and isinstance(action2, (list, tuple)):
            # Compare sequence actions
            return np.allclose(action1, action2, rtol=0.1)  # 10% tolerance
        else:
            # Compare scalar actions
            return action1 == action2

    def dict_equal(self, dict1, dict2, tolerance=0.1):
        """Compare dictionaries with numerical tolerance."""
        if set(dict1.keys()) != set(dict2.keys()):
            return False

        for key in dict1.keys():
            val1, val2 = dict1[key], dict2[key]
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if abs(val1 - val2) > tolerance:
                    return False
            elif val1 != val2:
                return False

        return True

    def evaluate_language_alignment(self, instructions, predicted_actions):
        """Evaluate how well actions align with language instructions."""
        alignment_scores = []

        for instruction, action in zip(instructions, predicted_actions):
            score = self.compute_alignment_score(instruction, action)
            alignment_scores.append(score)

        avg_alignment = np.mean(alignment_scores) if alignment_scores else 0.0

        return {
            'alignment_scores': alignment_scores,
            'average_alignment': avg_alignment,
            'std_alignment': np.std(alignment_scores) if alignment_scores else 0.0
        }

    def compute_alignment_score(self, instruction, action):
        """Compute alignment score between instruction and action."""
        # This is a simplified scoring - in practice, use more sophisticated methods
        instruction_lower = instruction.lower()
        action_str = str(action).lower()

        # Count relevant keywords
        keywords = ['move', 'go', 'turn', 'pick', 'place', 'grasp', 'navigate']
        instruction_keywords = sum(1 for keyword in keywords if keyword in instruction_lower)
        action_keywords = sum(1 for keyword in keywords if keyword in action_str)

        # Simple overlap score
        if instruction_keywords == 0:
            return 0.5  # Neutral if no keywords in instruction

        overlap = min(instruction_keywords, action_keywords)
        score = overlap / instruction_keywords

        return min(1.0, max(0.0, score))

    def evaluate_action_success_rate(self, execution_results):
        """Evaluate action execution success rate."""
        if not execution_results:
            return 0.0

        successful_executions = sum(1 for result in execution_results if result.get('success', False))
        success_rate = successful_executions / len(execution_results)

        return success_rate

    def evaluate_similarity(self, predicted_features, ground_truth_features):
        """Evaluate similarity between predicted and ground truth features."""
        if predicted_features is None or ground_truth_features is None:
            return 0.0

        # Compute cosine similarity
        if isinstance(predicted_features, torch.Tensor):
            predicted_features = predicted_features.detach().cpu().numpy()
        if isinstance(ground_truth_features, torch.Tensor):
            ground_truth_features = ground_truth_features.detach().cpu().numpy()

        # Normalize features
        pred_norm = predicted_features / (np.linalg.norm(predicted_features, axis=-1, keepdims=True) + 1e-8)
        gt_norm = ground_truth_features / (np.linalg.norm(ground_truth_features, axis=-1, keepdims=True) + 1e-8)

        # Compute similarity
        similarity = np.sum(pred_norm * gt_norm, axis=-1)
        avg_similarity = np.mean(similarity) if len(similarity) > 0 else 0.0

        return avg_similarity

    def run_comprehensive_evaluation(self, model, test_dataset):
        """Run comprehensive evaluation on test dataset."""
        print("Running comprehensive VLA evaluation...")

        all_metrics = defaultdict(list)
        execution_results = []

        for sample in test_dataset:
            # Process sample
            image = sample['image']
            instruction = sample['instruction']
            ground_truth_action = sample['action']

            # Get model prediction
            predicted_action = self.get_model_prediction(model, image, instruction)

            # Evaluate action prediction
            action_metrics = self.evaluate_action_prediction([predicted_action], [ground_truth_action])
            for metric, value in action_metrics.items():
                all_metrics[metric].append(value)

            # Evaluate language alignment
            alignment_metrics = self.evaluate_language_alignment([instruction], [predicted_action])
            all_metrics['language_alignment'].append(alignment_metrics['average_alignment'])

            # Record execution result
            execution_result = {
                'instruction': instruction,
                'predicted_action': predicted_action,
                'ground_truth': ground_truth_action,
                'success': self.action_equal(predicted_action, ground_truth_action)
            }
            execution_results.append(execution_result)

        # Calculate final metrics
        final_metrics = {}
        for metric, values in all_metrics.items():
            if values:
                final_metrics[metric] = np.mean(values)
                final_metrics[f'{metric}_std'] = np.std(values)

        # Calculate success rate
        final_metrics['action_success_rate'] = self.evaluate_action_success_rate(execution_results)

        # Add to evaluation history
        evaluation_result = {
            'timestamp': np.datetime64('now'),
            'metrics': final_metrics,
            'execution_results': execution_results,
            'dataset_size': len(test_dataset)
        }
        self.evaluation_history.append(evaluation_result)

        return final_metrics, evaluation_result

    def get_model_prediction(self, model, image, instruction):
        """Get prediction from model."""
        # This would call the actual model
        # For simulation, return a dummy prediction
        return {
            'type': 'navigation',
            'linear_velocity': 0.5,
            'angular_velocity': 0.0,
            'duration': 2.0
        }

    def visualize_evaluation_results(self, metrics, save_path='vla_evaluation_results.png'):
        """Visualize evaluation results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # Metric names to plot
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'language_alignment', 'action_success_rate']

        # Plot metrics
        for i, metric in enumerate(metric_names):
            if metric in metrics:
                value = metrics[metric]
                std_value = metrics.get(f'{metric}_std', 0)

                axes[i].bar([metric], [value], yerr=[std_value], capsize=5)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_ylim(0, 1)
                axes[i].set_ylabel('Score')

                # Add value label
                axes[i].text(0, value + std_value + 0.02, f'{value:.3f}',
                           ha='center', va='bottom', fontsize=10)

        # Remove unused subplot
        axes[5].remove()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Evaluation visualization saved to: {save_path}")

    def generate_evaluation_report(self, metrics, save_path='vla_evaluation_report.json'):
        """Generate comprehensive evaluation report."""
        report = {
            'timestamp': str(np.datetime64('now')),
            'metrics': metrics,
            'evaluation_summary': self.summarize_metrics(metrics),
            'recommendations': self.generate_recommendations(metrics)
        }

        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Evaluation report saved to: {save_path}")
        return report

    def summarize_metrics(self, metrics):
        """Summarize evaluation metrics."""
        summary = {
            'overall_performance': 'good' if metrics.get('accuracy', 0) > 0.7 else 'needs_improvement',
            'language_alignment_quality': 'good' if metrics.get('language_alignment', 0) > 0.6 else 'poor',
            'action_success_rate': f"{metrics.get('action_success_rate', 0):.2%}",
            'technical_indicators': {}
        }

        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and not metric.endswith('_std'):
                summary['technical_indicators'][metric] = f"{value:.3f}"

        return summary

    def generate_recommendations(self, metrics):
        """Generate recommendations based on metrics."""
        recommendations = []

        if metrics.get('accuracy', 0) < 0.7:
            recommendations.append("Model accuracy is below threshold. Consider retraining with more diverse data.")
        if metrics.get('language_alignment', 0) < 0.6:
            recommendations.append("Language alignment needs improvement. Consider better vision-language fusion.")
        if metrics.get('action_success_rate', 0) < 0.8:
            recommendations.append("Action success rate is low. Investigate action space mapping.")

        return recommendations if recommendations else ["Model performance is satisfactory."]


class SafetyEvaluator:
    """Safety evaluation for VLA systems."""

    def __init__(self):
        self.safety_rules = [
            self.check_collision_risk,
            self.check_stability,
            self.check_joint_limits,
            self.check_velocity_limits
        ]

    def check_collision_risk(self, action, environment_state):
        """Check if action poses collision risk."""
        # Simplified collision check
        if 'linear_velocity' in action and abs(action['linear_velocity']) > 0.8:
            return False  # High velocity might cause collision
        return True

    def check_stability(self, action, robot_state):
        """Check if action maintains robot stability."""
        # Simplified stability check
        if 'angular_velocity' in action and abs(action['angular_velocity']) > 0.5:
            return False  # High angular velocity might cause instability
        return True

    def check_joint_limits(self, action, robot_state):
        """Check if action violates joint limits."""
        if 'joint_positions' in action:
            joint_positions = action['joint_positions']
            # Check if any joint position is outside safe range
            for pos in joint_positions:
                if abs(pos) > 2.0:  # Example limit
                    return False
        return True

    def check_velocity_limits(self, action, robot_state):
        """Check if action violates velocity limits."""
        if 'linear_velocity' in action and abs(action['linear_velocity']) > 1.0:
            return False  # Exceeds velocity limit
        if 'angular_velocity' in action and abs(action['angular_velocity']) > 0.5:
            return False  # Exceeds angular velocity limit
        return True

    def evaluate_safety(self, action, environment_state, robot_state):
        """Evaluate action safety."""
        safety_results = {}
        is_safe = True

        for rule in self.safety_rules:
            try:
                rule_name = rule.__name__
                is_safe_rule = rule(action, robot_state)
                safety_results[rule_name] = is_safe_rule
                is_safe = is_safe and is_safe_rule
            except Exception as e:
                print(f"Safety rule {rule.__name__} failed: {e}")
                safety_results[rule.__name__] = False
                is_safe = False

        return {
            'is_safe': is_safe,
            'safety_results': safety_results,
            'risk_score': 1.0 - float(is_safe)  # Higher risk if not safe
        }


# Example usage
if __name__ == "__main__":
    print("VLA Evaluation System Example")

    # Create evaluator
    evaluator = VLAEvaluator()

    # Create safety evaluator
    safety_evaluator = SafetyEvaluator()

    # Simulate test dataset
    test_dataset = [
        {
            'image': torch.randn(3, 224, 224),
            'instruction': 'Move forward',
            'action': {'type': 'navigation', 'linear_velocity': 0.5, 'angular_velocity': 0.0}
        },
        {
            'image': torch.randn(3, 224, 224),
            'instruction': 'Turn left',
            'action': {'type': 'navigation', 'linear_velocity': 0.0, 'angular_velocity': 0.5}
        },
        {
            'image': torch.randn(3, 224, 224),
            'instruction': 'Pick up object',
            'action': {'type': 'manipulation', 'gripper_position': 1.0}
        }
    ]

    print(f"Testing with {len(test_dataset)} samples...")

    # Run evaluation (simulated)
    dummy_metrics = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.88,
        'f1_score': 0.85,
        'language_alignment': 0.78,
        'action_success_rate': 0.83,
        'accuracy_std': 0.05,
        'precision_std': 0.06,
        'recall_std': 0.04,
        'f1_score_std': 0.05,
        'language_alignment_std': 0.08,
        'action_success_rate_std': 0.07
    }

    print("\nEvaluation Results:")
    for metric, value in dummy_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Test safety evaluation
    print("\nTesting safety evaluation...")
    test_action = {
        'type': 'navigation',
        'linear_velocity': 0.5,
        'angular_velocity': 0.2,
        'joint_positions': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    }

    safety_result = safety_evaluator.evaluate_safety(test_action, {}, {})
    print(f"Safety evaluation result: {safety_result}")

    # Visualize results (would use dummy metrics in this example)
    print("\nGenerating evaluation visualization...")
    evaluator.visualize_evaluation_results(dummy_metrics)

    # Generate report
    print("\nGenerating evaluation report...")
    report = evaluator.generate_evaluation_report(dummy_metrics)

    print("\nVLA evaluation system example completed successfully")
```

## Troubleshooting

### Common Issues and Solutions

1. **Poor Language-Action Alignment**:
   - Increase training data diversity
   - Use better vision-language pre-trained models
   - Implement fine-grained alignment loss functions
   - Use reinforcement learning for policy improvement

2. **Action Space Mismatch**:
   - Ensure consistent action space definitions
   - Implement proper action space mapping
   - Use action discretization for complex continuous spaces
   - Validate action feasibility before execution

3. **Vision-Language Fusion Issues**:
   - Verify feature dimensions match across modalities
   - Use proper normalization for features
   - Implement attention mechanisms for better fusion
   - Add cross-modal regularization

4. **Real-World Transfer Problems**:
   - Use domain randomization in training
   - Implement sim-to-real adaptation techniques
   - Collect real-world data for fine-tuning
   - Use robust feature extraction methods

5. **Performance Bottlenecks**:
   - Optimize model inference with quantization
   - Use efficient attention mechanisms
   - Implement caching for repeated computations
   - Consider model compression techniques

## Assessment Questions

1. How do you evaluate the alignment between language instructions and robot actions?
2. What are the key challenges in fusing vision and language modalities?
3. How would you handle ambiguous language instructions in VLA systems?
4. What metrics would you use to evaluate VLA system performance?
5. How do you ensure safety in VLA system execution?

## Extension Exercises

1. Implement a transformer-based VLA architecture
2. Create a multimodal dataset for VLA training
3. Implement domain adaptation for sim-to-real transfer
4. Create a safety-aware VLA system
5. Implement reinforcement learning for VLA policy improvement

## Summary

In this lab, you successfully:
- Implemented VLA system architecture with vision, language, and action components
- Integrated CLIP for vision-language understanding
- Created advanced multimodal fusion techniques
- Developed action mapping from language to robot actions
- Implemented evaluation and validation systems for VLA performance
- Created safety evaluation mechanisms

These skills are fundamental for developing Vision-Language-Action systems that can understand natural language instructions, perceive visual information, and execute appropriate robotic actions. The combination of multimodal understanding, action planning, and safety considerations forms the foundation for building intelligent robotic systems that can interact naturally with humans in real-world environments.