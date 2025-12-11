# Lab 4.2: Multimodal Perception in Vision-Language-Action Systems

## Overview

This lab exercise focuses on implementing multimodal perception systems that combine visual, linguistic, and action data for humanoid robots. Students will learn to integrate multiple sensory inputs and create unified representations for decision-making.

## Learning Objectives

By the end of this lab, students will be able to:
1. Implement multimodal fusion architectures combining vision and language
2. Design attention mechanisms for cross-modal information processing
3. Create unified feature spaces for vision-language-action systems
4. Evaluate multimodal perception performance using relevant metrics
5. Integrate multimodal perception with action planning systems

## Prerequisites

- Completion of Lab 4.1: VLA Fundamentals
- Understanding of deep learning frameworks (PyTorch/TensorFlow)
- Basic knowledge of computer vision and natural language processing
- Familiarity with ROS 2 for robot control

## Theory Background

Multimodal perception combines information from multiple sensory modalities to create a more robust and comprehensive understanding of the environment. In VLA systems, this typically involves:

- **Visual Processing**: Extracting spatial and object information from camera feeds
- **Language Processing**: Understanding commands, descriptions, and contextual information
- **Cross-Modal Attention**: Mechanisms that allow different modalities to attend to relevant information
- **Unified Representations**: Embedding space where different modalities can interact meaningfully

## Lab Exercise

### Part 1: Cross-Modal Attention Implementation

First, let's implement a cross-modal attention mechanism that can attend to relevant visual regions based on language descriptions:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, visual_features, language_features):
        """
        Args:
            visual_features: [batch_size, seq_len_v, d_model] - Visual features from CNN
            language_features: [batch_size, seq_len_l, d_model] - Language features from encoder
        Returns:
            attended_features: [batch_size, seq_len_v, d_model] - Visual features attended by language
        """
        batch_size = visual_features.size(0)

        # Project features
        Q = self.q_proj(language_features)  # Language as queries
        K = self.k_proj(visual_features)    # Visual as keys
        V = self.v_proj(visual_features)    # Visual as values

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)

        # Reshape back to original dimensions
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Output projection
        output = self.out_proj(attended_values)

        return output

# Test the cross-modal attention
def test_cross_modal_attention():
    batch_size, seq_len_v, seq_len_l, d_model = 2, 49, 10, 256  # 7x7 visual grid, 10-word sentence

    visual_features = torch.randn(batch_size, seq_len_v, d_model)
    language_features = torch.randn(batch_size, seq_len_l, d_model)

    attention_module = CrossModalAttention(d_model)
    attended_output = attention_module(visual_features, language_features)

    print(f"Input visual shape: {visual_features.shape}")
    print(f"Input language shape: {language_features.shape}")
    print(f"Output attended shape: {attended_output.shape}")

    return attended_output

if __name__ == "__main__":
    test_cross_modal_attention()
```

### Part 2: Multimodal Fusion Network

Now let's create a multimodal fusion network that combines visual and language features:

```python
class MultimodalFusion(nn.Module):
    def __init__(self, visual_dim, language_dim, fused_dim):
        super().__init__()
        self.visual_dim = visual_dim
        self.language_dim = language_dim
        self.fused_dim = fused_dim

        # Individual modality encoders
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.language_encoder = nn.Sequential(
            nn.Linear(language_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Cross-attention modules
        self.visual_to_language = CrossModalAttention(fused_dim)
        self.language_to_visual = CrossModalAttention(fused_dim)

        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fused_dim, fused_dim)
        )

        # Residual connection
        self.residual_proj = nn.Linear(fused_dim * 2, fused_dim)

    def forward(self, visual_input, language_input):
        """
        Args:
            visual_input: [batch_size, visual_dim] or [batch_size, seq_len_v, visual_dim]
            language_input: [batch_size, language_dim] or [batch_size, seq_len_l, language_dim]
        Returns:
            fused_output: [batch_size, fused_dim] - Fused multimodal representation
        """
        # Encode individual modalities
        encoded_visual = self.visual_encoder(visual_input)
        encoded_language = self.language_encoder(language_input)

        # Cross-attention: language attends to visual, visual attends to language
        lang_attended_vis = self.visual_to_language(encoded_visual, encoded_language)
        vis_attended_lang = self.language_to_visual(encoded_language, encoded_visual)

        # Aggregate attended features (mean pooling for sequences)
        if len(lang_attended_vis.shape) > 2:
            lang_attended_vis = lang_attended_vis.mean(dim=1)  # Average over sequence dimension
        if len(vis_attended_lang.shape) > 2:
            vis_attended_lang = vis_attended_lang.mean(dim=1)  # Average over sequence dimension

        # Concatenate and fuse
        concat_features = torch.cat([lang_attended_vis, vis_attended_lang], dim=-1)

        # Apply fusion with residual connection
        fused_output = self.fusion_layer(concat_features)
        residual = self.residual_proj(concat_features)

        final_output = fused_output + residual

        return final_output

# Test the multimodal fusion
def test_multimodal_fusion():
    batch_size, visual_dim, language_dim, fused_dim = 4, 512, 512, 256

    visual_input = torch.randn(batch_size, visual_dim)
    language_input = torch.randn(batch_size, language_dim)

    fusion_module = MultimodalFusion(visual_dim, language_dim, fused_dim)
    fused_output = fusion_module(visual_input, language_input)

    print(f"Visual input shape: {visual_input.shape}")
    print(f"Language input shape: {language_input.shape}")
    print(f"Fused output shape: {fused_output.shape}")

    return fused_output

if __name__ == "__main__":
    test_multimodal_fusion()
```

### Part 3: Vision-Language-Action Integration

Now let's create an integrated system that connects perception to action:

```python
class VisionLanguageActionSystem(nn.Module):
    def __init__(self, visual_backbone, language_backbone, action_space_dim):
        super().__init__()
        self.visual_backbone = visual_backbone
        self.language_backbone = language_backbone

        # Feature dimensions after backbones
        self.visual_dim = 512  # From visual backbone
        self.language_dim = 512  # From language backbone
        self.fused_dim = 256

        # Multimodal fusion
        self.multimodal_fusion = MultimodalFusion(
            self.visual_dim, self.language_dim, self.fused_dim
        )

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(self.fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, action_space_dim)
        )

        # Action decoder
        self.action_decoder = nn.GRU(
            input_size=action_space_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )

    def forward(self, visual_obs, language_command):
        """
        Args:
            visual_obs: [batch_size, channels, height, width] - Visual observation
            language_command: [batch_size, seq_len, embed_dim] - Language command
        Returns:
            action_pred: [batch_size, action_space_dim] - Predicted action
        """
        # Extract features from backbones
        visual_features = self.visual_backbone(visual_obs)  # Shape: [batch, visual_dim]
        language_features = self.language_backbone(language_command)  # Shape: [batch, seq_len, language_dim]

        # Fuse multimodal information
        fused_representation = self.multimodal_fusion(visual_features, language_features)

        # Predict initial action
        action_pred = self.action_head(fused_representation)

        return action_pred

# Dummy backbone models for testing
class DummyVisualBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 512)
        )

    def forward(self, x):
        return self.conv_layers(x)

class DummyLanguageBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 512)  # 1000 vocab size
        self.lstm = nn.LSTM(512, 512, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return lstm_out  # Return sequence for attention

# Test the integrated system
def test_vla_system():
    batch_size, channels, height, width = 2, 3, 224, 224
    seq_len, vocab_size = 10, 1000
    action_space_dim = 14  # 14 DOF humanoid robot

    visual_obs = torch.randn(batch_size, channels, height, width)
    language_command = torch.randint(0, vocab_size, (batch_size, seq_len))

    vla_system = VisionLanguageActionSystem(
        DummyVisualBackbone(),
        DummyLanguageBackbone(),
        action_space_dim
    )

    action_pred = vla_system(visual_obs, language_command)

    print(f"Visual observation shape: {visual_obs.shape}")
    print(f"Language command shape: {language_command.shape}")
    print(f"Action prediction shape: {action_pred.shape}")
    print(f"Action prediction: {action_pred[0]}")

    return action_pred

if __name__ == "__main__":
    test_vla_system()
```

### Part 4: Evaluation Metrics for Multimodal Perception

Let's implement evaluation metrics for multimodal perception:

```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class MultimodalEvaluator:
    def __init__(self):
        pass

    def compute_modality_alignment(self, visual_features, language_features):
        """
        Compute alignment between visual and language modalities using cosine similarity
        """
        # Normalize features
        visual_norm = F.normalize(visual_features, p=2, dim=-1)
        language_norm = F.normalize(language_features, p=2, dim=-1)

        # Compute cosine similarity
        alignment_scores = torch.sum(visual_norm * language_norm, dim=-1)

        return alignment_scores.mean().item()

    def compute_attention_heatmap_similarity(self, attention_maps_1, attention_maps_2):
        """
        Compute similarity between attention heatmaps
        """
        # Flatten attention maps and compute correlation
        flat_map_1 = attention_maps_1.view(attention_maps_1.size(0), -1)
        flat_map_2 = attention_maps_2.view(attention_maps_2.size(0), -1)

        # Compute cosine similarity between flattened maps
        similarity = F.cosine_similarity(flat_map_1, flat_map_2, dim=1)

        return similarity.mean().item()

    def evaluate_perception_accuracy(self, predicted_actions, ground_truth_actions, threshold=0.1):
        """
        Evaluate perception accuracy based on action predictions
        """
        # Compute mean squared error
        mse = F.mse_loss(predicted_actions, ground_truth_actions)

        # Compute accuracy based on threshold
        diff = torch.abs(predicted_actions - ground_truth_actions)
        accuracy = (diff < threshold).float().mean()

        return {
            'mse': mse.item(),
            'accuracy': accuracy.item(),
            'rmse': torch.sqrt(mse).item()
        }

# Example usage of evaluator
def test_evaluation():
    # Simulate some predictions and ground truth
    batch_size, action_dim = 8, 14
    predicted_actions = torch.randn(batch_size, action_dim)
    ground_truth_actions = torch.randn(batch_size, action_dim)

    evaluator = MultimodalEvaluator()
    eval_results = evaluator.evaluate_perception_accuracy(
        predicted_actions,
        ground_truth_actions
    )

    print("Evaluation Results:")
    for metric, value in eval_results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    test_evaluation()
```

### Part 5: Integration with ROS 2

Finally, let's create a ROS 2 node that integrates the multimodal perception system:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
import torch
import numpy as np

class MultimodalPerceptionNode(Node):
    def __init__(self):
        super().__init__('multimodal_perception_node')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Load the trained VLA model
        self.load_model()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/robot/command',
            self.command_callback,
            10
        )

        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Timer for processing loop
        self.timer = self.create_timer(0.1, self.process_loop)

        # Internal state
        self.current_image = None
        self.current_command = None
        self.last_action = None

    def load_model(self):
        """Load the trained multimodal perception model"""
        # Initialize the model (using dummy backbones for this example)
        self.visual_backbone = DummyVisualBackbone()
        self.language_backbone = DummyLanguageBackbone()

        self.vla_model = VisionLanguageActionSystem(
            self.visual_backbone,
            self.language_backbone,
            action_space_dim=2  # For Twist message (linear.x, angular.z)
        )

        # Set model to evaluation mode
        self.vla_model.eval()
        self.get_logger().info('Multimodal perception model loaded')

    def image_callback(self, msg):
        """Process incoming image messages"""
        try:
            # Convert ROS image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Preprocess image for model
            self.current_image = self.preprocess_image(cv_image)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def command_callback(self, msg):
        """Process incoming command messages"""
        try:
            # Process natural language command
            self.current_command = self.process_language_command(msg.data)
        except Exception as e:
            self.get_logger().error(f'Error processing command: {str(e)}')

    def preprocess_image(self, cv_image):
        """Preprocess image for the model"""
        # Resize image to expected input size
        resized = cv2.resize(cv_image, (224, 224))

        # Convert BGR to RGB and normalize
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_image.astype(np.float32) / 255.0

        # Transpose to CHW format and add batch dimension
        tensor_image = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)

        return tensor_image

    def process_language_command(self, command_str):
        """Convert natural language command to tensor representation"""
        # Simple tokenization for demonstration
        # In practice, you'd use a proper tokenizer
        tokens = command_str.lower().split()[:10]  # Limit to 10 tokens

        # Convert to indices (dummy conversion for demo)
        vocab = {"move": 1, "forward": 2, "backward": 3, "left": 4, "right": 5, "stop": 6}
        token_indices = []

        for token in tokens:
            if token in vocab:
                token_indices.append(vocab[token])
            else:
                token_indices.append(0)  # Unknown token

        # Pad or truncate to fixed length
        while len(token_indices) < 10:
            token_indices.append(0)  # Padding token

        return torch.tensor([token_indices])  # Add batch dimension

    def process_loop(self):
        """Main processing loop"""
        if self.current_image is not None and self.current_command is not None:
            try:
                # Generate action prediction
                with torch.no_grad():
                    action_pred = self.vla_model(self.current_image, self.current_command)

                # Convert prediction to Twist message
                twist_msg = Twist()
                twist_msg.linear.x = float(action_pred[0, 0])  # Forward/backward
                twist_msg.angular.z = float(action_pred[0, 1])  # Turn left/right

                # Publish action
                self.action_pub.publish(twist_msg)
                self.last_action = action_pred

                self.get_logger().info(f'Published action: linear.x={twist_msg.linear.x:.3f}, '
                                     f'angular.z={twist_msg.angular.z:.3f}')

            except Exception as e:
                self.get_logger().error(f'Error in processing loop: {str(e)}')

    def destroy_node(self):
        """Cleanup when node is destroyed"""
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    perception_node = MultimodalPerceptionNode()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        pass
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Implementation Steps

1. Create the multimodal perception files in your workspace
2. Implement the cross-modal attention mechanism
3. Build the fusion network
4. Integrate with action prediction
5. Test the evaluation metrics
6. Deploy the ROS 2 node

## Expected Outcomes

After completing this lab, you should have:
1. A working cross-modal attention system that can attend to relevant visual regions based on language
2. A multimodal fusion network that combines visual and language features effectively
3. An integrated VLA system that connects perception to action
4. Evaluation metrics to assess multimodal perception performance
5. A ROS 2 node that implements multimodal perception for robot control

## Troubleshooting Tips

- If attention weights are NaN, check for proper normalization and gradient clipping
- If fusion performance is poor, try adjusting the fusion architecture or training procedure
- If ROS 2 integration fails, verify message types and topic names
- Monitor GPU memory usage during training and inference

## Further Exploration

- Experiment with different attention mechanisms (e.g., self-attention, co-attention)
- Implement temporal attention for video sequences
- Add tactile sensing modalities to the fusion system
- Create hierarchical fusion architectures for multi-level perception