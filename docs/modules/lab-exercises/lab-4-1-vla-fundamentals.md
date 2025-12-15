---
sidebar_label: 'Lab 4.1: VLA Fundamentals'
---

# Lab Exercise 4.1: Vision-Language-Action Fundamentals

This lab exercise covers the fundamental concepts of Vision-Language-Action systems and their implementation.

## Objectives

- Understand the multi-modal integration in VLA systems
- Implement basic vision-language fusion
- Create simple action generation from language commands
- Test VLA pipeline with basic commands

## Prerequisites

- Python with PyTorch/TensorFlow
- Basic understanding of computer vision and NLP
- ROS 2 Humble with necessary packages

## VLA System Architecture

### Multi-Modal Integration

Vision-Language-Action systems integrate three key modalities:
- **Vision**: Understanding visual scene information
- **Language**: Processing natural language commands
- **Action**: Executing appropriate robot behaviors

### Basic VLA Pipeline

```
Visual Input → Vision Encoder → Multi-Modal Fusion → Language Encoder → Action Decoder → Robot Actions
```

## Vision Processing Component

### Vision Encoder Implementation

```python
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50
import clip

class VisionEncoder(nn.Module):
    def __init__(self, model_type='resnet', pretrained=True):
        super().__init__()

        if model_type == 'resnet':
            self.backbone = resnet50(pretrained=pretrained)
            self.feature_dim = 2048
            # Remove the final classification layer
            self.backbone.fc = nn.Identity()
        elif model_type == 'clip':
            self.backbone, _ = clip.load("ViT-B/32", device='cpu')
            self.feature_dim = 512

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, images):
        if isinstance(self.backbone, type(clip.load("ViT-B/32", device='cpu')[0])):
            # CLIP model
            features = self.backbone.encode_image(images)
        else:
            # ResNet or other model
            features = self.backbone(images)

        return features

class VisionProcessor:
    def __init__(self, model_type='resnet'):
        self.encoder = VisionEncoder(model_type)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            self.encoder.normalize
        ])

    def process_image(self, image):
        """Process a single image and extract features"""
        transformed_image = self.transform(image).unsqueeze(0)  # Add batch dimension
        features = self.encoder(transformed_image)
        return features
```

## Language Processing Component

### Language Encoder Implementation

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import clip

class LanguageEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', use_clip=False):
        super().__init__()

        self.use_clip = use_clip

        if use_clip:
            # Use CLIP's text encoder
            clip_model, _ = clip.load("ViT-B/32", device='cpu')
            self.backbone = clip_model.encode_text
            self.feature_dim = 512
        else:
            # Use transformer model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.backbone = AutoModel.from_pretrained(model_name)
            self.feature_dim = self.backbone.config.hidden_size

    def forward(self, text_inputs):
        if self.use_clip:
            # CLIP expects tokenized text
            if isinstance(text_inputs, list):
                # Tokenize multiple texts
                texts = clip.tokenize(text_inputs)
                return self.backbone(texts)
            else:
                # Single text
                text = clip.tokenize([text_inputs])
                return self.backbone(text)
        else:
            # Transformer model
            outputs = self.backbone(**text_inputs)
            # Use [CLS] token representation
            return outputs.last_hidden_state[:, 0, :]

class LanguageProcessor:
    def __init__(self, model_name='bert-base-uncased', use_clip=False):
        self.encoder = LanguageEncoder(model_name, use_clip)
        self.use_clip = use_clip

        if not use_clip:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def process_text(self, text):
        """Process text and extract features"""
        if self.use_clip:
            # CLIP handles tokenization internally
            features = self.encoder(text)
        else:
            # Tokenize for transformer
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            )
            features = self.encoder(inputs)

        return features
```

## Multi-Modal Fusion

### Cross-Modal Attention Implementation

```python
import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert self.head_dim * num_heads == feature_dim, "Feature dim must be divisible by num heads"

        # Query, Key, Value projections for vision and language
        self.vision_q = nn.Linear(feature_dim, feature_dim)
        self.vision_k = nn.Linear(feature_dim, feature_dim)
        self.vision_v = nn.Linear(feature_dim, feature_dim)

        self.language_q = nn.Linear(feature_dim, feature_dim)
        self.language_k = nn.Linear(feature_dim, feature_dim)
        self.language_v = nn.Linear(feature_dim, feature_dim)

        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, vision_features, language_features):
        batch_size = vision_features.size(0)

        # Project features
        v_q = self.vision_q(vision_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_k = self.vision_k(vision_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_v = self.vision_v(vision_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        l_q = self.language_q(language_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        l_k = self.language_k(language_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        l_v = self.language_v(language_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Cross-attention: vision attending to language
        v_attn = torch.matmul(v_q, l_k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        v_attn = torch.softmax(v_attn, dim=-1)
        v_attn = self.dropout(v_attn)
        v_out = torch.matmul(v_attn, l_v)
        v_out = v_out.transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim)

        # Cross-attention: language attending to vision
        l_attn = torch.matmul(l_q, v_k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        l_attn = torch.softmax(l_attn, dim=-1)
        l_attn = self.dropout(l_attn)
        l_out = torch.matmul(l_attn, v_v)
        l_out = l_out.transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim)

        # Combine outputs
        combined_features = torch.cat([v_out, l_out], dim=-1)
        output = self.out_proj(combined_features)

        return output

class MultiModalFusion(nn.Module):
    def __init__(self, vision_feature_dim, language_feature_dim, fusion_dim=512):
        super().__init__()

        self.vision_proj = nn.Linear(vision_feature_dim, fusion_dim)
        self.language_proj = nn.Linear(language_feature_dim, fusion_dim)

        # Cross-modal attention
        self.cross_attention = CrossModalAttention(fusion_dim)

        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )

    def forward(self, vision_features, language_features):
        # Project features to common space
        vision_proj = self.vision_proj(vision_features)
        language_proj = self.language_proj(language_features)

        # Apply cross-modal attention
        attended_features = self.cross_attention(vision_proj, language_proj)

        # Global average pooling if needed
        if len(attended_features.shape) > 2:
            attended_features = attended_features.mean(dim=1)

        # Final fusion
        fused_features = self.fusion_layers(attended_features)

        return fused_features
```

## Action Generation Component

### Action Decoder Implementation

```python
import torch
import torch.nn as nn

class ActionDecoder(nn.Module):
    def __init__(self, fusion_dim, action_space_dim, hidden_dim=512):
        super().__init__()

        self.action_network = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space_dim)
        )

        # For continuous action spaces (e.g., robot joint commands)
        self.continuous_head = nn.Sequential(
            nn.Linear(action_space_dim, action_space_dim),
            nn.Tanh()  # Keep actions in reasonable range
        )

        # For discrete action spaces (e.g., navigation commands)
        self.discrete_head = nn.Linear(action_space_dim, action_space_dim)

    def forward(self, fused_features, action_type='continuous'):
        action_features = self.action_network(fused_features)

        if action_type == 'continuous':
            actions = self.continuous_head(action_features)
        else:  # discrete
            actions = self.discrete_head(action_features)
            actions = torch.softmax(actions, dim=-1)

        return actions

class ActionProcessor:
    def __init__(self, fusion_dim, action_space_dim):
        self.decoder = ActionDecoder(fusion_dim, action_space_dim)

    def generate_action(self, fused_features, action_type='continuous'):
        """Generate action from fused features"""
        return self.decoder(fused_features, action_type)
```

## Complete VLA System

### Integrated VLA Model

```python
import torch
import torch.nn as nn

class VLAModel(nn.Module):
    def __init__(self,
                 vision_feature_dim=2048,
                 language_feature_dim=768,
                 fusion_dim=512,
                 action_space_dim=4):
        super().__init__()

        # Components
        self.vision_encoder = VisionEncoder()
        self.language_encoder = LanguageEncoder()
        self.fusion_module = MultiModalFusion(
            vision_feature_dim,
            language_feature_dim,
            fusion_dim
        )
        self.action_decoder = ActionDecoder(fusion_dim, action_space_dim)

        # Dimension parameters
        self.action_space_dim = action_space_dim

    def forward(self, images, texts, action_type='continuous'):
        # Encode vision
        vision_features = self.vision_encoder(images)

        # Encode language
        language_features = self.language_encoder(texts)

        # Fuse modalities
        fused_features = self.fusion_module(vision_features, language_features)

        # Generate action
        actions = self.action_decoder(fused_features, action_type)

        return actions

    def process_command(self, image, command_text, action_type='continuous'):
        """Process a single image-command pair"""
        # Add batch dimension
        image_batch = image.unsqueeze(0) if len(image.shape) == 3 else image
        text_batch = [command_text] if isinstance(command_text, str) else command_text

        # Forward pass
        actions = self.forward(image_batch, text_batch, action_type)

        return actions.squeeze(0)  # Remove batch dimension

class VLAProcessor:
    def __init__(self, action_space_dim=4):
        self.vla_model = VLAModel(action_space_dim=action_space_dim)
        self.vision_processor = VisionProcessor()
        self.language_processor = LanguageProcessor()

    def execute_command(self, image, command):
        """Execute a vision-language command"""
        # Process image
        vision_features = self.vision_processor.process_image(image)

        # Process language command
        language_features = self.language_processor.process_text(command)

        # Generate action through the model
        action = self.vla_model(image.unsqueeze(0), [command])

        return action
```

## ROS 2 Integration

### VLA ROS Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
from PIL import Image as PILImage

class VLAROSNode(Node):
    def __init__(self):
        super().__init__('vla_ros_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.command_sub = self.create_subscription(
            String, '/vla/command', self.command_callback, 10
        )
        self.action_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Initialize VLA processor
        self.vla_processor = VLAProcessor(action_space_dim=2)  # [linear_x, angular_z]

        # Store latest image and command
        self.latest_image = None
        self.pending_command = None

    def image_callback(self, msg):
        """Process incoming image"""
        try:
            # Convert ROS image to PIL image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            pil_image = PILImage.fromarray(cv_image)

            # Store for processing with command
            self.latest_image = pil_image

            # If we have a pending command, process now
            if self.pending_command:
                self.process_vla_request()

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def command_callback(self, msg):
        """Process incoming command"""
        self.pending_command = msg.data

        # If we have a recent image, process now
        if self.latest_image:
            self.process_vla_request()

    def process_vla_request(self):
        """Process image-command pair"""
        if self.latest_image is None or self.pending_command is None:
            return

        try:
            # Convert PIL image to tensor
            transform = self.vla_processor.vision_processor.transform
            image_tensor = transform(self.latest_image)

            # Process with VLA model
            action = self.vla_processor.execute_command(
                image_tensor,
                self.pending_command
            )

            # Convert action to robot command (Twist message)
            cmd_vel = self.convert_action_to_twist(action)

            # Publish command
            self.action_pub.publish(cmd_vel)

            # Clear pending command
            self.pending_command = None

            self.get_logger().info(f'Executed command: {self.pending_command}')

        except Exception as e:
            self.get_logger().error(f'Error processing VLA request: {e}')

    def convert_action_to_twist(self, action_tensor):
        """Convert action tensor to Twist message"""
        cmd_vel = Twist()

        # Assuming action tensor has [linear_x, angular_z]
        action_values = action_tensor.detach().cpu().numpy()

        cmd_vel.linear.x = float(action_values[0]) if len(action_values) > 0 else 0.0
        cmd_vel.angular.z = float(action_values[1]) if len(action_values) > 1 else 0.0

        return cmd_vel
```

## Example Usage and Testing

### VLA Testing Script

```python
def test_vla_system():
    """Test the VLA system with sample inputs"""
    import numpy as np
    from PIL import Image as PILImage

    # Initialize VLA processor
    vla_processor = VLAProcessor(action_space_dim=2)

    # Create a dummy image (in practice, this would come from camera)
    dummy_image = PILImage.new('RGB', (224, 224), color='red')

    # Test commands
    test_commands = [
        "move forward",
        "turn left",
        "go to the blue object",
        "navigate to the kitchen"
    ]

    print("Testing VLA System:")
    for command in test_commands:
        print(f"\nCommand: '{command}'")

        # Process with VLA
        action = vla_processor.execute_command(dummy_image, command)
        print(f"Generated action: {action.detach().cpu().numpy()}")

def main():
    rclpy.init()

    # Create and run VLA node
    vla_node = VLAROSNode()

    try:
        rclpy.spin(vla_node)
    except KeyboardInterrupt:
        pass
    finally:
        vla_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercise Tasks

1. Implement the Vision, Language, and Action components of the VLA system
2. Create the multi-modal fusion module with cross-attention
3. Integrate the components into a complete VLA model
4. Test the system with simple image-command pairs
5. Create a ROS 2 node that processes VLA commands
6. Evaluate the system's ability to follow basic commands

## Troubleshooting

### Common Issues

- **Dimension mismatches**: Ensure all feature dimensions align between components
- **Memory issues**: Use appropriate batch sizes for your GPU memory
- **Tokenization problems**: Handle text preprocessing carefully
- **ROS integration**: Verify message types and topic names

## Summary

In this lab, you learned to implement the fundamental components of a Vision-Language-Action system. You created vision and language encoders, implemented multi-modal fusion with cross-attention, and connected the system to robot action generation. This forms the core architecture for advanced robotic systems that can understand and execute natural language commands based on visual input.