---
sidebar_label: 'Lab 4.2: Multimodal Perception'
---

# Lab Exercise 4.2: Multimodal Perception in VLA Systems

This lab exercise covers implementing multimodal perception systems that integrate vision, language, and other sensory inputs for VLA systems.

## Objectives

- Implement multimodal feature extraction
- Create cross-modal attention mechanisms
- Integrate multiple sensor modalities
- Test perception accuracy with multimodal inputs

## Prerequisites

- Completed VLA fundamentals lab
- Understanding of computer vision and NLP
- PyTorch/TensorFlow knowledge
- ROS 2 Humble with perception packages

## Multimodal Perception Architecture

### Sensor Integration Framework

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import torchvision.transforms as T
from PIL import Image

class MultimodalSensorFusion:
    def __init__(self):
        # Initialize sensor encoders
        self.vision_encoder = VisionEncoder()
        self.language_encoder = LanguageEncoder()
        self.depth_encoder = DepthEncoder()
        self.audio_encoder = AudioEncoder()  # If available

        # Fusion dimensions
        self.vision_dim = 512
        self.language_dim = 512
        self.depth_dim = 256
        self.fusion_dim = 1024

        # Cross-modal attention modules
        self.vision_language_attention = CrossModalAttention(self.vision_dim, self.language_dim)
        self.vision_depth_attention = CrossModalAttention(self.vision_dim, self.depth_dim)

    def forward(self, sensors: Dict[str, torch.Tensor], language_input: str):
        """Process multimodal sensor inputs with language context"""
        outputs = {}

        # Process individual modalities
        if 'rgb' in sensors:
            vision_features = self.vision_encoder(sensors['rgb'])
            outputs['vision'] = vision_features

        if 'depth' in sensors:
            depth_features = self.depth_encoder(sensors['depth'])
            outputs['depth'] = depth_features

        if language_input:
            language_features = self.language_encoder(language_input)
            outputs['language'] = language_features

        # Cross-modal fusion
        if 'vision' in outputs and 'language' in outputs:
            fused_vl = self.vision_language_attention(
                outputs['vision'], outputs['language']
            )
            outputs['vision_language'] = fused_vl

        if 'vision' in outputs and 'depth' in outputs:
            fused_vd = self.vision_depth_attention(
                outputs['vision'], outputs['depth']
            )
            outputs['vision_depth'] = fused_vd

        return outputs

class VisionEncoder(nn.Module):
    def __init__(self, backbone='resnet50'):
        super().__init__()

        if backbone == 'resnet50':
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=True)
            self.feature_dim = 2048
            self.backbone.fc = nn.Identity()  # Remove classification layer
        elif backbone == 'clip':
            import clip
            self.backbone, _ = clip.load("ViT-B/32", device='cpu')
            self.feature_dim = 512

        # Spatial feature extraction
        self.spatial_pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, images):
        features = self.backbone(images)

        if len(features.shape) == 4:  # CNN features
            features = self.spatial_pool(features)
            # Reshape to (batch, channels, height*width)
            batch_size, channels, h, w = features.shape
            features = features.view(batch_size, channels, h*w)
            features = features.transpose(1, 2)  # (batch, h*w, channels)

        return features

class DepthEncoder(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()

        # Simple CNN for depth processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 128

    def forward(self, depth_maps):
        features = self.conv_layers(depth_maps)
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)  # Flatten
        return features

class LanguageEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()

        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.feature_dim = self.backbone.config.hidden_size

    def forward(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )

        outputs = self.backbone(**inputs)
        # Use [CLS] token representation
        return outputs.last_hidden_state[:, 0, :]
```

## Cross-Modal Attention Mechanisms

### Vision-Language Attention

```python
class VisionLanguageAttention(nn.Module):
    def __init__(self, vision_dim, language_dim, hidden_dim=512):
        super().__init__()

        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.hidden_dim = hidden_dim

        # Projection layers
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, vision_features, language_features):
        """
        vision_features: (batch_size, num_patches, vision_dim)
        language_features: (batch_size, language_dim)
        """
        batch_size = vision_features.size(0)

        # Project features
        vision_proj = self.vision_proj(vision_features)  # (B, num_patches, H)
        language_proj = self.language_proj(language_features).unsqueeze(1)  # (B, 1, H)

        # Concatenate vision and language features
        all_features = torch.cat([language_proj, vision_proj], dim=1)  # (B, 1+num_patches, H)

        # Self-attention
        attended_features, attention_weights = self.attention(
            all_features, all_features, all_features
        )

        # Extract attended language features (first position)
        attended_language = attended_features[:, 0, :]  # (B, H)

        # Extract attended vision features (remaining positions)
        attended_vision = attended_features[:, 1:, :]  # (B, num_patches, H)

        # Apply output projection
        attended_language = self.output_proj(attended_language)
        attended_vision = self.output_proj(attended_vision)

        return attended_vision, attended_language, attention_weights

class CrossModalAttention(nn.Module):
    def __init__(self, dim1, dim2, hidden_dim=512):
        super().__init__()

        self.proj1 = nn.Linear(dim1, hidden_dim)
        self.proj2 = nn.Linear(dim2, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features1, features2):
        # Project to common dimension
        f1_proj = self.proj1(features1)
        f2_proj = self.proj2(features2)

        # Cross attention: features1 attending to features2
        attended_f1, _ = self.attention(f1_proj, f2_proj, f2_proj)
        attended_f1 = self.norm(attended_f1 + f1_proj)

        # Cross attention: features2 attending to features1
        attended_f2, _ = self.attention(f2_proj, f1_proj, f1_proj)
        attended_f2 = self.norm(attended_f2 + f2_proj)

        return attended_f1, attended_f2
```

## Spatial-Temporal Perception

### Spatio-Temporal Feature Integration

```python
class SpatioTemporalPerceiver(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, num_layers=2):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # Transformer layers for spatio-temporal processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Spatial attention
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, spatial_features, temporal_features=None):
        """
        spatial_features: (batch, num_patches, feature_dim)
        temporal_features: (batch, time_steps, feature_dim) or None
        """
        batch_size, num_patches, feature_dim = spatial_features.shape

        if temporal_features is not None:
            # Process temporal dimension
            attended_temporal = self.temporal_transformer(temporal_features)

            # Aggregate temporal information
            temporal_summary = attended_temporal.mean(dim=1, keepdim=True)  # (B, 1, F)

            # Add temporal context to spatial features
            spatial_with_temporal = spatial_features + temporal_summary
        else:
            spatial_with_temporal = spatial_features

        # Apply spatial attention
        attended_spatial, attention_weights = self.spatial_attention(
            spatial_with_temporal, spatial_with_temporal, spatial_with_temporal
        )

        return attended_spatial, attention_weights
```

## Multimodal Object Detection

### Vision-Language Object Detection

```python
class VisionLanguageObjectDetector(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()

        # Vision backbone
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        self.vision_detector = fasterrcnn_resnet50_fpn(pretrained=True)

        # Language encoder for class descriptions
        self.language_encoder = LanguageEncoder()

        # Fusion module for vision-language grounding
        self.grounding_module = nn.Sequential(
            nn.Linear(1024, 512),  # Assuming concatenated features
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.num_classes = num_classes

    def forward(self, images, class_descriptions=None):
        # Get vision-based detections
        vision_detections = self.vision_detector(images)

        if class_descriptions is not None:
            # Encode class descriptions
            class_features = self.language_encoder(class_descriptions)

            # Ground detections with language
            grounded_detections = self.ground_detections(
                vision_detections, class_features
            )

            return grounded_detections

        return vision_detections

    def ground_detections(self, vision_detections, class_features):
        """Ground vision detections with language descriptions"""
        results = []

        for detection in vision_detections:
            boxes = detection['boxes']  # (N, 4)
            labels = detection['labels']  # (N,)
            scores = detection['scores']  # (N,)

            # For each detection, compute grounding score with class descriptions
            grounded_scores = []
            for i in range(len(boxes)):
                # Get region features (simplified - in practice, extract from backbone)
                region_features = self.extract_region_features(boxes[i])

                # Compute grounding score
                grounding_score = self.compute_grounding_score(
                    region_features, class_features
                )

                grounded_scores.append(grounding_score)

            # Update scores with grounding information
            detection['grounding_scores'] = torch.tensor(grounded_scores)
            results.append(detection)

        return results

    def extract_region_features(self, box):
        """Extract features for a specific region (simplified)"""
        # In practice, this would use ROI pooling on the backbone features
        return torch.randn(512)  # Placeholder

    def compute_grounding_score(self, region_features, class_features):
        """Compute grounding score between region and class descriptions"""
        # Compute similarity (dot product or cosine similarity)
        similarity = torch.dot(region_features, class_features[0])  # Simplified
        return torch.sigmoid(similarity)
```

## Multimodal Perception Node

### ROS 2 Integration

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Point
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray, Detection2D
from cv_bridge import CvBridge
import numpy as np

class MultimodalPerceptionNode(Node):
    def __init__(self):
        super().__init__('multimodal_perception_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Publishers and subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10
        )
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/points2', self.pointcloud_callback, 10
        )
        self.command_sub = self.create_subscription(
            String, '/perception_command', self.command_callback, 10
        )

        self.detection_pub = self.create_publisher(
            Detection2DArray, '/multimodal_detections', 10
        )

        # Initialize multimodal perception system
        self.perception_system = MultimodalSensorFusion()

        # Storage for sensor data
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_command = None

    def rgb_callback(self, msg):
        """Process RGB image"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Convert to tensor and preprocess
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.latest_rgb = transform(cv_image).unsqueeze(0)

            # Process if we have command
            if self.latest_command:
                self.process_multimodal_request()

        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {e}')

    def depth_callback(self, msg):
        """Process depth image"""
        try:
            cv_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            # Convert to tensor
            depth_tensor = torch.from_numpy(cv_depth).unsqueeze(0).unsqueeze(0).float()
            self.latest_depth = depth_tensor

            # Process if we have other data
            if self.latest_rgb is not None and self.latest_command is not None:
                self.process_multimodal_request()

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def pointcloud_callback(self, msg):
        """Process point cloud data"""
        # Extract 3D information from point cloud
        # This would typically involve segmentation and object detection
        pass

    def command_callback(self, msg):
        """Process perception command"""
        self.latest_command = msg.data

        # Process if we have sensor data
        if self.latest_rgb is not None:
            self.process_multimodal_request()

    def process_multimodal_request(self):
        """Process multimodal perception request"""
        if (self.latest_rgb is None or
            self.latest_depth is None or
            self.latest_command is None):
            return

        try:
            # Prepare sensor data
            sensors = {
                'rgb': self.latest_rgb,
                'depth': self.latest_depth
            }

            # Process with multimodal perception system
            perception_results = self.perception_system(
                sensors, self.latest_command
            )

            # Publish results
            self.publish_perception_results(perception_results)

            # Clear command after processing
            self.latest_command = None

        except Exception as e:
            self.get_logger().error(f'Error in multimodal processing: {e}')

    def publish_perception_results(self, results):
        """Publish perception results"""
        detection_array = Detection2DArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = 'camera_rgb_optical_frame'

        # Convert results to Detection2DArray message
        # This would depend on your specific output format
        for key, value in results.items():
            if key.startswith('vision'):
                # Create detection message
                detection = Detection2D()
                # Fill in detection details based on your results
                detection_array.detections.append(detection)

        self.detection_pub.publish(detection_array)
```

## Scene Understanding and Grounding

### Multimodal Scene Graph

```python
class MultimodalSceneGraph:
    def __init__(self):
        self.objects = {}
        self.relations = []
        self.attributes = {}

    def add_object(self, obj_id, visual_features, language_description):
        """Add object with visual and language features"""
        self.objects[obj_id] = {
            'visual_features': visual_features,
            'language_description': language_description,
            'bbox': None,
            'category': None
        }

    def add_relation(self, obj1_id, obj2_id, relation_type, confidence=1.0):
        """Add relation between objects"""
        self.relations.append({
            'subject': obj1_id,
            'object': obj2_id,
            'relation': relation_type,
            'confidence': confidence
        })

    def ground_language_in_scene(self, language_query):
        """Ground language query in the scene"""
        # Match language descriptions to objects
        matches = []

        for obj_id, obj_data in self.objects.items():
            # Compute similarity between query and object description
            similarity = self.compute_language_similarity(
                language_query, obj_data['language_description']
            )

            if similarity > 0.5:  # Threshold
                matches.append({
                    'object_id': obj_id,
                    'similarity': similarity,
                    'bbox': obj_data['bbox']
                })

        return sorted(matches, key=lambda x: x['similarity'], reverse=True)

    def compute_language_similarity(self, query, description):
        """Compute similarity between language query and description"""
        # This would use a language model or embedding similarity
        # For now, return a simple similarity score
        query_words = set(query.lower().split())
        desc_words = set(description.lower().split())

        intersection = len(query_words.intersection(desc_words))
        union = len(query_words.union(desc_words))

        return intersection / union if union > 0 else 0.0

class SceneUnderstandingModule(nn.Module):
    def __init__(self):
        super().__init__()

        # Object detection and segmentation
        self.object_detector = VisionLanguageObjectDetector()

        # Spatial reasoning module
        self.spatial_reasoner = SpatioTemporalPerceiver()

        # Scene graph builder
        self.scene_graph = MultimodalSceneGraph()

    def forward(self, images, language_query):
        """Process image and language to build scene understanding"""
        # Detect objects in image
        detections = self.object_detector(images)

        # Extract visual features for detected objects
        visual_features = self.extract_object_features(images, detections)

        # Build scene graph
        for i, detection in enumerate(detections):
            obj_id = f'obj_{i}'
            self.scene_graph.add_object(
                obj_id,
                visual_features[i],
                detection.get('description', 'unknown')
            )

        # Ground language query in scene
        grounded_objects = self.scene_graph.ground_language_in_scene(language_query)

        return {
            'detections': detections,
            'grounded_objects': grounded_objects,
            'scene_graph': self.scene_graph
        }

    def extract_object_features(self, images, detections):
        """Extract features for detected objects"""
        # This would typically use ROI pooling or similar techniques
        features = []
        for detection in detections:
            # Extract features for each detected object
            features.append(torch.randn(512))  # Placeholder
        return features
```

## Exercise Tasks

1. Implement the multimodal sensor fusion architecture
2. Create cross-modal attention mechanisms for vision-language integration
3. Build a spatio-temporal perception module
4. Implement vision-language object detection
5. Create a ROS 2 node for multimodal perception
6. Test the system with various sensor inputs and language queries

## Evaluation Metrics

### Multimodal Perception Evaluation

```python
class MultimodalEvaluation:
    def __init__(self):
        self.metrics = {
            'vision_grounding_accuracy': [],
            'language_alignment_score': [],
            'spatial_reasoning_accuracy': [],
            'temporal_consistency': []
        }

    def evaluate_vision_grounding(self, predicted_objects, ground_truth_objects):
        """Evaluate how well vision detections align with language"""
        correct = 0
        total = len(ground_truth_objects)

        for gt_obj in ground_truth_objects:
            for pred_obj in predicted_objects:
                if self.objects_match(gt_obj, pred_obj):
                    correct += 1
                    break

        accuracy = correct / total if total > 0 else 0
        self.metrics['vision_grounding_accuracy'].append(accuracy)
        return accuracy

    def evaluate_language_alignment(self, vision_features, language_features):
        """Evaluate alignment between vision and language features"""
        # Compute similarity score
        similarity = torch.cosine_similarity(
            vision_features.flatten(),
            language_features.flatten(),
            dim=0
        )
        score = similarity.item()
        self.metrics['language_alignment_score'].append(score)
        return score

    def objects_match(self, obj1, obj2):
        """Check if two objects match based on spatial overlap and semantic similarity"""
        # Check spatial overlap (IoU)
        iou = self.calculate_iou(obj1['bbox'], obj2['bbox'])

        # Check semantic similarity
        semantic_sim = self.calculate_semantic_similarity(
            obj1['description'],
            obj2['description']
        )

        return iou > 0.5 and semantic_sim > 0.7

    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union"""
        # Implementation of IoU calculation
        pass

    def calculate_semantic_similarity(self, desc1, desc2):
        """Calculate semantic similarity between descriptions"""
        # Use language model embeddings or simple word overlap
        pass
```

## Troubleshooting

### Common Issues

- **Feature dimension mismatches**: Ensure all modalities project to compatible dimensions
- **Memory issues**: Use appropriate batch sizes for multimodal processing
- **Synchronization problems**: Ensure sensor data alignment in time
- **Grounding failures**: Improve language-vision alignment training

## Summary

In this lab, you implemented multimodal perception systems that integrate vision, language, and other sensory inputs. You created cross-modal attention mechanisms, built scene understanding modules, and integrated the system with ROS 2. This enables robots to understand complex scenes by combining multiple sensory modalities with natural language understanding.