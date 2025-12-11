# Training VLA Models

## Overview

Training Vision-Language-Action (VLA) models requires specialized techniques to handle the multimodal nature of the data and the complex mapping between language, vision, and robotic actions. This section covers data collection, model architectures, training methodologies, and evaluation strategies for VLA systems.

## Data Collection and Preparation

### Multimodal Dataset Requirements

VLA training requires datasets that include:
- **Visual data**: Images, videos, or 3D point clouds from robot perspective
- **Language data**: Natural language instructions and descriptions
- **Action data**: Robot actions, trajectories, or motor commands
- **Temporal context**: Sequential relationships between states and actions

### Data Collection Strategies

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image

class VLADataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        VLA dataset containing vision, language, and action data.

        Args:
            data_path: Path to dataset directory
            transform: Optional transforms to apply to images
        """
        self.data_path = data_path
        self.transform = transform

        # Load dataset metadata
        self.samples = self.load_dataset_metadata()

    def load_dataset_metadata(self):
        """Load dataset metadata from JSON or other format."""
        import json

        # Example metadata structure
        metadata_path = f"{self.data_path}/metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        samples = []
        for entry in metadata:
            sample = {
                'image_path': f"{self.data_path}/images/{entry['image_id']}.jpg",
                'instruction': entry['instruction'],
                'action': np.array(entry['action']),  # Robot action vector
                'language_tokens': entry['language_tokens'],
                'sequence_id': entry['sequence_id'],
                'timestamp': entry['timestamp']
            }
            samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Process instruction
        instruction = sample['instruction']

        # Load action
        action = torch.FloatTensor(sample['action'])

        return {
            'image': image,
            'instruction': instruction,
            'action': action,
            'sequence_id': sample['sequence_id']
        }

# Data loading with appropriate batching
def create_vla_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """Create data loader for VLA training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=vla_collate_fn
    )

def vla_collate_fn(batch):
    """Custom collate function for VLA data."""
    images = torch.stack([item['image'] for item in batch])
    instructions = [item['instruction'] for item in batch]
    actions = torch.stack([item['action'] for item in batch])
    sequence_ids = [item['sequence_id'] for item in batch]

    return {
        'images': images,
        'instructions': instructions,
        'actions': actions,
        'sequence_ids': sequence_ids
    }
```

### Data Augmentation for VLA

```python
import torchvision.transforms as transforms
import random

class VLATransforms:
    def __init__(self):
        # Vision augmentation
        self.vision_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Language augmentation (for training robustness)
        self.language_augmentations = [
            self.synonym_replacement,
            self.random_insertion,
            self.random_deletion
        ]

    def augment_language(self, instruction):
        """Apply language augmentations to instruction."""
        augmented_instruction = instruction

        # Apply random augmentations
        if random.random() < 0.3:  # 30% chance of augmentation
            aug_func = random.choice(self.language_augmentations)
            augmented_instruction = aug_func(augmented_instruction)

        return augmented_instruction

    def synonym_replacement(self, text):
        """Replace words with synonyms."""
        # This would use a thesaurus or word embeddings
        # Simplified example
        synonyms = {
            'pick': ['grasp', 'take', 'grab'],
            'move': ['go', 'navigate', 'travel'],
            'place': ['put', 'set', 'position']
        }

        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in synonyms:
                if random.random() < 0.3:  # 30% chance to replace
                    words[i] = random.choice(synonyms[word.lower()])

        return ' '.join(words)

    def random_insertion(self, text):
        """Randomly insert words."""
        words = text.split()
        insertions = ['please', 'carefully', 'gently', 'slowly']

        if len(words) > 0 and random.random() < 0.2:
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, random.choice(insertions))

        return ' '.join(words)

    def random_deletion(self, text):
        """Randomly delete non-critical words."""
        words = text.split()
        critical_words = ['pick', 'place', 'move', 'go', 'turn', 'stop']

        filtered_words = [word for word in words if word.lower() not in critical_words or random.random() > 0.3]

        return ' '.join(filtered_words) if filtered_words else text
```

### Synthetic Data Generation

```python
class SyntheticVLADataGenerator:
    def __init__(self, simulation_env):
        self.sim_env = simulation_env

    def generate_synthetic_data(self, num_samples=10000):
        """Generate synthetic VLA training data using simulation."""
        synthetic_data = []

        for i in range(num_samples):
            # Generate random scene
            scene = self.create_random_scene()

            # Generate natural language instruction
            instruction = self.generate_instruction(scene)

            # Execute instruction in simulation to get ground truth action
            action = self.execute_instruction_get_action(scene, instruction)

            # Capture image from robot perspective
            image = self.capture_robot_view(scene)

            synthetic_data.append({
                'image': image,
                'instruction': instruction,
                'action': action,
                'scene_context': scene
            })

        return synthetic_data

    def create_random_scene(self):
        """Create random scene for synthetic data."""
        # Place random objects in environment
        scene = {
            'objects': [],
            'robot_pose': [0, 0, 0],  # x, y, theta
            'target_positions': []
        }

        # Add random objects
        num_objects = np.random.randint(1, 5)
        for _ in range(num_objects):
            obj_type = random.choice(['cup', 'box', 'ball', 'bottle'])
            position = [
                np.random.uniform(-1, 1),  # x
                np.random.uniform(-1, 1),  # y
                np.random.uniform(0.5, 1.5)  # z
            ]
            scene['objects'].append({
                'type': obj_type,
                'position': position,
                'color': random.choice(['red', 'blue', 'green', 'yellow'])
            })

        return scene

    def generate_instruction(self, scene):
        """Generate natural language instruction for scene."""
        actions = ['pick', 'place', 'move', 'navigate', 'grasp']
        objects = [obj['type'] for obj in scene['objects']]
        colors = [obj['color'] for obj in scene['objects']]

        action = random.choice(actions)
        obj = random.choice(objects)
        color = random.choice(colors) if random.random() > 0.5 else None

        if color:
            instruction = f"{action} the {color} {obj}"
        else:
            instruction = f"{action} the {obj}"

        return instruction
```

## Model Architectures for VLA

### End-to-End VLA Model

```python
class EndToEndVLA(nn.Module):
    def __init__(self, vision_backbone='resnet50', language_model='bert-base-uncased', action_dim=12):
        super().__init__()

        # Vision encoder
        self.vision_encoder = VisionEncoder(vision_backbone)

        # Language encoder
        self.language_encoder = LanguageEncoder(language_model)

        # Multimodal fusion
        self.fusion_module = MultimodalFusion(feature_dim=512)

        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, action_dim)
        )

        # Optional: temporal modeling
        self.temporal_encoder = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )

    def forward(self, images, instructions, prev_actions=None):
        # Encode vision
        vision_features = self.vision_encoder(images)

        # Encode language
        language_features = self.language_encoder(instructions)

        # Fuse modalities
        fused_features = self.fusion_module(vision_features, language_features)

        # Optional: temporal context
        if prev_actions is not None:
            temporal_input = torch.cat([fused_features.unsqueeze(1), prev_actions], dim=1)
            temporal_output, _ = self.temporal_encoder(temporal_input)
            fused_features = temporal_output[:, -1, :]  # Use last output

        # Decode to action
        action = self.action_decoder(fused_features)

        return action

class VisionEncoder(nn.Module):
    def __init__(self, backbone='resnet50'):
        super().__init__()
        import torchvision.models as models

        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
        elif backbone == 'vit':
            from transformers import ViTModel
            self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')

        # Remove classification head
        if hasattr(self.backbone, 'fc'):
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            self.feature_dim = 768  # ViT feature dimension

        # Projection to common space
        self.projection = nn.Linear(self.feature_dim, 512)

    def forward(self, images):
        features = self.backbone(images)
        if hasattr(features, 'last_hidden_state'):
            # For ViT
            features = features.last_hidden_state.mean(dim=1)
        else:
            # For ResNet
            features = features

        projected = self.projection(features)
        return projected

class LanguageEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Projection to common space
        self.projection = nn.Linear(self.model.config.hidden_size, 512)

    def forward(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )

        outputs = self.model(**inputs)
        # Use [CLS] token representation
        features = outputs.last_hidden_state[:, 0, :]

        projected = self.projection(features)
        return projected
```

### Transformer-Based VLA Architecture

```python
import torch.nn.functional as F

class TransformerVLA(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, action_dim=12):
        super().__init__()

        self.d_model = d_model
        self.action_dim = action_dim

        # Vision tokenization
        self.vision_patch_embed = nn.Conv2d(3, d_model, kernel_size=16, stride=16)
        self.vision_pos_embed = nn.Parameter(torch.randn(1, 197, d_model))  # 14x14 patches + CLS token

        # Language tokenization
        self.language_embed = nn.Embedding(30522, d_model)  # BERT vocab size
        self.language_pos_embed = nn.Parameter(torch.randn(1, 128, d_model))  # Max sequence length

        # Cross-modal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Combined vision-language
            nn.ReLU(),
            nn.Linear(d_model, action_dim)
        )

        # Normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, images, input_ids, attention_mask=None):
        batch_size = images.size(0)

        # Process vision
        vision_patches = self.vision_patch_embed(images)  # [B, C, H, W] -> [B, C, H', W']
        vision_patches = vision_patches.flatten(2).transpose(1, 2)  # [B, N, C]

        # Add CLS token
        cls_token = torch.zeros(batch_size, 1, self.d_model, device=images.device)
        vision_tokens = torch.cat([cls_token, vision_patches], dim=1)

        # Add positional embeddings
        vision_tokens = vision_tokens + self.vision_pos_embed[:, :vision_tokens.size(1)]

        # Process language
        language_tokens = self.language_embed(input_ids)
        language_tokens = language_tokens + self.language_pos_embed[:, :language_tokens.size(1)]

        # Apply attention mask to language tokens
        if attention_mask is not None:
            language_tokens = language_tokens * attention_mask.unsqueeze(-1).float()

        # Combine vision and language
        combined_tokens = torch.cat([vision_tokens, language_tokens], dim=1)

        # Apply transformer
        fused_features = self.transformer(combined_tokens)

        # Use CLS token for vision and mean pooling for language
        vision_cls = fused_features[:, 0, :]  # Vision CLS token
        language_features = fused_features[:, vision_tokens.size(1):, :]

        if attention_mask is not None:
            # Masked mean pooling for language
            masked_lang = language_features * attention_mask.unsqueeze(-1).float()
            language_pooled = masked_lang.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            language_pooled = language_features.mean(dim=1)

        # Combine and predict action
        combined_features = torch.cat([vision_cls, language_pooled], dim=-1)
        action = self.action_head(combined_features)

        return action
```

## Training Strategies

### Imitation Learning

```python
class ImitationLearningTrainer:
    def __init__(self, model, learning_rate=1e-4, weight_decay=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )

    def train_step(self, batch):
        """Single training step for imitation learning."""
        images = batch['images']
        instructions = batch['instructions']
        expert_actions = batch['actions']

        # Forward pass
        predicted_actions = self.model(images, instructions)

        # Compute loss
        loss = self.criterion(predicted_actions, expert_actions)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1

            if num_batches % 100 == 0:
                print(f"Batch {num_batches}, Loss: {loss:.4f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, val_dataloader):
        """Validate model performance."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                images = batch['images']
                instructions = batch['instructions']
                expert_actions = batch['actions']

                predicted_actions = self.model(images, instructions)
                loss = self.criterion(predicted_actions, expert_actions)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss
```

### Behavioral Cloning with Augmentation

```python
class AugmentedBehavioralCloningTrainer(ImitationLearningTrainer):
    def __init__(self, model, vla_transforms, learning_rate=1e-4):
        super().__init__(model, learning_rate)
        self.transforms = vla_transforms
        self.augmentation_prob = 0.5

    def train_step(self, batch):
        """Training step with data augmentation."""
        images = batch['images']
        instructions = batch['instructions']
        expert_actions = batch['actions']

        # Apply augmentations with some probability
        if random.random() < self.augmentation_prob:
            augmented_instructions = [
                self.transforms.augment_language(inst) for inst in instructions
            ]
        else:
            augmented_instructions = instructions

        # Forward pass
        predicted_actions = self.model(images, augmented_instructions)

        # Compute loss
        loss = self.criterion(predicted_actions, expert_actions)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()
```

### Reinforcement Learning Integration

```python
class RLIntegratedVLA(nn.Module):
    def __init__(self, base_vla_model, action_space_dim):
        super().__init__()
        self.base_model = base_vla_model

        # Policy network (for RL fine-tuning)
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_space_dim * 2)  # Mean and std for Gaussian policy
        )

        # Value network
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, images, instructions):
        # Get features from base VLA model
        fused_features = self.base_model.get_fused_features(images, instructions)

        # Policy output
        policy_output = self.policy_head(fused_features)
        action_dim = policy_output.size(-1) // 2
        action_mean = policy_output[:, :action_dim]
        action_std = torch.exp(policy_output[:, action_dim:])

        # Value output
        value = self.value_head(fused_features)

        return action_mean, action_std, value

    def get_fused_features(self, images, instructions):
        """Extract fused features for RL components."""
        # This would need to be implemented in the base model
        # For now, assume it returns the fused representation
        pass

class PPOVLA(ImitationLearningTrainer):
    def __init__(self, model, learning_rate=3e-4, clip_epsilon=0.2):
        super().__init__(model, learning_rate)
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = 4

    def ppo_update(self, old_model, batch, advantages, returns):
        """PPO update step."""
        images = batch['images']
        instructions = batch['instructions']
        old_actions = batch['actions']

        for _ in range(self.ppo_epochs):
            # Get current policy outputs
            action_mean, action_std, values = self.model(images, instructions)

            # Calculate log probabilities
            dist = torch.distributions.Normal(action_mean, action_std)
            log_probs = dist.log_prob(old_actions).sum(dim=-1)

            # Get old policy outputs
            with torch.no_grad():
                old_action_mean, old_action_std, _ = old_model(images, instructions)
                old_dist = torch.distributions.Normal(old_action_mean, old_action_std)
                old_log_probs = old_dist.log_prob(old_actions).sum(dim=-1)

            # Calculate ratios
            ratios = torch.exp(log_probs - old_log_probs)

            # PPO objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)

            # Total loss
            total_loss = policy_loss + 0.5 * value_loss

            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
```

## Curriculum Learning

### Progressive Training Strategy

```python
class CurriculumVLA:
    def __init__(self, base_model):
        self.model = base_model
        self.stages = [
            'basic_navigation',
            'simple_manipulation',
            'complex_tasks',
            'multi_step_reasoning'
        ]
        self.current_stage = 0

    def train_curriculum_stage(self, stage_data, num_epochs=50):
        """Train on current curriculum stage."""
        trainer = ImitationLearningTrainer(self.model)

        for epoch in range(num_epochs):
            avg_loss = trainer.train_epoch(stage_data)
            val_loss = trainer.validate(stage_data)  # Use validation subset

            print(f"Stage {self.current_stage} - Epoch {epoch}: Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Check if stage is mastered
            if self.is_stage_mastered(val_loss):
                self.advance_stage()
                break

    def is_stage_mastered(self, validation_loss):
        """Check if current stage is mastered."""
        # Define mastery criteria based on validation loss
        mastery_thresholds = [0.1, 0.08, 0.06, 0.05]  # Lower thresholds for later stages

        if self.current_stage < len(mastery_thresholds):
            return validation_loss < mastery_thresholds[self.current_stage]
        return True

    def advance_stage(self):
        """Advance to next curriculum stage."""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            print(f"Advanced to stage: {self.stages[self.current_stage]}")
        else:
            print("Curriculum completed!")
```

## Multi-Task Learning

### Joint Training Framework

```python
class MultiTaskVLATrainer:
    def __init__(self, model, task_weights=None):
        self.model = model
        self.task_weights = task_weights or {
            'navigation': 1.0,
            'manipulation': 1.0,
            'grasping': 1.0,
            'language_understanding': 0.5
        }

        # Task-specific loss functions
        self.task_losses = {
            'navigation': nn.MSELoss(),
            'manipulation': nn.MSELoss(),
            'grasping': nn.BCEWithLogitsLoss(),
            'language': nn.CrossEntropyLoss()
        }

    def compute_multitask_loss(self, batch):
        """Compute loss for multiple tasks."""
        images = batch['images']
        instructions = batch['instructions']

        # Get model outputs
        outputs = self.model.forward_multitask(images, instructions)

        total_loss = 0
        task_losses = {}

        for task_name, weight in self.task_weights.items():
            if task_name in outputs and f'{task_name}_targets' in batch:
                target = batch[f'{task_name}_targets']
                output = outputs[task_name]

                loss = self.task_losses[task_name](output, target)
                task_losses[task_name] = loss.item()
                total_loss += weight * loss

        return total_loss, task_losses

class MultiTaskVLA(nn.Module):
    def __init__(self, shared_encoder_dim=512):
        super().__init__()

        # Shared vision-language encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(1024, shared_encoder_dim),  # Combined vision-language input
            nn.ReLU(),
            nn.Linear(shared_encoder_dim, shared_encoder_dim)
        )

        # Task-specific heads
        self.navigation_head = nn.Linear(shared_encoder_dim, 3)  # x, y, theta
        self.manipulation_head = nn.Linear(shared_encoder_dim, 6)  # joint positions
        self.grasping_head = nn.Linear(shared_encoder_dim, 1)     # gripper
        self.language_head = nn.Linear(shared_encoder_dim, 100)  # language classification

    def forward_multitask(self, images, instructions):
        """Forward pass for multiple tasks."""
        # Encode inputs (this would involve the full VLA pipeline)
        # For simplicity, assume we have combined features
        combined_features = self.encode_multimodal(images, instructions)

        shared_features = self.shared_encoder(combined_features)

        return {
            'navigation': self.navigation_head(shared_features),
            'manipulation': self.manipulation_head(shared_features),
            'grasping': self.grasping_head(shared_features),
            'language': self.language_head(shared_features)
        }

    def encode_multimodal(self, images, instructions):
        """Encode multimodal inputs."""
        # This would use the full vision-language pipeline
        # Simplified implementation
        return torch.randn(images.size(0), 1024)  # Placeholder
```

## Training with Uncertainty

### Bayesian VLA Training

```python
class BayesianVLA(nn.Module):
    def __init__(self, base_model, num_samples=10):
        super().__init__()
        self.base_model = base_model
        self.num_samples = num_samples

        # Add dropout layers for uncertainty estimation
        self.dropout = nn.Dropout(0.1)

    def forward(self, images, instructions, uncertainty_mode=False):
        if uncertainty_mode:
            # Monte Carlo sampling for uncertainty estimation
            predictions = []
            for _ in range(self.num_samples):
                pred = self.base_model(images, instructions)
                pred = self.dropout(pred)  # Apply dropout during inference
                predictions.append(pred)

            predictions = torch.stack(predictions)
            mean_pred = predictions.mean(dim=0)
            std_pred = predictions.std(dim=0)

            return mean_pred, std_pred
        else:
            return self.base_model(images, instructions)

def train_bayesian_vla(model, dataloader, num_epochs=100):
    """Train Bayesian VLA model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            images = batch['images']
            instructions = batch['instructions']
            targets = batch['actions']

            # Get prediction with uncertainty
            mean_pred, std_pred = model(images, instructions, uncertainty_mode=True)

            # Negative log-likelihood loss with uncertainty
            nll_loss = 0.5 * torch.mean(torch.log(std_pred**2) + (targets - mean_pred)**2 / std_pred**2)

            optimizer.zero_grad()
            nll_loss.backward()
            optimizer.step()

            total_loss += nll_loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
```

## Distributed Training

### Multi-GPU Training Setup

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class DistributedVLATrainer:
    def __init__(self, model, rank, world_size):
        self.rank = rank
        self.world_size = world_size

        # Initialize distributed training
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        # Move model to GPU and wrap with DDP
        torch.cuda.set_device(rank)
        self.model = model.cuda(rank)
        self.model = DDP(self.model, device_ids=[rank])

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def train_distributed(self, train_dataset, batch_size=32, epochs=10):
        """Train model with distributed data parallelism."""
        # Create distributed sampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=self.world_size, rank=self.rank
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=vla_collate_fn
        )

        for epoch in range(epochs):
            train_sampler.set_epoch(epoch)  # Shuffle data differently each epoch

            total_loss = 0
            num_batches = 0

            for batch in train_loader:
                # Move batch to GPU
                batch['images'] = batch['images'].cuda(self.rank, non_blocking=True)
                batch['actions'] = batch['actions'].cuda(self.rank, non_blocking=True)

                # Forward pass
                outputs = self.model(batch['images'], batch['instructions'])
                loss = nn.MSELoss()(outputs, batch['actions'])

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # Average loss across all processes
            avg_loss = total_loss / num_batches
            print(f"Rank {self.rank}, Epoch {epoch}, Avg Loss: {avg_loss:.4f}")

    def cleanup(self):
        """Clean up distributed training."""
        dist.destroy_process_group()
```

## Training Monitoring and Evaluation

### Training Progress Tracking

```python
import wandb
import matplotlib.pyplot as plt

class VLAExperimentTracker:
    def __init__(self, experiment_name, config):
        self.experiment_name = experiment_name
        self.config = config

        # Initialize Weights & Biases
        wandb.init(
            project="vla-training",
            name=experiment_name,
            config=config
        )

        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'action_accuracy': [],
            'language_alignment': [],
            'training_time': []
        }

    def log_training_step(self, step, train_loss, val_loss=None):
        """Log training metrics."""
        metrics = {
            'step': step,
            'train_loss': train_loss
        }

        if val_loss is not None:
            metrics['val_loss'] = val_loss

        wandb.log(metrics)

        # Store locally
        self.metrics['train_loss'].append(train_loss)
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)

    def log_evaluation(self, step, eval_metrics):
        """Log evaluation metrics."""
        wandb.log({
            'step': step,
            **eval_metrics
        })

        for key, value in eval_metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def plot_training_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].plot(self.metrics['train_loss'], label='Train Loss')
        if 'val_loss' in self.metrics:
            axes[0, 0].plot(self.metrics['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()

        if 'action_accuracy' in self.metrics:
            axes[0, 1].plot(self.metrics['action_accuracy'])
            axes[0, 1].set_title('Action Accuracy')

        if 'language_alignment' in self.metrics:
            axes[1, 0].plot(self.metrics['language_alignment'])
            axes[1, 0].set_title('Language Alignment')

        if 'training_time' in self.metrics:
            axes[1, 1].plot(self.metrics['training_time'])
            axes[1, 1].set_title('Training Time')

        plt.tight_layout()
        wandb.log({"training_curves": wandb.Image(fig)})
        plt.close(fig)

def train_with_monitoring(model, train_loader, val_loader, num_epochs=100):
    """Train VLA model with comprehensive monitoring."""
    trainer = ImitationLearningTrainer(model)
    tracker = VLAExperimentTracker("vla_experiment", {
        'model_type': 'transformer_vla',
        'learning_rate': 1e-4,
        'batch_size': 32,
        'epochs': num_epochs
    })

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(num_epochs):
        # Training
        train_loss = trainer.train_epoch(train_loader)

        # Validation
        val_loss = trainer.validate(val_loader)

        # Log metrics
        tracker.log_training_step(epoch, train_loss, val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_vla_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    tracker.plot_training_curves()
    wandb.finish()
```

## Transfer Learning and Domain Adaptation

### Domain Adaptation for VLA

```python
class DomainAdaptiveVLA(nn.Module):
    def __init__(self, source_model, num_domains=3):
        super().__init__()
        self.source_model = source_model
        self.num_domains = num_domains

        # Domain-specific adaptation layers
        self.domain_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 512)
            ) for _ in range(num_domains)
        ])

        # Domain classifier for adversarial training
        self.domain_classifier = nn.Linear(512, num_domains)

    def forward(self, images, instructions, domain_id=None):
        # Get features from source model
        fused_features = self.source_model.get_fused_features(images, instructions)

        if domain_id is not None:
            # Apply domain-specific adaptation
            adapted_features = self.domain_adapters[domain_id](fused_features)
        else:
            # Use all domain adapters (for inference)
            adapted_features = fused_features
            for adapter in self.domain_adapters:
                adapted_features = adapted_features + adapter(fused_features)
            adapted_features = adapted_features / len(self.domain_adapters)

        # Predict action
        action = self.source_model.action_decoder(adapted_features)

        # Domain classification (for adversarial training)
        domain_logits = self.domain_classifier(fused_features)

        return action, domain_logits

def train_domain_adaptive_vla(model, source_loader, target_loader, num_epochs=50):
    """Train domain adaptive VLA with adversarial loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    action_criterion = nn.MSELoss()
    domain_criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for (source_batch, target_batch) in zip(source_loader, target_loader):
            # Source domain training
            source_actions = model(source_batch['images'],
                                 source_batch['instructions'],
                                 domain_id=0)[0]
            source_action_loss = action_criterion(source_actions, source_batch['actions'])

            # Target domain (no action labels, only domain adaptation)
            _, source_domain_logits = model(source_batch['images'],
                                          source_batch['instructions'],
                                          domain_id=None)
            _, target_domain_logits = model(target_batch['images'],
                                          target_batch['instructions'],
                                          domain_id=None)

            # Domain classification loss (should be hard to classify domain)
            source_domain_labels = torch.zeros(source_batch['images'].size(0)).long()
            target_domain_labels = torch.ones(target_batch['images'].size(0)).long()

            domain_loss = domain_criterion(
                torch.cat([source_domain_logits, target_domain_logits]),
                torch.cat([source_domain_labels, target_domain_labels])
            )

            # Total loss (minimize domain classification, maximize source performance)
            total_loss = source_action_loss - domain_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

## Troubleshooting Training Issues

### Common Training Problems and Solutions

```python
class VLATrainingDiagnostics:
    def __init__(self):
        self.diagnostics = {
            'gradient_flow': [],
            'loss_patterns': [],
            'overfitting_indicators': [],
            'convergence_metrics': []
        }

    def check_gradient_flow(self, model):
        """Check for gradient flow issues."""
        total_norm = 0
        param_count = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        total_norm = total_norm ** (1. / 2)

        if total_norm < 1e-6:
            print("Warning: Very small gradients (possible vanishing gradients)")
        elif total_norm > 10:
            print("Warning: Very large gradients (possible exploding gradients)")

        return total_norm

    def detect_overfitting(self, train_loss, val_loss):
        """Detect overfitting patterns."""
        if len(val_loss) > 10:
            # Check if validation loss is increasing while training loss decreases
            if val_loss[-1] > val_loss[-2] and train_loss[-1] < train_loss[-2]:
                return True
        return False

    def analyze_loss_patterns(self, loss_history):
        """Analyze training loss patterns."""
        if len(loss_history) < 5:
            return "Insufficient data"

        # Check for oscillation
        diffs = [abs(loss_history[i] - loss_history[i-1]) for i in range(1, len(loss_history))]
        avg_diff = sum(diffs) / len(diffs)

        if avg_diff > 0.1:  # High oscillation threshold
            return "High oscillation - consider reducing learning rate"

        # Check for stagnation
        recent_loss = loss_history[-5:]
        if max(recent_loss) - min(recent_loss) < 0.001:
            return "Stagnation detected - consider learning rate adjustment or architecture changes"

        return "Normal training pattern"
```

---
[Next: VLA Integration](./vla-integration.md) | [Previous: VLA Architecture](./vla-architecture.md)