---
name: cv
description: This skill should be used when extracting structured information from computer vision papers. Trigger when the user provides a CV paper or asks about model architecture, backbone networks, image datasets, CV tasks (classification, detection, segmentation), metrics like mAP/accuracy/FPS, data augmentation, or training strategies.
---

# CV Paper Extraction

## Fields to extract

- **model_architecture** (required): The neural network architecture (e.g., ResNet, ViT, U-Net)
  - Examples: "Prototypical Network with Conv-4 backbone", "Mask R-CNN with ResNet-50 FPN"
- **backbone**: The backbone network used for feature extraction
- **task** (required): The CV task (classification, detection, segmentation, etc.)
- **datasets** (required): Datasets used for training and evaluation
- **metrics** (required): Key reported metrics (accuracy, mAP, FPS, etc.)
- **training_strategy**: Training approach (episodic, transfer learning, meta-learning, etc.)
- **data_augmentation**: Data augmentation techniques used
- **input_size**: Input image resolution (e.g., 224x224, 84x84)

## Quality checklist for review

- Is the benchmark standard for this task?
- Are the compared baselines fair and recent?
- Is the improvement statistically significant?
- Is the code publicly available?
