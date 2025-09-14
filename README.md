# 3D-MAD: Multi-modal set up for 3D Face Morphing Attack Detection
Face recognition systems (FRS) are increasingly deployed in high-security applications such as border control and financial services, yet they remain vulnerable to morphing attacks. These attacks, especially in 3D, pose a growing threat as 3D imaging becomes more accessible. This repository provides the implementation of a novel framework for detecting 3D face morphing attacks using both RGB and depth images. It integrates a Cross-Modal Attention (CMA) mechanism to align and refine features across modalities, an Adaptive Fusion module for effective feature integration, and a multi-scale architecture to capture both local and global morphing artifacts from RGB and Depth images.

## Repository Structure
The code is organized into a clean and modular structure to promote reusability and clarity.

main.py: The primary script for running the training and evaluation pipelines.

models/: Contains custom model definitions, including the feature extractors, attention modules, and the main network architecture.

dataset/: Holds the data loading scripts and custom PyTorch Dataset and DataLoader classes, designed to handle multi-modal inputs.

configs/: Configuration files for training parameters, model architecture, and other settings.

utils/: Utility functions for data preprocessing, visualization, and other helper tasks.

train.py: Contains the main training loop and validation logic.

logs/: Stores training logs, TensorBoard files, and other output from training runs.

scores/: Saves the evaluation metrics and results on the test set.

## Training & Testing
To train the model, simply run the main.py script with your desired configurations in config.py.

Modify the configuration file to adjust hyperparameters, model settings, and dataset paths.

### Protocols
The project supports various training and testing protocols to evaluate the model's performance under different conditions, specifically regarding the data source (iPhone 11 vs. iPhone 12).

Protocol 0: Training and testing are performed only on a single, unified dataset.

Protocol 1: Training on the iPhone 11 dataset, and testing on the iPhone 11 dataset.

Protocol 2: Training on the iPhone 12 dataset, and testing on the iPhone 12 dataset.

Protocol 3: Training on the iPhone 11 dataset, and testing on the iPhone 12 dataset (cross-device evaluation).

Protocol 4: Training on the iPhone 12 dataset, and testing on the iPhone 11 dataset (cross-device evaluation).

