import cv2
import numpy as np
import os
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datasets.datasetwrapper import DatasetWrapper

AUGMENT_TIMES = 2
batch_size = 16
radius = 8  # LBP radius
n_points = 3 * radius  # Number of points to consider in LBP
num_bins = 512

def extract_lbp_features(image, depth_image):
    """Extracts LBP features from both grayscale and depth images with 512-bin histograms."""
    
    # Convert tensors to numpy arrays and remove extra dimensions
    image = image.cpu().numpy().squeeze()
    depth_image = depth_image.cpu().numpy().squeeze()
    
    # Convert (C, H, W) -> (H, W) if needed
    if image.ndim == 3:
        image = image[0]
    if depth_image.ndim == 3:
        depth_image = depth_image[0]
    
    # Compute LBP
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    depth_lbp = local_binary_pattern(depth_image, n_points, radius, method='uniform')
    
    # Compute histograms with 512 bins
    hist_color, _ = np.histogram(lbp.ravel(), bins=num_bins, range=(0, num_bins), density=True)
    hist_depth, _ = np.histogram(depth_lbp.ravel(), bins=num_bins, range=(0, num_bins), density=True)
    
    # Concatenate histograms
    return np.concatenate([hist_color, hist_depth])

# Load datasets
dataset_wrapper_train = DatasetWrapper(root_dir="/mnt/extravolume/data/iPhone11_filled/color/digital")
dataset_wrapper_test = DatasetWrapper(root_dir="/mnt/extravolume/data/iPhone12_filled/color/digital")

# Combine train and test splits for iPhone11 as training data
train_dataset = dataset_wrapper_train.get_train_dataset(augment_times=AUGMENT_TIMES, batch_size=batch_size, morph_types=["lmaubo"], num_models=1, shuffle=True)
train_dataset2 = dataset_wrapper_train.get_test_dataset(augment_times=AUGMENT_TIMES, batch_size=batch_size, morph_types=["lmaubo"], num_models=1, shuffle=True)

# Combine train and test splits for iPhone12 as testing data
test_dataset = dataset_wrapper_test.get_train_dataset(augment_times=AUGMENT_TIMES, batch_size=batch_size, morph_types=["lmaubo"], num_models=1, shuffle=True)
test_dataset2 = dataset_wrapper_test.get_test_dataset(augment_times=AUGMENT_TIMES, batch_size=batch_size, morph_types=["lmaubo"], num_models=1, shuffle=True)


features, labels = [], []
for color_imgs, depth_imgs, label in tqdm(train_dataset, desc="Processing Training Data"):
    label_class = label.argmax(dim=1).cpu().numpy()
    for img, depth_img, lbl in zip(color_imgs, depth_imgs, label_class):
        features.append(extract_lbp_features(img, depth_img))
        labels.append(lbl)
for color_imgs, depth_imgs, label in tqdm(train_dataset2, desc="Processing Training Data"):
    label_class = label.argmax(dim=1).cpu().numpy()
    for img, depth_img, lbl in zip(color_imgs, depth_imgs, label_class):
        features.append(extract_lbp_features(img, depth_img))
        labels.append(lbl)

features_np = np.array(features)
labels_np = np.array(labels)

# Check class distribution in training data
print("Training class distribution:", np.bincount(labels_np))


# Process test data
test_features, test_labels = [], []
for color_imgs, depth_imgs, label in tqdm(test_dataset, desc="Processing Test Data"):
    label_class = label.argmax(dim=1).cpu().numpy()
    for img, depth_img, lbl in zip(color_imgs, depth_imgs, label_class):
        test_features.append(extract_lbp_features(img, depth_img))
        test_labels.append(lbl)
for color_imgs, depth_imgs, label in tqdm(test_dataset2, desc="Processing Test Data"):
    label_class = label.argmax(dim=1).cpu().numpy()
    for img, depth_img, lbl in zip(color_imgs, depth_imgs, label_class):
        test_features.append(extract_lbp_features(img, depth_img))
        test_labels.append(lbl)

test_features_np = np.array(test_features)
test_labels_np = np.array(test_labels)

# Check class distribution in test data
print("Test class distribution:", np.bincount(test_labels_np))


# Normalize features
scaler = StandardScaler()
features_np = scaler.fit_transform(features_np)
test_features_np = scaler.transform(test_features_np)

# Train SVM
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, class_weight='balanced')
# svm = SVC(probability=True, gamma='scale')
# svm = SVC(probability=True, gamma='auto')
svm.fit(features_np, labels_np)

# Evaluate model
preds = svm.predict(test_features_np)
probs = svm.predict_proba(test_features_np)
accuracy = accuracy_score(test_labels_np, preds)
print("LBP + SVM Accuracy:", accuracy)

# Save genuine and imposter scores
genuine_scores, imposter_scores = [], []
genuine_path, imposter_path = "scores/lbp_svm/iPhone12_filled/lmaubo/genuine.npy", "scores/lbp_svm/iPhone12_filled/lmaubo/imposter.npy"
os.makedirs("scores/lbp_svm/iPhone12_filled/lmaubo/", exist_ok=True)

for prob, label in zip(probs, test_labels_np):
    if label == 1:
        genuine_scores.append(prob[0])
    else:
        imposter_scores.append(prob[0])

np.save(genuine_path, np.array(genuine_scores))
np.save(imposter_path, np.array(imposter_scores))
print("Scores saved successfully!")

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.hist(features_np.ravel(), bins=50, alpha=0.75, color="blue", label="Train Features")
plt.hist(test_features_np.ravel(), bins=50, alpha=0.75, color="red", label="Test Features")
plt.legend()
plt.title("Feature Distribution of LBP Histograms")
plt.savefig(f"lbp_feature_distribution_{num_bins}.png", dpi=300, bbox_inches="tight")
plt.close()

import seaborn as sns

class_0 = np.mean(features_np[labels_np == 0], axis=0)
class_1 = np.mean(features_np[labels_np == 1], axis=0)

plt.figure(figsize=(12, 5))
sns.lineplot(x=range(len(class_0)), y=class_0, label="Class 0 (Bonafide)", color='blue')
sns.lineplot(x=range(len(class_1)), y=class_1, label="Class 1 (Morph)", color='red')
plt.title("Average LBP Histograms per Class")
plt.savefig(f"lbp_class_histograms_{num_bins}.png", dpi=300, bbox_inches="tight")
plt.close()

from sklearn.decomposition import PCA
import seaborn as sns

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features_np)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=labels_np, palette="coolwarm")
plt.title("PCA Projection of LBP Features")
plt.savefig(f"pca_lbp_features_{num_bins}.png", dpi=300, bbox_inches="tight")
plt.close()


# LBP + SVM Accuracy: 0.7202195285663521