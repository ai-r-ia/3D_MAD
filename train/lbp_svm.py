from tqdm import tqdm
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datasets.datasetwrapper import DatasetWrapper
from sklearn.pipeline import make_pipeline
import torch

AUGMENT_TIMES = 2
batch_size = 128
# LBP Parameters
radius = 1
n_points = 8 * radius

def extract_lbp_features(image):
    # If image is a tensor, convert it to NumPy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()  # Move to CPU & convert to NumPy
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # If (C, H, W)
            image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C)

    # Ensure image is uint8
    image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)

    # Check if image is already grayscale
    if image.ndim == 3 and image.shape[2] == 3:  # (H, W, 3) format
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:  # Already grayscale (H, W)
        gray = image
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    # Apply LBP (Example using OpenCV's Laplacian as a placeholder)
    lbp = cv2.Laplacian(gray, cv2.CV_64F).flatten()

    return lbp

dataset_wrapper = DatasetWrapper(root_dir="/mnt/extravolume/data/iPhone11/color/digital/")

train_dataset = dataset_wrapper.get_train_dataset(
    augment_times=AUGMENT_TIMES,
    batch_size=batch_size,
    morph_types=["lmaubo"],
    num_models=1,
    shuffle=True,
)
train_dataset2 = dataset_wrapper.get_test_dataset(
    augment_times=AUGMENT_TIMES,
    batch_size=batch_size,
    morph_types=["lmaubo"],
    num_models=1,
    shuffle=True,
)

# Initialize storage for features and labels
color_features = []
depth_features = []
labels = []

# Iterate over dataloader
for color_imgs, depth_imgs, label in tqdm(train_dataset, desc="Extracting LBP features"):
    for i in range(color_imgs.shape[0]):
     
        color_img = color_imgs[i].to("cpu")  # Move to CPU if on GPU
        depth_img = depth_imgs[i].to("cpu")  # Move to CPU if on GPU
        label_class = label[i].argmax(dim=0).item()        
        color_feat = extract_lbp_features(color_img)
        depth_feat = extract_lbp_features(depth_img)
        
        color_features.append(color_feat)
        depth_features.append(depth_feat)
        labels.append(label_class)

for color_imgs, depth_imgs, label in tqdm(train_dataset2, desc="Extracting LBP features"):
    for i in range(color_imgs.shape[0]):
     
        color_img = color_imgs[i].to("cpu")  # Move to CPU if on GPU
        depth_img = depth_imgs[i].to("cpu")  # Move to CPU if on GPU
        label_class = label[i].argmax(dim=0).item()        
        color_feat = extract_lbp_features(color_img)
        depth_feat = extract_lbp_features(depth_img)
        
        color_features.append(color_feat)
        depth_features.append(depth_feat)
        labels.append(label_class)
        
# Convert to numpy arrays
color_features = np.array(color_features)
depth_features = np.array(depth_features)
labels = np.array(labels)
import time

# Train SVM on color image LBP features
print("Training color SVM...")
start_time = time.time()
# color_svm = make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True))
color_svm = make_pipeline(StandardScaler(), LinearSVC())
color_svm.fit(color_features, labels)
end_time = time.time()
print(f"Color SVM training completed in {end_time - start_time:.2f} seconds")

# Train SVM on depth image LBP features
print("Training depth SVM...")
start_time = time.time()
# depth_svm = make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True))
depth_svm = make_pipeline(StandardScaler(), LinearSVC())
depth_svm.fit(depth_features, labels)
end_time = time.time()
print(f"Depth SVM training completed in {end_time - start_time:.2f} seconds")


dataset_wrapper2 = DatasetWrapper(root_dir="/mnt/extravolume/data/iPhone12/color/digital/")

test_dataset = dataset_wrapper2.get_train_dataset(
    augment_times=AUGMENT_TIMES,
    batch_size=batch_size,
    morph_types=["lmaubo"],
    num_models=1,
    shuffle=True,
)
test_dataset2 = dataset_wrapper2.get_test_dataset(
    augment_times=AUGMENT_TIMES,
    batch_size=batch_size,
    morph_types=["lmaubo"],
    num_models=1,
    shuffle=True,
)

# Store scores for evaluation
final_predictions = []
true_labels = []

# Iterate over test data
for color_imgs, depth_imgs, label in tqdm(test_dataset, desc="Evaluating LBP + SVM"):
    for i in range(color_imgs.shape[0]):
        color_img = color_imgs[i].to("cpu")  # Move to CPU if on GPU
        depth_img = depth_imgs[i].to("cpu")  # Move to CPU if on GPU
        label_class = label[i].argmax(dim=0).item()            
        # Extract LBP features
        color_feat = extract_lbp_features(color_img).reshape(1, -1)
        depth_feat = extract_lbp_features(depth_img).reshape(1, -1)
        
        # Get SVM probabilities
        color_prob = color_svm.predict_proba(color_feat)[0, 1]
        depth_prob = depth_svm.predict_proba(depth_feat)[0, 1]
        
        # Combine scores using average
        final_score = (color_prob + depth_prob) / 2
        final_pred = 1 if final_score > 0.5 else 0
        
        final_predictions.append(final_pred)
        true_labels.append(label_class.item())

for color_imgs, depth_imgs, label in tqdm(test_dataset2, desc="Evaluating LBP + SVM"):
    for i in range(color_imgs.shape[0]):
        color_img = color_imgs[i].to("cpu")  # Move to CPU if on GPU
        depth_img = depth_imgs[i].to("cpu")  # Move to CPU if on GPU
        label_class = label[i].argmax(dim=0).item()            
        # Extract LBP features
        color_feat = extract_lbp_features(color_img).reshape(1, -1)
        depth_feat = extract_lbp_features(depth_img).reshape(1, -1)
        
        # Get SVM probabilities
        color_prob = color_svm.predict_proba(color_feat)[0, 1]
        depth_prob = depth_svm.predict_proba(depth_feat)[0, 1]
        
        # Combine scores using average
        final_score = (color_prob + depth_prob) / 2
        final_pred = 1 if final_score > 0.5 else 0
        
        final_predictions.append(final_pred)
        true_labels.append(label_class.item())

# Calculate accuracy
accuracy = np.mean(np.array(final_predictions) == np.array(true_labels))
print(f"Final Accuracy: {accuracy:.4f}")
