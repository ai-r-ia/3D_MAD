from point_clouds.pc_data import get_pc
from point_clouds.pointnet_classifier import PointNetClassifier
from point_clouds.simpleview import MVModel
import torch
import os
from tqdm import tqdm
import torch.nn as nn
from pointnet2_cls import PointNet2, PointNet2FeatureExtractor  # Assuming PointNet2 is in the 'pointnet2' module
from pointnet import PointNet  # Assuming PointNet2 is in the 'pointnet2' module
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter

from datasets.datasetwrapper import DatasetWrapper

AUGMENT_TIMES = 2
NUM_POINTS = 50000
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Pretrained PointNet2 for color images
# pointnet2 = PointNet(num_classes=2)  
# ver = "1"
# pointnet2_pretrained = PointNetClassifier(num_points=NUM_POINTS)  # Initialize PointNet2 model
# checkpoint = torch.load('point_clouds/pointnet.pth')
# # print(pointnet2_pretrained)
# pointnet2_pretrained.load_state_dict(checkpoint, strict=False)
# pointnet2 = pointnet2_pretrained.to(device)
# pointnet2.eval()
# ver = "2"
# pointnet2_pretrained = PointNet2(num_class=40)  # Initialize PointNet2 model
# checkpoint = torch.load('point_clouds/pointnet2.pth')
# pointnet2_pretrained.load_state_dict(checkpoint['model_state_dict'], strict=False)
# # print(pointnet2_pretrained)
# pointnet2 = PointNet2FeatureExtractor(pointnet2_pretrained)
# pointnet2 = pointnet2.to(device)
# pointnet2.eval()
# ver = "2_simpleview"
ver = "dgcnn_simpleview"
pointnet2_pretrained = MVModel()  # Initialize PointNet2 model
# checkpoint = torch.load('pretrained/pointnet2_simpleview_run_1/model_625.pth')
checkpoint = torch.load('pretrained/dgcnn_simpleview_run_1/model_650.pth')
print(checkpoint.keys())
pointnet2_pretrained.load_state_dict(checkpoint['model_state'], strict=False)
# print(pointnet2_pretrained)
pointnet2 = pointnet2_pretrained.to(device)
pointnet2.eval()

def sample_points(point_cloud, num_points=NUM_POINTS):
    """Ensures each point cloud has exactly `num_points` by randomly sampling"""
    B, C, N = point_cloud.shape  # B=batch size, C=channels, N=num points
    if N > num_points:
        indices = torch.randperm(N)[:num_points]  # Randomly select `num_points`
        return point_cloud[:, :, indices]  # Return subsampled points
    elif N < num_points:
        # If too few points, duplicate randomly chosen points
        pad_indices = torch.randint(0, N, (num_points - N,), device=point_cloud.device)
        pad_points = point_cloud[:, :, pad_indices]
        return torch.cat([point_cloud, pad_points], dim=-1)  # (B, C, num_points)
    return point_cloud 

def normalize_point_cloud(pc):
    centroid = pc.mean(dim=-1, keepdim=True)  # Compute centroid
    pc = pc - centroid  # Center the point cloud
    scale = torch.max(torch.norm(pc, dim=1, keepdim=True))  # Compute max distance
    pc = pc / scale  # Scale to unit sphere
    return pc


dataset_wrapper = DatasetWrapper(root_dir="/mnt/extravolume/data/iPhone11_filled/color/digital/")

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

features = []
labels = []

for color_imgs, depth_imgs, label in tqdm(train_dataset, desc="training first trainds"):
    color_imgs = color_imgs.to(device)
    depth_imgs = depth_imgs.to(device)
    label_class = label.argmax(dim=1).detach().to(device)

    for i in range(color_imgs.size(0)):
        color_img = color_imgs[i]
        depth_img = depth_imgs[i]
        
        pointcloud = get_pc(color_img, depth_img)
        points = np.asarray(pointcloud.points)  # (num_points, 3)
        points = torch.from_numpy(points).float().cuda()  
        points = points.T.unsqueeze(0).to(device)
        points = sample_points(points, num_points=NUM_POINTS)  
        points = normalize_point_cloud(points)
        points = points.permute(0, 2, 1)  # for simpleview
        # print(f"points shape: {points.shape}")

        with torch.no_grad():
            preds, pc_features = pointnet2(points) 
        # print(f"pc_features shape: {pc_features.shape}")
        features.append(pc_features.squeeze(0).view(-1)) 
        labels.append(label_class[i].unsqueeze(0))
        # features.append(pc_features.squeeze(0))
        # print(f"features shape: {pc_features.squeeze(0).view(-1).shape}")

for color_imgs, depth_imgs, label in tqdm(train_dataset2, desc= "training second trainds"):
    color_imgs = color_imgs.to(device)
    depth_imgs = depth_imgs.to(device)
    label_class = label.argmax(dim=1).detach().to(device)

    for i in range(color_imgs.size(0)):
        color_img = color_imgs[i]
        depth_img = depth_imgs[i]
        
        pointcloud = get_pc(color_img, depth_img)
        points = np.asarray(pointcloud.points)  # (num_points, 3)
        points = torch.from_numpy(points).float().cuda()  
        points = points.T.unsqueeze(0).to(device)
        points = sample_points(points, num_points=NUM_POINTS)  
        points = normalize_point_cloud(points)
        points = points.permute(0, 2, 1)  # for simpleview
        with torch.no_grad():
            preds, pc_features = pointnet2(points) 
            
        features.append(pc_features.squeeze(0).view(-1)) 
        labels.append(label_class[i].unsqueeze(0))
label_counts = Counter(label.item() for label in labels)

print("Class Distribution:",label_counts)

# print("Class Distribution:", Counter(labels))
print(f"Features length: {len(features)}")
print(f"Labels length: {len(labels)}")
concatenated_features = torch.stack(features, dim=0)  # Stack features along a new dimension
concatenated_labels = torch.stack(labels, dim=0)      # Stack labels along a new dimension
concatenated_labels = concatenated_labels.squeeze()
# concatenated_features = torch.cat(features, dim=0)
# concatenated_labels = torch.cat(labels, dim=0)
features_np = concatenated_features.cpu().numpy()
labels_np = concatenated_labels.cpu().numpy()
print(features_np.shape)  
print(labels_np.shape)    


dataset_wrapper2 = DatasetWrapper(root_dir="/mnt/extravolume/data/iPhone12_filled/color/digital/")

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

test_features = []
test_labels = []
labels_one_hot = []

for color_imgs, depth_imgs, label in tqdm(test_dataset, desc= "first testds"):
    color_imgs = color_imgs.to(device)
    depth_imgs = depth_imgs.to(device)
    label_class = label.argmax(dim=1).detach().to(device)

    for i in range(color_imgs.size(0)):
        color_img = color_imgs[i]
        depth_img = depth_imgs[i]
        
        pointcloud = get_pc(color_img, depth_img)
        points = np.asarray(pointcloud.points)  # (num_points, 3)
        points = torch.from_numpy(points).float().cuda()  
        points = points.T.unsqueeze(0).to(device)
        points = sample_points(points, num_points=NUM_POINTS)  
        points = normalize_point_cloud(points)
        points = points.permute(0, 2, 1)  # for simpleview
        with torch.no_grad():
            preds, pc_features = pointnet2(points) 
             
        test_features.append(pc_features.squeeze(0).view(-1))
        test_labels.append(label_class[i].unsqueeze(0)) 
    labels_one_hot.append(label) 

for color_imgs, depth_imgs, label in tqdm(test_dataset2, desc= "second testds"):
    color_imgs = color_imgs.to(device)
    depth_imgs = depth_imgs.to(device)
    label_class = label.argmax(dim=1).detach().to(device)

    for i in range(color_imgs.size(0)):
        color_img = color_imgs[i]
        depth_img = depth_imgs[i]
        
        pointcloud = get_pc(color_img, depth_img)
        points = np.asarray(pointcloud.points)  # (num_points, 3)
        points = torch.from_numpy(points).float().cuda()  
        points = points.T.unsqueeze(0).to(device)
        points = sample_points(points, num_points=NUM_POINTS)  
        points = normalize_point_cloud(points)
        points = points.permute(0, 2, 1)  # for simpleview
        with torch.no_grad():
            preds, pc_features = pointnet2(points) 
            
        test_features.append(pc_features.squeeze(0).view(-1))
        test_labels.append(label_class[i].unsqueeze(0)) 
    labels_one_hot.append(label)

print("lbl one hot: ",len(labels_one_hot))
concatenated_features_test = torch.stack(test_features, dim=0)
concatenated_labels_test = torch.stack(test_labels, dim=0)
concatenated_labels_test = concatenated_labels_test.squeeze()
conat_one_hot =torch.cat(labels_one_hot, dim=0) 
test_features_np = concatenated_features_test.cpu().numpy()
test_labels_np = concatenated_labels_test.cpu().numpy()
labels_one_hot = conat_one_hot.cpu().numpy()
print("lbl one hot: ",labels_one_hot.shape)

scaler = StandardScaler()
features_np, X_test = scaler.fit_transform(features_np), scaler.transform(test_features_np)

svm = SVC(probability=True, gamma='auto',class_weight='balanced')
svm.fit(features_np, labels_np)
# SVC(gamma='auto') 

# svm = LinearSVC()
# svm.fit(features_np, labels_np)

preds = svm.predict(X_test)
print("preds: ",preds.shape)
probs = svm.predict_proba(X_test)


accuracy = accuracy_score(test_labels_np, preds)
print(f"{ver} + SVM Accuracy:", accuracy)
print("Calculating scores")

genuine_scores, imposter_scores = [], []
genuine_path, imposter_path = f"scores/{ver}_svm/iPhone12_filled/lmaubo/genuine.npy", f"scores/{ver}_svm/iPhone12_filled/lmaubo/imposter.npy" 
os.makedirs(f"scores/{ver}_svm/iPhone12_filled/lmaubo/", exist_ok = True)

for pred, label in zip(probs, labels_one_hot):
    label = label[0]
    if label == 1:
        genuine_scores.append(pred[0])
    else:
        imposter_scores.append(pred[0])

print("saving scores")
genuine_scores = [torch.tensor(score) for score in genuine_scores]
imposter_scores = [torch.tensor(score) for score in imposter_scores]
np.save(genuine_path, torch.stack(genuine_scores).numpy())
np.save(imposter_path, torch.stack(imposter_scores).numpy())

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce to 2D for visualization
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_np)
test_features_pca = pca.transform(test_features_np)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels_np, cmap='viridis', alpha=0.5, label='Train Data')
plt.scatter(test_features_pca[:, 0], test_features_pca[:, 1], c=test_labels_np, cmap='coolwarm', marker='x', label='Test Data')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Visualization of PointNet Features")
plt.legend()
plt.colorbar()
plt.savefig(f"pca_ftrs_{ver}", dpi=300, bbox_inches="tight")
plt.close()


#PointNet2 + SVM Accuracy: <calculated accuracy>
# https://github.com/meder411/PointNet-PyTorch/blob/master/models/pointnet_classifier.py