import torch
import os
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import numpy as np

from datasets.datasetwrapper import DatasetWrapper

AUGMENT_TIMES = 2
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Pretrained ResNet50 for color images
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
resnet = resnet.to(device)
# Dummy function to process depth images (use your own model or feature extractor for depth)
def extract_depth_features(depth_image):
    """Extract features from depth images. This is a placeholder."""
    # For depth images, you could use a simple CNN or a custom feature extractor.
    # Here, we're just returning a dummy vector. Replace this with actual processing.
    return np.random.rand(2048)  # Dummy feature vector (size 2048)

def extract_resnet_features(model, image):
    """Extract deep features from ResNet50"""
    model.eval()
    with torch.no_grad():
        features = model(image).squeeze()  # Remove unnecessary dimensions
    return features

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

for color_imgs, depth_imgs, label in tqdm(train_dataset, desc= "training first trainds"):
    color_imgs = color_imgs.to(device)
    depth_imgs = depth_imgs.to(device)
    label_class = label.argmax(dim=1).detach().to(device)
    
    with torch.no_grad():
        color_features_batch = extract_resnet_features(resnet, color_imgs)

    depth_features_batch =  extract_resnet_features(resnet, depth_imgs)

    combined_features = torch.cat((color_features_batch, depth_features_batch), dim = 1)  

    features.append(combined_features)
    labels.append(label_class)

for color_imgs, depth_imgs, label in tqdm(train_dataset2, desc= "training second trainds"):
    color_imgs = color_imgs.to(device)
    depth_imgs = depth_imgs.to(device)
    label_class = label.argmax(dim=1).detach().to(device)
    
    with torch.no_grad():
        color_features_batch = extract_resnet_features(resnet, color_imgs)

    depth_features_batch =  extract_resnet_features(resnet, depth_imgs)

    combined_features = torch.cat((color_features_batch, depth_features_batch), dim = 1)  

    features.append(combined_features)
    labels.append(label_class)

concatenated_features = torch.cat(features, dim=0)
concatenated_labels = torch.cat(labels, dim=0)
features_np = concatenated_features.cpu().numpy()
labels_np = concatenated_labels.cpu().numpy()

print(features_np.shape)  # Should be (batch_size, 2*feature_dim)
print(labels_np.shape)        # Should be (batch_size,)


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
    
    with torch.no_grad():
        color_features_batch = extract_resnet_features(resnet, color_imgs)

    depth_features_batch =  extract_resnet_features(resnet, depth_imgs)

    combined_features = torch.cat((color_features_batch, depth_features_batch), dim = 1)  

    test_features.append(combined_features)
    test_labels.append(label_class)
    labels_one_hot.append(label)

for color_imgs, depth_imgs, label in tqdm(test_dataset2, desc= "second testds"):
    color_imgs = color_imgs.to(device)
    depth_imgs = depth_imgs.to(device)
    label_class = label.argmax(dim=1).detach().to(device)
    
    with torch.no_grad():
        color_features_batch = extract_resnet_features(resnet, color_imgs)

    depth_features_batch =  extract_resnet_features(resnet, depth_imgs)

    combined_features = torch.cat((color_features_batch, depth_features_batch), dim = 1)  

    test_features.append(combined_features)
    test_labels.append(label_class)
    labels_one_hot.append(label)


concatenated_features_test = torch.cat(test_features, dim=0)
concatenated_labels_test = torch.cat(test_labels, dim=0)
conat_one_hot =torch.cat(labels_one_hot, dim=0) 
test_features_np = concatenated_features_test.cpu().numpy()
test_labels_np = concatenated_labels_test.cpu().numpy()
labels_one_hot = conat_one_hot.cpu().numpy()

scaler = StandardScaler()
features_np, X_test = scaler.fit_transform(features_np), scaler.transform(test_features_np)

svm = SVC(probability=True, gamma='auto')
svm.fit(features_np, labels_np)
# SVC(gamma='auto') 

# svm = LinearSVC()
# svm.fit(features_np, labels_np)

preds = svm.predict(X_test)
probs = svm.predict_proba(X_test)

accuracy = accuracy_score(test_labels_np, preds)
print("ResNet50 + SVM Accuracy:", accuracy)
print("Calculating scores")

genuine_scores, imposter_scores = [], []
genuine_path, imposter_path = "scores/resnet_svm/iPhone12_filled/lmaubo/genuine.npy", "scores/resnet_svm/iPhone12_filled/lmaubo/imposter.npy" 
os.makedirs("scores/resnet_svm/iPhone12_filled/lmaubo/", exist_ok = True)

for pred, label in zip(probs, labels_one_hot):
    # print(label)  
    label = label[0]

    # pred = pred.cpu().detach().view(-1)  
    # print(pred)
    if label == 1:
        genuine_scores.append(pred[0])
    else:
        imposter_scores.append(pred[0])

print("saving scores")
genuine_scores = [torch.tensor(score) for score in genuine_scores]
imposter_scores = [torch.tensor(score) for score in imposter_scores]
np.save(genuine_path, torch.stack(genuine_scores).numpy())
np.save(imposter_path, torch.stack(imposter_scores).numpy())

#LinearSVC: ResNet50 + SVM Accuracy: 0.9305879101354286