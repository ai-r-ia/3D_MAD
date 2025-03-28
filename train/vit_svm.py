from asyncio import protocols
from fileinput import filename
from configs.config import create_parser, get_logger, protocol_dict
from configs.seed import set_seed
import torch
import os
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

from datasets.datasetwrapper import DatasetWrapper

AUGMENT_TIMES = 2
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Pretrained ViT for feature extraction
vit = models.vit_b_16(pretrained=True)
vit.heads = nn.Identity()  # Remove the classification head
vit = vit.to(device)

def extract_vit_features(model, image):
    """Extract deep features from ViT"""
    model.eval()
    with torch.no_grad():
        features = model(image).squeeze()  # Extract feature embeddings
    return features

def main(args):
    protocol_num = protocol_dict[f"{args.trainds}_{args.testds}"]
    logger = get_logger(filename = "vit", protocol = protocol_num)
    logger.info(f"training vit for protocol {protocol_num}") 
    dataset_wrapper = DatasetWrapper(root_dir=f"{args.root_dir}/{args.trainds}_filled/color/digital/")

    train_dataset = dataset_wrapper.get_train_dataset(
        augment_times=AUGMENT_TIMES,
        batch_size=batch_size,
        morph_types=["lmaubo"],
        num_models=1,
        shuffle=True,
    )

    features, labels = [], []

    for color_imgs, depth_imgs, label in tqdm(train_dataset, desc="Training train dataset"):
        color_imgs, depth_imgs = color_imgs.to(device), depth_imgs.to(device)
        label_class = label.argmax(dim=1).detach().to(device)
        
        with torch.no_grad():
            color_features_batch = extract_vit_features(vit, color_imgs)
            depth_features_batch = extract_vit_features(vit, depth_imgs)

        combined_features = torch.cat((color_features_batch, depth_features_batch), dim=1)
        
        features.append(combined_features)
        labels.append(label_class)


    concatenated_features = torch.cat(features, dim=0)
    concatenated_labels = torch.cat(labels, dim=0)
    features_np, labels_np = concatenated_features.cpu().numpy(), concatenated_labels.cpu().numpy()

    dataset_wrapper2 = DatasetWrapper(root_dir=f"{args.root_dir}/{args.testds}_filled/color/digital/")

    test_dataset = dataset_wrapper2.get_test_dataset(
        augment_times=AUGMENT_TIMES,
        batch_size=batch_size,
        morph_types=["lmaubo"],
        num_models=1,
        shuffle=True,
    )

    test_features, test_labels, labels_one_hot = [], [], []

    for color_imgs, depth_imgs, label in tqdm(test_dataset, desc="test dataset"):
        color_imgs, depth_imgs = color_imgs.to(device), depth_imgs.to(device)
        label_class = label.argmax(dim=1).detach().to(device)
        
        with torch.no_grad():
            color_features_batch = extract_vit_features(vit, color_imgs)
            depth_features_batch = extract_vit_features(vit, depth_imgs)

        combined_features = torch.cat((color_features_batch, depth_features_batch), dim=1)
        
        test_features.append(combined_features)
        test_labels.append(label_class)
        labels_one_hot.append(label)

    concatenated_features_test = torch.cat(test_features, dim=0)
    concatenated_labels_test = torch.cat(test_labels, dim=0)
    conat_one_hot = torch.cat(labels_one_hot, dim=0)

    test_features_np, test_labels_np, labels_one_hot = (
        concatenated_features_test.cpu().numpy(),
        concatenated_labels_test.cpu().numpy(),
        conat_one_hot.cpu().numpy(),
    )

    scaler = StandardScaler()
    features_np, X_test = scaler.fit_transform(features_np), scaler.transform(test_features_np)

    svm = SVC(probability=True, gamma='auto', class_weight='balanced')
    svm.fit(features_np, labels_np)

    preds = svm.predict(X_test)
    probs = svm.predict_proba(X_test)
    accuracy = accuracy_score(test_labels_np, preds)
    logger.info(f"ViT + SVM Accuracy: {accuracy}")

    logger.info("Calculating scores")
    genuine_scores, imposter_scores = [], []
    genuine_path, imposter_path = f"scores/Protocol_{protocol_num}/vit_svm/{args.testds}/lmaubo/genuine.npy", f"scores/Protocol_{protocol_num}/vit_svm/{args.testds}/lmaubo/imposter.npy"
    os.makedirs(f"scores/Protocol_{protocol_num}/vit_svm/{args.testds}/lmaubo/", exist_ok=True)

    for pred, label in zip(probs, labels_one_hot):
        label = label[0]
        if label == 1:
            genuine_scores.append(pred[0])
        else:
            imposter_scores.append(pred[0])

    logger.info("Saving scores")
    genuine_scores = [torch.tensor(score) for score in genuine_scores]
    imposter_scores = [torch.tensor(score) for score in imposter_scores]
    np.save(genuine_path, torch.stack(genuine_scores).numpy())
    np.save(imposter_path, torch.stack(imposter_scores).numpy())

    

if __name__ == "__main__":
    set_seed()
    parser = create_parser()
    args = parser.parse_args()
    main(args)

# ViT + SVM Accuracy: 0.9580684644845422