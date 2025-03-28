import cv2
import numpy as np
import os
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datasets.datasetwrapper import DatasetWrapper
from configs.config import get_logger, protocol_dict, create_parser
from configs.seed import set_seed

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

def main(args):
    protocol_num = protocol_dict[f"{args.trainds}_{args.testds}"]
    logger = get_logger(filename = "lbp", protocol = protocol_num)
    logger.info(f"training lbp for protocol {protocol_num}")

    dataset_wrapper_train = DatasetWrapper(root_dir=f"{args.root_dir}/{args.trainds}_filled/color/digital")
    dataset_wrapper_test = DatasetWrapper(root_dir=f"{args.root_dir}/{args.testds}_filled/color/digital")

    train_dataset = dataset_wrapper_train.get_train_dataset(augment_times=AUGMENT_TIMES, batch_size=batch_size, morph_types=["lmaubo"], num_models=1, shuffle=True)

    test_dataset = dataset_wrapper_test.get_test_dataset(augment_times=AUGMENT_TIMES, batch_size=batch_size, morph_types=["lmaubo"], num_models=1, shuffle=True)


    features, labels = [], []
    for color_imgs, depth_imgs, label in tqdm(train_dataset, desc="Processing Training Data"):
        label_class = label.argmax(dim=1).cpu().numpy()
        for img, depth_img, lbl in zip(color_imgs, depth_imgs, label_class):
            features.append(extract_lbp_features(img, depth_img))
            labels.append(lbl)

    features_np = np.array(features)
    labels_np = np.array(labels)
    label_cnt = np.bincount(labels_np)
    logger.info(f"Training class distribution: {str(label_cnt)}")

    test_features, test_labels = [], []
    for color_imgs, depth_imgs, label in tqdm(test_dataset, desc="Processing Test Data"):
        label_class = label.argmax(dim=1).cpu().numpy()
        for img, depth_img, lbl in zip(color_imgs, depth_imgs, label_class):
            test_features.append(extract_lbp_features(img, depth_img))
            test_labels.append(lbl)

    test_features_np = np.array(test_features)
    test_labels_np = np.array(test_labels)

    test_label_cnt = np.bincount(test_labels_np)
    logger.info(f"Test class distribution: {str(test_label_cnt)}")

    scaler = StandardScaler()
    features_np = scaler.fit_transform(features_np)
    test_features_np = scaler.transform(test_features_np)

    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, class_weight='balanced')
    # svm = SVC(probability=True, gamma='scale')
    # svm = SVC(probability=True, gamma='auto')
    svm.fit(features_np, labels_np)

    preds = svm.predict(test_features_np)
    probs = svm.predict_proba(test_features_np)
    accuracy = accuracy_score(test_labels_np, preds)
    logger.info(f"LBP + SVM Accuracy: {accuracy}")
    logger.info("computing scores")
    genuine_scores, imposter_scores = [], []
    genuine_path, imposter_path = f"scores/Protocol_{protocol_num}/lbp_svm/{args.testds}/lmaubo/genuine.npy", f"scores/Protocol_{protocol_num}/lbp_svm/{args.testds}/lmaubo/imposter.npy"
    os.makedirs(f"scores/Protocol_{protocol_num}/lbp_svm/{args.testds}/lmaubo/", exist_ok=True)

    for prob, label in zip(probs, test_labels_np):
        if label == 1:
            genuine_scores.append(prob[0])
        else:
            imposter_scores.append(prob[0])

    np.save(genuine_path, np.array(genuine_scores))
    np.save(imposter_path, np.array(imposter_scores))
    logger.info("Scores saved successfully!")

if __name__ == "__main__":
    set_seed()
    parser = create_parser()
    args = parser.parse_args()
    main(args)





# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.hist(features_np.ravel(), bins=50, alpha=0.75, color="blue", label="Train Features")
# plt.hist(test_features_np.ravel(), bins=50, alpha=0.75, color="red", label="Test Features")
# plt.legend()
# plt.title("Feature Distribution of LBP Histograms")
# plt.savefig(f"lbp_feature_distribution_11_{num_bins}.png", dpi=300, bbox_inches="tight")
# plt.close()

# import seaborn as sns

# class_0 = np.mean(features_np[labels_np == 0], axis=0)
# class_1 = np.mean(features_np[labels_np == 1], axis=0)

# plt.figure(figsize=(12, 5))
# sns.lineplot(x=range(len(class_0)), y=class_0, label="Class 0 (Bonafide)", color='blue')
# sns.lineplot(x=range(len(class_1)), y=class_1, label="Class 1 (Morph)", color='red')
# plt.title("Average LBP Histograms per Class")
# plt.savefig(f"lbp_class_histograms_11_{num_bins}.png", dpi=300, bbox_inches="tight")
# plt.close()

# from sklearn.decomposition import PCA
# import seaborn as sns

# pca = PCA(n_components=2)
# reduced_features = pca.fit_transform(features_np)

# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=labels_np, palette="coolwarm")
# plt.title("PCA Projection of LBP Features")
# plt.savefig(f"pca_lbp_features_11_{num_bins}.png", dpi=300, bbox_inches="tight")
# plt.close()


    # LBP + SVM Accuracy: 0.7202195285663521