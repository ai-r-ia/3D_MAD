import torch
import cv2
import numpy as np
from models.dual_attn import DualAttentionModel
from models.resnet_attn import AttentionResNet2
import torch.nn.functional as F
from tqdm import tqdm
# Directories for saving results
# output_dir = "interpretability_results"
# os.makedirs(output_dir, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load and transform images
def transform_single_image(color_path, depth_path):
    # Load and preprocess color image
    color_img = cv2.imread(color_path, cv2.IMREAD_COLOR)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    color_img = cv2.resize(color_img, (224, 224))
    color_img = (color_img - color_img.min()) / (color_img.max() - color_img.min() + 1e-8)
    color_img = np.transpose(color_img.astype("float32"), (2, 0, 1))

    # Load and preprocess depth image
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype("float32")
    depth_img = cv2.resize(depth_img, (224, 224))
    depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min() + 1e-8)
    depth_img = np.expand_dims(depth_img, axis=0)  # (1, H, W)

    # Convert to tensors
    color_tensor = torch.tensor(color_img, dtype=torch.float32).unsqueeze(0).to(device)
    depth_tensor = torch.tensor(depth_img, dtype=torch.float32).unsqueeze(0).to(device)
    return color_tensor, depth_tensor

# Model setup
reduction, kernel_size = 4, 5
attn_type = ["spatial", "channel"]
model1 = AttentionResNet2(attention_types=attn_type, reduction=reduction, kernel_size=kernel_size)
model2 = AttentionResNet2(attention_types=attn_type, reduction=reduction, kernel_size=kernel_size)
model = DualAttentionModel(model1=model1, model2=model2).to(device)
model.eval()

# Load model weights
# pretrained_weights = torch.load("checkpoints/Protocol_0/spatial_channel_iPhone11_4_5/spatial_channel_iPhone11_4_5_best.pth")
pretrained_weights = torch.load("checkpoints/Protocol_0/spatial_channel_iPhone11_final/spatial_channel_iPhone11_final_best.pth")
model.load_state_dict(pretrained_weights, strict=False)
model.eval()


# left2right,top2bottom
# 14_92
# 52_61
# 72_30
# 76_39 - test
# 68_63
# 4_17
 

# Correct Bonafide: [['/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S39_28.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S67_22.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S54_19.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S90_2.jpg']_]
# Bonafide misclassified as Morph: [['/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S67_33.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/bonnafide/test/S67_22.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S44_10.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/tsest/S19_21.jpg']]
# Morph misclassified as Bonafide: [['/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S10_0_S18_0_W0.50_B0.50_AR_CE.jpg'], ['/mnt/extravolume/data/iPhoone11_filled/color/digital/morph/lmaubo/test/M_S10_0_S54_0_W0.50_B0.50_AR_CE.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S49_0_S70_0_W05.50_B0.50_AR_CE.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S90_0_S1_0_W0.50_B0.50_AR_CE.jpg']]
# Correct Morph: [['/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S88_0_S70_0_W0.50_B0.50_AR_CE.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color//digital/morph/lmaubo/test/M_S65_0_S81_0_W0.50_B0.50_AR_CE.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S42_0_S88_0_W0.50_B0.50_AR_CE.jp'g'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S65_0_S70_0_W0.50_B0.50_AR_CE.jpg']]
# 
# List of image paths (replace these with your actual image paths)
image_list = [
    # corr bona
    # '/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S39_28.jpg',
    # # '/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S67_22.jpg',
    # '/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S54_19.jpg', 
    # # '/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S90_2.jpg'
    
    # bona miss
    # '/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S67_33.jpg',
    # # '/mnt/extravolume/data/iPhone11_filled/color/digital/bonnafide/test/S67_22.jpg',
    # '/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S44_10.jpg',
    # # '/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/tsest/S19_21.jpg',
    # mor miss
    # # '/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S10_0_S18_0_W0.50_B0.50_AR_CE.jpg',
    # # '/mnt/extravolume/data/iPhoone11_filled/color/digital/morph/lmaubo/test/M_S10_0_S54_0_W0.50_B0.50_AR_CE.jpg',
    # # '/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S49_0_S70_0_W05.50_B0.50_AR_CE.jpg',
    # '/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S90_0_S1_0_W0.50_B0.50_AR_CE.jpg',
    
    '/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S49_0_S70_0_W0.50_B0.50_AR_CE.jpg',
    '/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S54_0_S84_0_W0.50_B0.50_AR_CE.jpg',
    '/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S49_0_S25_0_W0.50_B0.50_AR_CE.jpg',
    '/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S2_0_S84_0_W0.50_B0.50_AR_CE.jpg',
    '/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S53_0_S15_0_W0.50_B0.50_AR_CE.jpg',
    '/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S10_0_S18_0_W0.50_B0.50_AR_CE.jpg'
#   corr mor
    # '/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S88_0_S70_0_W0.50_B0.50_AR_CE.jpg',
    # '/mnt/extravolume/data/iPhone11_filled/color//digital/morph/lmaubo/test/M_S65_0_S81_0_W0.50_B0.50_AR_CE.jpg',
    # '/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S42_0_S88_0_W0.50_B0.50_AR_CE.jpg',
    # '/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S65_0_S70_0_W0.50_B0.50_AR_CE.jpg',
    
    
    # "/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/train/S14_1.jpg",
    # "/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/train/S52_1.jpg",
    # "/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/train/S92_1.jpg",
    # "/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S76_1.jpg",
    # "/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S39_1.jpg",
    # "/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/train/M_S14_0_S92_0_W0.50_B0.50_AR_CE.jpg",
    # "/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/train/M_S4_0_S17_0_W0.50_B0.50_AR_CE.jpg",
    # "/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/train/M_S72_0_S30_0_W0.50_B0.50_AR_CE.jpg",
    # "/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/train/M_S52_0_S61_0_W0.50_B0.50_AR_CE.jpg",
    # "/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/train/M_S68_0_S63_0_W0.50_B0.50_AR_CE.jpg",
    # "/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S76_0_S39_0_W0.50_B0.50_AR_CE.jpg",
    
]

def preprocess_image(color_path, depth_path):
    """Preprocess color and depth images before feeding into the model."""

    # Load and preprocess color image (3-channel RGB)
    color_img = cv2.imread(color_path, cv2.IMREAD_COLOR)
    if color_img is None:
        raise FileNotFoundError(f"Could not read file: {color_path}")

    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    color_img = cv2.resize(color_img, (224, 224))  # Resize to 224x224
    color_img = color_img.astype("float32") / 255.0  # Normalize to [0,1]
    color_img = np.transpose(color_img, (2, 0, 1))  # Change to (C, H, W)

    # Load and preprocess depth image (single-channel grayscale)
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # Keep original depth values
    if depth_img is None:
        raise FileNotFoundError(f"Could not read file: {depth_path}")

    depth_img = cv2.resize(depth_img, (224, 224))  # Resize
    depth_img = depth_img.astype("float32") / 255.0  # Normalize
    depth_img = np.expand_dims(depth_img, axis=0)  # Convert to (1, H, W)
    
    # Convert depth image to 3 channels by repeating the single-channel depth image
    depth_img = np.repeat(depth_img, 3, axis=0)  # (1, H, W) -> (3, H, W)

    # Convert to tensors and move to device
    color_tensor = torch.tensor(color_img, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 3, 224, 224)
    depth_tensor = torch.tensor(depth_img, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    return color_tensor, depth_tensor

# Classification results
classification_results = {}

for img_path in image_list:
    depth_path = img_path.replace("color", "depth")  # Assuming depth has same filename structure
    
    try:
        color_tensor, depth_tensor = preprocess_image(img_path, depth_path)

        # Forward pass through the model
        output = model(color_tensor, depth_tensor)
        probabilities = F.softmax(output, dim=1) 
        pred = torch.argmax(probabilities, dim=1).item()  # Predicted class
        # print(probabilities)
        # Store results
        classification_results[img_path] = "Bonafide" if pred == 0 else "Morph"

        print(f"{img_path}: {classification_results[img_path]} (Confidence: {probabilities.max().item():.4f})")

    except Exception as e:
        # print(f"Error processing {img_path}: {e}")
        print(f"Error processing {img_path}:")


# from datasets.datasetwrapper import DatasetWrapper

# batch_size = 1
# dataset_wrapper = DatasetWrapper(root_dir= f"/mnt/extravolume/data/iPhone11_filled/color/digital/")

# testds = dataset_wrapper.get_test_dataset(
#     augment_times=2,
#     batch_size=batch_size,
#     morph_types=["lmaubo"],
#     num_models=1,
#     shuffle=True,
# )
# correct_bonafide = []
# wrong_bonafide = []
# correct_morph = []
# wrong_morph = []
# correct_bonafide_num = 0
# wrong_bonafide_num = 0
# correct_morph_num = 0
# wrong_morph_num = 0

# model.eval()
# with torch.no_grad():
#     for i, (color_img, depth_img, label, img_path) in tqdm(enumerate(testds)):
#         color_img, depth_img = color_img.to(device), depth_img.to(device)
#         label = torch.argmax(label, dim=1).item()
        
        
#         # Forward pass
#         output = model(color_img, depth_img)
#         output = F.softmax(output, dim=1) 
#         pred = torch.argmax(output, dim=1).item()

#         # Store images based on classification
#         if label == 0 and pred == 0:  # Bonafide correctly classified
#             correct_bonafide.append(img_path)
#             correct_bonafide_num+=1
#         elif label == 0 and pred == 1:  # Bonafide misclassified as Morph
#             wrong_bonafide.append(img_path)
#             wrong_bonafide_num+=1
#         elif label == 1 and pred == 0:  # Morph misclassified as Bonafide
#             wrong_morph.append(img_path)
#             wrong_morph_num+=1
#         elif label == 1 and pred == 1:  # Morph correctly classified
#             correct_morph.append(img_path)
#             correct_morph_num+=1

# # Print some examples
# # print("Correct Bonafide:", correct_bonafide[:4])
# # print("Bonafide misclassified as Morph:", wrong_bonafide[:4])
# print("Morph misclassified as Bonafide:", wrong_morph[:6])
# # print("Correct Morph:", correct_morph[:4])
# # print("Correct Bonafide:", correct_bonafide_num)
# # print("Bonafide misclassified as Morph:", wrong_bonafide_num)
# print("Morph misclassified as Bonafide:", wrong_morph_num)
# # print("Correct Morph:", correct_morph_num)


# Correct Bonafide: [['/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S39_28.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S67_22.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S54_19.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S90_2.jpg']_]
# Bonafide misclassified as Morph: [['/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S67_33.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/bonnafide/test/S67_22.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S44_10.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/tsest/S19_21.jpg']]
# Morph misclassified as Bonafide: [['/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S10_0_S18_0_W0.50_B0.50_AR_CE.jpg'], ['/mnt/extravolume/data/iPhoone11_filled/color/digital/morph/lmaubo/test/M_S10_0_S54_0_W0.50_B0.50_AR_CE.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S49_0_S70_0_W05.50_B0.50_AR_CE.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S90_0_S1_0_W0.50_B0.50_AR_CE.jpg']]
# Correct Morph: [['/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S88_0_S70_0_W0.50_B0.50_AR_CE.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color//digital/morph/lmaubo/test/M_S65_0_S81_0_W0.50_B0.50_AR_CE.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S42_0_S88_0_W0.50_B0.50_AR_CE.jp'g'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S65_0_S70_0_W0.50_B0.50_AR_CE.jpg']]
# Correct Bonafide: 4892
# Bonafide misclassified as Morph: 168
# Morph misclassified as Bonafide: 82
# Correct Morph: 1106


# Morph misclassified as Bonafide: [['/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S49_0_S70_0_W0.50_B0.50_AR_CE.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S54_0_S84_0_W0.50_B0.50_AR_CE.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S49_0_S25_0_W0.50_B0.50_AR_CE.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S2_0_S84_0_W0.50_B0.50_AR_CE.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S53_0_S15_0_W0.50_B0.50_AR_CE.jpg'], ['/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S10_0_S18_0_W0.50_B0.50_AR_CE.jpg']