import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from lime import lime_image
from torchcam.methods import GradCAMpp  # Import Grad-CAM++
from torchvision.transforms import ToTensor
from datasets.datasetwrapper import DatasetWrapper
from models.dual_attn import DualAttentionModel
from models.resnet_attn import AttentionResNet2

AUGMENT_TIMES = 0
batch_size = 16
root_dir = "/mnt/extravolume/data"
trainds = "iPhone11"  # Change to "iPhone12" if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to transform images
def transform_single_image(color_path, depth_path, width=224, height=224):
    # Load color image
    color_img = cv2.imread(color_path, cv2.IMREAD_COLOR)
    if color_img is None:
        raise ValueError(f"Failed to load image: {color_path}")

    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    color_img = cv2.resize(color_img, (width, height))
    color_img = (color_img - color_img.min()) / ((color_img.max() - color_img.min()) or 1.0)
    color_img = np.transpose(color_img.astype("float32"), axes=(2, 0, 1))  # (3, H, W)

    # Load depth image
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise ValueError(f"Failed to load depth image: {depth_path}")

    depth_img = depth_img.astype("float32")
    depth_img = cv2.resize(depth_img, (width, height))
    depth_img = (depth_img - depth_img.min()) / ((depth_img.max() - depth_img.min()) or 1.0)
    depth_img = np.expand_dims(depth_img, axis=-1)  # Convert to (H, W, 1)
    depth_img = np.repeat(depth_img, 3, axis=-1)   # Convert to (H, W, 3)

    depth_img = np.transpose(depth_img, (2, 0, 1))  # (3, H, W)

    # Convert to PyTorch tensors
    color_tensor = torch.tensor(color_img, dtype=torch.float32).unsqueeze(0)  # Add batch dim
    depth_tensor = torch.tensor(depth_img, dtype=torch.float32).unsqueeze(0)  # Add batch dim

    return color_tensor, depth_tensor

# Loading the model
reduction = 4
kernel_size = 5
attn_type = ["spatial", "channel"]
model_name = f"spatial_channel_{trainds}_4_5"
model1 = AttentionResNet2(attention_types=attn_type, reduction=reduction, kernel_size=kernel_size)
model2 = AttentionResNet2(attention_types=attn_type, reduction=reduction, kernel_size=kernel_size)
model = DualAttentionModel(model1=model1, model2=model2)

pretrained_weights = torch.load(f'checkpoints/Protocol_0/{model_name}/{model_name}_best.pth')
model.load_state_dict(pretrained_weights, strict=False)
model.eval()

# Initialize Grad-CAM++ for both branches
color_layer = "model1.resnet.7"  # Replace with actual last conv layer
depth_layer = "model2.resnet.7"  # Replace with actual last conv layer
cam_color = GradCAMpp(model, color_layer)
cam_depth = GradCAMpp(model, depth_layer)

# LIME explainer setup
explainer = lime_image.LimeImageExplainer()

# Image loading and transformation
subject = "M_S3_0_S8_0_W0.50_B0.50_AR_CE.jpg"
color_path = f"{root_dir}/{trainds}_filled/color/digital/morph/lmaubo/train/{subject}"
depth_path = f"{root_dir}/{trainds}_filled/depth/digital/morph/lmaubo/train/{subject}"
color_tensor, depth_tensor = transform_single_image(color_path, depth_path)
print("Tensors: ", color_tensor.shape, depth_tensor.shape)

# Forward pass for Grad-CAM++
output = model(color_tensor, depth_tensor)
output.requires_grad_()
class_idx = output.argmax(dim=1).item()

# Compute activation maps using Grad-CAM++
activation_map_color = cam_color(class_idx, output, retain_graph=True)
activation_map_depth = cam_depth(class_idx, output)

# Convert and resize heatmaps
heatmap_color = activation_map_color[0].numpy()
heatmap_depth = activation_map_depth[0].numpy()

# Normalize the heatmaps
heatmap_color = cv2.resize(heatmap_color, (color_tensor.shape[2], color_tensor.shape[3]))
heatmap_depth = cv2.resize(heatmap_depth, (depth_tensor.shape[2], depth_tensor.shape[3]))
heatmap_color -= heatmap_color.min()
heatmap_color /= heatmap_color.max()
heatmap_depth -= heatmap_depth.min()
heatmap_depth /= heatmap_depth.max()

# LIME explanation for color and depth images
def predict_fn(images):
    model.eval()
    images = torch.from_numpy(images).float().to(device)  # Convert to tensor and move to device
    with torch.no_grad():
        output = model(images)
    return output.cpu().numpy()

# LIME explanations
explanation_color = explainer.explain_instance(
    color_tensor.squeeze(0).permute(1, 2, 0).numpy(),
    predict_fn, 
    top_labels=1, 
    hide_color=0, 
    num_samples=1000
)

explanation_depth = explainer.explain_instance(
    depth_tensor.squeeze(0).squeeze(0).numpy(), 
    predict_fn, 
    top_labels=1, 
    hide_color=0, 
    num_samples=1000
)

# Get LIME explanation images
lime_img_color = explanation_color.get_image_and_mask(
    explanation_color.top_labels[0], 
    positive_only=True, 
    num_features=5, 
    hide_rest=False
)[0]

lime_img_depth = explanation_depth.get_image_and_mask(
    explanation_depth.top_labels[0], 
    positive_only=True, 
    num_features=5, 
    hide_rest=False
)[0]

# Normalize the LIME images
lime_img_color = lime_img_color / 255.0
lime_img_depth = lime_img_depth / 255.0

# Plot the original and LIME-explained color image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(color_tensor.squeeze(0).permute(1, 2, 0).numpy())  # Show color image
plt.title("Original Color Image")

plt.subplot(1, 2, 2)
plt.imshow(lime_img_color)  # Show LIME explanation for color image
plt.title("LIME Explanation for Color Image")

# Save the figure as a PNG file
plt.savefig("lime_color_image.png", format="png", bbox_inches="tight", dpi=300)

# Plot the original and LIME-explained depth image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(depth_tensor.squeeze(0).squeeze(0).numpy(), cmap="gray", aspect="auto")  # Show depth image
plt.title("Original Depth Image")

plt.subplot(1, 2, 2)
plt.imshow(lime_img_depth)  # Show LIME explanation for depth image
plt.title("LIME Explanation for Depth Image")

# Save the figure as a PNG file
plt.savefig("lime_depth_image.png", format="png", bbox_inches="tight", dpi=300)

# Combined heatmaps (Grad-CAM++ + LIME)
combined_lime_img = (lime_img_color + lime_img_depth) / 2
plt.imshow(color_tensor.squeeze(0).permute(1, 2, 0).numpy())  # Show color image
plt.imshow(combined_lime_img, cmap="jet", alpha=0.5)  # Overlay combined heatmap
plt.axis("off")
plt.title("Combined LIME Explanation for Color & Depth")

# Save the combined LIME heatmap
plt.savefig("combined_lime_heatmap.png", format="png", bbox_inches="tight", dpi=300)

# Show the plots
plt.show()
