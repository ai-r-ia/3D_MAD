# do not augment images
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchcam.methods import GradCAMpp  # Import Grad-CAM++
from torchvision.transforms import ToTensor


from datasets.datasetwrapper import DatasetWrapper
from models.dual_attn import DualAttentionModel
from models.resnet_attn import AttentionResNet2

AUGMENT_TIMES = 0
batch_size = 16
root_dir = "/mnt/extravolume/data"
# trainds = "iPhone12"
trainds = "iPhone11"

def transform_single_image(color_path, depth_path, width=224, height=224, num_classes=10):
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
    if len(depth_img.shape) == 2:
        # depth_img = np.expand_dims(depth_img, axis=0)  # Keep as (H, W, 1)

        depth_img = np.expand_dims(depth_img, axis=-1)  # (H, W, 1)
        depth_img = np.repeat(depth_img, 3, axis=-1)   # Convert to (H, W, 3)

    depth_img = np.transpose(depth_img, (2, 0, 1))  # (3, H, W)
    # depth_img = np.expand_dims(depth_img, axis=-1)  # Keep as (H, W, 1)


    # Convert to PyTorch tensors
    color_tensor = torch.tensor(color_img, dtype=torch.float32).unsqueeze(0)  # Add batch dim
    depth_tensor = torch.tensor(depth_img, dtype=torch.float32).unsqueeze(0)  # Add batch dim

    return color_tensor, depth_tensor


subject = "M_S3_0_S8_0_W0.50_B0.50_AR_CE.jpg"
color_path = f"{root_dir}/{trainds}_filled/color/digital/morph/lmaubo/train/{subject}"
depth_path = f"{root_dir}/{trainds}_filled/depth/digital/morph/lmaubo/train/{subject}"
color_tensor, depth_tensor = transform_single_image(color_path, depth_path)
print("tensors: ", color_tensor.shape, depth_tensor.shape)

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

# SEPARATE HEATMAPS
# Select last convolutional layers from both branches
color_layer = "model1.resnet.7"  # Replace with actual last conv layer
depth_layer = "model2.resnet.7"  # Replace with actual last conv layer

# Initialize Grad-CAM++ for both branches
cam_color = GradCAMpp(model, color_layer)
cam_depth = GradCAMpp(model, depth_layer)

# Forward pass
output = model(color_tensor, depth_tensor)
output.requires_grad_()
class_idx = output.argmax(dim=1).item()

# Compute activation maps
activation_map_color = cam_color(class_idx, output, retain_graph=True)
activation_map_depth = cam_depth(class_idx, output)

# Convert and resize heatmaps
heatmap_color = activation_map_color[0].numpy()
heatmap_depth = activation_map_depth[0].numpy()
heatmap_color = cv2.resize(heatmap_color, (color_tensor.shape[2], color_tensor.shape[3]))
heatmap_depth = cv2.resize(heatmap_depth, (depth_tensor.shape[2], depth_tensor.shape[3]))

# Sum across the channels (if needed) or select specific channel (if needed)
heatmap_color = heatmap_color.sum(axis=-1)  # Sum across channels if needed
heatmap_depth = heatmap_depth.sum(axis=-1)  # Sum across channels if needed


# Save the color and depth heatmaps as PNG
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(color_tensor.squeeze(0).permute(1, 2, 0).numpy())  # Show color image
plt.imshow(heatmap_color, cmap="jet", alpha=0.5)  # Overlay Grad-CAM++ heatmap
plt.axis("off")
plt.title("Color Image - Grad-CAM++")
plt.savefig("color_image_heatmap.png", bbox_inches='tight', dpi=300)

plt.subplot(1, 2, 2)
plt.imshow(depth_tensor.squeeze(0).permute(1, 2, 0).numpy())  # Show color image
# plt.imshow(depth_tensor.squeeze(0).squeeze(0).numpy(), cmap="gray", aspect='auto')  # Show depth image
plt.imshow(heatmap_depth, cmap="jet", alpha=0.5)  # Overlay Grad-CAM++ heatmap
plt.axis("off")
plt.title("Depth Image - Grad-CAM++")
plt.savefig("depth_image_heatmap.png", bbox_inches='tight', dpi=300)
plt.subplot(1, 2, 2)

combined_heatmap = (heatmap_color + heatmap_depth) / 2
# Save the combined heatmap
plt.imshow(color_tensor.squeeze(0).permute(1, 2, 0).numpy())  # Show color image
plt.imshow(combined_heatmap, cmap="jet", alpha=0.5)  # Overlay combined heatmap
plt.axis("off")
plt.title("Combined Grad-CAM++ for Color & Depth")
plt.savefig("combined_heatmap.png", bbox_inches='tight', dpi=300)



# Overlay heatmaps separately
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(color_tensor.squeeze(0).permute(1, 2, 0).numpy())  # Show color image
plt.imshow(heatmap_color, cmap="jet", alpha=0.5)  # Overlay Grad-CAM++ heatmap
plt.axis("off")
plt.title("Color Image - Grad-CAM++")

plt.subplot(1, 2, 2)
plt.imshow(depth_tensor.squeeze(0).permute(1, 2, 0).numpy())  # Show color image
# plt.imshow(depth_tensor.squeeze(0).squeeze(0).numpy(), cmap="gray", aspect = 'auto')  # Show depth image
plt.imshow(heatmap_depth, cmap="jet", alpha=0.5)  # Overlay Grad-CAM++ heatmap
plt.axis("off")
plt.title("Depth Image - Grad-CAM++")

# Save the figure as a PNG file
plt.savefig("gradcam_output_separate.png", format="png", bbox_inches="tight")

plt.show()

# COMBINED HEATMAP

# Combine heatmaps (averaging)

plt.imshow(color_tensor.squeeze(0).permute(1, 2, 0).numpy())  # Show color image
plt.imshow(combined_heatmap, cmap="jet", alpha=0.5)  # Overlay combined heatmap
plt.axis("off")
plt.title("Combined Grad-CAM++ for Color & Depth")

# Save the figure as a PNG file
plt.savefig("gradcam_output_combined.png", format="png", bbox_inches="tight")

plt.show()






# # Apply Grad-CAM++
# cam_extractor = GradCAMpp(model, "layer_name")  # Set your last conv layer
# output = model(color_tensor, depth_tensor)
# class_idx = output.argmax(dim=1).item()
# activation_map = cam_extractor(class_idx, output)

# # Resize heatmap and overlay on the original image
# heatmap = activation_map[0].numpy()
# heatmap = cv2.resize(heatmap, (input_tensor.shape[2], input_tensor.shape[3]))

# plt.imshow(sample.permute(1, 2, 0).numpy())  # Show preprocessed image
# plt.imshow(heatmap, cmap="jet", alpha=0.5)  # Overlay heatmap
# plt.axis("off")
# plt.show()
