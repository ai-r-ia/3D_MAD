import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from lime import lime_image
from torchcam.methods import GradCAMpp  # Import Grad-CAM++
from torchvision.transforms import ToTensor
from datasets.datasetwrapper import DatasetWrapper
from models.dual_attn import DualAttentionModel
from models.resnet_attn import AttentionResNet2
import torch.nn.functional as F
# Directories for saving results
output_dir = "interpretability_results"
os.makedirs(output_dir, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load and transform images
def transform_single_image(color_path, depth_path):
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

# Model setup
reduction, kernel_size = 4, 5
attn_type = ["spatial", "channel"]
model1 = AttentionResNet2(attention_types=attn_type, reduction=reduction, kernel_size=kernel_size)
model2 = AttentionResNet2(attention_types=attn_type, reduction=reduction, kernel_size=kernel_size)
model = DualAttentionModel(model1=model1, model2=model2).to(device)
model.eval()

# Load model weights
pretrained_weights = torch.load("checkpoints/Protocol_0/spatial_channel_iPhone11_final/spatial_channel_iPhone11_final_best.pth")
model.load_state_dict(pretrained_weights, strict=False)

# GradCAM++ setup
cam_color = GradCAMpp(model, "model1.resnet.7")
cam_depth = GradCAMpp(model, "model2.resnet.7")

# LIME explainer
explainer = lime_image.LimeImageExplainer()

# List of test cases with corresponding labels
test_cases = {
    "bona_bona": [('/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S39_28.jpg',
                   '/mnt/extravolume/data/iPhone11_filled/depth/digital/bonafide/test/S39_28.jpg'),
                  
                  ('/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S54_19.jpg', 
                   '/mnt/extravolume/data/iPhone11_filled/depth/digital/bonafide/test/S54_19.jpg')],
    
    "bona_morph": [('/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S67_33.jpg',
                    '/mnt/extravolume/data/iPhone11_filled/depth/digital/bonafide/test/S67_33.jpg'), 
                    
                   ( '/mnt/extravolume/data/iPhone11_filled/color/digital/bonafide/test/S44_10.jpg', 
                    '/mnt/extravolume/data/iPhone11_filled/depth/digital/bonafide/test/S44_10.jpg',)],
   
    # "morph_bona": [('/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S49_0_S70_0_W05.50_B0.50_AR_CE.jpg',
    "morph_bona": [('/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S10_0_S18_0_W0.50_B0.50_AR_CE.jpg',
                    '/mnt/extravolume/data/iPhone11_filled/depth/digital/morph/lmaubo/test/M_S10_0_S18_0_W0.50_B0.50_AR_CE.jpg',), 
   
                   ( '/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S90_0_S1_0_W0.50_B0.50_AR_CE.jpg', 
                     '/mnt/extravolume/data/iPhone11_filled/depth/digital/morph/lmaubo/test/M_S90_0_S1_0_W0.50_B0.50_AR_CE.jpg')],
   
    "morph_morph" : [('/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S88_0_S70_0_W0.50_B0.50_AR_CE.jpg',
                    '/mnt/extravolume/data/iPhone11_filled/depth/digital/morph/lmaubo/test/M_S88_0_S70_0_W0.50_B0.50_AR_CE.jpg'),
    
                   ('/mnt/extravolume/data/iPhone11_filled/color/digital/morph/lmaubo/test/M_S65_0_S81_0_W0.50_B0.50_AR_CE.jpg',
                   '/mnt/extravolume/data/iPhone11_filled/depth/digital/morph/lmaubo/test/M_S65_0_S81_0_W0.50_B0.50_AR_CE.jpg')]
}

def compute_gradcam(model, cam, color_tensor, depth_tensor, class_idx):
    # Forward pass both inputs through the model
    output = model(color_tensor, depth_tensor)
    output = F.softmax(output, dim=1)

    # Compute GradCAM activation maps
    activation_map_color = cam(class_idx, output, retain_graph = True)  # Assuming `use_color` directs which input
    # activation_map_depth = cam(class_idx, output)

    # Process Color GradCAM
    heatmap_color = activation_map_color[0].cpu().numpy()
    if heatmap_color.ndim == 3:  # Ensure single channel
        heatmap_color = heatmap_color[0]
    heatmap_color = cv2.resize(heatmap_color, (224, 224))
    heatmap_color = (heatmap_color - heatmap_color.min()) / (heatmap_color.max() - heatmap_color.min() + 1e-8)

    # # Process Depth GradCAM
    # heatmap_depth = activation_map_depth[0].cpu().numpy()
    # if heatmap_depth.ndim == 3:  # Ensure single channel
    #     heatmap_depth = heatmap_depth[0]
    # heatmap_depth = cv2.resize(heatmap_depth, (224, 224))
    # heatmap_depth = (heatmap_depth - heatmap_depth.min()) / (heatmap_depth.max() - heatmap_depth.min() + 1e-8)

    # return heatmap_color, heatmap_depth
    return heatmap_color



import numpy as np
import torch
import torch.nn.functional as F
from lime import lime_image

def compute_lime(color_tensor, depth_tensor, model, explainer, device, input_type="color"):
    """
    This function computes LIME explanations for either color or depth image input
    (while keeping the other fixed), given the model, color, and depth inputs.
    
    Parameters:
    - color_tensor: Tensor containing the color image.
    - depth_tensor: Tensor containing the depth image.
    - model: The model that takes both color and depth images as input.
    - explainer: The LIME image explainer.
    - device: The device (CPU/GPU).
    - input_type: Specifies whether to compute LIME for "color" or "depth" image.
    
    Returns:
    - lime_img: LIME explanation for the input image (color or depth).
    """
    
    model.eval()  # Ensure the model is in evaluation mode

    if input_type == "color":
        # Compute LIME for the color image, keeping depth fixed
        color_tensor.requires_grad_()
        
        def predict_fn(images):
            """Predict function for LIME on color images (with fixed depth)."""
            images = torch.from_numpy(images).float().to(device)

            # Fix shape: LIME provides (B, H, W, C), but model needs (B, C, H, W)
            images = images.permute(0, 3, 1, 2)

            # Repeat the depth tensor to match the batch size
            depth_fixed = depth_tensor.repeat(images.shape[0], 1, 1, 1).to(device)

            with torch.no_grad():
                output = model(images, depth_fixed)
                output = F.softmax(output, dim=1)
            return output.cpu().numpy()

    elif input_type == "depth":
        # Compute LIME for the depth image, keeping color fixed
        depth_tensor.requires_grad_()
        
        def predict_fn(images):
            """Predict function for LIME on depth images (with fixed color)."""
            images = torch.from_numpy(images).float().to(device)

            # Fix shape: LIME provides (B, H, W, C), but model needs (B, C, H, W)
            images = images.permute(0, 3, 1, 2)

            # Repeat the color tensor to match the batch size
            color_fixed = color_tensor.repeat(images.shape[0], 1, 1, 1).to(device)

            with torch.no_grad():
                output = model(color_fixed, images)
                output = F.softmax(output, dim=1)
            return output.cpu().numpy()

    else:
        raise ValueError("input_type must be either 'color' or 'depth'")

    # Convert tensors to numpy for LIME (from torch tensor to numpy array)
    if input_type == "color":
        input_np = color_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # Ensure to move to CPU
    else:
        input_np = depth_tensor.squeeze(0).detach().cpu().numpy()

    # If depth is a single-channel, repeat across RGB channels for LIME
    if len(input_np.shape) == 2 and input_type == "depth":
        input_np = np.repeat(input_np[:, :, np.newaxis], 3, axis=-1)

    # Create the LIME explainer
    lime_explainer = lime_image.LimeImageExplainer()

    # Compute LIME explanation for the selected input image
    explanation = lime_explainer.explain_instance(
        input_np,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    lime_img, _ = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
    )

    return lime_img / 255.0  # Return the LIME explanation for the image

def ensure_requires_grad(model):
    """Ensure all layers in the model have requires_grad=True."""
    for param in model.parameters():
        param.requires_grad = True
    return model
def remove_hooks(model):
    """Function to remove hooks from the model layers."""
    for layer in model.modules():
        if hasattr(layer, 'hook_handles'):
            for handle in layer.hook_handles:
                handle.remove()

def gradcam():
    # Process each test case
    for case, images in test_cases.items():
        case_dir = os.path.join(output_dir, case)
        os.makedirs(case_dir, exist_ok=True)

        for i, (color_path, depth_path) in enumerate(images):
            color_tensor, depth_tensor = transform_single_image(color_path, depth_path)

            # Get model prediction
            output = model(color_tensor, depth_tensor)
            class_idx = output.argmax(dim=1).item()

            # heatmap_color, heatmap_depth = compute_gradcam(model, cam_color, color_tensor, depth_tensor, class_idx)
            heatmap_color = compute_gradcam(model, cam_color, color_tensor, depth_tensor, class_idx)
            heatmap_depth = compute_gradcam(model, cam_depth, color_tensor, depth_tensor, class_idx)

            fig, axes = plt.subplots(1, 2, figsize=(12, 8))
            
            def brighten_image(image, factor=2.5):
                bright = np.clip(image * factor, 0, 1)
                return bright
            temp_color = color_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # color_vis = brighten_image(temp_color)
            color_vis = temp_color

            axes[0].imshow(color_vis)  # Show color image
            axes[0].imshow(heatmap_color, cmap="jet", alpha=0.5)
            axes[0].set_title("Color Image", fontsize = 35)
            axes[0].axis('off')
            
            axes[1].imshow(depth_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())  # Show color image
            axes[1].imshow(heatmap_depth, cmap="jet", alpha=0.5)
            axes[1].set_title("Depth Image", fontsize = 35)
            axes[1].axis('off')

            plt.savefig(os.path.join(case_dir, f"{case}_gradcam_{i}.pdf"), bbox_inches="tight", dpi=300)
            plt.close()

from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
from skimage.segmentation import quickshift


explainer = LimeImageExplainer()

from lime import lime_image
from skimage.segmentation import quickshift, mark_boundaries
import matplotlib.pyplot as plt
import os
import torch
import numpy as np

def lime():
    explainer = LimeImageExplainer()

    for case, images in test_cases.items():
        case_dir = os.path.join(output_dir, case)
        os.makedirs(case_dir, exist_ok=True)

        for i, (color_path, depth_path) in enumerate(images):
            color_tensor, depth_tensor = transform_single_image(color_path, depth_path)
            
            # Convert to NumPy
            color_numpy = color_tensor.cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
            color_numpy = np.squeeze(color_numpy, axis=0)

            depth_numpy = depth_tensor.cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
            depth_numpy = np.squeeze(depth_numpy, axis=0)

            model.to(device)
            model.eval()

            # --- Color Predictor ---
            def predict_color_fn(color_images):
                color_tensor = torch.tensor(color_images).to(device).permute(0, 3, 1, 2).float()
                features = model.model1(color_tensor)
                return features.cpu().detach().numpy()

            # --- Depth Predictor ---
            def predict_depth_fn(depth_images):
                depth_tensor = torch.tensor(depth_images).to(device).permute(0, 3, 1, 2).float()
                features = model.model2(depth_tensor)
                return features.cpu().detach().numpy()

            # --- Explain Color ---
            explanation_color = explainer.explain_instance(
                color_numpy,
                predict_color_fn,
                top_labels=1,
                hide_color=0,
                num_samples=1000,
                segmentation_fn=lambda x: quickshift(x, kernel_size=4, max_dist=200, ratio=0.2)
            )
            temp_color, mask_color = explanation_color.get_image_and_mask(
                explanation_color.top_labels[0], positive_only=True, num_features=10, hide_rest=False
            )

            # --- Explain Depth ---
            explanation_depth = explainer.explain_instance(
                depth_numpy,
                predict_depth_fn,
                top_labels=1,
                hide_color=0,
                num_samples=1000,
                segmentation_fn=lambda x: quickshift(x, kernel_size=4, max_dist=200, ratio=0.2)
            )
            temp_depth, mask_depth = explanation_depth.get_image_and_mask(
                explanation_depth.top_labels[0], positive_only=True, num_features=10, hide_rest=False
            )

            # --- Plot Side-by-Side ---
            
            def brighten_image(image, factor=2.5):
                bright = np.clip(image * factor, 0, 1)
                return bright

            color_vis = brighten_image(temp_color)
            # depth_vis = brighten_image(temp_depth)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(mark_boundaries(color_vis, mask_color))
            axes[0].set_title("Color Image LIME Explanation")
            axes[0].axis('off')


            axes[1].imshow(mark_boundaries(temp_depth, mask_depth))
            axes[1].set_title("Depth Image LIME Explanation")
            axes[1].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(case_dir, f"lime_explanation_{i}.png"))
            plt.close()

        # plt.savefig(os.path.join(case_dir, f"lime_explanation_{i}.pdf"))
        # plt.close()

gradcam()
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")
# Number of trainable parameters: 53702826