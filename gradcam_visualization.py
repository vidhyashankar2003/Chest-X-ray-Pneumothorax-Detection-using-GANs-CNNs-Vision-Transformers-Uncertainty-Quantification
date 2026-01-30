"""
Grad-CAM Attention Visualization for Pneumothorax Detection
Shows WHERE the model looks when making predictions

Features:
1. Grad-CAM heatmaps overlaid on X-rays
2. Compare attention on real vs synthetic images
3. Clinical interpretability analysis
4. Generate publication-ready figures
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from PIL import Image
from torchvision import transforms
import argparse
from tqdm import tqdm

# Import your trained model
import sys

sys.path.append('.')
from train_simple_cnn import SimpleCNN
from pneumo_dataset_v2 import PneumoDataset


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping

    Shows which regions of the image are important for the model's decision
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Save forward pass activations"""
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        """Save backward pass gradients"""
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        """
        Generate Class Activation Map

        Args:
            input_image: (1, C, H, W) tensor
            target_class: Class to visualize (None = predicted class)

        Returns:
            cam: (H, W) numpy array with attention weights
        """
        # Forward pass
        model_output = self.model(input_image)

        if target_class is None:
            # Use predicted class
            target_class = (torch.sigmoid(model_output) > 0.5).long()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        model_output.backward(retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients[0].cpu().numpy()  # (C, H, W)
        activations = self.activations[0].cpu().numpy()  # (C, H, W)

        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU (only positive contributions)
        cam = np.maximum(cam, 0)

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def overlay_heatmap(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on original image

    Args:
        image: Original image (H, W, 3) in [0, 255]
        heatmap: Attention map (H, W) in [0, 1]
        alpha: Transparency of heatmap
        colormap: OpenCV colormap

    Returns:
        Overlayed image (H, W, 3)
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay
    overlayed = (1 - alpha) * image + alpha * heatmap_colored
    overlayed = overlayed.astype(np.uint8)

    return overlayed


def visualize_predictions(model, dataset, indices, output_dir, device):
    """
    Generate Grad-CAM visualizations for multiple images

    Args:
        model: Trained model
        dataset: Dataset to sample from
        indices: List of image indices to visualize
        output_dir: Where to save visualizations
        device: torch device
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get target layer (last conv layer before pooling)
    # For ResNet18: features[7][1] is the last conv block
    target_layer = model.features[7][1].conv2

    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)

    model.eval()

    print(f"\n🎨 Generating Grad-CAM visualizations...")

    for idx in tqdm(indices, desc="Visualizing"):
        # Get image
        img_tensor, label, filename = dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()
            pred = int(prob > 0.5)

        # Generate CAM
        cam = grad_cam.generate_cam(img_tensor)

        # Load original image for visualization
        img_path = Path(dataset.samples[idx][0])
        original_img = Image.open(img_path).convert('RGB')
        original_img = original_img.resize((224, 224))
        original_img_np = np.array(original_img)

        # Create overlay
        overlayed = overlay_heatmap(original_img_np, cam, alpha=0.4)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(original_img_np)
        axes[0].set_title('Original X-ray', fontsize=12)
        axes[0].axis('off')

        # Heatmap only
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Attention Map', fontsize=12)
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(overlayed)
        axes[2].set_title('Grad-CAM Overlay', fontsize=12)
        axes[2].axis('off')

        # Add prediction info
        true_label = "Pneumothorax" if label == 1 else "Healthy"
        pred_label = "Pneumothorax" if pred == 1 else "Healthy"
        is_correct = "✓" if pred == label else "✗"

        fig.suptitle(
            f'{filename}\n'
            f'True: {true_label} | Pred: {pred_label} ({prob:.3f}) {is_correct}',
            fontsize=14,
            fontweight='bold'
        )

        plt.tight_layout()

        # Save
        save_name = f"gradcam_{idx:04d}_{true_label.lower()}.png"
        plt.savefig(output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"✅ Saved visualizations to: {output_dir}")


def compare_real_vs_synthetic(model, real_dataset, synthetic_dir, output_dir, device, n_samples=5):
    """
    Compare attention patterns on real vs synthetic images

    Shows if model focuses on different regions for synthetic images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get target layer
    target_layer = model.features[7][1].conv2
    grad_cam = GradCAM(model, target_layer)

    model.eval()

    print(f"\n🔬 Comparing Real vs Synthetic attention patterns...")

    # Transform for images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Process real images
    real_cams = []
    for i in range(n_samples):
        img_tensor, label, _ = real_dataset[i]
        if label == 1:  # Pneumothorax
            img_tensor = img_tensor.unsqueeze(0).to(device)
            cam = grad_cam.generate_cam(img_tensor)
            real_cams.append(cam)

    # Process synthetic images
    synthetic_dir = Path(synthetic_dir)
    synthetic_files = list(synthetic_dir.glob("*.png"))[:n_samples]

    synthetic_cams = []
    for img_path in synthetic_files:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        cam = grad_cam.generate_cam(img_tensor)
        synthetic_cams.append(cam)

    # Create comparison figure
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))

    for i in range(n_samples):
        # Real
        if i < len(real_cams):
            axes[0, i].imshow(real_cams[i], cmap='jet')
            axes[0, i].set_title('Real Pneumothorax', fontsize=10)
            axes[0, i].axis('off')

        # Synthetic
        if i < len(synthetic_cams):
            axes[1, i].imshow(synthetic_cams[i], cmap='jet')
            axes[1, i].set_title('Synthetic Pneumothorax', fontsize=10)
            axes[1, i].axis('off')

    plt.suptitle('Attention Pattern Comparison: Real vs Synthetic', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_real_vs_synthetic.png", dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✅ Comparison saved to: {output_dir / 'comparison_real_vs_synthetic.png'}")

    # Statistical comparison
    avg_real = np.mean([cam for cam in real_cams], axis=0)
    avg_synthetic = np.mean([cam for cam in synthetic_cams], axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(avg_real, cmap='jet')
    axes[0].set_title('Average Attention - Real', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(avg_synthetic, cmap='jet')
    axes[1].set_title('Average Attention - Synthetic', fontsize=12)
    axes[1].axis('off')

    diff = np.abs(avg_real - avg_synthetic)
    axes[2].imshow(diff, cmap='RdYlGn_r')
    axes[2].set_title('Attention Difference', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "attention_statistics.png", dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✅ Statistics saved to: {output_dir / 'attention_statistics.png'}")


def generate_publication_figure(model, dataset, indices, output_path, device):
    """
    Generate publication-ready figure with multiple examples

    Shows model interpretability for paper/presentation
    """
    target_layer = model.features[7][1].conv2
    grad_cam = GradCAM(model, target_layer)
    model.eval()

    n_examples = len(indices)
    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 4 * n_examples))

    if n_examples == 1:
        axes = axes.reshape(1, -1)

    print(f"\n📄 Generating publication figure...")

    for row, idx in enumerate(indices):
        # Get image
        img_tensor, label, filename = dataset[idx]
        img_tensor_input = img_tensor.unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            output = model(img_tensor_input)
            prob = torch.sigmoid(output).item()
            pred = int(prob > 0.5)

        # Generate CAM
        cam = grad_cam.generate_cam(img_tensor_input)

        # Load original
        img_path = Path(dataset.samples[idx][0])
        original_img = Image.open(img_path).convert('RGB')
        original_img = original_img.resize((224, 224))
        original_img_np = np.array(original_img)

        # Overlay
        overlayed = overlay_heatmap(original_img_np, cam, alpha=0.5)

        # Plot
        axes[row, 0].imshow(original_img_np)
        axes[row, 0].axis('off')
        if row == 0:
            axes[row, 0].set_title('Original X-ray', fontsize=14, fontweight='bold')

        axes[row, 1].imshow(cam, cmap='jet')
        axes[row, 1].axis('off')
        if row == 0:
            axes[row, 1].set_title('Attention Heatmap', fontsize=14, fontweight='bold')

        axes[row, 2].imshow(overlayed)
        axes[row, 2].axis('off')
        if row == 0:
            axes[row, 2].set_title('Overlay', fontsize=14, fontweight='bold')

        # Label
        true_label = "Pneumothorax" if label == 1 else "Healthy"
        pred_label = "Pneumothorax" if pred == 1 else "Healthy"

        axes[row, 0].text(
            -0.15, 0.5, f'Case {row + 1}\n{true_label}\nConf: {prob:.2f}',
            transform=axes[row, 0].transAxes,
            fontsize=11,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Publication figure saved: {output_path}")


def main(args):
    """Main execution"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Convert output_dir to Path object
    args.output_dir = Path(args.output_dir)

    # Load trained model
    print(f"\n📂 Loading trained model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = SimpleCNN(dropout=0.5).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Model loaded (trained for {checkpoint['epoch']} epochs)")

    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = PneumoDataset(
        csv_path=args.csv,
        images_dir=args.images,
        list_txt=args.val_list,
        synthetic_dir=None,
        img_size=224,
        transform=transform
    )

    print(f"📊 Dataset loaded: {len(dataset)} images")

    # Generate visualizations
    if args.mode == 'individual':
        # Visualize specific indices
        indices = args.indices if args.indices else list(range(min(10, len(dataset))))
        visualize_predictions(model, dataset, indices, args.output_dir, device)

    elif args.mode == 'comparison':
        # Compare real vs synthetic
        compare_real_vs_synthetic(
            model, dataset, args.synthetic_dir,
            args.output_dir, device, n_samples=5
        )

    elif args.mode == 'publication':
        # Publication-ready figure
        indices = args.indices if args.indices else [0, 10, 20, 30]
        output_path = Path(args.output_dir) / "publication_figure.png"
        generate_publication_figure(model, dataset, indices, output_path, device)

    elif args.mode == 'all':
        # Generate everything
        print("\n🎨 Generating all visualizations...\n")

        # Individual
        indices = list(range(min(20, len(dataset))))
        visualize_predictions(model, dataset, indices, args.output_dir / "individual", device)

        # Comparison
        if args.synthetic_dir:
            compare_real_vs_synthetic(
                model, dataset, args.synthetic_dir,
                args.output_dir / "comparison", device, n_samples=5
            )

        # Publication
        pub_indices = [0, 5, 10, 15]
        generate_publication_figure(
            model, dataset, pub_indices,
            args.output_dir / "publication_figure.png", device
        )

    print(f"\n{'=' * 70}")
    print(f"✅ Visualization complete!")
    print(f"   Output directory: {args.output_dir}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM Attention Visualization")

    # Model and data
    parser.add_argument("--checkpoint", default="checkpoints/cnn_pneumo/cnn_pneumo_best.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--csv", default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/Data_Entry_2017_v2020.csv")
    parser.add_argument("--images", default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/images")
    parser.add_argument("--val_list",
                        default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/val_BALANCED.txt")
    parser.add_argument("--synthetic_dir", default="generated_images/synthetic_pneumothorax")

    # Visualization options
    parser.add_argument("--mode", choices=['individual', 'comparison', 'publication', 'all'],
                        default='all', help="Visualization mode")
    parser.add_argument("--indices", nargs='+', type=int, default=None,
                        help="Specific image indices to visualize")
    parser.add_argument("--output_dir", default="visualizations/gradcam",
                        help="Output directory for visualizations")

    args = parser.parse_args()
    main(args)