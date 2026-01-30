"""
Uncertainty Quantification for Pneumothorax Detection
Uses Monte Carlo Dropout to estimate prediction confidence

Novel Contribution:
- Provides uncertainty estimates alongside predictions
- Enables doctors to know when model is unsure
- Critical for clinical AI deployment
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import pandas as pd
import seaborn as sns

from train_simple_cnn import SimpleCNN
from pneumo_dataset_v2 import PneumoDataset


class UncertaintyEstimator:
    """
    Monte Carlo Dropout for uncertainty estimation

    Runs model multiple times with dropout enabled to get prediction distribution
    """

    def __init__(self, model, n_samples=30):
        self.model = model
        self.n_samples = n_samples

    def enable_dropout(self):
        """Enable dropout during inference"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict_with_uncertainty(self, x):
        """
        Get prediction with uncertainty estimate

        Args:
            x: Input image tensor (1, C, H, W)

        Returns:
            mean_prob: Mean predicted probability
            std: Standard deviation (uncertainty)
            predictions: All individual predictions
        """
        self.model.eval()  # Set to eval mode first
        self.enable_dropout()  # But keep dropout active

        predictions = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                output = self.model(x)
                prob = torch.sigmoid(output).cpu().item()
                predictions.append(prob)

        predictions = np.array(predictions)
        mean_prob = predictions.mean()
        std = predictions.std()

        return mean_prob, std, predictions


def visualize_uncertainty(mean_prob, std, predictions, true_label, filename, save_path):
    """
    Visualize prediction distribution

    Shows histogram of predictions from Monte Carlo dropout
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of predictions
    axes[0].hist(predictions, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0].axvline(mean_prob, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_prob:.3f}')
    axes[0].axvline(0.5, color='green', linestyle=':', linewidth=2,
                    label='Threshold: 0.5')
    axes[0].set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Prediction Distribution (MC Dropout)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # Confidence visualization
    confidence = 1 - std
    pred_label = "Pneumothorax" if mean_prob > 0.5 else "Healthy"
    true_str = "Pneumothorax" if true_label == 1 else "Healthy"
    is_correct = (mean_prob > 0.5) == (true_label == 1)
    status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
    status_color = 'green' if is_correct else 'red'

    # Bar plot
    bars = axes[1].barh(['Confidence', 'Uncertainty'], [confidence, std],
                        color=['green', 'red'], alpha=0.7, edgecolor='black')
    axes[1].set_xlim([0, 1])
    axes[1].set_xlabel('Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Uncertainty Metrics', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='x')

    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        axes[1].text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}', ha='left', va='center', fontsize=11, fontweight='bold')

    # Add prediction info
    info_text = f'Prediction: {pred_label}\n'
    info_text += f'True Label: {true_str}\n'
    info_text += f'Status: {status}\n'
    info_text += f'Confidence: {confidence:.3f}'

    axes[1].text(0.5, -0.3, info_text, fontsize=11, ha='center',
                 transform=axes[1].transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Uncertainty Analysis: {filename[:40]}\n'
                 f'Mean Probability: {mean_prob:.3f} ± {std:.3f}',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def analyze_dataset_uncertainty(model, dataset, device, output_dir, n_images=50):
    """
    Analyze uncertainty across entire dataset

    Shows which cases have high/low uncertainty
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    estimator = UncertaintyEstimator(model, n_samples=30)

    results = []

    print(f"\n📊 Analyzing uncertainty for {n_images} images...")

    for idx in tqdm(range(min(n_images, len(dataset)))):
        img_tensor, label, filename = dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        mean_prob, std, predictions = estimator.predict_with_uncertainty(img_tensor)

        pred_label = int(mean_prob > 0.5)
        is_correct = (pred_label == label)

        results.append({
            'idx': idx,
            'filename': filename,
            'true_label': label,
            'mean_prob': mean_prob,
            'std': std,
            'confidence': 1 - std,
            'correct': is_correct,
            'pred_label': pred_label
        })

    # Create summary visualizations
    print(f"\n📈 Creating summary visualizations...")

    # 1. Uncertainty vs Correctness
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    correct_results = [r for r in results if r['correct']]
    incorrect_results = [r for r in results if not r['correct']]

    # Uncertainty distribution
    correct_std = [r['std'] for r in correct_results]
    incorrect_std = [r['std'] for r in incorrect_results]

    axes[0, 0].hist(correct_std, bins=15, alpha=0.7, label='Correct',
                    color='green', edgecolor='black')
    axes[0, 0].hist(incorrect_std, bins=15, alpha=0.7, label='Incorrect',
                    color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Uncertainty (Std Dev)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Uncertainty Distribution by Correctness',
                         fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(alpha=0.3)

    # Mean uncertainty comparison
    if correct_std and incorrect_std:
        mean_correct = np.mean(correct_std)
        mean_incorrect = np.mean(incorrect_std)
        axes[0, 0].axvline(mean_correct, color='darkgreen', linestyle='--',
                           linewidth=2, alpha=0.7)
        axes[0, 0].axvline(mean_incorrect, color='darkred', linestyle='--',
                           linewidth=2, alpha=0.7)

    # Confidence vs Probability
    confidences = [r['confidence'] for r in results]
    probs = [r['mean_prob'] for r in results]
    colors = ['green' if r['correct'] else 'red' for r in results]

    axes[0, 1].scatter(probs, confidences, c=colors, alpha=0.6, s=50, edgecolors='black')
    axes[0, 1].axvline(0.5, color='black', linestyle='--', alpha=0.5, linewidth=2)
    axes[0, 1].set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Confidence (1 - Std)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Confidence vs Prediction', fontsize=13, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].set_ylim([0, 1])

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='Correct'),
                       Patch(facecolor='red', label='Incorrect')]
    axes[0, 1].legend(handles=legend_elements, fontsize=11)

    # Average metrics
    avg_correct_conf = np.mean([r['confidence'] for r in correct_results]) if correct_results else 0
    avg_incorrect_conf = np.mean([r['confidence'] for r in incorrect_results]) if incorrect_results else 0

    bars = axes[1, 0].bar(['Correct\nPredictions', 'Incorrect\nPredictions'],
                          [avg_correct_conf, avg_incorrect_conf],
                          color=['green', 'red'], alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('Average Confidence', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Confidence by Prediction Correctness',
                         fontsize=13, fontweight='bold')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom',
                        fontsize=11, fontweight='bold')

    # Statistics panel
    stats_text = f"""UNCERTAINTY ANALYSIS STATISTICS

Total Images Analyzed: {len(results)}
Correct Predictions: {len(correct_results)} ({len(correct_results) / len(results) * 100:.1f}%)
Incorrect Predictions: {len(incorrect_results)} ({len(incorrect_results) / len(results) * 100:.1f}%)

CONFIDENCE METRICS:
  Avg Confidence (Correct): {avg_correct_conf:.3f}
  Avg Confidence (Incorrect): {avg_incorrect_conf:.3f}
  Difference: {abs(avg_correct_conf - avg_incorrect_conf):.3f}

UNCERTAINTY METRICS:
  Mean Uncertainty (Correct): {np.mean(correct_std) if correct_std else 0:.3f}
  Mean Uncertainty (Incorrect): {np.mean(incorrect_std) if incorrect_std else 0:.3f}

INTERPRETATION:
  {"✅ Model is MORE confident on correct predictions" if avg_correct_conf > avg_incorrect_conf else "⚠️ Model is LESS confident on correct predictions"}
  {"✅ Good calibration - uncertainty signals errors" if (np.mean(incorrect_std) > np.mean(correct_std) if incorrect_std and correct_std else False) else "⚠️ Poor calibration - check model"}
"""

    axes[1, 1].text(0.05, 0.95, stats_text, fontsize=10,
                    verticalalignment='top', family='monospace',
                    transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    axes[1, 1].axis('off')

    plt.suptitle('Uncertainty Quantification Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "uncertainty_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Summary saved: {output_dir / 'uncertainty_summary.png'}")

    # Save detailed results to CSV
    df_results = pd.DataFrame(results)
    csv_path = output_dir / "uncertainty_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"✅ Detailed results saved: {csv_path}")

    # Print top uncertain cases
    sorted_results = sorted(results, key=lambda x: x['std'], reverse=True)

    print(f"\n⚠️  Top 10 Most Uncertain Cases:")
    print(f"{'#':<4} {'Filename':<35} {'True':<8} {'Pred':<8} {'Prob':<12} {'Uncertainty':<12} {'Status'}")
    print(f"{'-' * 95}")

    for i, r in enumerate(sorted_results[:10]):
        status = "✓" if r['correct'] else "✗"
        true_str = "Pneumo" if r['true_label'] == 1 else "Healthy"
        pred_str = "Pneumo" if r['pred_label'] == 1 else "Healthy"

        print(f"{i + 1:<4} {r['filename'][:35]:<35} {true_str:<8} {pred_str:<8} "
              f"{r['mean_prob']:.3f}±{r['std']:.3f}  {r['std']:.4f}      {status}")

    return results


def create_uncertainty_heatmap(results, output_dir):
    """Create heatmap showing uncertainty patterns"""

    output_dir = Path(output_dir)

    # Create bins for probability and uncertainty
    prob_bins = np.linspace(0, 1, 11)
    uncertainty_bins = np.linspace(0, 0.3, 11)

    # Create 2D histogram
    probs = [r['mean_prob'] for r in results]
    uncertainties = [r['std'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 8))

    hist, xedges, yedges = np.histogram2d(probs, uncertainties,
                                          bins=[prob_bins, uncertainty_bins])

    im = ax.imshow(hist.T, origin='lower', aspect='auto', cmap='YlOrRd',
                   extent=[prob_bins[0], prob_bins[-1],
                           uncertainty_bins[0], uncertainty_bins[-1]])

    ax.set_xlabel('Predicted Probability', fontsize=13, fontweight='bold')
    ax.set_ylabel('Uncertainty (Std Dev)', fontsize=13, fontweight='bold')
    ax.set_title('Uncertainty Heatmap\n(Darker = More samples)',
                 fontsize=15, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=12, fontweight='bold')

    # Add threshold line
    ax.axvline(0.5, color='white', linestyle='--', linewidth=2, alpha=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / "uncertainty_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Heatmap saved: {output_dir / 'uncertainty_heatmap.png'}")


def compare_uncertainty_by_label(results, output_dir):
    """Compare uncertainty for different true labels"""

    output_dir = Path(output_dir)

    healthy_results = [r for r in results if r['true_label'] == 0]
    pneumo_results = [r for r in results if r['true_label'] == 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Uncertainty by true label
    healthy_std = [r['std'] for r in healthy_results]
    pneumo_std = [r['std'] for r in pneumo_results]

    axes[0].hist(healthy_std, bins=15, alpha=0.7, label='Healthy',
                 color='blue', edgecolor='black')
    axes[0].hist(pneumo_std, bins=15, alpha=0.7, label='Pneumothorax',
                 color='orange', edgecolor='black')
    axes[0].set_xlabel('Uncertainty (Std Dev)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Uncertainty by True Label', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Box plot comparison
    data_to_plot = [healthy_std, pneumo_std]
    bp = axes[1].boxplot(data_to_plot, labels=['Healthy', 'Pneumothorax'],
                         patch_artist=True, widths=0.6)

    # Color boxes
    colors = ['lightblue', 'lightsalmon']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1].set_ylabel('Uncertainty (Std Dev)', fontsize=12, fontweight='bold')
    axes[1].set_title('Uncertainty Distribution by True Label',
                      fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')

    plt.suptitle('Uncertainty Analysis by Disease Status',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "uncertainty_by_label.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Label comparison saved: {output_dir / 'uncertainty_by_label.png'}")


def main(args):
    """Main execution"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 70}")
    print(f"🎯 Uncertainty Quantification for Pneumothorax Detection")
    print(f"   Using Monte Carlo Dropout (n={args.n_forward_passes} passes)")
    print(f"{'=' * 70}\n")
    print(f"Device: {device}")

    # Load model
    print(f"\n📂 Loading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = SimpleCNN(dropout=0.5).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"✅ Model loaded successfully")
    print(f"   Trained for {checkpoint['epoch']} epochs")
    if 'metrics' in checkpoint:
        print(f"   F1-Score: {checkpoint['metrics']['f1']:.4f}")
        print(f"   AUC: {checkpoint['metrics'].get('auc', 'N/A')}")

    # Load dataset
    print(f"\n📂 Loading validation dataset...")
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

    print(f"✅ Dataset loaded: {len(dataset)} images")

    # Analyze uncertainty
    results = analyze_dataset_uncertainty(
        model, dataset, device,
        args.output_dir, n_images=args.n_samples
    )

    # Create additional visualizations
    print(f"\n📊 Creating additional analyses...")
    create_uncertainty_heatmap(results, args.output_dir)
    compare_uncertainty_by_label(results, args.output_dir)

    # Generate individual visualizations for most uncertain cases
    estimator = UncertaintyEstimator(model, n_samples=args.n_forward_passes)

    print(f"\n🎨 Generating individual uncertainty visualizations...")

    # Sort by uncertainty
    sorted_results = sorted(results, key=lambda x: x['std'], reverse=True)

    # Create subdirectory for individual cases
    individual_dir = Path(args.output_dir) / "individual_cases"
    individual_dir.mkdir(exist_ok=True)

    for i, result in enumerate(tqdm(sorted_results[:args.n_visualize],
                                    desc="Visualizing uncertain cases")):
        idx = result['idx']
        img_tensor, label, filename = dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        mean_prob, std, predictions = estimator.predict_with_uncertainty(img_tensor)

        # Clean filename for saving
        clean_filename = filename.replace('.png', '').replace('/', '_')
        save_path = individual_dir / f"case_{i + 1:02d}_{clean_filename}.png"

        visualize_uncertainty(mean_prob, std, predictions, label, filename, save_path)

    print(f"✅ Individual cases saved to: {individual_dir}")

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"✅ Uncertainty Analysis Complete!")
    print(f"{'=' * 70}")
    print(f"\n📁 Output Directory: {args.output_dir}")
    print(f"\n📊 Generated Files:")
    print(f"   1. uncertainty_summary.png - Overall statistics")
    print(f"   2. uncertainty_heatmap.png - Probability vs uncertainty")
    print(f"   3. uncertainty_by_label.png - Comparison by disease status")
    print(f"   4. uncertainty_results.csv - Detailed numerical results")
    print(f"   5. individual_cases/ - Top {args.n_visualize} uncertain cases")

    # Key insights
    print(f"\n💡 Key Insights:")
    correct_results = [r for r in results if r['correct']]
    incorrect_results = [r for r in results if not r['correct']]

    if correct_results and incorrect_results:
        avg_correct_unc = np.mean([r['std'] for r in correct_results])
        avg_incorrect_unc = np.mean([r['std'] for r in incorrect_results])

        print(f"   • Average uncertainty on CORRECT predictions: {avg_correct_unc:.4f}")
        print(f"   • Average uncertainty on INCORRECT predictions: {avg_incorrect_unc:.4f}")

        if avg_incorrect_unc > avg_correct_unc:
            diff = (avg_incorrect_unc - avg_correct_unc) / avg_correct_unc * 100
            print(f"   ✅ Model is {diff:.1f}% MORE uncertain on incorrect predictions")
            print(f"   → Good calibration! High uncertainty signals potential errors.")
        else:
            diff = (avg_correct_unc - avg_incorrect_unc) / avg_incorrect_unc * 100
            print(f"   ⚠️  Model is {diff:.1f}% MORE uncertain on correct predictions")
            print(f"   → Poor calibration. Consider recalibration techniques.")

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Uncertainty Quantification using Monte Carlo Dropout"
    )

    # Model and data paths
    parser.add_argument("--checkpoint",
                        default="checkpoints/cnn_pneumo/cnn_pneumo_best.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--csv",
                        default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/Data_Entry_2017_v2020.csv",
                        help="Path to CSV with image labels")
    parser.add_argument("--images",
                        default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/images",
                        help="Directory containing images")
    parser.add_argument("--val_list",
                        default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/val_BALANCED.txt",
                        help="Path to validation split file")

    # Uncertainty quantification parameters
    parser.add_argument("--n_forward_passes", type=int, default=30,
                        help="Number of forward passes for MC Dropout (default: 30)")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of images to analyze (default: 100)")
    parser.add_argument("--n_visualize", type=int, default=10,
                        help="Number of most uncertain cases to visualize (default: 10)")

    # Output
    parser.add_argument("--output_dir",
                        default="visualizations/uncertainty",
                        help="Output directory for visualizations")

    args = parser.parse_args()
    main(args)