"""
Visualize Model Metrics with Charts
Creates bar charts and comparison plots for ViT and CNN performance
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_vit_metrics(checkpoint_path="checkpoints/vit/vit_best.pth", save_path="vit_metrics.png"):
    """Plot ViT classifier metrics as bar chart"""

    if not Path(checkpoint_path).exists():
        print(f"❌ ViT checkpoint not found: {checkpoint_path}")
        return

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract metrics
    metrics = {
        'AUC': checkpoint['auc'],
        'Accuracy': checkpoint['metrics']['accuracy'],
        'Precision': checkpoint['metrics']['precision'],
        'Recall': checkpoint['metrics']['recall'],
        'F1-Score': checkpoint['metrics']['f1']
    }

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(metrics.keys(), metrics.values(), color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Vision Transformer Performance (Pneumothorax Detection)\nTrained for {checkpoint["epoch"]} epochs',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ ViT metrics chart saved: {save_path}")

    return fig


def plot_cnn_metrics(checkpoint_path="checkpoints/cnn/cnn_best.pth", save_path="cnn_metrics.png"):
    """Plot CNN classifier metrics as bar chart"""

    if not Path(checkpoint_path).exists():
        print(f"❌ CNN checkpoint not found: {checkpoint_path}")
        return

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract metrics
    metrics = {
        'Accuracy': checkpoint['accuracy'],
        'Precision': checkpoint['metrics']['precision'],
        'Recall': checkpoint['metrics']['recall'],
        'F1-Score': checkpoint['metrics']['f1'],
        'AUC': checkpoint['metrics']['auc']
    }

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(metrics.keys(), metrics.values(), color=['#1abc9c', '#16a085', '#27ae60', '#2980b9', '#8e44ad'])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'CNN Classifier Performance (Real vs Synthetic Detection)\nTrained for {checkpoint["epoch"]} epochs',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    # Add interpretation text
    acc = checkpoint['accuracy']
    if acc >= 0.95:
        quality = "Synthetic images easily detectable (not ideal)"
        color = 'red'
    elif acc >= 0.85:
        quality = "Synthetic images somewhat realistic (acceptable)"
        color = 'orange'
    else:
        quality = "Synthetic images very realistic (excellent!)"
        color = 'green'

    ax.text(0.5, 0.95, quality, transform=ax.transAxes,
            ha='center', va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ CNN metrics chart saved: {save_path}")

    return fig


def plot_comparison(vit_path="checkpoints/vit/vit_best.pth",
                    cnn_path="checkpoints/cnn/cnn_best.pth",
                    save_path="model_comparison.png"):
    """Compare ViT and CNN metrics side by side"""

    if not Path(vit_path).exists() or not Path(cnn_path).exists():
        print("❌ One or both checkpoints not found")
        return

    # Load checkpoints
    vit_ckpt = torch.load(vit_path, map_location='cpu')
    cnn_ckpt = torch.load(cnn_path, map_location='cpu')

    # Extract common metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    vit_values = [
        vit_ckpt['metrics']['accuracy'],
        vit_ckpt['metrics']['precision'],
        vit_ckpt['metrics']['recall'],
        vit_ckpt['metrics']['f1']
    ]
    cnn_values = [
        cnn_ckpt['accuracy'],
        cnn_ckpt['metrics']['precision'],
        cnn_ckpt['metrics']['recall'],
        cnn_ckpt['metrics']['f1']
    ]

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width / 2, vit_values, width, label='ViT (Pneumothorax Detection)', color='#3498db')
    bars2 = ax.bar(x + width / 2, cnn_values, width, label='CNN (Real vs Synthetic)', color='#e74c3c')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison chart saved: {save_path}")

    return fig


def create_summary_report(output_file="model_report.txt"):
    """Create a text report with all metrics"""

    report = []
    report.append("=" * 70)
    report.append("MODEL PERFORMANCE REPORT")
    report.append("=" * 70)
    report.append("")

    # CycleGAN section
    report.append("1. CYCLEGAN (Image Generation)")
    report.append("-" * 70)
    g_ab_files = list(Path("checkpoints").glob("G_AB_epoch*.pth"))
    if g_ab_files:
        max_epoch = max([int(f.stem.split("epoch")[1]) for f in g_ab_files])
        report.append(f"   Status: ✅ Trained")
        report.append(f"   Epochs: {max_epoch}")
        report.append(f"   Checkpoints: {len(g_ab_files)}")
        report.append(f"   Location: checkpoints/")
    else:
        report.append(f"   Status: ❌ Not trained")
    report.append("")

    # ViT section
    report.append("2. VISION TRANSFORMER (Pneumothorax Detection)")
    report.append("-" * 70)
    vit_path = Path("checkpoints/vit/vit_best.pth")
    if vit_path.exists():
        ckpt = torch.load(vit_path, map_location='cpu')
        report.append(f"   Status: ✅ Trained")
        report.append(f"   Best Epoch: {ckpt['epoch']}")
        report.append(f"   AUC: {ckpt['auc']:.4f}")
        report.append(f"   Accuracy: {ckpt['metrics']['accuracy']:.4f}")
        report.append(f"   Precision: {ckpt['metrics']['precision']:.4f}")
        report.append(f"   Recall: {ckpt['metrics']['recall']:.4f}")
        report.append(f"   F1-Score: {ckpt['metrics']['f1']:.4f}")
    else:
        report.append(f"   Status: ❌ Not trained")
    report.append("")

    # CNN section
    report.append("3. CNN CLASSIFIER (Real vs Synthetic Detection)")
    report.append("-" * 70)
    cnn_path = Path("checkpoints/cnn/cnn_best.pth")
    if cnn_path.exists():
        ckpt = torch.load(cnn_path, map_location='cpu')
        report.append(f"   Status: ✅ Trained")
        report.append(f"   Best Epoch: {ckpt['epoch']}")
        report.append(f"   Accuracy: {ckpt['accuracy']:.4f}")
        report.append(f"   Precision: {ckpt['metrics']['precision']:.4f}")
        report.append(f"   Recall: {ckpt['metrics']['recall']:.4f}")
        report.append(f"   F1-Score: {ckpt['metrics']['f1']:.4f}")
        report.append(f"   AUC: {ckpt['metrics']['auc']:.4f}")

        # Interpretation
        acc = ckpt['accuracy']
        report.append("")
        report.append("   Interpretation:")
        if acc >= 0.95:
            report.append("   ⚠️  Synthetic images are easily detectable (CNN accuracy ≥95%)")
            report.append("       → May not be ideal for data augmentation")
        elif acc >= 0.85:
            report.append("   ✅ Synthetic images are somewhat realistic (CNN accuracy 85-95%)")
            report.append("       → Acceptable for data augmentation")
        else:
            report.append("   ✅ Synthetic images are very realistic (CNN accuracy <85%)")
            report.append("       → Excellent for data augmentation!")
    else:
        report.append(f"   Status: ❌ Not trained")

    report.append("")
    report.append("=" * 70)

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"✅ Text report saved: {output_file}")

    # Also print to console
    print('\n'.join(report))


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("📊 MODEL METRICS VISUALIZATION")
    print("=" * 70 + "\n")

    # Create visualizations
    plot_vit_metrics()
    plot_cnn_metrics()
    plot_comparison()

    # Create text report
    create_summary_report()

    print("\n" + "=" * 70)
    print("✅ Visualization complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("   📊 vit_metrics.png - ViT performance chart")
    print("   📊 cnn_metrics.png - CNN performance chart")
    print("   📊 model_comparison.png - Side-by-side comparison")
    print("   📄 model_report.txt - Text summary report")
    print()