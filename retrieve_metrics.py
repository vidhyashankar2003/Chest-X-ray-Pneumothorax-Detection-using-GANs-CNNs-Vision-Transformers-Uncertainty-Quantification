"""
Retrieve and Display Metrics from All Trained Models
Shows performance metrics for CycleGAN, ViT, and CNN classifiers
"""
import torch
import json
from pathlib import Path
import pandas as pd
from tabulate import tabulate


def get_cyclegan_metrics(checkpoint_dir="checkpoints"):
    """
    Retrieve CycleGAN training metrics
    Note: CycleGAN doesn't have validation metrics, only training losses
    """
    checkpoint_dir = Path(checkpoint_dir)

    print("\n" + "=" * 70)
    print("🔄 CycleGAN Training Metrics")
    print("=" * 70)

    # Find all generator checkpoints
    g_ab_files = sorted(checkpoint_dir.glob("G_AB_epoch*.pth"))

    if not g_ab_files:
        print("❌ No CycleGAN checkpoints found!")
        return None

    print(f"Found {len(g_ab_files)} checkpoints\n")

    # For CycleGAN, we can only show which epochs were saved
    epochs_saved = []
    for file in g_ab_files:
        epoch_num = file.stem.split("epoch")[1]
        file_size_mb = file.stat().st_size / (1024 * 1024)
        epochs_saved.append({
            'Epoch': int(epoch_num),
            'Generator File': file.name,
            'Size (MB)': f"{file_size_mb:.1f}"
        })

    df = pd.DataFrame(epochs_saved)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

    # Recommend which epoch to use for generation
    final_epoch = max([int(f.stem.split("epoch")[1]) for f in g_ab_files])
    print(f"\n💡 Recommendation: Use epoch {final_epoch} for generating synthetic images")
    print(f"   Command: python generator.py --epoch {final_epoch}")

    return epochs_saved


def get_vit_metrics(checkpoint_path="checkpoints/vit/vit_best.pth"):
    """Retrieve ViT classifier metrics"""
    checkpoint_path = Path(checkpoint_path)

    print("\n" + "=" * 70)
    print("🧠 Vision Transformer (ViT) Metrics")
    print("=" * 70)

    if not checkpoint_path.exists():
        print(f"❌ ViT checkpoint not found: {checkpoint_path}")
        print("   Run: python train_VIT.py")
        return None

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract metrics
    metrics = {
        'Model': 'Vision Transformer',
        'Task': 'Pneumothorax Detection',
        'Best Epoch': checkpoint['epoch'],
        'AUC': f"{checkpoint['auc']:.4f}",
        'Accuracy': f"{checkpoint['metrics']['accuracy']:.4f}",
        'Precision': f"{checkpoint['metrics']['precision']:.4f}",
        'Recall': f"{checkpoint['metrics']['recall']:.4f}",
        'F1-Score': f"{checkpoint['metrics']['f1']:.4f}"
    }

    # Display as table
    df = pd.DataFrame([metrics])
    print(tabulate(df.T, tablefmt='grid', headers=['Metric', 'Value']))

    # Interpretation
    auc = checkpoint['auc']
    print(f"\n📊 Performance Interpretation:")
    if auc >= 0.85:
        print(f"   ✅ Excellent performance (AUC ≥ 0.85)")
    elif auc >= 0.75:
        print(f"   ✅ Good performance (AUC ≥ 0.75)")
    elif auc >= 0.65:
        print(f"   ⚠️  Fair performance (AUC ≥ 0.65)")
    else:
        print(f"   ❌ Poor performance (AUC < 0.65)")

    print(f"\n💡 This model classifies chest X-rays as:")
    print(f"   Class 0: No Pneumothorax (healthy)")
    print(f"   Class 1: Pneumothorax present (disease)")

    return metrics


def get_cnn_metrics(checkpoint_path="checkpoints/cnn/cnn_best.pth"):
    """Retrieve CNN classifier metrics"""
    checkpoint_path = Path(checkpoint_path)

    print("\n" + "=" * 70)
    print("🤖 CNN Classifier Metrics")
    print("=" * 70)

    if not checkpoint_path.exists():
        print(f"❌ CNN checkpoint not found: {checkpoint_path}")
        print("   Run: python CNN_Model.py")
        return None

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract metrics
    metrics = {
        'Model': 'CNN Classifier',
        'Task': 'Real vs Synthetic Detection',
        'Best Epoch': checkpoint['epoch'],
        'Accuracy': f"{checkpoint['accuracy']:.4f}",
        'Precision': f"{checkpoint['metrics']['precision']:.4f}",
        'Recall': f"{checkpoint['metrics']['recall']:.4f}",
        'F1-Score': f"{checkpoint['metrics']['f1']:.4f}",
        'AUC': f"{checkpoint['metrics']['auc']:.4f}"
    }

    # Display as table
    df = pd.DataFrame([metrics])
    print(tabulate(df.T, tablefmt='grid', headers=['Metric', 'Value']))

    # Interpretation
    accuracy = checkpoint['accuracy']
    print(f"\n📊 Performance Interpretation:")
    if accuracy >= 0.95:
        print(f"   ⚠️  Very high accuracy (≥95%): Synthetic images are easily detectable")
        print(f"      → Not ideal for data augmentation")
    elif accuracy >= 0.85:
        print(f"   ✅ Good balance (85-95%): Synthetic images are somewhat realistic")
        print(f"      → Acceptable for data augmentation")
    else:
        print(f"   ✅ Excellent! (< 85%): Synthetic images are very realistic")
        print(f"      → Great for data augmentation")

    print(f"\n💡 This model classifies images as:")
    print(f"   Class 0: Real X-ray")
    print(f"   Class 1: Synthetic (GAN-generated)")

    return metrics


def compare_with_without_synthetic(vit_without_path=None, vit_with_path="checkpoints/vit/vit_best.pth"):
    """
    Compare ViT performance with and without synthetic data
    (Only works if you trained ViT twice: once without synthetic, once with)
    """
    print("\n" + "=" * 70)
    print("📊 Impact of Synthetic Data on ViT Performance")
    print("=" * 70)

    if vit_without_path is None:
        print("\n⚠️  To see the impact of synthetic data, you need to:")
        print("   1. Train ViT WITHOUT synthetic images:")
        print("      python train_VIT.py --synthetic \"\" --ckpt_dir checkpoints/vit_no_synthetic")
        print("   2. Train ViT WITH synthetic images:")
        print("      python train_VIT.py --synthetic generated_images/synthetic_pneumothorax")
        print("   3. Run this script again with both checkpoints")
        return

    # Load both checkpoints
    try:
        ckpt_without = torch.load(vit_without_path, map_location='cpu')
        ckpt_with = torch.load(vit_with_path, map_location='cpu')
    except:
        print("❌ Could not load one or both checkpoints")
        return

    # Compare metrics
    comparison = {
        'Metric': ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Without Synthetic': [
            f"{ckpt_without['auc']:.4f}",
            f"{ckpt_without['metrics']['accuracy']:.4f}",
            f"{ckpt_without['metrics']['precision']:.4f}",
            f"{ckpt_without['metrics']['recall']:.4f}",
            f"{ckpt_without['metrics']['f1']:.4f}"
        ],
        'With Synthetic': [
            f"{ckpt_with['auc']:.4f}",
            f"{ckpt_with['metrics']['accuracy']:.4f}",
            f"{ckpt_with['metrics']['precision']:.4f}",
            f"{ckpt_with['metrics']['recall']:.4f}",
            f"{ckpt_with['metrics']['f1']:.4f}"
        ],
        'Improvement': [
            f"{(ckpt_with['auc'] - ckpt_without['auc']):.4f}",
            f"{(ckpt_with['metrics']['accuracy'] - ckpt_without['metrics']['accuracy']):.4f}",
            f"{(ckpt_with['metrics']['precision'] - ckpt_without['metrics']['precision']):.4f}",
            f"{(ckpt_with['metrics']['recall'] - ckpt_without['metrics']['recall']):.4f}",
            f"{(ckpt_with['metrics']['f1'] - ckpt_without['metrics']['f1']):.4f}"
        ]
    }

    df = pd.DataFrame(comparison)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

    # Overall conclusion
    auc_improvement = ckpt_with['auc'] - ckpt_without['auc']
    print(f"\n🎯 Overall Impact:")
    if auc_improvement > 0.02:
        print(f"   ✅ Positive impact! AUC improved by {auc_improvement:.4f}")
        print(f"      → Synthetic data helped improve the model")
    elif auc_improvement > -0.02:
        print(f"   ➖ Neutral impact (AUC change: {auc_improvement:.4f})")
        print(f"      → Synthetic data didn't significantly help or hurt")
    else:
        print(f"   ❌ Negative impact! AUC decreased by {abs(auc_improvement):.4f}")
        print(f"      → Synthetic data quality may need improvement")


def export_metrics_to_json(output_file="model_metrics.json"):
    """Export all metrics to JSON file"""
    print("\n" + "=" * 70)
    print("💾 Exporting Metrics to JSON")
    print("=" * 70)

    all_metrics = {}

    # Get all metrics
    all_metrics['cyclegan'] = get_cyclegan_metrics()
    all_metrics['vit'] = get_vit_metrics()
    all_metrics['cnn'] = get_cnn_metrics()

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=4, default=str)

    print(f"\n✅ Metrics exported to: {output_file}")


def display_summary():
    """Display summary of all models"""
    print("\n" + "=" * 70)
    print("📋 PROJECT SUMMARY")
    print("=" * 70)

    summary = []

    # Check CycleGAN
    if Path("checkpoints").exists() and list(Path("checkpoints").glob("G_AB_epoch*.pth")):
        g_ab_files = list(Path("checkpoints").glob("G_AB_epoch*.pth"))
        max_epoch = max([int(f.stem.split("epoch")[1]) for f in g_ab_files])
        summary.append({
            'Component': 'CycleGAN',
            'Status': '✅ Trained',
            'Details': f'{len(g_ab_files)} checkpoints (epoch {max_epoch})',
            'Location': 'checkpoints/'
        })
    else:
        summary.append({
            'Component': 'CycleGAN',
            'Status': '❌ Not trained',
            'Details': 'Run: python train.py',
            'Location': '-'
        })

    # Check synthetic images
    synth_dir = Path("generated_images/synthetic_pneumothorax")
    if synth_dir.exists():
        count = len(list(synth_dir.glob("*.png")))
        summary.append({
            'Component': 'Synthetic Images',
            'Status': '✅ Generated',
            'Details': f'{count} images',
            'Location': 'generated_images/'
        })
    else:
        summary.append({
            'Component': 'Synthetic Images',
            'Status': '❌ Not generated',
            'Details': 'Run: python generator.py',
            'Location': '-'
        })

    # Check ViT
    vit_path = Path("checkpoints/vit/vit_best.pth")
    if vit_path.exists():
        ckpt = torch.load(vit_path, map_location='cpu')
        summary.append({
            'Component': 'ViT Classifier',
            'Status': '✅ Trained',
            'Details': f"AUC: {ckpt['auc']:.4f}",
            'Location': 'checkpoints/vit/'
        })
    else:
        summary.append({
            'Component': 'ViT Classifier',
            'Status': '❌ Not trained',
            'Details': 'Run: python train_VIT.py',
            'Location': '-'
        })

    # Check CNN
    cnn_path = Path("checkpoints/cnn/cnn_best.pth")
    if cnn_path.exists():
        ckpt = torch.load(cnn_path, map_location='cpu')
        summary.append({
            'Component': 'CNN Classifier',
            'Status': '✅ Trained',
            'Details': f"Acc: {ckpt['accuracy']:.4f}",
            'Location': 'checkpoints/cnn/'
        })
    else:
        summary.append({
            'Component': 'CNN Classifier',
            'Status': '❌ Not trained',
            'Details': 'Run: python CNN_Model.py',
            'Location': '-'
        })

    df = pd.DataFrame(summary)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("📊 MODEL METRICS RETRIEVAL TOOL")
    print("=" * 70)

    # Display project summary
    display_summary()

    # Get all metrics
    get_cyclegan_metrics()
    vit_metrics = get_vit_metrics()
    cnn_metrics = get_cnn_metrics()

    # Optional: Compare with/without synthetic
    # compare_with_without_synthetic()

    # Export to JSON
    export_metrics_to_json()

    print("\n" + "=" * 70)
    print("✅ Metrics retrieval complete!")
    print("=" * 70)
    print("\n💡 Tips:")
    print("   - View metrics anytime: python retrieve_metrics.py")
    print("   - Metrics saved to: model_metrics.json")
    print("   - To compare with/without synthetic data, train ViT twice")
    print()