"""
Comprehensive Comparison: Simple CNN vs Multi-Task CNN (Updated for v3)
Generates publication-ready comparison visualizations

Compares:
1. Performance metrics (F1, AUC, Precision, Recall)
2. Training curves (loss over epochs)
3. Confusion matrices
4. ROC curves
5. Prediction confidence distributions
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import seaborn as sns

from train_simple_cnn import SimpleCNN
from multitask_cnn import ImprovedMultiTaskCNN  # Updated import
from pneumo_dataset_v2 import PneumoDataset


def load_model(checkpoint_path, model_type='simple', device='cuda'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if model_type == 'simple':
        model = SimpleCNN(dropout=0.5).to(device)
    else:  # multitask v3
        # Check if checkpoint has attention/uncertainty info
        use_attention = checkpoint.get('use_attention', True)
        use_uncertainty = checkpoint.get('use_uncertainty', True)

        model = ImprovedMultiTaskCNN(
            dropout=0.5,
            use_attention=use_attention
        ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def evaluate_model(model, loader, device, model_type='simple'):
    """Evaluate model and get all predictions"""
    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc=f"Evaluating {model_type}", leave=False):
            images = images.to(device)

            if model_type == 'simple':
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
            else:  # multitask v3
                pneumo_logits, _ = model(images)  # Ignore severity output
                probs = torch.sigmoid(pneumo_logits).cpu().numpy()

            preds = (probs >= 0.5).astype(int)

            all_labels.append(labels.numpy())
            all_probs.append(probs)
            all_preds.append(preds)

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    y_pred = np.concatenate(all_preds)

    return y_true, y_prob, y_pred


def plot_metrics_comparison(simple_metrics, multitask_metrics, output_path):
    """Compare performance metrics side by side"""

    metrics_names = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    simple_values = [
        simple_metrics['auc'],
        simple_metrics['accuracy'],
        simple_metrics['precision'],
        simple_metrics['recall'],
        simple_metrics['f1']
    ]
    multitask_values = [
        multitask_metrics['auc'],
        multitask_metrics['accuracy'],
        multitask_metrics['precision'],
        multitask_metrics['recall'],
        multitask_metrics['f1']
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width / 2, simple_values, width, label='Simple CNN',
                   color='steelblue', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width / 2, multitask_values, width, label='Multi-Task CNN v3',
                   color='coral', edgecolor='black', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Calculate improvements
    improvements = [(mt - s) / s * 100 for s, mt in zip(simple_values, multitask_values)]

    # Add improvement percentages
    for i, imp in enumerate(improvements):
        if imp > 0:
            ax.text(i, max(simple_values[i], multitask_values[i]) + 0.02,
                    f'+{imp:.1f}%', ha='center', fontsize=9, color='green', fontweight='bold')
        elif imp < 0:
            ax.text(i, max(simple_values[i], multitask_values[i]) + 0.02,
                    f'{imp:.1f}%', ha='center', fontsize=9, color='red', fontweight='bold')

    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Performance Comparison: Simple CNN vs Multi-Task CNN v3',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Metrics comparison saved: {output_path}")


def plot_roc_comparison(simple_data, multitask_data, output_path):
    """Compare ROC curves"""

    simple_y_true, simple_y_prob, _ = simple_data
    multitask_y_true, multitask_y_prob, _ = multitask_data

    # Calculate ROC curves
    simple_fpr, simple_tpr, _ = roc_curve(simple_y_true, simple_y_prob)
    simple_auc = auc(simple_fpr, simple_tpr)

    multitask_fpr, multitask_tpr, _ = roc_curve(multitask_y_true, multitask_y_prob)
    multitask_auc = auc(multitask_fpr, multitask_tpr)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot ROC curves
    ax.plot(simple_fpr, simple_tpr, linewidth=2.5, label=f'Simple CNN (AUC = {simple_auc:.3f})',
            color='steelblue')
    ax.plot(multitask_fpr, multitask_tpr, linewidth=2.5, label=f'Multi-Task CNN v3 (AUC = {multitask_auc:.3f})',
            color='coral')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier', alpha=0.5)

    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curve Comparison', fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')

    # Add improvement text
    improvement = (multitask_auc - simple_auc) / simple_auc * 100
    color = 'green' if improvement > 0 else 'red'
    ax.text(0.6, 0.2, f'AUC Improvement: {improvement:+.2f}%',
            fontsize=12, fontweight='bold', color=color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ ROC comparison saved: {output_path}")


def plot_confusion_matrices(simple_data, multitask_data, output_path):
    """Compare confusion matrices side by side"""

    simple_y_true, _, simple_y_pred = simple_data
    multitask_y_true, _, multitask_y_pred = multitask_data

    simple_cm = confusion_matrix(simple_y_true, simple_y_pred)
    multitask_cm = confusion_matrix(multitask_y_true, multitask_y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Simple CNN confusion matrix
    sns.heatmap(simple_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Pneumothorax'],
                yticklabels=['Healthy', 'Pneumothorax'],
                ax=axes[0], cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    axes[0].set_title('Simple CNN', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')

    # Add accuracy
    simple_acc = (simple_cm[0, 0] + simple_cm[1, 1]) / simple_cm.sum()
    axes[0].text(0.5, -0.15, f'Accuracy: {simple_acc:.3f}',
                 ha='center', transform=axes[0].transAxes,
                 fontsize=12, fontweight='bold')

    # Multi-Task CNN v3 confusion matrix
    sns.heatmap(multitask_cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Healthy', 'Pneumothorax'],
                yticklabels=['Healthy', 'Pneumothorax'],
                ax=axes[1], cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    axes[1].set_title('Multi-Task CNN v3', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')

    # Add accuracy
    multitask_acc = (multitask_cm[0, 0] + multitask_cm[1, 1]) / multitask_cm.sum()
    axes[1].text(0.5, -0.15, f'Accuracy: {multitask_acc:.3f}',
                 ha='center', transform=axes[1].transAxes,
                 fontsize=12, fontweight='bold')

    plt.suptitle('Confusion Matrix Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Confusion matrices saved: {output_path}")


def plot_confidence_distributions(simple_data, multitask_data, output_path):
    """Compare prediction confidence distributions"""

    simple_y_true, simple_y_prob, _ = simple_data
    multitask_y_true, multitask_y_prob, _ = multitask_data

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Simple CNN - Correct predictions
    simple_correct = simple_y_prob[(simple_y_prob >= 0.5) == simple_y_true]
    axes[0, 0].hist(simple_correct, bins=20, color='green', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Simple CNN - Correct Predictions', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Confidence', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    if len(simple_correct) > 0:
        axes[0, 0].axvline(np.mean(simple_correct), color='darkgreen', linestyle='--',
                           linewidth=2, label=f'Mean: {np.mean(simple_correct):.3f}')
        axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Simple CNN - Incorrect predictions
    simple_incorrect = simple_y_prob[(simple_y_prob >= 0.5) != simple_y_true]
    axes[0, 1].hist(simple_incorrect, bins=20, color='red', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Simple CNN - Incorrect Predictions', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Confidence', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    if len(simple_incorrect) > 0:
        axes[0, 1].axvline(np.mean(simple_incorrect), color='darkred', linestyle='--',
                           linewidth=2, label=f'Mean: {np.mean(simple_incorrect):.3f}')
        axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Multi-Task CNN v3 - Correct predictions
    multitask_correct = multitask_y_prob[(multitask_y_prob >= 0.5) == multitask_y_true]
    axes[1, 0].hist(multitask_correct, bins=20, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Multi-Task CNN v3 - Correct Predictions', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Confidence', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    if len(multitask_correct) > 0:
        axes[1, 0].axvline(np.mean(multitask_correct), color='darkgreen', linestyle='--',
                           linewidth=2, label=f'Mean: {np.mean(multitask_correct):.3f}')
        axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Multi-Task CNN v3 - Incorrect predictions
    multitask_incorrect = multitask_y_prob[(multitask_y_prob >= 0.5) != multitask_y_true]
    axes[1, 1].hist(multitask_incorrect, bins=20, color='red', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Multi-Task CNN v3 - Incorrect Predictions', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Confidence', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    if len(multitask_incorrect) > 0:
        axes[1, 1].axvline(np.mean(multitask_incorrect), color='darkred', linestyle='--',
                           linewidth=2, label=f'Mean: {np.mean(multitask_incorrect):.3f}')
        axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle('Prediction Confidence Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Confidence distributions saved: {output_path}")


def create_summary_table(simple_metrics, multitask_metrics, simple_ckpt, multitask_ckpt, output_path):
    """Create comprehensive comparison table"""

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')

    # Check if multitask has attention/uncertainty info
    use_attention = multitask_ckpt.get('use_attention', 'Unknown')
    use_uncertainty = multitask_ckpt.get('use_uncertainty', 'Unknown')

    # Table data
    table_data = [
        ['Metric', 'Simple CNN', 'Multi-Task CNN v3', 'Improvement'],
        ['AUC', f"{simple_metrics['auc']:.4f}", f"{multitask_metrics['auc']:.4f}",
         f"{(multitask_metrics['auc'] - simple_metrics['auc']) / simple_metrics['auc'] * 100:+.2f}%"],
        ['Accuracy', f"{simple_metrics['accuracy']:.4f}", f"{multitask_metrics['accuracy']:.4f}",
         f"{(multitask_metrics['accuracy'] - simple_metrics['accuracy']) / simple_metrics['accuracy'] * 100:+.2f}%"],
        ['Precision', f"{simple_metrics['precision']:.4f}", f"{multitask_metrics['precision']:.4f}",
         f"{(multitask_metrics['precision'] - simple_metrics['precision']) / simple_metrics['precision'] * 100:+.2f}%"],
        ['Recall', f"{simple_metrics['recall']:.4f}", f"{multitask_metrics['recall']:.4f}",
         f"{(multitask_metrics['recall'] - simple_metrics['recall']) / simple_metrics['recall'] * 100:+.2f}%"],
        ['F1-Score', f"{simple_metrics['f1']:.4f}", f"{multitask_metrics['f1']:.4f}",
         f"{(multitask_metrics['f1'] - simple_metrics['f1']) / simple_metrics['f1'] * 100:+.2f}%"],
        ['─' * 15] * 4,
        ['Architecture', '', '', ''],
        ['Epochs Trained', f"{simple_ckpt['epoch']}", f"{multitask_ckpt['epoch']}", '-'],
        ['Task Attention', 'No', str(use_attention), '-'],
        ['Uncertainty Weight', 'No', str(use_uncertainty), '-'],
        ['Auxiliary Task', 'None', 'Severity Est.', '-'],
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.25, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)

    # Style section headers
    for i in range(4):
        table[(7, i)].set_facecolor('#D9E1F2')
        table[(7, i)].set_text_props(weight='bold')

    # Color improvements
    for i in range(1, 6):
        improvement_text = table_data[i][3]
        if '+' in improvement_text:
            table[(i, 3)].set_facecolor('#C6EFCE')
            table[(i, 3)].set_text_props(color='#006100', weight='bold')
        elif '-' in improvement_text and improvement_text != '-':
            table[(i, 3)].set_facecolor('#FFC7CE')
            table[(i, 3)].set_text_props(color='#9C0006', weight='bold')

    plt.title('Comprehensive Model Comparison\nSimple CNN vs Multi-Task CNN v3',
              fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Summary table saved: {output_path}")


def main(args):
    """Main comparison execution"""

    print(f"\n{'=' * 70}")
    print(f"📊 Model Comparison: Simple CNN vs Multi-Task CNN v3")
    print(f"{'=' * 70}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    print(f"\n📂 Loading models...")
    simple_model, simple_ckpt = load_model(args.simple_checkpoint, 'simple', device)
    multitask_model, multitask_ckpt = load_model(args.multitask_checkpoint, 'multitask', device)

    print(f"✅ Simple CNN loaded (epoch {simple_ckpt['epoch']})")
    print(f"✅ Multi-Task CNN v3 loaded (epoch {multitask_ckpt['epoch']})")

    # Print multitask config
    if 'use_attention' in multitask_ckpt:
        print(f"   - Task Attention: {multitask_ckpt['use_attention']}")
    if 'use_uncertainty' in multitask_ckpt:
        print(f"   - Uncertainty Weighting: {multitask_ckpt['use_uncertainty']}")

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

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"📊 Evaluation dataset: {len(dataset)} images\n")

    # Evaluate both models
    print("🔍 Evaluating models...")
    simple_y_true, simple_y_prob, simple_y_pred = evaluate_model(simple_model, loader, device, 'simple')
    multitask_y_true, multitask_y_prob, multitask_y_pred = evaluate_model(multitask_model, loader, device, 'multitask')

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    simple_metrics = {
        'auc': roc_auc_score(simple_y_true, simple_y_prob),
        'accuracy': accuracy_score(simple_y_true, simple_y_pred),
    }
    prec, rec, f1, _ = precision_recall_fscore_support(simple_y_true, simple_y_pred, average='binary', zero_division=0)
    simple_metrics.update({'precision': prec, 'recall': rec, 'f1': f1})

    multitask_metrics = {
        'auc': roc_auc_score(multitask_y_true, multitask_y_prob),
        'accuracy': accuracy_score(multitask_y_true, multitask_y_pred),
    }
    prec, rec, f1, _ = precision_recall_fscore_support(multitask_y_true, multitask_y_pred, average='binary',
                                                       zero_division=0)
    multitask_metrics.update({'precision': prec, 'recall': rec, 'f1': f1})

    print(f"\n📈 Results:")
    print(f"   Simple CNN      - F1: {simple_metrics['f1']:.4f}, AUC: {simple_metrics['auc']:.4f}")
    print(f"   Multi-Task v3   - F1: {multitask_metrics['f1']:.4f}, AUC: {multitask_metrics['auc']:.4f}")

    # Calculate improvement
    f1_improvement = (multitask_metrics['f1'] - simple_metrics['f1']) / simple_metrics['f1'] * 100
    auc_improvement = (multitask_metrics['auc'] - simple_metrics['auc']) / simple_metrics['auc'] * 100

    print(f"\n📊 Improvements:")
    print(f"   F1-Score: {f1_improvement:+.2f}%")
    print(f"   AUC:      {auc_improvement:+.2f}%")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all comparison visualizations
    print(f"\n🎨 Generating comparison visualizations...")

    plot_metrics_comparison(simple_metrics, multitask_metrics,
                            output_dir / "metrics_comparison.png")

    plot_roc_comparison((simple_y_true, simple_y_prob, simple_y_pred),
                        (multitask_y_true, multitask_y_prob, multitask_y_pred),
                        output_dir / "roc_comparison.png")

    plot_confusion_matrices((simple_y_true, simple_y_prob, simple_y_pred),
                            (multitask_y_true, multitask_y_prob, multitask_y_pred),
                            output_dir / "confusion_matrices.png")

    plot_confidence_distributions((simple_y_true, simple_y_prob, simple_y_pred),
                                  (multitask_y_true, multitask_y_prob, multitask_y_pred),
                                  output_dir / "confidence_distributions.png")

    create_summary_table(simple_metrics, multitask_metrics, simple_ckpt, multitask_ckpt,
                         output_dir / "summary_table.png")

    print(f"\n{'=' * 70}")
    print(f"✅ Comparison complete!")
    print(f"   Output directory: {output_dir}")
    print(f"   Generated 5 comparison figures")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Simple CNN vs Multi-Task CNN v3")

    parser.add_argument("--simple_checkpoint", default="checkpoints/cnn_pneumo/cnn_pneumo_best.pth",
                        help="Path to Simple CNN checkpoint")
    parser.add_argument("--multitask_checkpoint", default="checkpoints/multitask_v3/multitask_v3_best.pth",
                        help="Path to Multi-Task CNN v3 checkpoint")
    parser.add_argument("--csv", default="../data/chestxray/Data_Entry_2017_v2020.csv")
    parser.add_argument("--images", default="../data/chestxray/images")
    parser.add_argument("--val_list", default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/val_BALANCED.txt")
    parser.add_argument("--output_dir", default="visualizations/model_comparison_v3",
                        help="Output directory for comparison visualizations")

    args = parser.parse_args()
    main(args)