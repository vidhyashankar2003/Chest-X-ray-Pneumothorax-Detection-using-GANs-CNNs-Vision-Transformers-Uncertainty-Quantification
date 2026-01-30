"""
Comprehensive Ensemble Visualization
Compares individual models (ResNet18, DenseNet121, EfficientNet-B0) vs Ensemble

Generates:
1. Performance metrics bar chart (F1, AUC, Precision, Recall, Accuracy)
2. ROC curves comparison
3. Confidence calibration plots
4. Confusion matrices comparison
5. Summary table with statistics
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, 
    precision_recall_fscore_support, accuracy_score
)
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from tqdm import tqdm

# Import dataset
import sys
sys.path.append('.')
from pneumo_dataset_v2 import PneumoDataset
from ensemble_learning import ResNet18Model, DenseNet121Model, EfficientNetB0Model


def load_trained_models(checkpoint_dir, device='cuda'):
    """Load all trained individual models and ensemble weights"""
    checkpoint_dir = Path(checkpoint_dir)
    
    models = []
    architectures = ['resnet18', 'densenet121', 'efficientnet']
    
    print(f"\n📂 Loading trained models from: {checkpoint_dir}")
    
    for arch in architectures:
        model_path = checkpoint_dir / arch / "best.pth"
        
        if not model_path.exists():
            print(f"   ❌ {arch} checkpoint not found: {model_path}")
            continue
        
        # Load model
        if arch == 'resnet18':
            model = ResNet18Model(dropout=0.5).to(device)
        elif arch == 'densenet121':
            model = DenseNet121Model(dropout=0.5).to(device)
        elif arch == 'efficientnet':
            model = EfficientNetB0Model(dropout=0.5).to(device)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        models.append((model, arch, checkpoint['f1']))
        print(f"   ✅ {arch}: F1={checkpoint['f1']:.4f}")
    
    # Load ensemble weights
    ensemble_path = checkpoint_dir / "ensemble_model.pth"
    if ensemble_path.exists():
        ensemble_data = torch.load(ensemble_path, map_location=device, weights_only=False)
        weights = ensemble_data['model_weights']
        print(f"\n📊 Ensemble weights: {weights}")
    else:
        # Equal weights if ensemble not saved
        weights = [1.0 / len(models)] * len(models)
        print(f"\n⚠️  Ensemble checkpoint not found, using equal weights")
    
    return models, weights


def evaluate_individual_model(model, loader, device):
    """Evaluate a single model"""
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_labels.append(labels.numpy())
            all_probs.append(probs)
    
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    roc_auc = auc(*roc_curve(y_true, y_prob)[:2])
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'y_true': y_true,
        'y_prob': y_prob,
        'y_pred': y_pred,
        'cm': cm
    }


def evaluate_ensemble(models, weights, loader, device):
    """Evaluate ensemble model"""
    all_labels = []
    all_ensemble_probs = []
    
    for images, labels, _ in tqdm(loader, desc="Evaluating Ensemble"):
        images = images.to(device)
        
        # Get predictions from each model
        ensemble_prob = np.zeros(images.size(0))
        
        for (model, _, _), weight in zip(models, weights):
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                ensemble_prob += probs * weight
        
        all_labels.append(labels.numpy())
        all_ensemble_probs.append(ensemble_prob)
    
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_ensemble_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    roc_auc = auc(*roc_curve(y_true, y_prob)[:2])
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'y_true': y_true,
        'y_prob': y_prob,
        'y_pred': y_pred,
        'cm': cm
    }


def plot_metrics_comparison(individual_results, ensemble_results, output_path):
    """Bar chart comparing all metrics"""
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    
    # Prepare data
    model_names = list(individual_results.keys()) + ['Ensemble']
    data = {metric: [] for metric in metrics}
    
    for name in model_names:
        if name == 'Ensemble':
            results = ensemble_results
        else:
            results = individual_results[name]
        
        data['Accuracy'].append(results['accuracy'])
        data['Precision'].append(results['precision'])
        data['Recall'].append(results['recall'])
        data['F1-Score'].append(results['f1'])
        data['AUC'].append(results['auc'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(metrics))
    width = 0.18
    colors = ['steelblue', 'coral', 'lightgreen', 'gold']
    
    for i, name in enumerate(model_names):
        offset = (i - len(model_names)/2) * width + width/2
        values = [data[metric][i] for metric in metrics]
        
        bars = ax.bar(x + offset, values, width, label=name, 
                     color=colors[i], edgecolor='black', alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', 
                   fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Performance Comparison: Individual Models vs Ensemble', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Metrics comparison saved: {output_path}")


def plot_roc_curves(individual_results, ensemble_results, output_path):
    """ROC curves for all models"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['steelblue', 'coral', 'lightgreen', 'red']
    linestyles = ['-', '--', '-.', '-']
    
    # Plot individual models
    for i, (name, results) in enumerate(individual_results.items()):
        fpr, tpr, _ = roc_curve(results['y_true'], results['y_prob'])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, linewidth=2.5, label=f'{name} (AUC={roc_auc:.3f})',
               color=colors[i], linestyle=linestyles[i])
    
    # Plot ensemble
    fpr, tpr, _ = roc_curve(ensemble_results['y_true'], ensemble_results['y_prob'])
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, linewidth=3.5, label=f'Ensemble (AUC={roc_auc:.3f})',
           color=colors[-1], linestyle=linestyles[-1])
    
    # Random classifier
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random', alpha=0.5)
    
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curve Comparison', fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ ROC curves saved: {output_path}")


def plot_confusion_matrices(individual_results, ensemble_results, output_path):
    """Confusion matrices for all models"""
    
    n_models = len(individual_results) + 1
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    # Individual models
    for idx, (name, results) in enumerate(individual_results.items()):
        cm = results['cm']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Healthy', 'Pneumothorax'],
                   yticklabels=['Healthy', 'Pneumothorax'],
                   ax=axes[idx], cbar_kws={'label': 'Count'},
                   annot_kws={'fontsize': 12, 'fontweight': 'bold'})
        
        axes[idx].set_title(f'{name}\n(F1={results["f1"]:.3f})', 
                           fontsize=13, fontweight='bold', pad=15)
        axes[idx].set_ylabel('True Label', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    # Ensemble
    cm = ensemble_results['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
               xticklabels=['Healthy', 'Pneumothorax'],
               yticklabels=['Healthy', 'Pneumothorax'],
               ax=axes[-1], cbar_kws={'label': 'Count'},
               annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    
    axes[-1].set_title(f'Ensemble\n(F1={ensemble_results["f1"]:.3f})', 
                      fontsize=13, fontweight='bold', pad=15)
    axes[-1].set_ylabel('True Label', fontsize=11, fontweight='bold')
    axes[-1].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    plt.suptitle('Confusion Matrix Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrices saved: {output_path}")


def plot_calibration_curves(individual_results, ensemble_results, output_path):
    """Reliability diagrams (calibration curves)"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['steelblue', 'coral', 'lightgreen', 'red']
    
    # Plot individual models
    for i, (name, results) in enumerate(individual_results.items()):
        y_true = results['y_true']
        y_prob = results['y_prob']
        
        # Bin predictions
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_counts = np.zeros(len(bins) - 1)
        bin_true_rates = np.zeros(len(bins) - 1)
        
        for j in range(len(bins) - 1):
            mask = (y_prob >= bins[j]) & (y_prob < bins[j+1])
            if mask.sum() > 0:
                bin_counts[j] = mask.sum()
                bin_true_rates[j] = y_true[mask].mean()
        
        ax.plot(bin_centers, bin_true_rates, 'o-', linewidth=2, 
               label=name, color=colors[i], markersize=8)
    
    # Plot ensemble
    y_true = ensemble_results['y_true']
    y_prob = ensemble_results['y_prob']
    
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_counts = np.zeros(len(bins) - 1)
    bin_true_rates = np.zeros(len(bins) - 1)
    
    for j in range(len(bins) - 1):
        mask = (y_prob >= bins[j]) & (y_prob < bins[j+1])
        if mask.sum() > 0:
            bin_counts[j] = mask.sum()
            bin_true_rates[j] = y_true[mask].mean()
    
    ax.plot(bin_centers, bin_true_rates, 'o-', linewidth=3, 
           label='Ensemble', color=colors[-1], markersize=10)
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.5)
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=13, fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontsize=13, fontweight='bold')
    ax.set_title('Calibration Curves (Reliability Diagram)', 
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Calibration curves saved: {output_path}")


def create_summary_table(individual_results, ensemble_results, weights, output_path):
    """Create comprehensive summary table"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Prepare table data
    table_data = [['Model', 'Weight', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']]
    
    # Individual models
    for i, (name, results) in enumerate(individual_results.items()):
        table_data.append([
            name,
            f'{weights[i]:.3f}',
            f'{results["accuracy"]:.4f}',
            f'{results["precision"]:.4f}',
            f'{results["recall"]:.4f}',
            f'{results["f1"]:.4f}',
            f'{results["auc"]:.4f}'
        ])
    
    # Separator
    table_data.append(['─' * 15] * 7)
    
    # Ensemble
    table_data.append([
        'ENSEMBLE',
        '1.000',
        f'{ensemble_results["accuracy"]:.4f}',
        f'{ensemble_results["precision"]:.4f}',
        f'{ensemble_results["recall"]:.4f}',
        f'{ensemble_results["f1"]:.4f}',
        f'{ensemble_results["auc"]:.4f}'
    ])
    
    # Separator
    table_data.append(['─' * 15] * 7)
    
    # Improvements
    best_individual_f1 = max([r['f1'] for r in individual_results.values()])
    improvement = (ensemble_results['f1'] - best_individual_f1) / best_individual_f1 * 100
    
    table_data.append(['IMPROVEMENT', '-', '-', '-', '-', f'+{improvement:.2f}%', '-'])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.1, 0.13, 0.13, 0.13, 0.13, 0.13])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(7):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    
    # Style ensemble row
    ensemble_row_idx = len(individual_results) + 2
    for i in range(7):
        table[(ensemble_row_idx, i)].set_facecolor('#FFC000')
        table[(ensemble_row_idx, i)].set_text_props(weight='bold', fontsize=12)
    
    # Style improvement row
    improvement_row_idx = ensemble_row_idx + 2
    for i in range(7):
        if improvement > 0:
            table[(improvement_row_idx, i)].set_facecolor('#C6EFCE')
            table[(improvement_row_idx, i)].set_text_props(color='#006100', weight='bold')
        else:
            table[(improvement_row_idx, i)].set_facecolor('#FFC7CE')
            table[(improvement_row_idx, i)].set_text_props(color='#9C0006', weight='bold')
    
    plt.title('Ensemble Performance Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Summary table saved: {output_path}")


def main(args):
    """Main execution"""
    
    print(f"\n{'='*70}")
    print(f"📊 Ensemble Visualization")
    print(f"{'='*70}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load models
    models, weights = load_trained_models(args.checkpoint_dir, device)
    
    if len(models) == 0:
        print("\n❌ No trained models found! Train ensemble first:")
        print("   python ensemble_learning.py")
        return
    
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
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False, 
                       num_workers=4, pin_memory=True)
    
    print(f"   Validation samples: {len(dataset)}")
    
    # Evaluate individual models
    print(f"\n📈 Evaluating individual models...")
    individual_results = {}
    
    for model, name, _ in models:
        print(f"   Evaluating {name}...")
        results = evaluate_individual_model(model, loader, device)
        individual_results[name] = results
        print(f"      F1: {results['f1']:.4f}, AUC: {results['auc']:.4f}")
    
    # Evaluate ensemble
    print(f"\n📈 Evaluating ensemble...")
    ensemble_results = evaluate_ensemble(models, weights, loader, device)
    print(f"   F1: {ensemble_results['f1']:.4f}, AUC: {ensemble_results['auc']:.4f}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print(f"\n🎨 Generating visualizations...")
    
    plot_metrics_comparison(individual_results, ensemble_results,
                           output_dir / "ensemble_metrics_comparison.png")
    
    plot_roc_curves(individual_results, ensemble_results,
                   output_dir / "ensemble_roc_curves.png")
    
    plot_confusion_matrices(individual_results, ensemble_results,
                           output_dir / "ensemble_confusion_matrices.png")
    
    plot_calibration_curves(individual_results, ensemble_results,
                           output_dir / "ensemble_calibration.png")
    
    create_summary_table(individual_results, ensemble_results, weights,
                        output_dir / "ensemble_summary_table.png")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"✅ Ensemble Visualization Complete!")
    print(f"{'='*70}")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n📊 Generated files:")
    print(f"   1. ensemble_metrics_comparison.png - Bar chart of all metrics")
    print(f"   2. ensemble_roc_curves.png - ROC curves comparison")
    print(f"   3. ensemble_confusion_matrices.png - Confusion matrices")
    print(f"   4. ensemble_calibration.png - Calibration curves")
    print(f"   5. ensemble_summary_table.png - Summary table")
    
    # Print key findings
    best_individual = max(individual_results.items(), key=lambda x: x[1]['f1'])
    improvement = (ensemble_results['f1'] - best_individual[1]['f1']) / best_individual[1]['f1'] * 100
    
    print(f"\n🎯 Key Findings:")
    print(f"   Best Individual: {best_individual[0]} (F1={best_individual[1]['f1']:.4f})")
    print(f"   Ensemble: F1={ensemble_results['f1']:.4f}")
    print(f"   Improvement: {improvement:+.2f}%")
    
    if improvement > 1:
        print(f"   ✅ Ensemble outperforms all individual models!")
    elif improvement > 0:
        print(f"   ✅ Ensemble provides modest improvement")
    else:
        print(f"   ⚠️  Ensemble does not improve over best individual model")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Ensemble Performance")
    
    parser.add_argument("--checkpoint_dir", default="checkpoints/ensemble",
                       help="Directory containing ensemble checkpoints")
    parser.add_argument("--csv", default="../data/chestxray/Data_Entry_2017_v2020.csv")
    parser.add_argument("--images", default="../data/chestxray/images")
    parser.add_argument("--val_list", 
                       default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/val_BALANCED.txt")
    parser.add_argument("--output_dir", default="visualizations/ensemble",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    main(args)
