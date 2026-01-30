"""
Comprehensive 3-Way Comparison: Simple CNN vs Multi-Task CNN vs Ensemble
Compares three different approaches for pneumothorax detection

Generates:
1. Performance metrics comparison (bar charts)
2. ROC curves comparison
3. Confusion matrices side-by-side
4. Training efficiency comparison (params, time, memory)
5. Confidence distribution comparison
6. Summary table with statistical analysis
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

# Import models and dataset
import sys
sys.path.append('.')
from pneumo_dataset_v2 import PneumoDataset
from train_simple_cnn import SimpleCNN
from multitask_cnn import ImprovedMultiTaskCNN

from ensemble_learning import ResNet18Model, DenseNet121Model, EfficientNetB0Model


def load_simple_cnn(checkpoint_path, device):
    """Load Simple CNN model"""
    model = SimpleCNN(dropout=0.5).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def load_multitask_cnn(checkpoint_path, device):
    """Load Multi-Task CNN model"""
    model = ImprovedMultiTaskCNN(dropout=0.5).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def load_ensemble(checkpoint_dir, device):
    """Load ensemble models"""
    checkpoint_dir = Path(checkpoint_dir)
    
    models = []
    architectures = ['resnet18', 'densenet121', 'efficientnet']
    
    for arch in architectures:
        model_path = checkpoint_dir / arch / "best.pth"
        
        if not model_path.exists():
            print(f"   ⚠️  {arch} checkpoint not found")
            continue
        
        if arch == 'resnet18':
            model = ResNet18Model(dropout=0.5).to(device)
        elif arch == 'densenet121':
            model = DenseNet121Model(dropout=0.5).to(device)
        elif arch == 'efficientnet':
            model = EfficientNetB0Model(dropout=0.5).to(device)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        models.append((model, arch))
    
    # Load ensemble weights
    ensemble_path = checkpoint_dir / "ensemble_model.pth"
    if ensemble_path.exists():
        ensemble_data = torch.load(ensemble_path, map_location=device, weights_only=False)
        weights = ensemble_data['model_weights']
        ensemble_checkpoint = ensemble_data
    else:
        weights = [1.0 / len(models)] * len(models)
        ensemble_checkpoint = {'ensemble_results': {}}
    
    return models, weights, ensemble_checkpoint


def evaluate_simple_cnn(model, loader, device):
    """Evaluate Simple CNN"""
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Evaluating Simple CNN", leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_labels.append(labels.numpy())
            all_probs.append(probs)
    
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    
    return compute_metrics(y_true, y_prob, y_pred)


def evaluate_multitask_cnn(model, loader, device):
    """Evaluate Multi-Task CNN"""
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Evaluating Multi-Task CNN", leave=False):
            images = images.to(device)
            pneumo_logits, _ = model(images)  # Ignore quality output
            probs = torch.sigmoid(pneumo_logits).cpu().numpy()
            
            all_labels.append(labels.numpy())
            all_probs.append(probs)
    
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    
    return compute_metrics(y_true, y_prob, y_pred)


def evaluate_ensemble(models, weights, loader, device):
    """Evaluate Ensemble"""
    all_labels = []
    all_ensemble_probs = []
    
    for images, labels, _ in tqdm(loader, desc="Evaluating Ensemble", leave=False):
        images = images.to(device)
        
        ensemble_prob = np.zeros(images.size(0))
        
        for (model, _), weight in zip(models, weights):
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
    
    return compute_metrics(y_true, y_prob, y_pred)


def compute_metrics(y_true, y_prob, y_pred):
    """Compute all evaluation metrics"""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Specificity and Sensitivity
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1,
        'auc': roc_auc,
        'y_true': y_true,
        'y_prob': y_prob,
        'y_pred': y_pred,
        'cm': cm,
        'fpr': fpr,
        'tpr': tpr
    }


def plot_metrics_comparison(results_dict, output_path):
    """3-way bar chart comparison"""
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC']
    model_names = ['Simple CNN', 'Multi-Task CNN', 'Ensemble']
    
    # Prepare data
    data = {metric: [] for metric in metrics}
    for name in model_names:
        results = results_dict[name]
        data['Accuracy'].append(results['accuracy'])
        data['Precision'].append(results['precision'])
        data['Recall'].append(results['recall'])
        data['Sensitivity'].append(results['sensitivity'])
        data['Specificity'].append(results['specificity'])
        data['F1-Score'].append(results['f1'])
        data['AUC'].append(results['auc'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(metrics))
    width = 0.25
    colors = ['steelblue', 'coral', 'mediumseagreen']
    
    for i, name in enumerate(model_names):
        offset = (i - 1) * width
        values = [data[metric][i] for metric in metrics]
        
        bars = ax.bar(x + offset, values, width, label=name,
                     color=colors[i], edgecolor='black', alpha=0.85)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Performance Comparison: Simple CNN vs Multi-Task CNN vs Ensemble',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=13, loc='lower right')
    ax.set_ylim([0, 1.15])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add improvement annotations
    simple_f1 = results_dict['Simple CNN']['f1']
    multitask_f1 = results_dict['Multi-Task CNN']['f1']
    ensemble_f1 = results_dict['Ensemble']['f1']
    
    mt_improvement = ((multitask_f1 - simple_f1) / simple_f1 * 100)
    ens_improvement = ((ensemble_f1 - simple_f1) / simple_f1 * 100)
    
    # Add text box with improvements
    textstr = f'Improvements over Simple CNN:\n'
    textstr += f'Multi-Task: {mt_improvement:+.2f}%\n'
    textstr += f'Ensemble: {ens_improvement:+.2f}%'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Metrics comparison saved: {output_path}")


def plot_roc_comparison(results_dict, output_path):
    """ROC curves for all three approaches"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['steelblue', 'coral', 'mediumseagreen']
    linestyles = ['-', '--', '-.']
    linewidths = [2.5, 2.5, 3.5]
    
    for i, (name, results) in enumerate(results_dict.items()):
        ax.plot(results['fpr'], results['tpr'],
               linewidth=linewidths[i],
               label=f'{name} (AUC={results["auc"]:.3f})',
               color=colors[i],
               linestyle=linestyles[i])
    
    # Random classifier
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random', alpha=0.5)
    
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curve Comparison', fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ ROC curves saved: {output_path}")


def plot_confusion_matrices(results_dict, output_path):
    """Side-by-side confusion matrices"""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    cmaps = ['Blues', 'Oranges', 'Greens']
    
    for idx, ((name, results), cmap) in enumerate(zip(results_dict.items(), cmaps)):
        cm = results['cm']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                   xticklabels=['Healthy', 'Pneumothorax'],
                   yticklabels=['Healthy', 'Pneumothorax'],
                   ax=axes[idx], cbar_kws={'label': 'Count'},
                   annot_kws={'fontsize': 14, 'fontweight': 'bold'})
        
        # Add metrics to title
        axes[idx].set_title(
            f'{name}\nF1={results["f1"]:.3f} | AUC={results["auc"]:.3f}',
            fontsize=13, fontweight='bold', pad=15
        )
        axes[idx].set_ylabel('True Label', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    plt.suptitle('Confusion Matrix Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrices saved: {output_path}")


def plot_confidence_distributions(results_dict, output_path):
    """Compare prediction confidence distributions"""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    colors = ['steelblue', 'coral', 'mediumseagreen']
    
    for idx, (name, results) in enumerate(results_dict.items()):
        y_true = results['y_true']
        y_prob = results['y_prob']
        
        # Correct predictions
        correct_mask = (y_prob >= 0.5) == y_true
        correct_probs = y_prob[correct_mask]
        
        axes[0, idx].hist(correct_probs, bins=20, color='green',
                         alpha=0.7, edgecolor='black')
        axes[0, idx].set_title(f'{name} - Correct Predictions',
                              fontsize=12, fontweight='bold')
        axes[0, idx].set_xlabel('Confidence', fontsize=10)
        axes[0, idx].set_ylabel('Frequency', fontsize=10)
        axes[0, idx].axvline(np.mean(correct_probs), color='darkgreen',
                            linestyle='--', linewidth=2,
                            label=f'Mean: {np.mean(correct_probs):.3f}')
        axes[0, idx].legend()
        axes[0, idx].grid(alpha=0.3)
        
        # Incorrect predictions
        incorrect_probs = y_prob[~correct_mask]
        
        if len(incorrect_probs) > 0:
            axes[1, idx].hist(incorrect_probs, bins=20, color='red',
                             alpha=0.7, edgecolor='black')
            axes[1, idx].axvline(np.mean(incorrect_probs), color='darkred',
                                linestyle='--', linewidth=2,
                                label=f'Mean: {np.mean(incorrect_probs):.3f}')
            axes[1, idx].legend()
        
        axes[1, idx].set_title(f'{name} - Incorrect Predictions',
                              fontsize=12, fontweight='bold')
        axes[1, idx].set_xlabel('Confidence', fontsize=10)
        axes[1, idx].set_ylabel('Frequency', fontsize=10)
        axes[1, idx].grid(alpha=0.3)
    
    plt.suptitle('Prediction Confidence Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confidence distributions saved: {output_path}")


def plot_model_complexity(model_info, output_path):
    """Compare model complexity (parameters, inference time)"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models = list(model_info.keys())
    params = [info['params'] / 1e6 for info in model_info.values()]  # Convert to millions
    
    colors = ['steelblue', 'coral', 'mediumseagreen']
    
    # Parameters comparison
    bars1 = axes[0].bar(models, params, color=colors, edgecolor='black', alpha=0.8)
    axes[0].set_ylabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Complexity', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}M', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    
    # Performance vs Complexity
    f1_scores = [info['f1'] for info in model_info.values()]
    
    axes[1].scatter(params, f1_scores, s=300, c=colors, edgecolors='black',
                   linewidths=2, alpha=0.8)
    
    for i, model in enumerate(models):
        axes[1].annotate(model, (params[i], f1_scores[i]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3))
    
    axes[1].set_xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Performance vs Complexity', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Model complexity comparison saved: {output_path}")


def create_summary_table(results_dict, model_info, output_path):
    """Comprehensive summary table"""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    # Prepare table data
    table_data = [
        ['Metric', 'Simple CNN', 'Multi-Task CNN', 'Ensemble',
         'MT vs Simple', 'Ens vs Simple']
    ]
    
    simple = results_dict['Simple CNN']
    multitask = results_dict['Multi-Task CNN']
    ensemble = results_dict['Ensemble']
    
    metrics = [
        ('Accuracy', 'accuracy'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('Sensitivity', 'sensitivity'),
        ('Specificity', 'specificity'),
        ('F1-Score', 'f1'),
        ('AUC', 'auc')
    ]
    
    for metric_name, metric_key in metrics:
        simple_val = simple[metric_key]
        mt_val = multitask[metric_key]
        ens_val = ensemble[metric_key]
        
        mt_diff = ((mt_val - simple_val) / simple_val * 100)
        ens_diff = ((ens_val - simple_val) / simple_val * 100)
        
        table_data.append([
            metric_name,
            f'{simple_val:.4f}',
            f'{mt_val:.4f}',
            f'{ens_val:.4f}',
            f'{mt_diff:+.2f}%',
            f'{ens_diff:+.2f}%'
        ])
    
    # Add separator
    table_data.append(['─' * 15] * 6)
    
    # Add model complexity
    table_data.append([
        'Parameters',
        f'{model_info["Simple CNN"]["params"]/1e6:.1f}M',
        f'{model_info["Multi-Task CNN"]["params"]/1e6:.1f}M',
        f'{model_info["Ensemble"]["params"]/1e6:.1f}M',
        '-', '-'
    ])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    
    # Style improvement columns
    for i in range(1, len(table_data)):
        # MT vs Simple
        if table_data[i][4] != '-' and '+' in table_data[i][4]:
            table[(i, 4)].set_facecolor('#C6EFCE')
            table[(i, 4)].set_text_props(color='#006100', weight='bold')
        elif table_data[i][4] != '-':
            table[(i, 4)].set_facecolor('#FFC7CE')
            table[(i, 4)].set_text_props(color='#9C0006', weight='bold')
        
        # Ens vs Simple
        if table_data[i][5] != '-' and '+' in table_data[i][5]:
            table[(i, 5)].set_facecolor('#C6EFCE')
            table[(i, 5)].set_text_props(color='#006100', weight='bold')
        elif table_data[i][5] != '-':
            table[(i, 5)].set_facecolor('#FFC7CE')
            table[(i, 5)].set_text_props(color='#9C0006', weight='bold')
    
    plt.title('Comprehensive Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Summary table saved: {output_path}")


def main(args):
    """Main execution"""
    
    print(f"\n{'='*70}")
    print(f"📊 Three-Way Comparison: Simple CNN vs Multi-Task CNN vs Ensemble")
    print(f"{'='*70}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load models
    print(f"\n📂 Loading models...")
    
    # Simple CNN
    simple_model, simple_ckpt = load_simple_cnn(args.simple_checkpoint, device)
    print(f"   ✅ Simple CNN loaded")
    
    # Multi-Task CNN
    multitask_model, multitask_ckpt = load_multitask_cnn(args.multitask_checkpoint, device)
    print(f"   ✅ Multi-Task CNN loaded")
    
    # Ensemble
    ensemble_models, ensemble_weights, ensemble_ckpt = load_ensemble(args.ensemble_dir, device)
    print(f"   ✅ Ensemble loaded ({len(ensemble_models)} models)")
    
    # Count parameters
    model_info = {
        'Simple CNN': {
            'params': sum(p.numel() for p in simple_model.parameters()),
            'f1': 0  # Will be filled after evaluation
        },
        'Multi-Task CNN': {
            'params': sum(p.numel() for p in multitask_model.parameters()),
            'f1': 0
        },
        'Ensemble': {
            'params': sum(sum(p.numel() for p in m.parameters()) for m, _ in ensemble_models),
            'f1': 0
        }
    }
    
    print(f"\n📊 Model Complexity:")
    for name, info in model_info.items():
        print(f"   {name}: {info['params']/1e6:.1f}M parameters")
    
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
    
    # Evaluate all models
    print(f"\n📈 Evaluating models...")
    
    results_dict = {}
    
    print(f"   Evaluating Simple CNN...")
    results_dict['Simple CNN'] = evaluate_simple_cnn(simple_model, loader, device)
    model_info['Simple CNN']['f1'] = results_dict['Simple CNN']['f1']
    print(f"      F1: {results_dict['Simple CNN']['f1']:.4f}, AUC: {results_dict['Simple CNN']['auc']:.4f}")
    
    print(f"   Evaluating Multi-Task CNN...")
    results_dict['Multi-Task CNN'] = evaluate_multitask_cnn(multitask_model, loader, device)
    model_info['Multi-Task CNN']['f1'] = results_dict['Multi-Task CNN']['f1']
    print(f"      F1: {results_dict['Multi-Task CNN']['f1']:.4f}, AUC: {results_dict['Multi-Task CNN']['auc']:.4f}")
    
    print(f"   Evaluating Ensemble...")
    results_dict['Ensemble'] = evaluate_ensemble(ensemble_models, ensemble_weights, loader, device)
    model_info['Ensemble']['f1'] = results_dict['Ensemble']['f1']
    print(f"      F1: {results_dict['Ensemble']['f1']:.4f}, AUC: {results_dict['Ensemble']['auc']:.4f}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print(f"\n🎨 Generating visualizations...")
    
    plot_metrics_comparison(results_dict, output_dir / "three_way_metrics.png")
    plot_roc_comparison(results_dict, output_dir / "three_way_roc.png")
    plot_confusion_matrices(results_dict, output_dir / "three_way_confusion.png")
    plot_confidence_distributions(results_dict, output_dir / "three_way_confidence.png")
    plot_model_complexity(model_info, output_dir / "three_way_complexity.png")
    create_summary_table(results_dict, model_info, output_dir / "three_way_summary.png")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"✅ Three-Way Comparison Complete!")
    print(f"{'='*70}")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n📊 Generated files:")
    print(f"   1. three_way_metrics.png - Performance metrics comparison")
    print(f"   2. three_way_roc.png - ROC curves")
    print(f"   3. three_way_confusion.png - Confusion matrices")
    print(f"   4. three_way_confidence.png - Confidence distributions")
    print(f"   5. three_way_complexity.png - Model complexity analysis")
    print(f"   6. three_way_summary.png - Comprehensive summary table")
    
    # Print key findings
    simple_f1 = results_dict['Simple CNN']['f1']
    multitask_f1 = results_dict['Multi-Task CNN']['f1']
    ensemble_f1 = results_dict['Ensemble']['f1']
    
    mt_improvement = ((multitask_f1 - simple_f1) / simple_f1 * 100)
    ens_improvement = ((ensemble_f1 - simple_f1) / simple_f1 * 100)
    
    print(f"\n🎯 Key Findings:")
    print(f"   Simple CNN (Baseline):")
    print(f"      F1: {simple_f1:.4f}")
    print(f"      AUC: {results_dict['Simple CNN']['auc']:.4f}")
    print(f"      Parameters: {model_info['Simple CNN']['params']/1e6:.1f}M")
    
    print(f"\n   Multi-Task CNN:")
    print(f"      F1: {multitask_f1:.4f} ({mt_improvement:+.2f}%)")
    print(f"      AUC: {results_dict['Multi-Task CNN']['auc']:.4f}")
    print(f"      Parameters: {model_info['Multi-Task CNN']['params']/1e6:.1f}M")
    
    print(f"\n   Ensemble:")
    print(f"      F1: {ensemble_f1:.4f} ({ens_improvement:+.2f}%)")
    print(f"      AUC: {results_dict['Ensemble']['auc']:.4f}")
    print(f"      Parameters: {model_info['Ensemble']['params']/1e6:.1f}M")
    
    # Determine winner
    print(f"\n🏆 Winner:")
    best_model = max(results_dict.items(), key=lambda x: x[1]['f1'])
    print(f"   {best_model[0]} achieves the best F1-Score: {best_model[1]['f1']:.4f}")
    
    # Analysis
    print(f"\n📝 Analysis:")
    if mt_improvement > 1:
        print(f"   ✅ Multi-task learning provides significant improvement (+{mt_improvement:.2f}%)")
    elif mt_improvement > 0:
        print(f"   ✅ Multi-task learning provides modest improvement (+{mt_improvement:.2f}%)")
    else:
        print(f"   ❌ Multi-task learning does not improve performance ({mt_improvement:.2f}%)")
    
    if ens_improvement > 1:
        print(f"   ✅ Ensemble provides significant improvement (+{ens_improvement:.2f}%)")
    elif ens_improvement > 0:
        print(f"   ✅ Ensemble provides modest improvement (+{ens_improvement:.2f}%)")
    else:
        print(f"   ❌ Ensemble does not improve performance ({ens_improvement:.2f}%)")
    
    # Cost-benefit analysis
    print(f"\n💰 Cost-Benefit Analysis:")
    simple_params = model_info['Simple CNN']['params']
    mt_params = model_info['Multi-Task CNN']['params']
    ens_params = model_info['Ensemble']['params']
    
    mt_param_increase = ((mt_params - simple_params) / simple_params * 100)
    ens_param_increase = ((ens_params - simple_params) / simple_params * 100)
    
    print(f"   Multi-Task CNN:")
    print(f"      Performance gain: {mt_improvement:+.2f}%")
    print(f"      Parameter increase: {mt_param_increase:+.2f}%")
    if mt_param_increase > 0:
        efficiency = mt_improvement / mt_param_increase
        print(f"      Efficiency: {efficiency:.3f} (gain per param%)")
    
    print(f"\n   Ensemble:")
    print(f"      Performance gain: {ens_improvement:+.2f}%")
    print(f"      Parameter increase: {ens_param_increase:+.2f}%")
    if ens_param_increase > 0:
        efficiency = ens_improvement / ens_param_increase
        print(f"      Efficiency: {efficiency:.3f} (gain per param%)")
    
    # Recommendation
    print(f"\n💡 Recommendation:")
    if mt_improvement > ens_improvement and mt_param_increase < ens_param_increase:
        print(f"   ✅ Multi-Task CNN offers the best trade-off:")
        print(f"      - Better efficiency ({mt_improvement/mt_param_increase:.3f} vs {ens_improvement/ens_param_increase:.3f})")
        print(f"      - Lower computational cost")
    elif ens_improvement > mt_improvement and best_model[0] == 'Ensemble':
        print(f"   ✅ Ensemble achieves the best performance:")
        print(f"      - Highest F1-Score: {ensemble_f1:.4f}")
        print(f"      - Worth the extra computational cost for critical applications")
    else:
        print(f"   ⚠️  Consider Simple CNN:")
        print(f"      - Simpler architecture")
        print(f"      - Minimal performance difference")
        print(f"      - Lower computational requirements")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Simple CNN vs Multi-Task CNN vs Ensemble"
    )
    
    # Model checkpoints
    parser.add_argument("--simple_checkpoint",
                       default="checkpoints/cnn_pneumo/cnn_pneumo_best.pth",
                       help="Path to Simple CNN checkpoint")
    parser.add_argument("--multitask_checkpoint",
                       default="D:\ChestXray_GAN_CNN_VIT_NEW\src\checkpoints\multitask_v3\multitask_v3_best.pth",
                       help="Path to Multi-Task CNN checkpoint")
    parser.add_argument("--ensemble_dir",
                       default="checkpoints/ensemble",
                       help="Directory containing ensemble checkpoints")
    
    # Data paths
    parser.add_argument("--csv",
                       default="../data/chestxray/Data_Entry_2017_v2020.csv")
    parser.add_argument("--images",
                       default="../data/chestxray/images")
    parser.add_argument("--val_list",
                       default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/val_BALANCED.txt")
    
    # Output
    parser.add_argument("--output_dir",
                       default="visualizations/three_way_comparison",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    main(args)