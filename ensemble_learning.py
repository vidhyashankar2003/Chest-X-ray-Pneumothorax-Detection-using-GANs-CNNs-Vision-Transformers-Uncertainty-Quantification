"""
Ensemble Learning for Pneumothorax Detection
Combines predictions from multiple architectures:
1. ResNet18
2. DenseNet121
3. EfficientNet-B0

Novel Contribution:
- Robust predictions through model diversity
- Weighted voting based on validation performance
- Reduces individual model errors
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from pneumo_dataset_v2 import PneumoDataset
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import argparse
import matplotlib.pyplot as plt


class ResNet18Model(nn.Module):
    """ResNet18-based classifier"""
    def __init__(self, dropout=0.5):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)


class DenseNet121Model(nn.Module):
    """DenseNet121-based classifier"""
    def __init__(self, dropout=0.5):
        super().__init__()
        densenet = models.densenet121(pretrained=True)
        self.features = densenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x.squeeze(1)


class EfficientNetB0Model(nn.Module):
    """EfficientNet-B0-based classifier"""
    def __init__(self, dropout=0.5):
        super().__init__()
        efficientnet = models.efficientnet_b0(pretrained=True)
        self.features = efficientnet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x.squeeze(1)


def get_model(architecture, dropout=0.5, device='cuda'):
    """Get model by architecture name"""
    if architecture == 'resnet18':
        return ResNet18Model(dropout).to(device)
    elif architecture == 'densenet121':
        return DenseNet121Model(dropout).to(device)
    elif architecture == 'efficientnet':
        return EfficientNetB0Model(dropout).to(device)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def train_single_model(model, train_loader, val_loader, device, args, model_name):
    """Train a single model"""
    
    print(f"\n{'='*70}")
    print(f"🚀 Training {model_name}")
    print(f"{'='*70}\n")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_f1 = 0
    checkpoint_dir = Path(args.checkpoint_dir) / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                
                val_labels.append(labels.numpy())
                val_probs.append(probs)
        
        y_true = np.concatenate(val_labels)
        y_prob = np.concatenate(val_probs)
        y_pred = (y_prob >= 0.5).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        auc = roc_auc_score(y_true, y_prob)
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        
        scheduler.step(f1)
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'f1': f1,
                'auc': auc
            }, checkpoint_dir / "best.pth")
            print(f"💾 Best model saved! F1: {best_f1:.4f}")
    
    print(f"\n✅ {model_name} training complete! Best F1: {best_f1:.4f}\n")
    return best_f1


class EnsembleModel:
    """
    Ensemble of multiple models with weighted voting
    
    Combines predictions from multiple architectures
    """
    
    def __init__(self, models, weights=None):
        """
        Args:
            models: List of (model, name) tuples
            weights: List of weights for each model (optional)
        """
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def predict(self, x, device='cuda'):
        """
        Get ensemble prediction
        
        Args:
            x: Input tensor
        
        Returns:
            Weighted average probability
        """
        predictions = []
        
        for (model, _), weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                output = model(x.to(device))
                prob = torch.sigmoid(output).cpu().numpy()
                predictions.append(prob * weight)
        
        # Weighted average
        ensemble_prob = np.sum(predictions, axis=0)
        return ensemble_prob
    
    def evaluate(self, loader, device='cuda'):
        """Evaluate ensemble on dataset"""
        all_labels = []
        all_probs = []
        
        for images, labels, _ in tqdm(loader, desc="Evaluating ensemble"):
            images = images.to(device)
            ensemble_probs = self.predict(images, device)
            
            all_labels.append(labels.numpy())
            all_probs.append(ensemble_probs)
        
        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)
        y_pred = (y_prob >= 0.5).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        auc = roc_auc_score(y_true, y_prob)
        
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }


def evaluate_individual_models(models, loader, device):
    """Evaluate each individual model"""
    results = {}
    
    for model, name in models:
        model.eval()
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(loader, desc=f"Evaluating {name}", leave=False):
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                
                all_labels.append(labels.numpy())
                all_probs.append(probs)
        
        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)
        y_pred = (y_prob >= 0.5).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        auc = roc_auc_score(y_true, y_prob)
        
        results[name] = {
            'f1': f1,
            'auc': auc,
            'precision': precision,
            'recall': recall
        }
    
    return results


def visualize_ensemble_results(individual_results, ensemble_results, output_path):
    """Create visualization comparing individual models vs ensemble"""
    
    model_names = list(individual_results.keys()) + ['Ensemble']
    f1_scores = [individual_results[name]['f1'] for name in individual_results.keys()]
    f1_scores.append(ensemble_results['f1'])
    
    auc_scores = [individual_results[name]['auc'] for name in individual_results.keys()]
    auc_scores.append(ensemble_results['auc'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # F1 scores
    colors = ['steelblue', 'coral', 'lightgreen', 'gold']
    bars1 = axes[0].bar(model_names, f1_scores, color=colors, edgecolor='black', alpha=0.8)
    axes[0].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    axes[0].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # AUC scores
    bars2 = axes[1].bar(model_names, auc_scores, color=colors, edgecolor='black', alpha=0.8)
    axes[1].set_ylabel('AUC', fontsize=12, fontweight='bold')
    axes[1].set_title('AUC Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Ensemble Learning Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Ensemble comparison saved: {output_path}")


def main(args):
    """Main execution"""
    
    print(f"\n{'='*70}")
    print(f"🎯 Ensemble Learning for Pneumothorax Detection")
    print(f"   Models: ResNet18, DenseNet121, EfficientNet-B0")
    print(f"{'='*70}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print(f"\n📂 Loading datasets...")
    train_ds = PneumoDataset(
        csv_path=args.csv,
        images_dir=args.images,
        list_txt=args.train_list,
        synthetic_dir=args.synthetic,
        synthetic_label=1,
        img_size=224,
        transform=train_transform
    )
    
    val_ds = PneumoDataset(
        csv_path=args.csv,
        images_dir=args.images,
        list_txt=args.val_list,
        synthetic_dir=None,
        img_size=224,
        transform=val_transform
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, 
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                           num_workers=args.workers, pin_memory=True)
    
    print(f"   Training: {len(train_ds)} samples")
    print(f"   Validation: {len(val_ds)} samples")
    
    # Train individual models
    architectures = ['resnet18', 'densenet121', 'efficientnet']
    trained_models = []
    model_weights = []
    
    if args.train_models:
        for arch in architectures:
            model = get_model(arch, args.dropout, device)
            f1_score = train_single_model(model, train_loader, val_loader, 
                                         device, args, arch)
            trained_models.append((model, arch))
            model_weights.append(f1_score)  # Weight by F1 score
    else:
        # Load pre-trained models
        print(f"\n📂 Loading pre-trained models...")
        for arch in architectures:
            model = get_model(arch, args.dropout, device)
            checkpoint_path = Path(args.checkpoint_dir) / arch / "best.pth"
            
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                f1_score = checkpoint['f1']
                model_weights.append(f1_score)
                trained_models.append((model, arch))
                print(f"✅ Loaded {arch} (F1: {f1_score:.4f})")
            else:
                print(f"❌ Checkpoint not found for {arch}")
    
    # Create ensemble
    print(f"\n🔗 Creating ensemble...")
    ensemble = EnsembleModel(trained_models, weights=model_weights)
    print(f"   Model weights:")
    for (_, name), weight in zip(trained_models, ensemble.weights):
        print(f"      {name}: {weight:.4f}")
    
    # Evaluate individual models
    print(f"\n📊 Evaluating individual models...")
    individual_results = evaluate_individual_models(trained_models, val_loader, device)
    
    for name, metrics in individual_results.items():
        print(f"   {name}:")
        print(f"      F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
    
    # Evaluate ensemble
    print(f"\n📊 Evaluating ensemble...")
    ensemble_results = ensemble.evaluate(val_loader, device)
    
    print(f"\n{'='*70}")
    print(f"🎉 ENSEMBLE RESULTS:")
    print(f"   Accuracy:  {ensemble_results['accuracy']:.4f}")
    print(f"   Precision: {ensemble_results['precision']:.4f}")
    print(f"   Recall:    {ensemble_results['recall']:.4f}")
    print(f"   F1-Score:  {ensemble_results['f1']:.4f}")
    print(f"   AUC:       {ensemble_results['auc']:.4f}")
    print(f"{'='*70}\n")
    
    # Compare with best individual model
    best_individual = max(individual_results.items(), key=lambda x: x[1]['f1'])
    best_name, best_metrics = best_individual
    
    improvement = (ensemble_results['f1'] - best_metrics['f1']) / best_metrics['f1'] * 100
    
    print(f"📈 Comparison:")
    print(f"   Best Individual ({best_name}): F1={best_metrics['f1']:.4f}")
    print(f"   Ensemble: F1={ensemble_results['f1']:.4f}")
    print(f"   Improvement: {improvement:+.2f}%\n")
    
    # Visualize results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_ensemble_results(individual_results, ensemble_results,
                              output_dir / "ensemble_comparison.png")
    
    # Save ensemble
    torch.save({
        'model_weights': ensemble.weights,
        'architectures': architectures,
        'ensemble_results': ensemble_results
    }, output_dir / "ensemble_model.pth")
    
    print(f"✅ Ensemble saved: {output_dir / 'ensemble_model.pth'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ensemble of models")
    
    parser.add_argument("--csv", default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/Data_Entry_2017_v2020.csv")
    parser.add_argument("--images", default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/images")
    parser.add_argument("--train_list", default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/train_BALANCED.txt")
    parser.add_argument("--val_list", default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/val_BALANCED.txt")
    parser.add_argument("--synthetic", default="generated_images/synthetic_pneumothorax")
    
    parser.add_argument("--bs", type=int, default=32, help="Batch size")
    parser.add_argument("--workers", type=int, default=4, help="Num workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per model")
    
    parser.add_argument("--train_models", action="store_true", default=True,
                       help="Train models from scratch (otherwise load checkpoints)")
    parser.add_argument("--checkpoint_dir", default="checkpoints/ensemble")
    parser.add_argument("--output_dir", default="visualizations/ensemble")
    
    args = parser.parse_args()
    main(args)
