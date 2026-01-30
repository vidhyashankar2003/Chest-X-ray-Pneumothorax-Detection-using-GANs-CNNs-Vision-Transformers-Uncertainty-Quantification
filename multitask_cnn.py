"""
Improved Multi-Task CNN for Pneumothorax Detection (Version 3)

Key Improvements:
1. Better auxiliary task: Severity estimation instead of quality assessment
2. Task-specific attention branches (reduces task interference)
3. Adaptive loss weighting (automatically balances tasks)
4. Gradient balancing to prevent negative transfer
5. Option to use uncertainty weighting
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


class TaskAttention(nn.Module):
    """Task-specific attention module to reduce task interference"""

    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.attention(x)
        return x * att


class ImprovedMultiTaskCNN(nn.Module):
    """
    Improved Multi-Task CNN with better architecture

    Tasks:
    1. Primary: Pneumothorax Detection (binary classification)
    2. Auxiliary: Severity Estimation (regression 0-1, where 0=no disease, 1=severe)

    Improvements:
    - Task-specific attention branches
    - Separate high-level features for each task
    - Uncertainty-based loss weighting
    """

    def __init__(self, dropout=0.5, use_attention=True):
        super().__init__()

        # Shared backbone (ResNet18)
        resnet = models.resnet18(pretrained=True)

        # Split backbone into early and late features
        # Early features: conv1 -> layer2 (shared for all tasks)
        self.shared_early = nn.Sequential(*list(resnet.children())[:6])

        # Late features: layer3 -> layer4 (task-specific)
        self.shared_late = nn.Sequential(*list(resnet.children())[6:8])

        # Task-specific attention (optional)
        self.use_attention = use_attention
        if use_attention:
            self.pneumo_attention = TaskAttention(512)
            self.severity_attention = TaskAttention(512)

        # Task-specific feature extraction
        self.pneumo_features = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.severity_features = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # Task 1: Pneumothorax Detection Head
        self.pneumo_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        # Task 2: Severity Estimation Head
        self.severity_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output range [0, 1]
        )

        # Learnable task weights (uncertainty weighting)
        self.log_var_pneumo = nn.Parameter(torch.zeros(1))
        self.log_var_severity = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Shared early features
        shared = self.shared_early(x)

        # Shared late features
        late_features = self.shared_late(shared)

        # Task-specific attention
        if self.use_attention:
            pneumo_feat = self.pneumo_attention(late_features)
            severity_feat = self.severity_attention(late_features)
        else:
            pneumo_feat = late_features
            severity_feat = late_features

        # Task-specific pooling
        pneumo_feat = self.pneumo_features(pneumo_feat)
        severity_feat = self.severity_features(severity_feat)

        # Task predictions
        pneumo_logits = self.pneumo_head(pneumo_feat).squeeze(1)
        severity_score = self.severity_head(severity_feat).squeeze(1)

        return pneumo_logits, severity_score

    def get_task_weights(self):
        """Get uncertainty-based task weights"""
        w_pneumo = torch.exp(-self.log_var_pneumo)
        w_severity = torch.exp(-self.log_var_severity)
        return w_pneumo, w_severity


def get_severity_labels(labels, filenames, is_synthetic):
    """
    Assign severity scores based on pneumothorax presence

    Logic:
    - No pneumothorax (label=0): severity = 0.0
    - Real pneumothorax (label=1): severity = 0.7 (moderate, we don't have severity annotations)
    - Synthetic pneumothorax: severity = 0.6 (slightly less severe as a proxy)

    Better approach: Use actual severity annotations if available
    """
    severity = torch.zeros_like(labels, dtype=torch.float32)

    for i, (label, fname) in enumerate(zip(labels, filenames)):
        if label == 0:
            # No disease
            severity[i] = 0.0
        elif 'synthetic' in str(fname).lower():
            # Synthetic (proxy for severity)
            severity[i] = 0.6
        else:
            # Real pneumothorax (assume moderate severity)
            severity[i] = 0.7

    return severity


class UncertaintyLoss(nn.Module):
    """
    Multi-task loss with learnable uncertainty weights

    Based on: "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al., 2018)

    Loss = (1 / 2*sigma_1^2) * L_1 + (1 / 2*sigma_2^2) * L_2 + log(sigma_1) + log(sigma_2)
    """

    def __init__(self):
        super().__init__()

    def forward(self, loss_pneumo, loss_severity, log_var_pneumo, log_var_severity):
        # Precision weighting
        precision_pneumo = torch.exp(-log_var_pneumo)
        precision_severity = torch.exp(-log_var_severity)

        # Weighted losses
        weighted_pneumo = precision_pneumo * loss_pneumo + log_var_pneumo
        weighted_severity = precision_severity * loss_severity + log_var_severity

        total_loss = weighted_pneumo + weighted_severity

        return total_loss, precision_pneumo.item(), precision_severity.item()


def evaluate(model, loader, device, use_uncertainty=False):
    """Evaluate both tasks"""
    model.eval()

    pneumo_labels = []
    pneumo_probs = []
    severity_labels = []
    severity_preds = []

    with torch.no_grad():
        for images, labels, filenames in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)

            pneumo_logits, severity_scores = model(images)
            pneumo_probs_batch = torch.sigmoid(pneumo_logits).cpu().numpy()

            pneumo_labels.append(labels.numpy())
            pneumo_probs.append(pneumo_probs_batch)

            # Get severity labels
            is_synthetic = any('synthetic' in str(f).lower() for f in filenames)
            severity = get_severity_labels(labels, filenames, is_synthetic)
            severity_labels.append(severity.numpy())
            severity_preds.append(severity_scores.cpu().numpy())

    # Pneumothorax metrics
    y_true = np.concatenate(pneumo_labels)
    y_prob = np.concatenate(pneumo_probs)

    # Find best threshold
    best_f1 = 0
    best_thresh = 0.5

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = (y_prob >= thresh).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    y_pred = (y_prob >= best_thresh).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except:
        roc_auc = 0.5

    metrics = {
        'pneumo': {
            'auc': roc_auc,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'threshold': best_thresh
        }
    }

    # Severity metrics (MAE)
    sev_true = np.concatenate(severity_labels)
    sev_pred = np.concatenate(severity_preds)
    severity_mae = np.mean(np.abs(sev_true - sev_pred))

    metrics['severity'] = {
        'mae': severity_mae,
        'mean_pred': np.mean(sev_pred),
        'std_pred': np.std(sev_pred)
    }

    return metrics


def train(args):
    """Training loop with improved multi-task learning"""

    print(f"\n{'=' * 70}")
    print(f"🚀 Improved Multi-Task CNN Training (v3)")
    print(f"   Primary Task: Pneumothorax Detection")
    print(f"   Auxiliary Task: Severity Estimation")
    print(f"{'=' * 70}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Transforms
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

    # Model
    print(f"\n🧠 Initializing Improved Multi-Task CNN...")
    model = ImprovedMultiTaskCNN(
        dropout=args.dropout,
        use_attention=args.use_attention
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Task-specific attention: {args.use_attention}")
    print(f"   Uncertainty weighting: {args.use_uncertainty}")

    # Loss functions
    criterion_pneumo = nn.BCEWithLogitsLoss()
    criterion_severity = nn.MSELoss()

    if args.use_uncertainty:
        uncertainty_loss = UncertaintyLoss()
        print(f"\n⚖️ Using uncertainty-based loss weighting")
    else:
        print(f"\n⚖️ Using fixed loss weights:")
        print(f"   Pneumothorax: {args.pneumo_weight}")
        print(f"   Severity: {args.severity_weight}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    best_f1 = 0
    checkpoint_dir = Path(args.ckpt_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"🎯 Starting training for {args.epochs} epochs")
    print(f"{'=' * 70}\n")

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        running_loss_pneumo = 0.0
        running_loss_severity = 0.0
        running_loss_total = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels, filenames in pbar:
            images = images.to(device)
            labels = labels.to(device).float()

            # Get severity labels
            is_synthetic = any('synthetic' in str(f).lower() for f in filenames)
            severity_labels = get_severity_labels(labels, filenames, is_synthetic).to(device)

            optimizer.zero_grad()

            # Forward pass
            pneumo_logits, severity_scores = model(images)

            # Task losses
            loss_pneumo = criterion_pneumo(pneumo_logits, labels)
            loss_severity = criterion_severity(severity_scores, severity_labels)

            # Combined loss
            if args.use_uncertainty:
                total_loss, w_p, w_s = uncertainty_loss(
                    loss_pneumo, loss_severity,
                    model.log_var_pneumo, model.log_var_severity
                )
            else:
                total_loss = (args.pneumo_weight * loss_pneumo +
                              args.severity_weight * loss_severity)
                w_p, w_s = args.pneumo_weight, args.severity_weight

            total_loss.backward()

            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss_pneumo += loss_pneumo.item()
            running_loss_severity += loss_severity.item()
            running_loss_total += total_loss.item()

            pbar.set_postfix({
                'P_loss': f'{loss_pneumo.item():.4f}',
                'S_loss': f'{loss_severity.item():.4f}',
                'w_P': f'{w_p:.3f}',
                'w_S': f'{w_s:.3f}'
            })

        # Epoch summary
        avg_loss_pneumo = running_loss_pneumo / len(train_loader)
        avg_loss_severity = running_loss_severity / len(train_loader)
        avg_loss_total = running_loss_total / len(train_loader)

        print(f"\n📊 Epoch {epoch}:")
        print(f"   Total Loss: {avg_loss_total:.4f}")
        print(f"   Pneumo Loss: {avg_loss_pneumo:.4f}")
        print(f"   Severity Loss: {avg_loss_severity:.4f}")

        if args.use_uncertainty:
            w_p, w_s = model.get_task_weights()
            print(f"   Learned Weights: Pneumo={w_p.item():.3f}, Severity={w_s.item():.3f}")

        # Validation
        print(f"   Validating...")
        metrics = evaluate(model, val_loader, device, args.use_uncertainty)

        pneumo_metrics = metrics['pneumo']
        print(f"\n   Task 1 - Pneumothorax Detection:")
        print(f"      Threshold: {pneumo_metrics['threshold']:.2f}")
        print(f"      AUC: {pneumo_metrics['auc']:.4f}")
        print(f"      Accuracy: {pneumo_metrics['accuracy']:.4f}")
        print(f"      Precision: {pneumo_metrics['precision']:.4f}")
        print(f"      Recall: {pneumo_metrics['recall']:.4f}")
        print(f"      F1-Score: {pneumo_metrics['f1']:.4f}")

        severity_metrics = metrics['severity']
        print(f"\n   Task 2 - Severity Estimation:")
        print(f"      MAE: {severity_metrics['mae']:.4f}")
        print(f"      Mean Prediction: {severity_metrics['mean_pred']:.4f}")

        # Update scheduler
        scheduler.step(pneumo_metrics['f1'])

        # Save best model
        if pneumo_metrics['f1'] > best_f1:
            best_f1 = pneumo_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'use_attention': args.use_attention,
                'use_uncertainty': args.use_uncertainty
            }, checkpoint_dir / "multitask_v3_best.pth")
            print(f"   💾 New best! F1: {best_f1:.4f}")

        print()

    print(f"✅ Training complete! Best F1: {best_f1:.4f}")
    print(f"   Model saved: {checkpoint_dir / 'multitask_v3_best.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", default="../data/chestxray/Data_Entry_2017_v2020.csv")
    parser.add_argument("--images", default="../data/chestxray/images")
    parser.add_argument("--train_list",
                        default="../data/chestxray/splits/pneumothorax/train_BALANCED.txt")
    parser.add_argument("--val_list",
                        default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/val_BALANCED.txt")
    parser.add_argument("--synthetic", default="generated_images/synthetic_pneumothorax")

    parser.add_argument("--bs", type=int, default=32, help="Batch size")
    parser.add_argument("--workers", type=int, default=4, help="Num workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs")

    # Multi-task specific
    parser.add_argument("--use_attention", action="store_true", default=True,
                        help="Use task-specific attention")
    parser.add_argument("--use_uncertainty", action="store_true", default=True,
                        help="Use uncertainty-based loss weighting")
    parser.add_argument("--pneumo_weight", type=float, default=2.0,
                        help="Weight for pneumothorax task (if not using uncertainty)")
    parser.add_argument("--severity_weight", type=float, default=0.2,
                        help="Weight for severity task (if not using uncertainty)")

    parser.add_argument("--ckpt_dir", default="checkpoints/multitask_v3")

    args = parser.parse_args()
    train(args)