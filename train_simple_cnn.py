"""
Simple CNN for Pneumothorax Detection
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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
import argparse


class SimpleCNN(nn.Module):
    """Simple but effective CNN for binary classification"""

    def __init__(self, dropout=0.5):
        super().__init__()

        # Use pretrained ResNet18 backbone
        resnet = models.resnet18(pretrained=True)

        # Remove final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # New classification head
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


def evaluate(model, loader, device):
    """Evaluate with multiple thresholds"""
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_labels.append(labels.numpy())
            all_probs.append(probs)

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    # Try multiple thresholds
    best_f1 = 0
    best_thresh = 0.5
    all_results = {}

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = (y_prob >= thresh).astype(int)

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        all_results[thresh] = {
            'acc': acc, 'prec': precision,
            'rec': recall, 'f1': f1
        }

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    # Use best threshold
    y_pred = (y_prob >= best_thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5

    return {
        'auc': auc,
        'threshold': best_thresh,
        'accuracy': all_results[best_thresh]['acc'],
        'precision': all_results[best_thresh]['prec'],
        'recall': all_results[best_thresh]['rec'],
        'f1': best_f1,
        'confusion_matrix': cm,
        'all_thresholds': all_results
    }


def train(args):
    """Training loop"""

    print(f"\n{'=' * 70}")
    print(f"🚀 Simple CNN Training (ResNet18 Backbone)")
    print(f"{'=' * 70}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data transforms with augmentation
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
    print(f"\n🧠 Initializing ResNet18-based CNN...")
    model = SimpleCNN(dropout=args.dropout).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    print(f"\n⚖️  Using standard BCE Loss")
    print(f"   Learning rate: {args.lr}")
    print(f"   Optimizer: Adam with weight decay")

    best_f1 = 0
    checkpoint_dir = Path(args.ckpt_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"🎯 Starting training for {args.epochs} epochs")
    print(f"{'=' * 70}\n")

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Track predictions
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Training stats
        train_acc = accuracy_score(train_labels, train_preds)
        avg_loss = running_loss / len(train_loader)

        print(f"\n📊 Epoch {epoch}:")
        print(f"   Train Loss: {avg_loss:.4f}")
        print(f"   Train Accuracy: {train_acc:.4f}")

        # Validation
        print(f"   Validating...")
        metrics = evaluate(model, val_loader, device)

        print(f"   Val Threshold: {metrics['threshold']:.2f}")
        print(f"   Val AUC: {metrics['auc']:.4f}")
        print(f"   Val Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Val Precision: {metrics['precision']:.4f}")
        print(f"   Val Recall: {metrics['recall']:.4f}")
        print(f"   Val F1-Score: {metrics['f1']:.4f}")

        cm = metrics['confusion_matrix']
        print(f"   Confusion: TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}")

        # Show threshold analysis
        print(f"   Threshold Analysis:")
        for t, m in sorted(metrics['all_thresholds'].items()):
            print(f"      {t:.1f}: F1={m['f1']:.4f}, P={m['prec']:.4f}, R={m['rec']:.4f}")

        # Update LR scheduler
        scheduler.step(metrics['f1'])

        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }, checkpoint_dir / "cnn_pneumo_best.pth")
            print(f"   💾 New best! F1: {best_f1:.4f}")

        print()

    print(f"✅ Training complete! Best F1: {best_f1:.4f}")
    print(f"   Model saved: {checkpoint_dir / 'cnn_pneumo_best.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/Data_Entry_2017_v2020.csv")
    parser.add_argument("--images", default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/images")
    parser.add_argument("--train_list",
                        default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/train_BALANCED.txt")
    parser.add_argument("--val_list",
                        default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/val_BALANCED.txt")
    parser.add_argument("--synthetic", default="generated_images/synthetic_pneumothorax")

    parser.add_argument("--bs", type=int, default=32, help="Batch size")
    parser.add_argument("--workers", type=int, default=4, help="Num workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs")
    parser.add_argument("--ckpt_dir", default="checkpoints/cnn_pneumo", help="Checkpoint dir")

    args = parser.parse_args()
    train(args)