"""
CNN Classifier for Real vs Synthetic Detection
Binary classification: 0 = Real X-ray, 1 = Synthetic (GAN-generated)
Uses ResNet18 backbone for stability and performance
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from xraydataset_v2 import XrayDataset
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
import argparse


class RealVsFakeCNN(nn.Module):
    """CNN for detecting synthetic images - ResNet18 backbone"""

    def __init__(self, dropout=0.5):
        super().__init__()

        # Use pretrained ResNet18
        resnet = models.resnet18(pretrained=True)

        # Remove final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # New classification head for binary classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 2)  # 2 classes: Real, Synthetic
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def evaluate(model, loader, device):
    """Evaluate model on validation set"""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Prob of class 1 (synthetic)
            _, preds = torch.max(outputs, 1)

            all_labels.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }


def train(args):
    """Training loop"""

    print(f"\n{'=' * 70}")
    print(f"🚀 Real vs Synthetic CNN Training")
    print(f"{'=' * 70}\n")

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
    train_dataset = XrayDataset(
        real_images_dir=args.real_dir,
        synthetic_images_dir=args.synthetic_dir,
        list_A=args.train_A,
        list_B=args.train_B,
        transform=train_transform
    )

    val_dataset = XrayDataset(
        real_images_dir=args.real_dir,
        synthetic_images_dir=args.synthetic_dir,
        list_A=args.val_A,
        list_B=args.val_B,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")

    # Model
    print(f"\n🧠 Initializing ResNet18-based CNN...")
    model = RealVsFakeCNN(dropout=args.dropout).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    print(f"\n⚖️  Using CrossEntropyLoss")
    print(f"   Learning rate: {args.lr}")

    best_acc = 0
    checkpoint_dir = Path(args.ckpt_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"🎯 Starting training for {args.epochs} epochs")
    print(f"{'=' * 70}\n")

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

        # Training stats
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        print(f"\n📊 Epoch {epoch}:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Train Accuracy: {train_acc:.2f}%")

        # Validation
        print(f"   Validating...")
        metrics = evaluate(model, val_loader, device)

        print(f"   Val Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Val Precision: {metrics['precision']:.4f}")
        print(f"   Val Recall: {metrics['recall']:.4f}")
        print(f"   Val F1-Score: {metrics['f1']:.4f}")
        print(f"   Val AUC: {metrics['auc']:.4f}")

        cm = metrics['confusion_matrix']
        print(f"   Confusion Matrix:")
        print(f"      TN={cm[0, 0]} (Real→Real), FP={cm[0, 1]} (Real→Fake)")
        print(f"      FN={cm[1, 0]} (Fake→Real), TP={cm[1, 1]} (Fake→Fake)")

        # Interpretation
        if metrics['accuracy'] >= 0.95:
            print(f"   ⚠️  High accuracy (≥95%): Synthetic images easily detectable")
        elif metrics['accuracy'] >= 0.85:
            print(f"   ✅ Good balance (85-95%): Synthetic images somewhat realistic")
        else:
            print(f"   ✅ Excellent! (<85%): Synthetic images very realistic")

        # Update scheduler
        scheduler.step(metrics['accuracy'])

        # Save best model
        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }, checkpoint_dir / "real_vs_fake_best.pth")
            print(f"   💾 New best! Accuracy: {best_acc:.4f}")

        print()

    print(f"✅ Training complete! Best Accuracy: {best_acc:.4f}")
    print(f"   Model saved: {checkpoint_dir / 'real_vs_fake_best.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN for Real vs Synthetic detection")

    # Data paths
    parser.add_argument("--real_dir", default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/images")
    parser.add_argument("--synthetic_dir", default="generated_images/synthetic_pneumothorax")
    parser.add_argument("--train_A", default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/train_A.txt")
    parser.add_argument("--train_B", default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/train_B.txt")
    parser.add_argument("--val_A", default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/val_A.txt")
    parser.add_argument("--val_B", default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/val_B.txt")

    # Training parameters
    parser.add_argument("--bs", type=int, default=32, help="Batch size")
    parser.add_argument("--workers", type=int, default=4, help="Num workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--ckpt_dir", default="checkpoints/real_vs_fake", help="Checkpoint directory")

    args = parser.parse_args()
    train(args)