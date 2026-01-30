"""
DataLoader creation for CNN Real vs Synthetic classifier
"""
import torch
from torch.utils.data import DataLoader
from xraydataset_v2 import XrayDataset
from torchvision import transforms


def create_dataloader(real_images_dir, synthetic_images_dir, list_A, list_B,
                      batch_size=16, num_workers=4, img_size=256, train=True):
    """
    Create DataLoader for Real vs Synthetic classification
    
    Args:
        real_images_dir: Directory containing real images
        synthetic_images_dir: Directory containing synthetic images
        list_A: Path to Domain A split file
        list_B: Path to Domain B split file
        batch_size: Batch size
        num_workers: Number of data loading workers
        img_size: Image size
        train: Whether this is for training (affects augmentation)
    
    Returns:
        DataLoader
    """
    
    # Define transformations
    transform_list = []
    
    # Augmentation for training only
    if train:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
        ])
    
    # Always apply these
    transform_list.extend([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    transform = transforms.Compose(transform_list)
    
    # Create dataset
    dataset = XrayDataset(
        real_images_dir=real_images_dir,
        synthetic_images_dir=synthetic_images_dir,
        list_A=list_A,
        list_B=list_B,
        transform=transform
    )
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train  # Drop last incomplete batch for training
    )
    
    return loader


# Pre-configured loaders for convenience
def get_train_loader(batch_size=16, num_workers=4):
    """Get training DataLoader with default paths"""
    return create_dataloader(
        real_images_dir="D:/ChestXray_GAN_CNN_ViT/data/chestxray/images",
        synthetic_images_dir="generated_images/synthetic_pneumothorax",
        list_A="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/train_A.txt",
        list_B="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/train_B.txt",
        batch_size=batch_size,
        num_workers=num_workers,
        train=True
    )


def get_val_loader(batch_size=16, num_workers=4):
    """Get validation DataLoader with default paths"""
    return create_dataloader(
        real_images_dir="D:/ChestXray_GAN_CNN_ViT/data/chestxray/images",
        synthetic_images_dir="generated_images/synthetic_pneumothorax",
        list_A="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/val_A.txt",
        list_B="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/val_B.txt",
        batch_size=batch_size,
        num_workers=num_workers,
        train=False
    )


if __name__ == "__main__":
    print("🧪 Testing DataLoader creation...\n")
    
    train_loader = get_train_loader(batch_size=8, num_workers=0)
    val_loader = get_val_loader(batch_size=8, num_workers=0)
    
    print(f"\n📊 DataLoader statistics:")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"\n   Sample batch:")
    print(f"      Shape: {images.shape}")
    print(f"      Labels: {labels}")
    print(f"      Value range: [{images.min():.3f}, {images.max():.3f}]")
    
    print("\n✅ DataLoader test passed!")
