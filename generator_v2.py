"""
Synthetic Image Generation Script
Loads trained CycleGAN models and generates synthetic chest X-rays
- Domain A → B: Generate synthetic Pneumothorax images
- Domain B → A: Generate synthetic healthy images
"""
import torch
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image
from networks_v2 import ResnetGenerator
from dataset_v2 import UnpairedXrayDataset
from torch.utils.data import DataLoader


def denormalize(tensor):
    """
    Convert tensor from [-1, 1] to [0, 1] range for proper image saving
    CRITICAL: This was missing in the original code!
    """
    return (tensor + 1.0) / 2.0


def generate_images(args):
    """Generate synthetic images using trained CycleGAN models"""
    
    print(f"\n{'='*70}")
    print(f"🎨 Generating Synthetic Images")
    print(f"{'='*70}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create output directories
    output_dir_A = Path(args.output_dir) / "synthetic_healthy"
    output_dir_B = Path(args.output_dir) / "synthetic_pneumothorax"
    output_dir_A.mkdir(parents=True, exist_ok=True)
    output_dir_B.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directories:")
    print(f"   Synthetic healthy (Domain A): {output_dir_A}")
    print(f"   Synthetic Pneumothorax (Domain B): {output_dir_B}")
    
    # Load trained models
    print(f"\n📂 Loading models from epoch {args.epoch}...")
    G_AB = ResnetGenerator(in_c=1, out_c=1, n_blocks=args.n_blocks).to(device)
    G_BA = ResnetGenerator(in_c=1, out_c=1, n_blocks=args.n_blocks).to(device)
    
    checkpoint_dir = Path(args.checkpoint_dir)
    G_AB_path = checkpoint_dir / f"G_AB_epoch{args.epoch}.pth"
    G_BA_path = checkpoint_dir / f"G_BA_epoch{args.epoch}.pth"
    
    if not G_AB_path.exists() or not G_BA_path.exists():
        raise FileNotFoundError(
            f"Model checkpoints not found!\n"
            f"Looking for:\n"
            f"  {G_AB_path}\n"
            f"  {G_BA_path}\n"
            f"Available epochs: {sorted([int(p.stem.split('epoch')[1]) for p in checkpoint_dir.glob('G_AB_epoch*.pth')])}"
        )
    
    G_AB.load_state_dict(torch.load(G_AB_path, map_location=device))
    G_BA.load_state_dict(torch.load(G_BA_path, map_location=device))
    
    G_AB.eval()
    G_BA.eval()
    
    print("✅ Models loaded successfully")
    
    # Load dataset
    print(f"\n📊 Loading dataset...")
    dataset = UnpairedXrayDataset(
        img_dir=args.img_dir,
        list_A=args.list_a,
        list_B=args.list_b,
        size=args.size,
        train=False  # No augmentation for generation
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    print(f"Total batches: {len(dataloader)}")
    
    # Generate images
    print(f"\n🎨 Generating images...")
    count_A = 0
    count_B = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)
            
            # Generate synthetic images
            fake_B = G_AB(real_A)  # Healthy → Pneumothorax
            fake_A = G_BA(real_B)  # Pneumothorax → Healthy
            
            # CRITICAL FIX: Denormalize from [-1, 1] to [0, 1]
            fake_B = denormalize(fake_B)
            fake_A = denormalize(fake_A)
            
            # Clamp to valid range
            fake_B = torch.clamp(fake_B, 0, 1)
            fake_A = torch.clamp(fake_A, 0, 1)
            
            # Save each image individually
            for j in range(fake_B.size(0)):
                # Save synthetic Pneumothorax (Domain B)
                filename_B = f"synthetic_pneumo_{count_B + 1:05d}.png"
                save_image(fake_B[j], output_dir_B / filename_B)
                count_B += 1
                
                # Save synthetic healthy (Domain A)
                filename_A = f"synthetic_healthy_{count_A + 1:05d}.png"
                save_image(fake_A[j], output_dir_A / filename_A)
                count_A += 1
            
            # Optional: Limit number of generated images
            if args.max_images > 0 and count_B >= args.max_images:
                break
    
    print(f"\n{'='*70}")
    print(f"✅ Generation complete!")
    print(f"   Synthetic Pneumothorax images: {count_B}")
    print(f"   Synthetic healthy images: {count_A}")
    print(f"{'='*70}\n")
    
    # Verify a sample image
    sample_B = list(output_dir_B.glob("*.png"))[0]
    print(f"📋 Sample verification:")
    print(f"   File: {sample_B.name}")
    print(f"   Location: {sample_B}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic chest X-rays using trained CycleGAN")
    
    # Model parameters
    parser.add_argument("--epoch", type=int, default=15, help="Epoch number to load (default: 15)")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Directory containing model checkpoints")
    parser.add_argument("--n_blocks", type=int, default=6, help="Number of ResNet blocks (must match training)")
    
    # Data parameters
    parser.add_argument("--img_dir", default="D:/ChestXray_GAN_CNN_ViT/data/chestxray", help="Root image directory")
    parser.add_argument("--list_a", default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/val_A.txt",
                        help="Domain A image list (use validation set)")
    parser.add_argument("--list_b", default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/val_B.txt",
                        help="Domain B image list (use validation set)")
    parser.add_argument("--size", type=int, default=256, help="Image size")
    
    # Generation parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--max_images", type=int, default=-1, help="Max images to generate (-1 for all)")
    parser.add_argument("--output_dir", default="generated_images", help="Output directory for synthetic images")
    
    args = parser.parse_args()
    
    generate_images(args)