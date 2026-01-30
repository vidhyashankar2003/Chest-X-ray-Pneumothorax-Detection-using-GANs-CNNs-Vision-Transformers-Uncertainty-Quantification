"""
CycleGAN Training Script for Chest X-ray Domain Translation
Trains unpaired image-to-image translation between:
  - Domain A: No Finding (healthy X-rays)
  - Domain B: Pneumothorax (disease X-rays)

Optimized for 8-hour training sessions with subset data
"""
import os
import itertools
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from networks_v2 import ResnetGenerator, PatchDiscriminator
from dataset_v2 import UnpairedXrayDataset

# Speed optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(img_dir, list_a, list_b, size, bs, workers):
    """Create DataLoader for training"""
    ds = UnpairedXrayDataset(img_dir, list_a, list_b, size=size, train=True)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
        prefetch_factor=2 if workers > 0 else None
    )


def save_checkpoint(epoch, G_AB, G_BA, D_A, D_B, opt_G, opt_D_A, opt_D_B, save_dir):
    """Save model checkpoints"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    torch.save(G_AB.state_dict(), save_dir / f"G_AB_epoch{epoch}.pth")
    torch.save(G_BA.state_dict(), save_dir / f"G_BA_epoch{epoch}.pth")
    torch.save(D_A.state_dict(), save_dir / f"D_A_epoch{epoch}.pth")
    torch.save(D_B.state_dict(), save_dir / f"D_B_epoch{epoch}.pth")

    # Save optimizers (for resuming training)
    torch.save({
        'epoch': epoch,
        'opt_G': opt_G.state_dict(),
        'opt_D_A': opt_D_A.state_dict(),
        'opt_D_B': opt_D_B.state_dict(),
    }, save_dir / f"optimizers_epoch{epoch}.pth")


def train(args):
    """Main training loop"""

    print(f"\n{'=' * 70}")
    print(f"🚀 Starting CycleGAN Training")
    print(f"{'=' * 70}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.size}")
    print(f"Learning rate: {args.lr}")
    print(f"{'=' * 70}\n")

    # Create DataLoader
    loader = make_loader(
        args.img_dir, args.list_a, args.list_b,
        args.size, args.batch_size, args.workers
    )

    print(f"📊 Training batches per epoch: {len(loader)}")
    print(f"⏱️  Estimated time per epoch: ~{len(loader) * args.batch_size / 1000:.1f} minutes\n")

    # Initialize models
    G_AB = ResnetGenerator(in_c=1, out_c=1, n_blocks=args.n_blocks).to(device)
    G_BA = ResnetGenerator(in_c=1, out_c=1, n_blocks=args.n_blocks).to(device)
    D_A = PatchDiscriminator(in_c=1).to(device)
    D_B = PatchDiscriminator(in_c=1).to(device)

    # Optional: Compile models (PyTorch 2.x - significant speedup)
    if args.torch_compile and hasattr(torch, 'compile'):
        print("⚡ Compiling models with torch.compile()...")
        G_AB = torch.compile(G_AB)
        G_BA = torch.compile(G_BA)
        D_A = torch.compile(D_A)
        D_B = torch.compile(D_B)

    # Use channels_last for better Tensor Core utilization
    for m in (G_AB, G_BA, D_A, D_B):
        m = m.to(memory_format=torch.channels_last)

    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Optimizers
    opt_G = optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999)
    )
    opt_D_A = optim.Adam(D_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D_B = optim.Adam(D_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # Learning rate scheduler (optional but recommended)
    if args.lr_decay:
        scheduler_G = optim.lr_scheduler.LambdaLR(
            opt_G,
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - args.epochs // 2) / (args.epochs // 2)
        )
        scheduler_D_A = optim.lr_scheduler.LambdaLR(
            opt_D_A,
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - args.epochs // 2) / (args.epochs // 2)
        )
        scheduler_D_B = optim.lr_scheduler.LambdaLR(
            opt_D_B,
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - args.epochs // 2) / (args.epochs // 2)
        )

    # Training loop
    for epoch in range(1, args.epochs + 1):
        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()

        epoch_loss_G = 0.0
        epoch_loss_D_A = 0.0
        epoch_loss_D_B = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")

        for i, batch in enumerate(pbar):
            real_A = batch["A"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
            real_B = batch["B"].to(device, non_blocking=True).to(memory_format=torch.channels_last)

            batch_size = real_A.size(0)

            # ------------------
            #  Train Generators
            # ------------------
            opt_G.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                # Identity loss (helps preserve color/intensity)
                id_A = G_BA(real_A)
                id_B = G_AB(real_B)
                loss_identity = (
                                        criterion_identity(id_A, real_A) +
                                        criterion_identity(id_B, real_B)
                                ) * 0.5 * args.lambda_identity

                # GAN loss
                fake_B = G_AB(real_A)
                fake_A = G_BA(real_B)

                pred_fake_B = D_B(fake_B)
                pred_fake_A = D_A(fake_A)

                # Create target labels with same shape as discriminator output
                valid_B = torch.ones_like(pred_fake_B, device=device) * 0.9
                valid_A = torch.ones_like(pred_fake_A, device=device) * 0.9

                loss_GAN_AB = criterion_GAN(pred_fake_B, valid_B)
                loss_GAN_BA = criterion_GAN(pred_fake_A, valid_A)

                # Cycle consistency loss
                rec_A = G_BA(fake_B)
                rec_B = G_AB(fake_A)

                loss_cycle = (
                                     criterion_cycle(rec_A, real_A) +
                                     criterion_cycle(rec_B, real_B)
                             ) * args.lambda_cycle

                # Total generator loss
                loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle + loss_identity

            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            scaler.update()

            epoch_loss_G += loss_G.item()

            # -----------------------
            #  Train Discriminators
            # -----------------------
            if (i % args.d_every) == 0:
                # Train D_A
                opt_D_A.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    pred_real_A = D_A(real_A)
                    fake_A_det = fake_A.detach()
                    pred_fake_A = D_A(fake_A_det)

                    # Create labels with correct shape
                    valid_A_d = torch.ones_like(pred_real_A, device=device) * 0.9
                    fake_A_label = torch.zeros_like(pred_fake_A, device=device) + 0.1

                    loss_real_A = criterion_GAN(pred_real_A, valid_A_d)
                    loss_fake_A = criterion_GAN(pred_fake_A, fake_A_label)
                    loss_D_A = (loss_real_A + loss_fake_A) * 0.5

                scaler.scale(loss_D_A).backward()
                scaler.step(opt_D_A)
                scaler.update()

                # Train D_B
                opt_D_B.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    pred_real_B = D_B(real_B)
                    fake_B_det = fake_B.detach()
                    pred_fake_B = D_B(fake_B_det)

                    # Create labels with correct shape
                    valid_B_d = torch.ones_like(pred_real_B, device=device) * 0.9
                    fake_B_label = torch.zeros_like(pred_fake_B, device=device) + 0.1

                    loss_real_B = criterion_GAN(pred_real_B, valid_B_d)
                    loss_fake_B = criterion_GAN(pred_fake_B, fake_B_label)
                    loss_D_B = (loss_real_B + loss_fake_B) * 0.5

                scaler.scale(loss_D_B).backward()
                scaler.step(opt_D_B)
                scaler.update()

                epoch_loss_D_A += loss_D_A.item()
                epoch_loss_D_B += loss_D_B.item()
            else:
                loss_D_A = torch.tensor(0.0)
                loss_D_B = torch.tensor(0.0)

            # Update progress bar
            pbar.set_postfix({
                'G': f'{loss_G.item():.4f}',
                'D_A': f'{loss_D_A.item():.4f}',
                'D_B': f'{loss_D_B.item():.4f}'
            })

        # Epoch summary
        avg_loss_G = epoch_loss_G / len(loader)
        avg_loss_D_A = epoch_loss_D_A / (len(loader) // args.d_every)
        avg_loss_D_B = epoch_loss_D_B / (len(loader) // args.d_every)

        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Avg Loss_G: {avg_loss_G:.4f}")
        print(f"   Avg Loss_D_A: {avg_loss_D_A:.4f}")
        print(f"   Avg Loss_D_B: {avg_loss_D_B:.4f}")

        # Update learning rate
        if args.lr_decay:
            scheduler_G.step()
            scheduler_D_A.step()
            scheduler_D_B.step()
            print(f"   Learning rate: {scheduler_G.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(epoch, G_AB, G_BA, D_A, D_B, opt_G, opt_D_A, opt_D_B, args.checkpoint_dir)
            print(f"💾 Checkpoint saved at epoch {epoch}")

        print()

    # Save final checkpoint
    save_checkpoint(args.epochs, G_AB, G_BA, D_A, D_B, opt_G, opt_D_A, opt_D_B, args.checkpoint_dir)

    print(f"\n{'=' * 70}")
    print(f"✅ Training completed!")
    print(f"   Total epochs: {args.epochs}")
    print(f"   Final checkpoints saved to: {args.checkpoint_dir}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CycleGAN Training for Chest X-rays")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs (15 epochs with 20% data ~= 3 epochs with full data)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (increase if you have more GPU memory)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--size", type=int, default=256, help="Image size")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers")

    # Loss weights
    parser.add_argument("--lambda_cycle", type=float, default=10.0, help="Cycle consistency loss weight")
    parser.add_argument("--lambda_identity", type=float, default=5.0, help="Identity loss weight")

    # Model architecture
    parser.add_argument("--n_blocks", type=int, default=6, help="Number of ResNet blocks (6 or 9)")

    # Paths
    parser.add_argument("--img_dir", default="D:/ChestXray_GAN_CNN_ViT/data/chestxray", help="Root image directory")
    parser.add_argument("--list_a",
                        default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/train_A_small.txt")
    parser.add_argument("--list_b",
                        default="D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/train_B_small.txt")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Directory to save checkpoints")

    # Optimization
    parser.add_argument("--d_every", type=int, default=1, help="Update discriminators every N steps")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs (default: every 5 epochs)")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use automatic mixed precision")
    parser.add_argument("--torch_compile", action="store_true", help="Use torch.compile() for speedup")
    parser.add_argument("--lr_decay", action="store_true", default=True, help="Use learning rate decay")

    args = parser.parse_args()

    train(args)