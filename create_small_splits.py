"""
Create smaller training splits for faster experimentation
Use 20% of training data to get 10 epochs in ~8 hours instead of 2 epochs with full data
"""
from pathlib import Path

SPLIT_DIR = Path("D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax")
SUBSET_RATIO = 0.2  # Use 20% of data

def create_small_split(input_file, output_file, ratio=0.2):
    """Create smaller split file"""
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Take first N% of lines
    n = int(len(lines) * ratio)
    subset = lines[:n]
    
    with open(output_file, 'w') as f:
        f.writelines(subset)
    
    print(f"Created {output_file.name}: {n} images (from {len(lines)})")

# Create small splits
create_small_split(
    SPLIT_DIR / "train_A.txt",
    SPLIT_DIR / "train_A_small.txt",
    SUBSET_RATIO
)

create_small_split(
    SPLIT_DIR / "train_B.txt", 
    SPLIT_DIR / "train_B_small.txt",
    SUBSET_RATIO
)

print(f"\n✅ Small splits created! Now run:")
print(f"python train.py --epochs 10 --batch_size 8 \\")
print(f"  --list_a {SPLIT_DIR}/train_A_small.txt \\")
print(f"  --list_b {SPLIT_DIR}/train_B_small.txt")
