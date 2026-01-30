"""
Merge Domain Splits for Classification Training
Combines Domain A and B splits into unified train_ALL.txt and val_ALL.txt
Used for supervised classification (ViT/CNN training)
"""
from pathlib import Path

# Configuration
SPLIT_DIR = Path("D:/ChestXray_GAN_CNN_VIT_NEW/data/chestxray/splits/pneumothorax")

def merge_splits():
    """Merge train_A + train_B and val_A + val_B"""
    
    # Define file paths
    train_A = SPLIT_DIR / "train_A.txt"
    train_B = SPLIT_DIR / "train_B.txt"
    val_A = SPLIT_DIR / "val_A.txt"
    val_B = SPLIT_DIR / "val_B.txt"
    
    train_ALL = SPLIT_DIR / "train_ALL.txt"
    val_ALL = SPLIT_DIR / "val_ALL.txt"
    
    # Check if input files exist
    required_files = [train_A, train_B, val_A, val_B]
    missing_files = [f for f in required_files if not f.exists()]
    
    if missing_files:
        print("❌ Error: Missing required split files:")
        for f in missing_files:
            print(f"   {f.name}")
        print("\n💡 Run build_domain_splits.py first!")
        return
    
    print("📂 Merging domain splits...")
    
    # Read and merge train splits
    with open(train_A, "r") as f1, open(train_B, "r") as f2:
        train_lines = f1.readlines() + f2.readlines()
    
    # Read and merge val splits
    with open(val_A, "r") as f1, open(val_B, "r") as f2:
        val_lines = f1.readlines() + f2.readlines()
    
    # Write merged files
    with open(train_ALL, "w") as f:
        f.writelines(train_lines)
    
    with open(val_ALL, "w") as f:
        f.writelines(val_lines)
    
    # Statistics
    train_A_count = len(open(train_A).readlines())
    train_B_count = len(open(train_B).readlines())
    val_A_count = len(open(val_A).readlines())
    val_B_count = len(open(val_B).readlines())
    
    print(f"\n✅ Merged splits created:")
    print(f"   {train_ALL.name}:")
    print(f"      Domain A: {train_A_count}")
    print(f"      Domain B: {train_B_count}")
    print(f"      Total: {train_A_count + train_B_count}")
    print(f"   {val_ALL.name}:")
    print(f"      Domain A: {val_A_count}")
    print(f"      Domain B: {val_B_count}")
    print(f"      Total: {val_A_count + val_B_count}")
    
    # Check for duplicates
    train_set = set(line.strip() for line in train_lines)
    val_set = set(line.strip() for line in val_lines)
    
    if len(train_set) < len(train_lines):
        print(f"\n⚠️  Warning: {len(train_lines) - len(train_set)} duplicate(s) in train_ALL.txt")
    
    if len(val_set) < len(val_lines):
        print(f"⚠️  Warning: {len(val_lines) - len(val_set)} duplicate(s) in val_ALL.txt")
    
    # Check for train/val overlap
    overlap = train_set & val_set
    if overlap:
        print(f"\n❌ ERROR: {len(overlap)} image(s) appear in both train and val!")
        print("   This indicates a problem with domain splitting.")
    else:
        print(f"\n✅ No train/val overlap detected")
    
    print(f"\n{'='*60}")
    print(f"✅ All splits ready for classification training!")
    print(f"{'='*60}")

if __name__ == "__main__":
    merge_splits()
