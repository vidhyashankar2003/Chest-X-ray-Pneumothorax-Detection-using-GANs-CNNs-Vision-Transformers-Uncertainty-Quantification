"""
Create balanced validation split to match training distribution
"""
import pandas as pd
import random
from pathlib import Path

CSV = "D:/ChestXray_GAN_CNN_ViT/data/chestxray/Data_Entry_2017_v2020.csv"
SPLIT_DIR = Path("D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax")

df = pd.read_csv(CSV)

# Load existing validation split
with open(SPLIT_DIR / "val_ALL.txt") as f:
    val_files = [line.strip() for line in f]

# Separate into positive and negative from validation set
val_pneumo = []
val_healthy = []

for fname in val_files:
    row = df[df['Image Index'] == fname]
    if not row.empty:
        labels = row['Finding Labels'].values[0]
        if pd.notna(labels) and 'Pneumothorax' in labels:
            val_pneumo.append(fname)
        elif labels == 'No Finding':
            val_healthy.append(fname)

print(f"Original validation set:")
print(f"   Pneumothorax: {len(val_pneumo)}")
print(f"   Healthy: {len(val_healthy)}")

# Create balanced validation (same size as minority class)
n_samples = min(len(val_pneumo), len(val_healthy))
n_samples = min(n_samples, 500)  # Cap at 500 for faster validation

random.seed(42)
val_pneumo_balanced = random.sample(val_pneumo, n_samples) if len(val_pneumo) > n_samples else val_pneumo
val_healthy_balanced = random.sample(val_healthy, n_samples)

# Combine and shuffle
balanced_val = val_pneumo_balanced + val_healthy_balanced
random.shuffle(balanced_val)

# Save
output_file = SPLIT_DIR / "val_BALANCED.txt"
with open(output_file, "w") as f:
    f.write("\n".join(balanced_val))

print(f"\nBalanced validation set created:")
print(f"   Pneumothorax: {len(val_pneumo_balanced)}")
print(f"   Healthy: {len(val_healthy_balanced)}")
print(f"   Total: {len(balanced_val)}")
print(f"   Saved to: {output_file}")