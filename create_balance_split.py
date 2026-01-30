# create_balanced_splits.py
import pandas as pd
import random
from pathlib import Path

CSV = "D:/ChestXray_GAN_CNN_ViT/data/chestxray/Data_Entry_2017_v2020.csv"
SPLIT_DIR = Path("D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax")

df = pd.read_csv(CSV)

# Get Pneumothorax cases
pneumo = df[df['Finding Labels'].str.contains('Pneumothorax', na=False)]['Image Index'].tolist()

# Get equal number of healthy cases
no_finding = df[df['Finding Labels'] == 'No Finding']['Image Index'].tolist()
random.seed(42)
no_finding_balanced = random.sample(no_finding, len(pneumo))

# Combine
balanced_train = pneumo + no_finding_balanced
random.shuffle(balanced_train)

# Save
with open("D:\\ChestXray_GAN_CNN_VIT_NEW\\data\\chestxray\\splits\\pneumothorax\\train_BALANCED.txt", "w") as f:
    f.write("\n".join(balanced_train))

print(f"✅ Created balanced split: {len(pneumo)} Pneumothorax + {len(no_finding_balanced)} healthy")