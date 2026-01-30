"""
Domain Split Creation for CycleGAN Training
Creates train/val splits for Domain A (No Finding) and Domain B (Pneumothorax)
IMPORTANT: Splits by PATIENT ID to prevent data leakage
"""
import os
import pandas as pd
import random
from pathlib import Path

# Configuration
CSV_PATH = Path("D:/ChestXray_GAN_CNN_VIT_NEW/data/chestxray/Data_Entry_2017_v2020.csv")
IMG_DIR = Path("D:/ChestXray_GAN_CNN_VIT_NEW/data/chestxray/images")
SPLIT_DIR = Path("D:/ChestXray_GAN_CNN_VIT_NEW/data/chestxray/splits/pneumothorax")
TARGET_PATHOLOGY = "Pneumothorax"
RATIO_TRAIN = 0.9  # 90% train, 10% validation
SEED = 42
PATIENT_LEVEL_SPLIT = True  # CRITICAL: Prevents data leakage

def has_finding(finding_labels, target):
    """Check if target finding is present in pipe-separated labels"""
    if pd.isna(finding_labels):
        return False
    labels = [label.strip() for label in str(finding_labels).split("|")]
    return target in labels

def create_domain_splits():
    """Create train/val splits for both domains"""
    
    # Create output directory
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load CSV
    print(f"📂 Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"   Total records: {len(df)}")
    print(f"   Unique patients: {df['Patient ID'].nunique()}")
    
    # Filter Domain A: No Finding (healthy)
    df_no_finding = df[df["Finding Labels"] == "No Finding"].copy()
    
    # Filter Domain B: Target pathology present
    df_target = df[df["Finding Labels"].apply(
        lambda x: has_finding(x, TARGET_PATHOLOGY)
    )].copy()
    
    print(f"\n📊 Domain Statistics:")
    print(f"   Domain A (No Finding):")
    print(f"      Images: {len(df_no_finding)}")
    print(f"      Patients: {df_no_finding['Patient ID'].nunique()}")
    print(f"   Domain B ({TARGET_PATHOLOGY}):")
    print(f"      Images: {len(df_target)}")
    print(f"      Patients: {df_target['Patient ID'].nunique()}")
    
    # Save master lists (all images per domain)
    master_no_finding = SPLIT_DIR / "A_no_finding.txt"
    master_target = SPLIT_DIR / f"B_{TARGET_PATHOLOGY.lower()}.txt"
    
    with open(master_no_finding, "w") as f:
        f.write("\n".join(df_no_finding["Image Index"].tolist()))
    
    with open(master_target, "w") as f:
        f.write("\n".join(df_target["Image Index"].tolist()))
    
    print(f"\n✅ Master lists created:")
    print(f"   {master_no_finding.name}")
    print(f"   {master_target.name}")
    
    # Split by patient (prevents data leakage)
    if PATIENT_LEVEL_SPLIT:
        print(f"\n🔒 Using patient-level splitting (prevents data leakage)")
        train_A, val_A = split_by_patient(df_no_finding, RATIO_TRAIN, SEED)
        train_B, val_B = split_by_patient(df_target, RATIO_TRAIN, SEED)
    else:
        print(f"\n⚠️  Using image-level splitting (may cause data leakage)")
        train_A, val_A = split_by_image(df_no_finding["Image Index"].tolist(), RATIO_TRAIN, SEED)
        train_B, val_B = split_by_image(df_target["Image Index"].tolist(), RATIO_TRAIN, SEED)
    
    # Save splits
    splits = {
        "train_A.txt": train_A,
        "val_A.txt": val_A,
        "train_B.txt": train_B,
        "val_B.txt": val_B
    }
    
    print(f"\n📝 Creating split files:")
    for filename, images in splits.items():
        filepath = SPLIT_DIR / filename
        with open(filepath, "w") as f:
            f.write("\n".join(images))
        print(f"   {filename}: {len(images)} images")
    
    # Verify no overlap between train and val
    train_A_set = set(train_A)
    val_A_set = set(val_A)
    train_B_set = set(train_B)
    val_B_set = set(val_B)
    
    overlap_A = train_A_set & val_A_set
    overlap_B = train_B_set & val_B_set
    
    if overlap_A or overlap_B:
        print(f"\n⚠️  WARNING: Train/val overlap detected!")
        print(f"   Domain A overlap: {len(overlap_A)}")
        print(f"   Domain B overlap: {len(overlap_B)}")
    else:
        print(f"\n✅ No train/val overlap - splits are clean!")
    
    print(f"\n{'='*60}")
    print(f"✅ Domain splits ready: {SPLIT_DIR}")
    print(f"{'='*60}")

def split_by_patient(df, ratio_train, seed):
    """Split dataset by patient ID to prevent data leakage"""
    random.seed(seed)
    
    # Get unique patients
    unique_patients = df['Patient ID'].unique().tolist()
    random.shuffle(unique_patients)
    
    # Split patients
    n_train = int(len(unique_patients) * ratio_train)
    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train:])
    
    # Get images for each patient set
    train_images = df[df['Patient ID'].isin(train_patients)]['Image Index'].tolist()
    val_images = df[df['Patient ID'].isin(val_patients)]['Image Index'].tolist()
    
    return train_images, val_images

def split_by_image(image_list, ratio_train, seed):
    """Split dataset by individual images (old method - not recommended)"""
    random.seed(seed)
    images = image_list.copy()
    random.shuffle(images)
    
    n_train = int(len(images) * ratio_train)
    train_images = images[:n_train]
    val_images = images[n_train:]
    
    return train_images, val_images

if __name__ == "__main__":
    create_domain_splits()
