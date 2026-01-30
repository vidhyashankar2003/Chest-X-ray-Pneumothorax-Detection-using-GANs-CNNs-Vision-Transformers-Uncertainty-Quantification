"""
Image Preprocessing Script for ChestX-ray14 Dataset
Resizes images to 256×256 and saves as .npy files for GAN training
"""
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path
import warnings

# Configuration
RAW_IMAGES_DIR = Path("D:/ChestXray_GAN_CNN_VIT_NEW/data/chestxray/images")
PROCESSED_DIR = Path("D:/ChestXray_GAN_CNN_VIT_NEW/data/chestxray/processed")
IMG_SIZE = (256, 256)  # CRITICAL FIX: Changed from 128 to 256 to match model input
NORMALIZE = True  # Scale pixel values to [0, 1]

def preprocess_images():
    """Preprocess all PNG images: resize, normalize, save as .npy"""
    
    # Create output directory
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all PNG files (including subdirectories if any)
    image_files = list(RAW_IMAGES_DIR.rglob("*.png"))
    
    if not image_files:
        print(f"⚠️  No PNG files found in {RAW_IMAGES_DIR}")
        return
    
    print(f"🖼️  Found {len(image_files)} images to process")
    print(f"📐 Target size: {IMG_SIZE}")
    print(f"📂 Output directory: {PROCESSED_DIR}")
    
    successful = 0
    failed = 0
    skipped = 0
    
    # Process each image with progress bar
    for img_path in tqdm(image_files, desc="Preprocessing images"):
        try:
            # Generate output filename
            img_name = img_path.stem  # Filename without extension
            output_path = PROCESSED_DIR / f"{img_name}.npy"
            
            # Skip if already processed
            if output_path.exists():
                skipped += 1
                continue
            
            # Load image in grayscale (L mode for single channel)
            img = Image.open(img_path).convert("L")
            
            # Resize using high-quality Lanczos resampling
            img = img.resize(IMG_SIZE, Image.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32)
            
            # Normalize to [0, 1] range
            if NORMALIZE:
                img_array = img_array / 255.0
            
            # Validate array shape and range
            assert img_array.shape == IMG_SIZE, f"Unexpected shape: {img_array.shape}"
            assert img_array.min() >= 0 and img_array.max() <= 1, "Values out of range"
            
            # Save as .npy
            np.save(output_path, img_array)
            successful += 1
            
        except Exception as e:
            failed += 1
            warnings.warn(f"❌ Error processing {img_path.name}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Successfully processed: {successful}")
    print(f"⏭️  Skipped (already exist): {skipped}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total images in output: {len(list(PROCESSED_DIR.glob('*.npy')))}")
    print(f"{'='*60}")
    
    # Sample verification
    if successful > 0:
        sample_file = list(PROCESSED_DIR.glob("*.npy"))[0]
        sample_array = np.load(sample_file)
        print(f"\n📋 Sample file verification:")
        print(f"   File: {sample_file.name}")
        print(f"   Shape: {sample_array.shape}")
        print(f"   Dtype: {sample_array.dtype}")
        print(f"   Range: [{sample_array.min():.3f}, {sample_array.max():.3f}]")

if __name__ == "__main__":
    preprocess_images()
