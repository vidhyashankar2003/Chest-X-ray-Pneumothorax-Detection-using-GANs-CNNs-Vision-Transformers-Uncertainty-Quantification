"""
Unpaired Dataset for CycleGAN Training
Loads images from Domain A (No Finding) and Domain B (Pneumothorax)
Images are unpaired - random combinations of A and B for each batch
"""
import os
import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
from PIL import Image
import warnings
from pathlib import Path

class UnpairedXrayDataset(Dataset):
    """
    Dataset for CycleGAN with unpaired images from two domains
    
    Args:
        img_dir: Root directory containing 'processed' subfolder with .npy files
        list_A: Path to text file with Domain A image filenames
        list_B: Path to text file with Domain B image filenames
        size: Target image size (default 256)
        train: If True, applies data augmentation
    """
    
    def __init__(self, img_dir, list_A, list_B, size=256, train=True):
        self.img_dir = Path(img_dir)
        self.processed_dir = self.img_dir / "processed"
        self.size = size
        self.train = train
        
        # Load image lists
        with open(list_A, "r") as f:
            self.A = [line.strip() for line in f if line.strip()]
        
        with open(list_B, "r") as f:
            self.B = [line.strip() for line in f if line.strip()]
        
        print(f"📂 Dataset initialized:")
        print(f"   Domain A: {len(self.A)} images")
        print(f"   Domain B: {len(self.B)} images")
        print(f"   Mode: {'Training' if train else 'Validation'}")
        
        # Verify processed directory exists
        if not self.processed_dir.exists():
            raise RuntimeError(f"Processed directory not found: {self.processed_dir}\n"
                             f"Run preprocessor.py first!")
        
        # Define transforms
        transform_list = []
        
        # Augmentation for training only
        if train:
            transform_list.append(T.RandomHorizontalFlip(p=0.5))
        
        # Always resize and normalize
        transform_list.extend([
            T.Resize((size, size), antialias=True),
            T.ToTensor(),  # Converts to [0, 1] and adds channel dimension
            T.Normalize([0.5], [0.5])  # Normalize to [-1, 1] for GAN training
        ])
        
        self.transform = T.Compose(transform_list)
        
        # Verify a few sample files exist
        self._verify_files()
    
    def _verify_files(self, num_samples=5):
        """Verify that sample files can be loaded"""
        samples_to_check = min(num_samples, len(self.A), len(self.B))
        missing = []
        
        for filename in (self.A[:samples_to_check] + self.B[:samples_to_check]):
            npy_path = self._get_npy_path(filename)
            if not npy_path.exists():
                missing.append(filename)
        
        if missing:
            warnings.warn(f"⚠️  {len(missing)} sample files not found in processed/:\n"
                         f"   {missing[:3]}...")
    
    def _get_npy_path(self, filename):
        """Convert image filename to .npy path"""
        # Remove extension and add .npy
        base_name = Path(filename).stem
        return self.processed_dir / f"{base_name}.npy"
    
    def _safe_load(self, filename):
        """
        Safely load an image from .npy file
        Returns PIL Image or None if loading fails
        """
        npy_path = self._get_npy_path(filename)
        
        try:
            # Load numpy array
            img_array = np.load(npy_path)
            
            # Validate shape (should be 2D grayscale)
            if img_array.ndim != 2:
                warnings.warn(f"Unexpected shape {img_array.shape} for {filename}")
                return None
            
            # Validate value range [0, 1]
            if img_array.min() < 0 or img_array.max() > 1:
                warnings.warn(f"Values out of range for {filename}: [{img_array.min()}, {img_array.max()}]")
                img_array = np.clip(img_array, 0, 1)
            
            # Convert to PIL Image (scale back to 0-255 for PIL)
            img_array_uint8 = (img_array * 255).astype(np.uint8)
            img = Image.fromarray(img_array_uint8, mode='L')
            
            return img
            
        except Exception as e:
            warnings.warn(f"Error loading {filename}: {e}")
            return None
    
    def _load_with_fallback(self, filename, fallback_list, max_attempts=10):
        """
        Load image with fallback mechanism
        If loading fails, try random alternatives from fallback_list
        """
        img = self._safe_load(filename)
        
        if img is not None:
            return img, filename
        
        # Try fallback alternatives
        for attempt in range(max_attempts):
            alt_filename = random.choice(fallback_list)
            img = self._safe_load(alt_filename)
            
            if img is not None:
                return img, alt_filename
        
        # If all attempts fail, return black image
        warnings.warn(f"All load attempts failed for {filename}, using blank image")
        img = Image.new('L', (self.size, self.size), color=0)
        return img, "blank.png"
    
    def __len__(self):
        """Dataset length is the maximum of both domains"""
        return max(len(self.A), len(self.B))
    
    def __getitem__(self, idx):
        """
        Get one unpaired sample from each domain
        
        Returns:
            dict with keys: 'A', 'B', 'A_path', 'B_path'
        """
        # Cycle through Domain A
        filename_A = self.A[idx % len(self.A)]
        img_A, path_A = self._load_with_fallback(filename_A, self.A)
        
        # Random sample from Domain B (unpaired)
        filename_B = random.choice(self.B)
        img_B, path_B = self._load_with_fallback(filename_B, self.B)
        
        # Apply transforms
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        
        return {
            "A": img_A,
            "B": img_B,
            "A_path": path_A,
            "B_path": path_B
        }


# Test function
def test_dataset():
    """Test dataset loading"""
    from torch.utils.data import DataLoader
    
    IMG_DIR = "D:/ChestXray_GAN_CNN_ViT/data/chestxray"
    LIST_A = "D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/train_A.txt"
    LIST_B = "D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/train_B.txt"
    
    dataset = UnpairedXrayDataset(IMG_DIR, LIST_A, LIST_B, size=256, train=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    print(f"\n🧪 Testing dataset...")
    batch = next(iter(loader))
    
    print(f"   Batch shapes:")
    print(f"      A: {batch['A'].shape}")
    print(f"      B: {batch['B'].shape}")
    print(f"   Value ranges:")
    print(f"      A: [{batch['A'].min():.3f}, {batch['A'].max():.3f}]")
    print(f"      B: [{batch['B'].min():.3f}, {batch['B'].max():.3f}]")
    print(f"✅ Dataset test passed!")


if __name__ == "__main__":
    test_dataset()
