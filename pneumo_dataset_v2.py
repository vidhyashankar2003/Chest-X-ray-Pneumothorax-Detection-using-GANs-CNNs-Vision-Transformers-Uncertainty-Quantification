"""
Pneumothorax Classification Dataset
For training ViT classifier on real + synthetic images
Binary classification: 0 = No Pneumothorax, 1 = Pneumothorax
"""
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import warnings
from pathlib import Path


class PneumoDataset(Dataset):
    """
    Dataset for Pneumothorax binary classification
    
    Combines:
    - Real images from NIH ChestX-ray14 dataset
    - Synthetic images from CycleGAN (optional)
    
    Args:
        csv_path: Path to Data_Entry_2017_v2020.csv
        images_dir: Directory containing real images
        list_txt: Path to split file (train_ALL.txt or val_ALL.txt)
        synthetic_dir: Directory containing synthetic images (optional)
        synthetic_label: Label for synthetic images (0 or 1)
        img_size: Target image size
        transform: Torchvision transforms to apply
    """
    
    def __init__(self, csv_path, images_dir, list_txt, 
                 synthetic_dir=None, synthetic_label=1,
                 img_size=256, transform=None):
        
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.img_size = img_size
        
        # Read CSV and create label mapping
        print(f"📂 Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Create binary Pneumothorax labels
        df['Pneumothorax'] = df['Finding Labels'].fillna('').apply(
            lambda x: 1 if 'Pneumothorax' in str(x).split('|') else 0
        )
        
        self.label_map = dict(zip(df['Image Index'], df['Pneumothorax']))
        
        print(f"   Total images in CSV: {len(df)}")
        print(f"   Pneumothorax cases: {df['Pneumothorax'].sum()}")
        print(f"   No Pneumothorax: {len(df) - df['Pneumothorax'].sum()}")
        
        # Load split file
        print(f"\n📋 Loading split: {list_txt}")
        with open(list_txt, 'r') as f:
            split_filenames = [line.strip() for line in f if line.strip()]
        
        print(f"   Images in split: {len(split_filenames)}")
        
        # Create sample list: (path, label)
        self.samples = []
        missing_count = 0
        
        for filename in split_filenames:
            img_path = self.images_dir / filename
            
            if img_path.exists():
                label = self.label_map.get(filename, 0)
                self.samples.append((str(img_path), int(label)))
            else:
                missing_count += 1
                if missing_count <= 5:  # Only warn for first few
                    warnings.warn(f"Missing: {filename}")
        
        if missing_count > 0:
            print(f"   ⚠️  Missing files: {missing_count}")
        
        # Add synthetic images (optional)
        if synthetic_dir and os.path.exists(synthetic_dir):
            synthetic_dir = Path(synthetic_dir)
            print(f"\n🎨 Adding synthetic images from: {synthetic_dir}")
            
            # Count before adding
            initial_count = len(self.samples)
            
            # Add all synthetic images with specified label
            for img_file in sorted(synthetic_dir.glob("*.png")):
                self.samples.append((str(img_file), synthetic_label))
            
            synthetic_count = len(self.samples) - initial_count
            print(f"   Synthetic images added: {synthetic_count}")
            print(f"   Synthetic label: {synthetic_label} ({'Pneumothorax' if synthetic_label == 1 else 'No Pneumothorax'})")
        
        if not self.samples:
            raise RuntimeError("No valid images found in dataset!")
        
        # Final statistics
        labels = [label for _, label in self.samples]
        print(f"\n📊 Final dataset statistics:")
        print(f"   Total samples: {len(self.samples)}")
        print(f"   Class 0 (No Pneumothorax): {labels.count(0)}")
        print(f"   Class 1 (Pneumothorax): {labels.count(1)}")
        print(f"   Class balance: {labels.count(1) / len(labels) * 100:.1f}% positive")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed PIL Image
            label: Binary label (0 or 1)
            filename: Image filename
        """
        img_path, label = self.samples[idx]
        
        # Try loading the image
        try:
            img = Image.open(img_path).convert("RGB")
            
            # Apply transforms
            if self.transform:
                img = self.transform(img)
            
            return img, int(label), os.path.basename(img_path)
        
        except Exception as e:
            warnings.warn(f"Error loading {img_path}: {e}")
            
            # Fallback: try next image
            next_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(next_idx)


def test_dataset():
    """Test dataset loading"""
    from torchvision import transforms
    
    CSV_PATH = "D:/ChestXray_GAN_CNN_ViT/data/chestxray/Data_Entry_2017_v2020.csv"
    IMAGES_DIR = "D:/ChestXray_GAN_CNN_ViT/data/chestxray/images"
    TRAIN_LIST = "D:/ChestXray_GAN_CNN_ViT/data/chestxray/splits/pneumothorax/train_ALL.txt"
    SYNTHETIC_DIR = "generated_images/synthetic_pneumothorax"
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = PneumoDataset(
        csv_path=CSV_PATH,
        images_dir=IMAGES_DIR,
        list_txt=TRAIN_LIST,
        synthetic_dir=SYNTHETIC_DIR,
        synthetic_label=1,  # Synthetic images ARE Pneumothorax
        transform=transform
    )
    
    print(f"\n🧪 Testing dataset...")
    img, label, filename = dataset[0]
    print(f"   Sample shape: {img.shape}")
    print(f"   Sample label: {label}")
    print(f"   Sample filename: {filename}")
    print(f"✅ Dataset test passed!")


if __name__ == "__main__":
    test_dataset()
