"""
Vision Transformer (ViT) for Pneumothorax Classification
Binary classification: 0 = No Pneumothorax, 1 = Pneumothorax
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    Splits image into non-overlapping patches and projects to embedding dimension
    
    Args:
        in_ch: Input channels (3 for RGB)
        embed_dim: Embedding dimension
        patch_size: Size of each patch
    """
    
    def __init__(self, in_ch=3, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, N, embed_dim) where N = (H/patch_size) * (W/patch_size)
        """
        x = self.proj(x)                   # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)   # (B, N, embed_dim)
        return x


class SimpleViT(nn.Module):
    """
    Simplified Vision Transformer for binary classification
    
    Architecture:
        1. Patch Embedding: Split image into patches
        2. Add CLS token and positional embeddings
        3. Transformer Encoder: Self-attention layers
        4. Classification Head: MLP on CLS token
    
    Args:
        img_size: Input image size (square)
        patch_size: Patch size (e.g., 16 for 16x16 patches)
        in_chans: Input channels (3 for RGB)
        embed_dim: Embedding dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        num_classes: Output classes (1 for binary with BCEWithLogitsLoss)
        dropout: Dropout rate
    """
    
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=512, 
                 depth=6, num_heads=8, mlp_ratio=4.0, num_classes=1, dropout=0.0):
        super().__init__()
        
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        
        self.patch_embed = PatchEmbed(in_ch=in_chans, embed_dim=embed_dim, patch_size=patch_size)
        num_patches = (img_size // patch_size) ** 2
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - Input images
        
        Returns:
            (B,) - Logits for binary classification
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)          # (B, 1+N, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer encoder
        x = self.encoder(x)  # (B, 1+N, embed_dim)
        
        # Layer norm and extract CLS token
        x = self.norm(x[:, 0])  # (B, embed_dim)
        
        # Classification head
        logits = self.head(x)   # (B, num_classes)
        
        return logits.squeeze(1)  # (B,) for binary classification


def test_vit():
    """Test ViT model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🧪 Testing Vision Transformer...\n")
    
    model = SimpleViT(
        img_size=256,
        patch_size=16,
        in_chans=3,
        embed_dim=512,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        num_classes=1,
        dropout=0.0
    ).to(device)
    
    # Test forward pass
    x = torch.randn(4, 3, 256, 256).to(device)
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Test with BCEWithLogitsLoss
    labels = torch.randint(0, 2, (4,)).float().to(device)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, labels)
    
    print(f"\nLoss test:")
    print(f"   Labels: {labels}")
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n✅ ViT test passed!")


if __name__ == "__main__":
    test_vit()
