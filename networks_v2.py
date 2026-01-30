"""
CycleGAN Network Architectures
- ResNet-based Generator for image-to-image translation
- PatchGAN Discriminator for adversarial training
"""
import torch
import torch.nn as nn


def conv_block(in_c, out_c, k, s, p, norm=True):
    """
    Convolutional block: Conv2d -> InstanceNorm2d -> ReLU
    
    Args:
        in_c: Input channels
        out_c: Output channels
        k: Kernel size
        s: Stride
        p: Padding
        norm: Whether to apply instance normalization
    """
    layers = [nn.Conv2d(in_c, out_c, k, s, p)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_c))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def deconv_block(in_c, out_c, k, s, p, outpad=0):
    """
    Transposed convolutional block: ConvTranspose2d -> InstanceNorm2d -> ReLU
    
    Args:
        in_c: Input channels
        out_c: Output channels
        k: Kernel size
        s: Stride
        p: Padding
        outpad: Output padding for size matching
    """
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, k, s, p, output_padding=outpad),
        nn.InstanceNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class ResnetBlock(nn.Module):
    """
    Residual block with two convolutional layers
    Uses skip connection: output = input + F(input)
    """
    
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """
    ResNet-based Generator for CycleGAN
    
    Architecture:
        - Initial convolution (7x7)
        - Downsampling (2 conv layers with stride 2)
        - Residual blocks (6 or 9 blocks)
        - Upsampling (2 transposed conv layers)
        - Output convolution (7x7) with Tanh
    
    Args:
        in_c: Input channels (1 for grayscale)
        out_c: Output channels (1 for grayscale)
        n_blocks: Number of residual blocks (6 or 9)
    
    Input: (B, in_c, 256, 256)
    Output: (B, out_c, 256, 256) with values in [-1, 1]
    """
    
    def __init__(self, in_c=1, out_c=1, n_blocks=6):
        super().__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_c, 64, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling layers
        model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        ]

        # Residual blocks
        for _ in range(n_blocks):
            model.append(ResnetBlock(256))

        # Upsampling layers
        model += [
            deconv_block(256, 128, k=3, s=2, p=1, outpad=1),
            deconv_block(128, 64, k=3, s=2, p=1, outpad=1),
        ]
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_c, kernel_size=7, stride=1, padding=0),
            nn.Tanh(),  # Output range [-1, 1]
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator
    
    Classifies whether 70x70 overlapping patches are real or fake
    instead of classifying the entire image.
    
    Args:
        in_c: Input channels (1 for grayscale)
    
    Input: (B, in_c, 256, 256)
    Output: (B, 1, 30, 30) - patch predictions
    """
    
    def __init__(self, in_c=1):
        super().__init__()

        def discriminator_block(in_channels, out_channels, normalize=True):
            """Discriminator block: Conv2d -> [InstanceNorm2d] -> LeakyReLU"""
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            # No normalization on first layer
            discriminator_block(in_c, 64, normalize=False),  # 256 -> 128
            discriminator_block(64, 128),                     # 128 -> 64
            discriminator_block(128, 256),                    # 64 -> 32
            nn.Conv2d(256, 1, 4, padding=1),                  # 32 -> 30 (no activation)
        )

    def forward(self, x):
        return self.model(x)


def test_networks():
    """Test network architectures"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🧪 Testing CycleGAN Networks...\n")
    
    # Test Generator
    print("Testing Generator:")
    G = ResnetGenerator(in_c=1, out_c=1, n_blocks=6).to(device)
    x = torch.randn(2, 1, 256, 256).to(device)
    y = G(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {y.shape}")
    print(f"   Output range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"   Parameters: {sum(p.numel() for p in G.parameters()):,}")
    
    # Test Discriminator
    print("\nTesting Discriminator:")
    D = PatchDiscriminator(in_c=1).to(device)
    pred = D(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {pred.shape}")
    print(f"   Output range: [{pred.min():.3f}, {pred.max():.3f}]")
    print(f"   Parameters: {sum(p.numel() for p in D.parameters()):,}")
    
    print("\n✅ Network tests passed!")


if __name__ == "__main__":
    test_networks()
