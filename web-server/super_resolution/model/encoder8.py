import torch.nn as nn

from .blocks import ResBlock


class Encoder(nn.Module):
    """tex-VQVAE Encoder: extracts textural features from HR images.

    Input:  (B, 3, H, W)
    Output: (B, embed_dim, H/8, W/8)

    Example:
        (B, 3, 256, 256) -> (B, embed_dim, 32, 32)
        (B, 3, 1080, 1920) -> (B, embed_dim, 135, 240)
    """

    def __init__(self, in_ch=3, ch=128, embed_dim=64, n_res_blocks=4):
        super().__init__()

        # Total downsampling: /8 = 2 x 2 x 2
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, stride=2, padding=1),   # H -> H/2
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1),      # H/2 -> H/4
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1),      # H/4 -> H/8
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock(ch) for _ in range(n_res_blocks)]
        )

        self.proj = nn.Conv2d(ch, embed_dim, 1)

    def forward(self, x):
        h = self.head(x)
        h = self.res_blocks(h)
        return self.proj(h)