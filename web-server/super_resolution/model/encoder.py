import torch.nn as nn

from .blocks import ResBlock


class Encoder(nn.Module):
    """tex-VQVAE Encoder: extracts textural features from HR images.

    Input:  (B, 3, H, H)   e.g. (B, 3, 160, 160)
    Output: (B, embed_dim, H/2, H/2)  e.g. (B, 64, 80, 80)
    """

    def __init__(self, in_ch=3, ch=128, embed_dim=64, n_res_blocks=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, stride=2, padding=1),
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
