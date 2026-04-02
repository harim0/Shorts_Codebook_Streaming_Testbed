"""Lightweight Decoder – same interface as Decoder but:
  - coarse_encoder: single Conv2d (stride-2) instead of two
  - ch=48, n_res_blocks=1
Weights are stored in light-Coarse-decoder.pt['model']['decoder.*'].
"""

import torch
import torch.nn as nn

from .blocks import ResBlock


class DecoderLight(nn.Module):
    """Lightweight tex-VQVAE Decoder.

    Inputs:
        z_q:      (B, embed_dim, H/2, W/2)  quantized texture features
        x_coarse: (B, 3, H, W)              coarse SR output
    Output:
        x_hat:    (B, 3, H, W)              reconstructed HR image

    Differences from full Decoder:
        - coarse_encoder has 1 Conv (vs 2 Convs in Decoder)
        - ch=48, n_res_blocks=1 (vs ch=128, n_res_blocks=4)
    """

    def __init__(self, embed_dim=64, coarse_feat_ch=32, ch=48,
                 n_res_blocks=1, out_ch=3):
        super().__init__()
        # Single stride-2 Conv to extract coarse features (lighter than Decoder)
        self.coarse_encoder = nn.Sequential(
            nn.Conv2d(out_ch, coarse_feat_ch, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(embed_dim + coarse_feat_ch, ch, 3, padding=1)

        self.res_blocks = nn.Sequential(
            *[ResBlock(ch) for _ in range(n_res_blocks)]
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Conv2d(ch, out_ch, 3, padding=1)

    def forward(self, z_q, x_coarse):
        coarse_feat = self.coarse_encoder(x_coarse)
        h = torch.cat([z_q, coarse_feat], dim=1)
        h = self.head(h)
        h = self.res_blocks(h)
        h = self.upsample(h)
        return self.tail(h)
