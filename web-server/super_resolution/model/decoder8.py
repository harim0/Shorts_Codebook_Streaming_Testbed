import torch
import torch.nn as nn

from .blocks import ResBlock


class Decoder(nn.Module):
    """tex-VQVAE Decoder: reconstructs HR from quantized texture + coarse SR.

    Inputs:
        z_q:      (B, embed_dim, H/8, W/8)   quantized texture features
        x_coarse: (B, 3, H, W)               coarse SR output

    Output:
        x_hat:    (B, 3, H, W)
    """

    def __init__(self, embed_dim=64, coarse_feat_ch=64, ch=128, n_res_blocks=4, out_ch=3):
        super().__init__()

        # Extract features from coarse SR at latent resolution (/8)
        self.coarse_encoder = nn.Sequential(
            nn.Conv2d(out_ch, coarse_feat_ch, 3, stride=2, padding=1),   # H -> H/2
            nn.ReLU(inplace=True),
            nn.Conv2d(coarse_feat_ch, coarse_feat_ch, 3, stride=2, padding=1),  # H/2 -> H/4
            nn.ReLU(inplace=True),
            nn.Conv2d(coarse_feat_ch, coarse_feat_ch, 3, stride=2, padding=1),  # H/4 -> H/8
            nn.ReLU(inplace=True),
            nn.Conv2d(coarse_feat_ch, coarse_feat_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Merge quantized texture + coarse features
        self.head = nn.Conv2d(embed_dim + coarse_feat_ch, ch, 3, padding=1)

        self.res_blocks = nn.Sequential(
            *[ResBlock(ch) for _ in range(n_res_blocks)]
        )

        # Upsample 8x back to HR
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1),   # H/8 -> H/4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1),   # H/4 -> H/2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1),   # H/2 -> H
            nn.ReLU(inplace=True),
        )

        self.tail = nn.Conv2d(ch, out_ch, 3, padding=1)

    def forward(self, z_q, x_coarse):
        coarse_feat = self.coarse_encoder(x_coarse)

        if z_q.shape[-2:] != coarse_feat.shape[-2:]:
            raise RuntimeError(
                f"Shape mismatch: z_q={tuple(z_q.shape)}, coarse_feat={tuple(coarse_feat.shape)}"
            )

        h = torch.cat([z_q, coarse_feat], dim=1)
        h = self.head(h)
        h = self.res_blocks(h)
        h = self.upsample(h)
        return self.tail(h)