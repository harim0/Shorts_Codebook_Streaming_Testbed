import torch.nn as nn

from .blocks import ResBlock


class CoarseSRNet(nn.Module):
    """SRResNet with temporal fusion for coarse super-resolution.

    Takes 3 concatenated LR frames (t-1, t, t+1) and produces a coarse HR.

    Input:  (B, 9, lr_h, lr_w)   e.g. (B, 9, 40, 40)
    Output: (B, 3, hr_h, hr_w)   e.g. (B, 3, 160, 160)
    """

    def __init__(self, in_ch=9, n_feats=64, n_res_blocks=16, scale=4, out_ch=3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock(n_feats) for _ in range(n_res_blocks)]
        )
        self.res_tail = nn.Conv2d(n_feats, n_feats, 3, padding=1)

        # Upsampling: 4x = 2x + 2x via PixelShuffle
        upsample_layers = []
        n_upscale = 0
        s = scale
        while s > 1:
            upsample_layers += [
                nn.Conv2d(n_feats, n_feats * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            ]
            s //= 2
            n_upscale += 1
        self.upsample = nn.Sequential(*upsample_layers)
        self._layers_per_stage = 3  # Conv + PixelShuffle + ReLU

        self.tail = nn.Conv2d(n_feats, out_ch, 3, padding=1)

    def forward(self, x):
        h = self.head(x)
        res = self.res_blocks(h)
        res = self.res_tail(res)
        h = h + res  # global skip connection
        h = self.upsample(h)
        return self.tail(h)

    def forward_with_middle(self, x):
        """Forward with x_middle feature tap for MoE pipeline.

        x_middle = feature after first 2x PixelShuffle.
        Same spatial resolution as VQ latent z (H/2 × W/2), 64ch.

        Returns:
            x_uped:   (B, 3, H, W)           — coarse SR output
            x_middle: (B, n_feats, H/2, W/2) — intermediate feature
        """
        h = self.head(x)
        res = self.res_blocks(h)
        res = self.res_tail(res)
        h = h + res
        s = self._layers_per_stage
        x_middle = self.upsample[:s](h)   # first 2x: (B, 64, H/2, W/2)
        h = self.upsample[s:](x_middle)   # second 2x: (B, 64, H, W)
        x_uped = self.tail(h)             # (B, 3, H, W)
        return x_uped, x_middle
