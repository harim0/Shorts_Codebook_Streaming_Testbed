"""Lightweight CoarseSR – same architecture as CoarseSRNet but with
n_feats=32, n_res_blocks=8 (half the channels and depth of the full model).
Weights are stored in light-Coarse-decoder.pt['model']['coarse_sr.*'].
"""

from .coarse_sr import CoarseSRNet


class CoarseSRLight(CoarseSRNet):
    """Lightweight CoarseSR (n_feats=32, n_res_blocks=8, scale=4)."""

    def __init__(self, in_ch=9, n_feats=32, n_res_blocks=8, scale=4, out_ch=3):
        super().__init__(in_ch=in_ch, n_feats=n_feats,
                         n_res_blocks=n_res_blocks, scale=scale, out_ch=out_ch)
