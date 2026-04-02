"""Stage 1: tex-VQVAE + Coarse-SR integrated model."""

import torch
import torch.nn as nn

from .coarse_sr import CoarseSRNet
from .decoder import Decoder
from .encoder import Encoder
from .quantize import FactorizedL2VQ


class TexVQVAE(nn.Module):
    """Full Stage 1 model combining tex-VQVAE and Coarse-SR.

    Pipeline:
        1. Coarse-SR: concat(LR_prev, LR_curr, LR_next) -> x_coarse (HR size)
        2. Encoder:   HR -> z (latent features)
        3. VQ:        z  -> z_q (quantized) + indices + losses
        4. Decoder:   (z_q, x_coarse) -> x_hat (reconstructed HR)

    Losses:
        L_recon   = L1(HR, x_hat)
        L_coarse  = L1(HR, x_coarse)
        L_commit  = commitment_cost * MSE(sg[z_q], z)
        L_entropy = entropy_weight * (log(K) - H(usage))
    """

    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]

        self.encoder = Encoder(
            in_ch=m["encoder_in_ch"],
            ch=m["encoder_ch"],
            embed_dim=m["encoder_embed_dim"],
            n_res_blocks=m["encoder_n_res_blocks"],
        )

        self.quantize = FactorizedL2VQ(
            embed_dim=m["embed_dim"],
            codebook_dim=m["codebook_dim"],
            n_embed=m["codebook_size"],
            decay=m["vq_decay"],
            commitment_cost=m["commitment_cost"],
            dead_code_threshold=m["dead_code_threshold"],
        )

        self.decoder = Decoder(
            embed_dim=m["embed_dim"],
            coarse_feat_ch=m["coarse_sr_n_feats"],
            ch=m["decoder_ch"],
            n_res_blocks=m["decoder_n_res_blocks"],
        )

        self.coarse_sr = CoarseSRNet(
            in_ch=m["coarse_sr_in_ch"],
            n_feats=m["coarse_sr_n_feats"],
            n_res_blocks=m["coarse_sr_n_res_blocks"],
            scale=m["coarse_sr_scale"],
        )

        self.dead_code_replace_freq = m["dead_code_replace_freq"]

    def forward(self, lr_prev, lr_curr, lr_next, hr_curr):
        """
        Args:
            lr_prev, lr_curr, lr_next: (B, 3, lr_h, lr_w) LR temporal triplet
            hr_curr: (B, 3, hr_h, hr_w) HR ground truth

        Returns:
            dict with x_hat, x_coarse, losses, and codebook metrics
        """
        # 1. Coarse SR from temporal LR triplet
        lr_concat = torch.cat([lr_prev, lr_curr, lr_next], dim=1)  # (B, 9, ...)
        x_coarse = self.coarse_sr(lr_concat)

        # 2. Encode HR texture
        z = self.encoder(hr_curr)

        # 3. Quantize
        z_q, commit_loss, indices, cb_metrics = self.quantize(z)

        # 4. Decode (texture + structure)
        x_hat = self.decoder(z_q, x_coarse)

        # 5. Losses
        recon_loss = nn.functional.l1_loss(x_hat, hr_curr)
        coarse_loss = nn.functional.l1_loss(x_coarse, hr_curr)

        # Dead code replacement (periodic)
        n_replaced = 0
        if self.training:
            iter_count = int(self.quantize.iter_count.item())
            if iter_count % self.dead_code_replace_freq == 0 and iter_count > 0:
                z_low = self.quantize.proj_down(z)
                z_flat = z_low.permute(0, 2, 3, 1).reshape(
                    -1, self.quantize.codebook_dim
                )
                z_flat = self.quantize._l2_normalize(z_flat, dim=-1)
                n_replaced = self.quantize.replace_dead_codes(z_flat)

        return {
            "x_hat": x_hat,
            "x_coarse": x_coarse,
            "indices": indices,
            "recon_loss": recon_loss,
            "coarse_loss": coarse_loss,
            "commit_loss": commit_loss,
            "entropy_loss": cb_metrics["entropy_loss"],
            "metrics": cb_metrics,
            "n_replaced": n_replaced,
        }

