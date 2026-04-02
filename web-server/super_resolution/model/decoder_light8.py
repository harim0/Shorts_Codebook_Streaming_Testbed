# decoder_light8.py
# Lightweight decoder for scale/8 models (texVQVAE_1080_8_light).
# Architecture is identical to decoder8.Decoder — different hyperparams at init:
#   ch=48, n_res_blocks=1, coarse_feat_ch=32  (vs 128/4/64 in the full version).
from .decoder8 import Decoder as DecoderLight  # noqa: F401
