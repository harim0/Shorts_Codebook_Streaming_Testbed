#!/usr/bin/env python3
"""
extract_model.py
texVQVAE_1080_8_light.pt['model'] 에서 decoder / coarse_sr 컴포넌트를 분리해
CDN model/{class}/ 아래에 배포한다.

Usage:
    python3 extract_model.py

Output per class:
    cdn-server/contentServer/dash/model/{class}/decoder.pt
    cdn-server/contentServer/dash/model/{class}/coarse_sr.pt
"""

import os
import torch

CKPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../web-server/super_resolution/model/scale4_patch8/texVQVAE_1080_8_light.pt"
)
DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
MODEL_ROOT = os.path.join(os.path.dirname(__file__), "model")


def extract_component(state_dict, prefix):
    """state_dict에서 prefix.* 키만 추출하고 prefix. 제거해서 반환."""
    prefix_dot = prefix + "."
    return {
        k[len(prefix_dot):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix_dot)
    }


def main():
    print(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model_sd = ckpt["model"]

    decoder_sd  = extract_component(model_sd, "decoder")
    coarse_sr_sd = extract_component(model_sd, "coarse_sr")

    print(f"  decoder   : {len(decoder_sd)} tensors")
    print(f"  coarse_sr : {len(coarse_sr_sd)} tensors")

    classes = sorted(
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    )
    print(f"\nDeploying to {len(classes)} classes: {classes}")

    for cls in classes:
        out_dir = os.path.join(MODEL_ROOT, cls)
        os.makedirs(out_dir, exist_ok=True)

        decoder_path   = os.path.join(out_dir, "decoder.pt")
        coarse_sr_path = os.path.join(out_dir, "coarse_sr.pt")

        torch.save(decoder_sd,   decoder_path)
        torch.save(coarse_sr_sd, coarse_sr_path)

        dec_mb  = os.path.getsize(decoder_path)   / 1024 / 1024
        csr_mb  = os.path.getsize(coarse_sr_path) / 1024 / 1024
        print(f"  [{cls}] decoder.pt={dec_mb:.1f}MB  coarse_sr.pt={csr_mb:.1f}MB")

    print("\nDone.")


if __name__ == "__main__":
    main()
