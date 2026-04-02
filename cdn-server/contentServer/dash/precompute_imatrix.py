#!/usr/bin/env python3
"""
precompute_imatrix.py  (v2 - HR-based)

i-matrix 추출 파이프라인 (올바른 버전):
  HR frame → encoder(hr_curr) → proj_down → L2-norm → cdist → argmin → indices

TexVQVAE.forward() 에서 확인:
  z = self.encoder(hr_curr)   ← HR 직접 입력 (NOT coarse_sr(LR))

잘못된 이전 버전:
  x_coarse = model.coarse_sr(lr_concat)   ← LR→coarse SR
  z = model.encoder(x_coarse)             ← coarse SR 피처를 인코딩 (오류)

Output per video:
  data/{class}/{vid}/imatrix.pt   shape (n_frames, H_z, W_z) int16

Usage:
  python3 precompute_imatrix.py [--overwrite]
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2

_CODEBOOK_REPO = "/home/harim/Short-form-Video-Codebook-Switching"
if _CODEBOOK_REPO not in sys.path:
    sys.path.insert(0, _CODEBOOK_REPO)

from models.tex_vqvae8_light import TexVQVAE  # noqa

CKPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../web-server/super_resolution/model/scale4_patch8/texVQVAE_1080_8_light.pt"
)
DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# HR source: data/{class}/{vid_name}.mp4  (1080×1920 portrait)
# LR segments: data/{class}/{vid_name}/h264_400k.mp4 (270×480)


def load_model(ckpt_path):
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model = TexVQVAE(ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def frame_to_tensor(frame_hwc_uint8, device):
    """numpy HWC uint8 → BCHW float [0,1]"""
    t = torch.from_numpy(frame_hwc_uint8).float().div(255.0)
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def read_frames(video_path, target_wh=None):
    """
    Read all frames from video_path.
    target_wh: (width, height) for cv2.resize; None = keep native.
    Returns list of numpy HWC uint8.
    """
    cap    = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if target_wh is not None:
            frame = cv2.resize(frame, target_wh, interpolation=cv2.INTER_CUBIC)
        frames.append(frame)
    cap.release()
    return frames


@torch.no_grad()
def compute_imatrix_for_video(model, hr_video_path):
    """
    HR frames → encoder → quantize → i-matrix.

    Args:
        model: TexVQVAE (eval)
        hr_video_path: path to HR source .mp4 (1080×1920)

    Returns:
        imatrix: (n_frames, H_z, W_z) int16  e.g. (150, 240, 135)
    """
    hr_frames = read_frames(hr_video_path)  # native 1080×1920
    n_frames  = len(hr_frames)
    if n_frames == 0:
        print(f"  [WARN] no frames: {hr_video_path}")
        return None

    # Pre-compute L2-normalized codebook
    e_norm = F.normalize(
        model.quantize.embedding.weight.float(), p=2, dim=-1
    ).to(DEVICE)  # (K, codebook_dim)

    all_indices = []

    for i, hr_frame in enumerate(hr_frames):
        # HR → BCHW float [0,1]
        hr_t = frame_to_tensor(hr_frame, DEVICE)  # (1, 3, 1920, 1080)

        # Encoder: HR texture → latent (TexVQVAE.forward line: z = self.encoder(hr_curr))
        z     = model.encoder(hr_t)              # (1, embed_dim, H_z, W_z)
        z_low = model.quantize.proj_down(z)      # (1, codebook_dim, H_z, W_z)

        B, C, H_z, W_z = z_low.shape
        z_low_flat = z_low.permute(0, 2, 3, 1).reshape(-1, C).float()
        z_low_norm = F.normalize(z_low_flat, p=2, dim=-1)  # (H_z*W_z, codebook_dim)

        dist    = torch.cdist(z_low_norm, e_norm)          # (H_z*W_z, K)
        indices = dist.argmin(dim=-1).reshape(H_z, W_z)    # (H_z, W_z)

        all_indices.append(indices.cpu().to(torch.int16))

        if (i + 1) % 30 == 0:
            print(f"    frame {i+1}/{n_frames} indices shape=({H_z},{W_z})", flush=True)

    imatrix = torch.stack(all_indices, dim=0)  # (n_frames, H_z, W_z)
    return imatrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--overwrite', action='store_true',
                    help='Overwrite existing imatrix.pt files (needed after v2 fix)')
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {CKPT_PATH}")
    model = load_model(CKPT_PATH)
    print("Model loaded.\n")

    for class_name in sorted(os.listdir(DATA_ROOT)):
        class_dir = os.path.join(DATA_ROOT, class_name)
        if not os.path.isdir(class_dir):
            continue

        for vid_name in sorted(os.listdir(class_dir)):
            vid_dir = os.path.join(class_dir, vid_name)
            if not os.path.isdir(vid_dir):
                continue

            # HR source at class level: data/{class}/{vid_name}.mp4
            hr_video_path = os.path.join(class_dir, f"{vid_name}.mp4")
            imatrix_out   = os.path.join(vid_dir, "imatrix.pt")

            if not os.path.exists(hr_video_path):
                print(f"[SKIP] {class_name}/{vid_name}: no HR source {hr_video_path}")
                continue

            if os.path.exists(imatrix_out) and not args.overwrite:
                print(f"[SKIP] {class_name}/{vid_name}: imatrix.pt exists (use --overwrite)")
                continue

            print(f">>> {class_name}/{vid_name}  HR={hr_video_path}", flush=True)
            imatrix = compute_imatrix_for_video(model, hr_video_path)
            if imatrix is None:
                continue

            torch.save(imatrix, imatrix_out)
            mb = os.path.getsize(imatrix_out) / 1024 / 1024
            print(f"    saved: shape={tuple(imatrix.shape)}  {mb:.1f}MB → {imatrix_out}\n")

    print("Done.")


if __name__ == "__main__":
    main()
