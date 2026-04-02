"""
codebook_sr.py
tex-VQVAE + CoarseSR 기반 Codebook SR 래퍼.

- 초기화: 기본 checkpoint(texVQVAE_1080_8_light.pt) 에서 전체 모델 로드
- load_model(): /dnn 수신 후 decoder + coarse_sr 컴포넌트 교체 (class별 adaptation)
- infer(): LR 3-프레임 triplet → SR 1-프레임 (numpy HWC uint8)
"""

import os
import sys
import torch
import numpy as np

# codebook switching repo models 참조
_CODEBOOK_REPO = "/home/harim/Short-form-Video-Codebook-Switching"
if _CODEBOOK_REPO not in sys.path:
    sys.path.insert(0, _CODEBOOK_REPO)

from models.tex_vqvae8_light import TexVQVAE  # noqa: E402

DEFAULT_CKPT = os.path.join(
    os.path.dirname(__file__),
    "model/scale4_patch8/texVQVAE_1080_8_light.pt"
)


class CodebookSR:
    def __init__(self, ckpt_path=None, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        ckpt_path = ckpt_path or DEFAULT_CKPT

        print(f"[CodebookSR] loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg  = ckpt["config"]

        self.model = TexVQVAE(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self._imatrix = None   # (n_frames, H_z, W_z) int64, set per video
        self._refresh_codebook_cache()
        print(f"[CodebookSR] model ready on {self.device}")

    def load_model(self, decoder_path: str, coarse_sr_path: str):
        """class별 decoder + coarse_sr weight 교체."""
        decoder_sd   = torch.load(decoder_path,   map_location="cpu")
        coarse_sr_sd = torch.load(coarse_sr_path, map_location="cpu")
        self.model.decoder.load_state_dict(decoder_sd)
        self.model.coarse_sr.load_state_dict(coarse_sr_sd)
        # 교체 후 L2-normalized codebook 캐시 갱신
        self._refresh_codebook_cache()
        print(f"[CodebookSR] updated decoder+coarse_sr from {os.path.dirname(decoder_path)}")

    def _refresh_codebook_cache(self):
        """L2-normalized embedding.weight 캐싱 (inference time lookup용)."""
        w = self.model.quantize.embedding.weight.float()
        self._e_norm = torch.nn.functional.normalize(w, p=2, dim=-1).to(self.device)

    def set_imatrix(self, imatrix: torch.Tensor):
        """
        Per-video pre-computed index map 설정.
        Args:
            imatrix: (n_frames, H_z, W_z) int16 tensor (precompute_imatrix.py 출력)
        """
        self._imatrix = imatrix.long().to(self.device)  # (n_frames, 135, 240)
        print(f"[CodebookSR] imatrix set: shape={tuple(self._imatrix.shape)}")

    def _to_tensor(self, frame_hwc_uint8: np.ndarray) -> torch.Tensor:
        """numpy HWC uint8 → BCHW float32 [0,1] on device."""
        t = torch.from_numpy(frame_hwc_uint8).float().div(255.0)  # HWC
        t = t.permute(2, 0, 1).unsqueeze(0)                       # 1CHW
        return t.to(self.device)

    @torch.no_grad()
    def infer(self,
              lr_prev: np.ndarray,
              lr_curr: np.ndarray,
              lr_next: np.ndarray) -> np.ndarray:
        """
        Args:
            lr_prev, lr_curr, lr_next: numpy HWC uint8 (270×480×3)
        Returns:
            SR frame: numpy HWC uint8 (1080×1920×3)
        """
        prev_t = self._to_tensor(lr_prev)
        curr_t = self._to_tensor(lr_curr)
        next_t = self._to_tensor(lr_next)

        x_hat, _ = self.model.inference(prev_t, curr_t, next_t)

        # BCHW float [0,1] → HWC uint8
        x_hat = x_hat.squeeze(0).permute(1, 2, 0)
        x_hat = (x_hat * 255.0).clamp(0, 255).byte()
        return x_hat.cpu().numpy()

    @torch.no_grad()
    def infer_with_imatrix(self,
                           lr_prev_hwc: torch.Tensor,
                           lr_curr_hwc: torch.Tensor,
                           frame_idx: int) -> torch.Tensor:
        """
        i-matrix lookup 기반 추론 (encoder 실행 없음).
        Client 동작 시뮬레이션:
          z_q_low = L2_normalize(embedding.weight)[imatrix[frame_idx]]
          z_q     = proj_up(z_q_low)
          x_hat   = decoder(z_q, coarse_sr(LR_triplet))

        Args:
            lr_prev/curr_hwc: CUDA tensor HWC uint8
            frame_idx: current frame index into imatrix
        Returns:
            SR frame: CUDA tensor HWC uint8
        """
        if self._imatrix is None:
            raise RuntimeError("imatrix not set — call set_imatrix() first")

        def prep(t):
            return t.float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)

        prev_t = prep(lr_prev_hwc)
        curr_t = prep(lr_curr_hwc)

        # 1. Coarse SR (temporal triplet, next=curr)
        lr_concat = torch.cat([prev_t, curr_t, curr_t], dim=1)
        x_coarse  = self.model.coarse_sr(lr_concat)           # (1, 32, 1080, 1920)

        # 2. Codebook lookup — NO encoder
        indices = self._imatrix[frame_idx % len(self._imatrix)]  # (135, 240)
        H_z, W_z = indices.shape
        e_norm   = self._e_norm                                # (n_embed, codebook_dim)
        z_q_low_flat = e_norm[indices.flatten()]               # (H_z*W_z, codebook_dim)
        z_q_low  = z_q_low_flat.reshape(1, H_z, W_z, -1).permute(0, 3, 1, 2)  # (1, D, H, W)
        z_q      = self.model.quantize.proj_up(z_q_low)       # (1, embed_dim, H, W)

        # 3. Decode
        x_hat = self.model.decoder(z_q, x_coarse)             # (1, 3, 1080, 1920)

        # BCHW → HWC uint8
        x_hat = x_hat.squeeze(0).permute(1, 2, 0)
        return (x_hat * 255.0).clamp(0, 255).byte()

    @torch.no_grad()
    def infer_tensor(self,
                     lr_prev_hwc: torch.Tensor,
                     lr_curr_hwc: torch.Tensor,
                     lr_next_hwc: torch.Tensor) -> torch.Tensor:
        """
        GPU-native 경로: shared CUDA tensor (HWC uint8) 를 직접 처리.
        Args:
            lr_prev/curr/next_hwc: CUDA tensor HWC uint8
        Returns:
            SR frame: CUDA tensor HWC uint8 (1080×1920×3)
        """
        def prep(t):
            return t.float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)

        x_hat, _ = self.model.inference(prep(lr_prev_hwc),
                                        prep(lr_curr_hwc),
                                        prep(lr_next_hwc))
        # BCHW → HWC uint8
        x_hat = x_hat.squeeze(0).permute(1, 2, 0)
        return (x_hat * 255.0).clamp(0, 255).byte()
