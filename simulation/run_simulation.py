#!/usr/bin/env python3
"""
Codebook SR E2E Simulation
- imatrix: HR-based (encoder(hr_curr), corrected)
- Per-frame PSNR: CoarseSR baseline vs Codebook E2E
- Phase timing: decode / SR / encode
- Download size per file type (for Dashlet-style timeline)
- Output: results/sim_results.json
"""

import os, sys, time, json, subprocess
import numpy as np
import torch
import cv2

sys.path.insert(0, '/home/harim/Short-form-Video-Codebook-Switching')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../web-server/super_resolution'))

from codebook_sr import CodebookSR

CDN_ROOT   = '/home/harim/Shorts_Codebook_Streaming_Testbed/cdn-server/contentServer/dash'
MODEL_ROOT = os.path.join(CDN_ROOT, 'model')
DATA_ROOT  = os.path.join(CDN_ROOT, 'data')
SEQ_JSON   = '/home/harim/Shorts_Codebook_Streaming_Testbed/web-server/static/sequence.json'
OUT_DIR    = '/home/harim/Shorts_Codebook_Streaming_Testbed/simulation/results'
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[SIM] device={DEVICE}')

# ─── Model ───────────────────────────────────────────────────────────────────
sr_model     = CodebookSR(device=DEVICE)
loaded_class = None

def ensure_class_model(class_name):
    global loaded_class
    if loaded_class == class_name:
        return
    dec = os.path.join(MODEL_ROOT, class_name, 'decoder.pt')
    csr = os.path.join(MODEL_ROOT, class_name, 'coarse_sr.pt')
    sr_model.load_model(dec, csr)
    loaded_class = class_name

# ─── PSNR ────────────────────────────────────────────────────────────────────
def psnr(a, b):
    """numpy HWC uint8 → float PSNR (dB)"""
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 100.0 if mse < 1e-10 else 10 * np.log10(255.0**2 / mse)

# ─── Frame reader ─────────────────────────────────────────────────────────────
def read_frames(path, target_wh=None):
    cap, out = cv2.VideoCapture(path), []
    while True:
        ret, f = cap.read()
        if not ret: break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if target_wh:
            f = cv2.resize(f, target_wh, interpolation=cv2.INTER_CUBIC)
        out.append(f)
    cap.release()
    return out

# ─── File size helper ────────────────────────────────────────────────────────
def fsize(path):
    return os.path.getsize(path) if os.path.exists(path) else 0

# ─── Per-video ────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_video(vid_id, class_name):
    vid_dir   = os.path.join(DATA_ROOT, vid_id)
    lr_path   = os.path.join(vid_dir, 'h264_400k.mp4')
    hr_path   = os.path.join(DATA_ROOT, class_name, f'{vid_id.split("/")[1]}.mp4')
    imat_path = os.path.join(vid_dir, 'imatrix.pt')
    seg1_path = os.path.join(vid_dir, 'segment_0_1.m4s')
    seg2_path = os.path.join(vid_dir, 'segment_0_2.m4s')
    dec_path  = os.path.join(MODEL_ROOT, class_name, 'decoder.pt')
    csr_path  = os.path.join(MODEL_ROOT, class_name, 'coarse_sr.pt')

    ensure_class_model(class_name)

    # File sizes (bytes) — for download timeline simulation
    dl_sizes = {
        'imatrix_bytes':   fsize(imat_path),
        'decoder_bytes':   fsize(dec_path),
        'coarse_sr_bytes': fsize(csr_path),
        'seg1_bytes':      fsize(seg1_path),
        'seg2_bytes':      fsize(seg2_path),
    }

    # ── Decode ──
    t0 = time.time()
    lr_frames = read_frames(lr_path)                       # (480,270,3) list
    hr_frames = read_frames(hr_path)                       # (1920,1080,3) list
    decode_sec = time.time() - t0

    n = min(len(lr_frames), len(hr_frames), 150)
    lr_frames  = lr_frames[:n]
    hr_frames  = hr_frames[:n]

    imatrix = torch.load(imat_path, map_location='cpu')[:n]  # (n, 240, 135) HR-based
    sr_model.set_imatrix(imatrix)

    # ── SR + PSNR ──
    psnr_e2e_list, psnr_coarse_list = [], []
    sr_frames_cpu = []

    t1 = time.time()
    for i in range(n):
        prev_t = torch.from_numpy(lr_frames[i-1] if i > 0 else lr_frames[0]).byte().to(DEVICE)
        curr_t = torch.from_numpy(lr_frames[i]).byte().to(DEVICE)

        # ── E2E codebook SR ──
        out_t = sr_model.infer_with_imatrix(prev_t, curr_t, i)   # (1920,1080,3) CUDA uint8
        out_np = out_t.cpu().numpy()
        sr_frames_cpu.append(out_np)
        psnr_e2e_list.append(psnr(out_np, hr_frames[i]))

        # ── CoarseSR PSNR baseline ──
        def prep(t): return t.float().div(255.0).permute(2,0,1).unsqueeze(0)
        lr_cat  = torch.cat([prep(prev_t), prep(curr_t), prep(curr_t)], dim=1).to(DEVICE)
        x_csr   = sr_model.model.coarse_sr(lr_cat)               # (1, C, H, W)
        if x_csr.shape[1] == 3:
            csr_np = (x_csr.squeeze(0).permute(1,2,0).clamp(0,1)*255).byte().cpu().numpy()
            psnr_coarse_list.append(psnr(csr_np, hr_frames[i]))

    torch.cuda.synchronize()
    sr_sec = time.time() - t1

    # ── Encode ──
    out_mp4 = os.path.join(OUT_DIR, vid_id.replace('/', '_') + '.mp4')
    t2 = time.time()
    cmd = ['/usr/bin/ffmpeg', '-y', '-loglevel', 'error',
           '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-s', '1080x1920', '-pix_fmt', 'rgb24', '-r', '30', '-i', '-',
           '-vcodec', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', out_mp4]
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in sr_frames_cpu:
        pipe.stdin.write(f.tobytes())
    pipe.stdin.flush(); pipe.stdin.close(); pipe.wait()
    encode_sec = time.time() - t2

    r = {
        'vid': vid_id, 'class': class_name,
        'n_frames':   n,
        'decode_sec': round(decode_sec, 4),
        'sr_sec':     round(sr_sec, 4),
        'encode_sec': round(encode_sec, 4),
        'total_sec':  round(decode_sec + sr_sec + encode_sec, 4),
        'sr_fps':     round(n / max(sr_sec, 1e-6), 2),
        'psnr_e2e_mean':    round(float(np.mean(psnr_e2e_list)), 3),
        'psnr_e2e_std':     round(float(np.std(psnr_e2e_list)), 3),
        'psnr_coarse_mean': round(float(np.mean(psnr_coarse_list)), 3) if psnr_coarse_list else None,
        **dl_sizes,
    }
    msg = (f'  [{vid_id}] {n}fr  dec={decode_sec:.3f}s  sr={sr_sec:.3f}s  enc={encode_sec:.3f}s'
           f'  PSNR_e2e={r["psnr_e2e_mean"]:.2f}dB')
    if r['psnr_coarse_mean']:
        msg += f'  PSNR_coarse={r["psnr_coarse_mean"]:.2f}dB'
    print(msg, flush=True)
    return r

def main():
    with open(SEQ_JSON) as f:
        sequence = json.load(f)

    all_results, sim_start = [], time.time()

    for i, item in enumerate(sequence):
        vid_id     = item['pid']
        class_name = item.get('class', vid_id.split('/')[0])
        print(f'[{i+1:2d}/{len(sequence)}] {vid_id}')
        try:
            all_results.append(run_video(vid_id, class_name))
        except Exception:
            import traceback; traceback.print_exc()

    total_wall = time.time() - sim_start
    print(f'\n[SIM] Done in {total_wall:.1f}s')

    out_path = os.path.join(OUT_DIR, 'sim_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'[SIM] saved → {out_path}')

    # Summary
    e2e  = np.array([r['psnr_e2e_mean'] for r in all_results])
    dec  = np.array([r['decode_sec']    for r in all_results])
    sr   = np.array([r['sr_sec']        for r in all_results])
    enc  = np.array([r['encode_sec']    for r in all_results])
    tot  = dec + sr + enc
    fps  = np.array([r['sr_fps']        for r in all_results])
    csr_arr = [r['psnr_coarse_mean'] for r in all_results if r['psnr_coarse_mean']]

    print('\n=== Phase Timing & PSNR Summary ===')
    print(f'{"Metric":<20} {"Mean":>8} {"Std":>8} {"Min":>8} {"Max":>8}')
    print('-'*52)
    for name, arr in [('Decode(s)', dec), ('SR(s)', sr), ('Encode(s)', enc),
                      ('Total(s)', tot), ('SR fps', fps), ('PSNR E2E(dB)', e2e)]:
        print(f'{name:<20} {arr.mean():>8.3f} {arr.std():>8.3f} {arr.min():>8.3f} {arr.max():>8.3f}')
    if csr_arr:
        ca = np.array(csr_arr)
        print(f'{"PSNR CoarseSR(dB)":<20} {ca.mean():>8.3f} {ca.std():>8.3f} {ca.min():>8.3f} {ca.max():>8.3f}')

if __name__ == '__main__':
    main()
