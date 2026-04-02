#!/usr/bin/env python3
"""
High-quality figures for Codebook SR Streaming Testbed
Fixes applied:
 - Replaced 3-digit hex colors (e.g., '#333') with 6-digit hex ('#333333')
 - Dashlet Fig 3 logic: 1 Class = 5 Videos. Model DL -> i-matrix DL -> Seg DL.
 - NAS Fig 17 logic: Download timeline vs SR Processor timeline.
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors

RESULTS_JSON = '/home/harim/Shorts_Codebook_Streaming_Testbed/simulation/results/sim_results.json'
OUT_DIR      = '/home/harim/Shorts_Codebook_Streaming_Testbed/simulation/results_gemini'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,  # High Quality for papers
    'axes.spines.right': False,
    'axes.spines.top': False,
})

CLASS_COLOR = {
    'Animal':   '#4C72B0',
    'Cooking':  '#DD8452',
    'Dance':    '#55A868',
    'Gameplay': '#C44E52',
    'Hobby':    '#8172B2',
    'Speech':   '#937860',
}

DL_COLOR = {
    'imatrix':  '#5B9BD5',   # blue
    'model':    '#ED7D31',   # orange
    'segment':  '#70AD47',   # green
    'sr':       '#FFC000',   # yellow (processing)
}

try:
    with open(RESULTS_JSON) as f:
        results = json.load(f)
except FileNotFoundError:
    print(f"Error: {RESULTS_JSON} not found. Run simulation first.")
    exit(1)

N = len(results)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Event Simulation Logic (NAS & Dashlet Hybrid)
# ─────────────────────────────────────────────────────────────────────────────
BW_MBps  = 1.25  # 10 Mbps
PLAY_TIME = 4.0   # 120 frames at 30fps = 4.0s

def dl_sec(nbytes):
    return nbytes / (BW_MBps * 1024 * 1024)

events = []
playback_times = []
sr_done_times = []

t_dl = 0.0  # Download network is sequential (no race conditions)
t_sr = 0.0  # SR processor is sequential (1 GPU)
t_play = 0.0 # Player timeline

model_cached = set()

for i, r in enumerate(results):
    class_name = r['class']
    vid_label = r['vid'].split('/')[-1][:12]

    # --- DOWNLOAD PHASE ---
    # 1. Model Download (Once per class, strictly before i-matrix)
    if class_name not in model_cached:
        # Simulate slight gap for user swipe / mpd load
        if i > 0: t_dl += 0.5 
        
        m_bytes = r['decoder_bytes'] + r['coarse_sr_bytes']
        t_dl_end = t_dl + dl_sec(m_bytes)
        events.append({'vid': i, 'type': 'model', 't0': t_dl, 't1': t_dl_end, 'label': class_name})
        t_dl = t_dl_end
        model_cached.add(class_name)

    # 2. i-matrix Download (Right after model or previous segment)
    t_dl_end = t_dl + dl_sec(r['imatrix_bytes'])
    events.append({'vid': i, 'type': 'imatrix', 't0': t_dl, 't1': t_dl_end, 'label': vid_label})
    t_dl = t_dl_end

    # 3. Segment Download
    t_dl_end = t_dl + dl_sec(r.get('seg1_bytes', 0) + r.get('seg2_bytes', 0))
    if t_dl_end == t_dl: t_dl_end += 0.5 # fallback if size is 0
    events.append({'vid': i, 'type': 'segment', 't0': t_dl, 't1': t_dl_end, 'label': vid_label})
    dl_finish_time = t_dl_end
    t_dl = t_dl_end

    # --- SR PROCESSING PHASE ---
    # SR starts when both download is finished AND GPU is free
    sr_start = max(dl_finish_time, t_sr)
    sr_end = sr_start + r['sr_sec'] + r['encode_sec'] + r['decode_sec']
    events.append({'vid': i, 'type': 'sr', 't0': sr_start, 't1': sr_end, 'label': vid_label})
    t_sr = sr_end
    sr_done_times.append(sr_end)

    # --- PLAYBACK PHASE ---
    # Playback starts when SR is done AND previous playback is finished
    if i == 0:
        play_start = sr_end
    else:
        play_start = max(t_play, sr_end)
    
    playback_times.append(play_start)
    t_play = play_start + PLAY_TIME

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1. NAS Figure 17-style: DNN Processor + Player two-row timeline
# ─────────────────────────────────────────────────────────────────────────────
fig1, axes = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={'height_ratios': [1, 2], 'hspace': 0.4})
ax_top, ax_bot = axes

SHOW_N = min(N, 15) # Show first 15 videos to avoid visual clutter

# Top: SR Processing
for ev in [e for e in events if e['type'] == 'sr' and e['vid'] < SHOW_N]:
    i = ev['vid']
    cname = results[i]['class']
    ax_top.barh(0, ev['t1'] - ev['t0'], left=ev['t0'], height=0.6,
                color=CLASS_COLOR[cname], edgecolor='white', linewidth=0.5)
    ax_top.text((ev['t0']+ev['t1'])/2, 0, f"V{i+1}", ha='center', va='center', 
                fontsize=7, color='white', fontweight='bold')

ax_top.set_yticks([0]); ax_top.set_yticklabels(['DNN\nProcessor'], fontweight='bold')
ax_top.set_xlim(0, sr_done_times[SHOW_N-1] * 1.02)
ax_top.set_title('(a) Codebook SR Processor Timeline (NAS style)', loc='left', fontweight='bold')
ax_top.axhline(-0.4, color='#888888', linewidth=1, linestyle='--')
ax_top.set_ylim(-0.6, 0.6)

# Bottom: Download events per video
type_h = {'imatrix': 0.3, 'model': 0.3, 'segment': 0.3}
type_y = {'imatrix': 0.35, 'model': 0.0, 'segment': -0.35}

for ev in events:
    i = ev['vid']
    if i >= SHOW_N: continue
    t = ev['type']
    if t == 'sr': continue
    y = i + type_y.get(t, 0)
    ax_bot.barh(y, ev['t1']-ev['t0'], left=ev['t0'], height=type_h[t],
                color=DL_COLOR[t], edgecolor='#333333', linewidth=0.3, alpha=0.9)

# Playback cursor
play_x, play_y = [], []
for i in range(SHOW_N):
    play_x.extend([playback_times[i], playback_times[i] + PLAY_TIME])
    play_y.extend([i, i])
ax_bot.plot(play_x, play_y, color='red', linewidth=2.0, label='Playback', zorder=5)

ax_bot.set_yticks(range(SHOW_N))
ax_bot.set_yticklabels([f"V{i+1}: {results[i]['class'][:4]}" for i in range(SHOW_N)], fontsize=8)
ax_bot.set_xlabel('Wall-clock time (s)', fontweight='bold')
ax_bot.set_title('(b) Player Download & Playback Timeline', loc='left', fontweight='bold')
ax_bot.invert_yaxis()
ax_bot.set_xlim(0, sr_done_times[SHOW_N-1] * 1.02)
ax_bot.grid(axis='x', linestyle=':', alpha=0.5)

dl_patches  = [mpatches.Patch(color=DL_COLOR[t], label=t.capitalize()) for t in ['model','imatrix','segment']]
dl_patches += [plt.Line2D([0],[0], color='red', lw=2, label='Playback')]
ax_bot.legend(handles=dl_patches, loc='lower right', framealpha=0.9)

plt.savefig(os.path.join(OUT_DIR, 'fig1_nas_timeline.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'fig1_nas_timeline.png'), bbox_inches='tight')
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2. Dashlet Figure 3-style: per-slot download + buffer occupancy
# ─────────────────────────────────────────────────────────────────────────────
fig2 = plt.figure(figsize=(15, 8))
gs   = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1], hspace=0.3)
ax_dl  = fig2.add_subplot(gs[0])
ax_buf = fig2.add_subplot(gs[1])

SHOW_N2 = min(N, 10) # 2 Classes (5 videos each)

for ev in events:
    i = ev['vid']
    if i >= SHOW_N2: continue
    t = ev['type']
    if t == 'sr': continue
    
    # Model bar is slightly thicker to stand out
    bw = 0.6 if t == 'model' else 0.4
    
    # Draw Model DL at the row of the first video of that class
    if t == 'model':
        ax_dl.barh(i, ev['t1']-ev['t0'], left=ev['t0'], height=bw, color=DL_COLOR[t], edgecolor='#333333', zorder=4)
        ax_dl.text(ev['t0'], i-0.4, f" {ev['label']} Model", fontsize=8, color='#333333', fontweight='bold')
    else:
        # imatrix and segment side by side
        ax_dl.barh(i, ev['t1']-ev['t0'], left=ev['t0'], height=bw, color=DL_COLOR[t], edgecolor='white', zorder=3)

# Thin lines connecting DL chunks
for i in range(SHOW_N2):
    vid_evs = [e for e in events if e['vid'] == i and e['type'] in ('imatrix','segment')]
    if len(vid_evs) >= 2:
        ax_dl.plot([vid_evs[0]['t0'], vid_evs[-1]['t1']], [i, i], color='#555555', lw=0.8, zorder=1, alpha=0.5)

# Playback line (Dashlet style staircase)
play_x2, play_y2 = [], []
for i in range(SHOW_N2):
    play_x2.extend([playback_times[i], playback_times[i]+PLAY_TIME, None])
    play_y2.extend([i, i, None])
ax_dl.plot(play_x2, play_y2, color='red', lw=2.5, label='Video Play', zorder=6)

# Highlight Gap (SR Processing time)
if len(playback_times) > 0:
    ax_dl.annotate('Waiting for SR Processing (~18s)', 
                   xy=(events[2]['t1'], 0), xytext=(events[2]['t1']+5, -0.5),
                   arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5), fontsize=9, fontweight='bold', color='#C44E52')

ax_dl.set_yticks(range(SHOW_N2))
ax_dl.set_yticklabels([f"V{i+1} [{results[i]['class']}]" for i in range(SHOW_N2)], fontsize=9)
ax_dl.set_title('(a) Video chunk downloading and playing timeline (Dashlet style)', loc='left', fontweight='bold')
ax_dl.invert_yaxis()
ax_dl.grid(axis='x', linestyle='--', alpha=0.4)
ax_dl.set_xlim(-1, playback_times[SHOW_N2-1] + PLAY_TIME + 2)

dl_patches2 = [mpatches.Patch(color=DL_COLOR[t], label=t.capitalize()) for t in ['model','imatrix','segment']]
dl_patches2 += [plt.Line2D([0],[0], color='red', lw=2.5, label='Video Play')]
ax_dl.legend(handles=dl_patches2, loc='lower right', framealpha=0.9)

# Buffer occupancy (bottom)
t_max = playback_times[SHOW_N2-1] + PLAY_TIME + 2
t_axis = np.linspace(0, t_max, 1000)
buf = []

for t in t_axis:
    buffered = 0
    for i in range(SHOW_N2):
        # Buffered if SR is done, but playback hasn't finished
        if sr_done_times[i] <= t and t < (playback_times[i] + PLAY_TIME):
            buffered += 1
    buf.append(buffered)

ax_buf.fill_between(t_axis, buf, step='post', alpha=0.3, color='#4C72B0')
ax_buf.step(t_axis, buf, where='post', color='#4C72B0', lw=2)
ax_buf.set_xlabel('Wall-clock time (s)', fontweight='bold')
ax_buf.set_ylabel('# ready videos\n(SR Done)', fontweight='bold')
ax_buf.set_title('(b) Client-side buffer occupancy (Ready to play)', loc='left', fontweight='bold')
ax_buf.set_ylim(-0.2, max(buf) + 1)
ax_buf.set_xlim(-1, t_max)
ax_buf.grid(linestyle='--', alpha=0.4)

plt.savefig(os.path.join(OUT_DIR, 'fig2_dashlet_timeline.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'fig2_dashlet_timeline.png'), bbox_inches='tight')
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3. PSNR Comparison: CoarseSR vs Codebook E2E
# ─────────────────────────────────────────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(15, 5))

x      = np.arange(N)
e2e    = np.array([r['psnr_e2e_mean'] for r in results])
coarse = np.array([r['psnr_coarse_mean'] if r['psnr_coarse_mean'] else np.nan for r in results])
gain   = e2e - coarse

bw = 0.38
ax.bar(x - bw/2, coarse, bw, label='CoarseSR Baseline', color='#AEC7E8', edgecolor='#333333', lw=0.5)
ax.bar(x + bw/2, e2e,    bw, label='Codebook SR E2E',   color='#4C72B0', edgecolor='#333333', lw=0.5)

for i in range(N):
    if not np.isnan(gain[i]):
        ax.text(x[i], max(e2e[i], coarse[i]) + 0.5, f'+{gain[i]:.2f}',
                ha='center', va='bottom', fontsize=7, color='#C44E52', fontweight='bold', rotation=45)

ax.axhline(np.nanmean(coarse), color='#AEC7E8', lw=1.5, linestyle='--')
ax.axhline(np.nanmean(e2e),    color='#4C72B0', lw=1.5, linestyle='--')

ax.set_xticks(x)
ax.set_xticklabels([r['vid'].split('/')[-1][:12] for r in results], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('PSNR (dB)', fontweight='bold')
ax.set_title('Video Quality Comparison (270p LR → 1080p HR)', fontweight='bold')
ax.legend(loc='upper left', framealpha=0.9)
ax.set_ylim(min(np.nanmin(coarse)-1, 22), max(np.nanmax(e2e)+3, 42))
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Background Class Color Bands
for i, r in enumerate(results):
    cl = r['class']
    ax.axvspan(i-0.5, i+0.5, alpha=0.08, color=CLASS_COLOR[cl], zorder=0)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig3_psnr_comparison.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'fig3_psnr_comparison.png'), bbox_inches='tight')
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4. Table 7-style: Processing time per phase
# ─────────────────────────────────────────────────────────────────────────────
dec_arr = np.array([r['decode_sec'] for r in results])
sr_arr  = np.array([r['sr_sec']     for r in results])
enc_arr = np.array([r['encode_sec'] for r in results])

fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
phase_info = [
    (dec_arr, 'Decode (s)',  '#4C72B0'),
    (sr_arr,  'SR Inference (s)', '#DD8452'),
    (enc_arr, 'Encode (s)',  '#55A868'),
]

for ax4, (arr, title, col) in zip(axes4, phase_info):
    x4 = np.arange(N)
    colors4 = [CLASS_COLOR[r['class']] for r in results]
    ax4.bar(x4, arr, color=colors4, alpha=0.9, edgecolor='#333333', lw=0.5)
    ax4.axhline(arr.mean(), color='red', lw=1.5, linestyle='--', label=f'Avg = {arr.mean():.2f}s')
    
    ax4.set_xticks(x4)
    ax4.set_xticklabels([r['vid'].split('/')[-1][:9] for r in results], rotation=90, fontsize=6)
    ax4.set_title(title, fontweight='bold')
    ax4.legend()
    ax4.grid(axis='y', linestyle='--', alpha=0.5)

fig4.suptitle('Video Processing Time per Phase (120 frames per video)', fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig4_phase_time.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'fig4_phase_time.png'), bbox_inches='tight')
plt.close()

print('\n[PLOT] All 4 high-quality figures generated successfully in:', OUT_DIR)