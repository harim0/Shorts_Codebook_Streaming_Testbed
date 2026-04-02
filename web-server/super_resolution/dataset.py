"""Video SR dataset with temporal triplet loading and video-aware sampling."""

import csv
import os
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


class VideoSRDataset(Dataset):
    """Loads temporal triplets (t-1, t, t+1) with HR/LR pair generation.

    Each sample returns:
        lr_prev, lr_curr, lr_next: (3, lr_size, lr_size) tensors
        hr_curr: (3, hr_size, hr_size) tensor
    """

    def __init__(self, frame_records, hr_crop_size=160, scale_factor=4, augment=True):
        """
        Args:
            frame_records: list of dicts with keys:
                source, video_name, frame_file, frame_path, frame_index
            hr_crop_size: HR crop size (must be divisible by scale_factor)
            scale_factor: SR scale factor
            augment: whether to apply random flip
        """
        self.hr_size = hr_crop_size
        self.lr_size = hr_crop_size // scale_factor
        self.scale_factor = scale_factor
        self.augment = augment

        # Group frames by video and sort by frame file name
        self.video_frames = defaultdict(list)
        for rec in frame_records:
            key = (rec["source"], rec["video_name"])
            self.video_frames[key].append(rec["frame_path"])

        for key in self.video_frames:
            self.video_frames[key].sort()

        # Build flat index: (video_key, frame_position_in_video)
        self.samples = []
        self.video_ids = []
        for key, frames in self.video_frames.items():
            for pos in range(len(frames)):
                self.samples.append((key, pos))
                self.video_ids.append(key)

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def get_video_id(self, idx):
        return self.video_ids[idx]

    def _load_image(self, path):
        return Image.open(path).convert("RGB")

    def _get_triplet_paths(self, video_key, pos):
        frames = self.video_frames[video_key]
        n = len(frames)
        prev_pos = max(0, pos - 1)
        next_pos = min(n - 1, pos + 1)
        return frames[prev_pos], frames[pos], frames[next_pos]

    def __getitem__(self, idx):
        video_key, pos = self.samples[idx]
        path_prev, path_curr, path_next = self._get_triplet_paths(video_key, pos)

        img_prev = self._load_image(path_prev)
        img_curr = self._load_image(path_curr)
        img_next = self._load_image(path_next)

        # Random crop (same region for all 3 frames)
        w, h = img_curr.size
        crop_h = min(self.hr_size, h)
        crop_w = min(self.hr_size, w)
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        hr_prev = img_prev.crop((left, top, left + crop_w, top + crop_h))
        hr_curr = img_curr.crop((left, top, left + crop_w, top + crop_h))
        hr_next = img_next.crop((left, top, left + crop_w, top + crop_h))

        # Resize HR crop to exact hr_size if needed (when source is smaller)
        if crop_h != self.hr_size or crop_w != self.hr_size:
            hr_prev = hr_prev.resize((self.hr_size, self.hr_size), Image.BICUBIC)
            hr_curr = hr_curr.resize((self.hr_size, self.hr_size), Image.BICUBIC)
            hr_next = hr_next.resize((self.hr_size, self.hr_size), Image.BICUBIC)

        # Generate LR by bicubic downsampling
        lr_prev = hr_prev.resize((self.lr_size, self.lr_size), Image.BICUBIC)
        lr_curr = hr_curr.resize((self.lr_size, self.lr_size), Image.BICUBIC)
        lr_next = hr_next.resize((self.lr_size, self.lr_size), Image.BICUBIC)

        # Random horizontal flip (consistent across triplet)
        if self.augment and random.random() > 0.5:
            hr_prev = hr_prev.transpose(Image.FLIP_LEFT_RIGHT)
            hr_curr = hr_curr.transpose(Image.FLIP_LEFT_RIGHT)
            hr_next = hr_next.transpose(Image.FLIP_LEFT_RIGHT)
            lr_prev = lr_prev.transpose(Image.FLIP_LEFT_RIGHT)
            lr_curr = lr_curr.transpose(Image.FLIP_LEFT_RIGHT)
            lr_next = lr_next.transpose(Image.FLIP_LEFT_RIGHT)

        # To tensor [0, 1]
        hr_curr_t = self.to_tensor(hr_curr)
        lr_prev_t = self.to_tensor(lr_prev)
        lr_curr_t = self.to_tensor(lr_curr)
        lr_next_t = self.to_tensor(lr_next)

        return lr_prev_t, lr_curr_t, lr_next_t, hr_curr_t


class VideoAwareSampler(Sampler):
    """Yields indices in an order that ensures each batch has diverse videos.

    Uses round-robin across videos so that consecutive batch_size indices
    come from different videos.
    """

    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle

        # Group sample indices by video
        self.video_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            vid = dataset.get_video_id(idx)
            self.video_to_indices[vid].append(idx)

        self.video_keys = list(self.video_to_indices.keys())

    def __iter__(self):
        # Build per-video queues
        queues = {}
        for vid, indices in self.video_to_indices.items():
            q = indices.copy()
            if self.shuffle:
                random.shuffle(q)
            queues[vid] = q

        # Shuffle video order
        keys = self.video_keys.copy()
        if self.shuffle:
            random.shuffle(keys)

        # Round-robin: cycle through videos, take one frame each
        result = []
        active = [k for k in keys if queues[k]]
        while active:
            if self.shuffle:
                random.shuffle(active)
            next_active = []
            for vid in active:
                result.append(queues[vid].pop())
                if queues[vid]:
                    next_active.append(vid)
            active = next_active

        return iter(result)

    def __len__(self):
        return len(self.dataset)


def load_frame_records(csv_path):
    """Load frame metadata from CSV."""
    records = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


class HROnlyDataset(Dataset):
    """Lightweight dataset that loads only HR crops (no LR, no triplets).

    For tasks that only need HR images (e.g., image clustering).
    ~3x faster than VideoSRDataset since it loads 1 image instead of 3
    and skips all LR downsampling.
    """

    def __init__(self, frame_records, hr_crop_size=256, augment=True):
        self.hr_size = hr_crop_size
        self.augment = augment

        self.video_frames = defaultdict(list)
        for rec in frame_records:
            key = (rec["source"], rec["video_name"])
            self.video_frames[key].append(rec["frame_path"])
        for key in self.video_frames:
            self.video_frames[key].sort()

        self.samples = []
        self.video_ids = []
        for key, frames in self.video_frames.items():
            for pos in range(len(frames)):
                self.samples.append(frames[pos])
                self.video_ids.append(key)

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def get_video_id(self, idx):
        return self.video_ids[idx]

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        w, h = img.size

        crop_h = min(self.hr_size, h)
        crop_w = min(self.hr_size, w)
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        hr = img.crop((left, top, left + crop_w, top + crop_h))

        if crop_h != self.hr_size or crop_w != self.hr_size:
            hr = hr.resize((self.hr_size, self.hr_size), Image.BICUBIC)

        if self.augment and random.random() > 0.5:
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)

        return self.to_tensor(hr)


class PreCachedHRDataset(Dataset):
    """All HR crops pre-loaded into a single uint8 tensor. Zero I/O training.

    First run: loads all images → random crop → saves cache (.npy).
    Subsequent runs: loads cache from disk (~10s for 34GB via NVMe).
    Training: tensor indexing + float conversion + augmentation. No disk I/O.
    num_workers=0 recommended (no I/O to parallelize).

    When two_views=True, returns (view1, view2, video_int_id) for
    augmentation consistency and video contrastive training.
    """

    def __init__(self, frame_records, hr_crop_size=256, augment=True,
                 cache_path=None, two_views=False):
        self.hr_size = hr_crop_size
        self.augment = augment
        self.two_views = two_views

        # Build path list and video IDs
        self.video_frames = defaultdict(list)
        for rec in frame_records:
            key = (rec["source"], rec["video_name"])
            self.video_frames[key].append(rec["frame_path"])
        for key in self.video_frames:
            self.video_frames[key].sort()

        self.paths = []
        self.video_keys = []
        for key, frames in self.video_frames.items():
            for path in frames:
                self.paths.append(path)
                self.video_keys.append(key)

        # Integer video IDs for contrastive loss
        unique_videos = sorted(set(self.video_keys))
        self.vid_to_int = {v: i for i, v in enumerate(unique_videos)}
        self.video_int_ids = [self.vid_to_int[k] for k in self.video_keys]

        # Load or build cache
        if cache_path and os.path.exists(cache_path):
            print(f"  Loading cache from {cache_path}...")
            self.data = np.load(cache_path, mmap_mode='r')
            assert len(self.data) == len(self.paths), \
                f"Cache size mismatch: {len(self.data)} vs {len(self.paths)} paths"
        else:
            self.data = self._build_cache()
            if cache_path:
                os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
                np.save(cache_path, self.data)
                print(f"  Cache saved: {cache_path} "
                      f"({os.path.getsize(cache_path) / 1e9:.1f} GB)")

    def _build_cache(self):
        from multiprocessing import Pool
        from functools import partial
        import time

        n = len(self.paths)
        s = self.hr_size
        print(f"  Building HR cache: {n} frames × {s}×{s}...")
        t0 = time.time()

        # Pre-allocate output array to avoid double memory from np.stack
        data = np.empty((n, 3, s, s), dtype=np.uint8)

        fn = partial(_crop_one_frame, hr_size=s)
        with Pool(min(16, os.cpu_count())) as pool:
            for i, crop in enumerate(pool.imap(fn, self.paths, chunksize=256)):
                data[i] = crop
                if (i + 1) % 10000 == 0:
                    print(f"    {i+1}/{n} ({(i+1)/n:.0%})")

        elapsed = time.time() - t0
        print(f"  Cache built in {elapsed:.0f}s "
              f"({data.nbytes / 1e9:.1f} GB, {n / elapsed:.0f} frames/s)")
        return data

    def __len__(self):
        return len(self.paths)

    def get_video_id(self, idx):
        return self.video_keys[idx]

    def _to_tensor(self, idx):
        return torch.from_numpy(self.data[idx].copy()).float().div_(255.0)

    @staticmethod
    def _augment_view(hr):
        """Random flip + mild color jitter for one view."""
        if random.random() > 0.5:
            hr = hr.flip(-1)  # horizontal flip
        # Mild brightness/contrast jitter
        if random.random() > 0.5:
            factor = 1.0 + random.uniform(-0.1, 0.1)
            hr = (hr * factor).clamp_(0, 1)
        return hr

    def __getitem__(self, idx):
        hr = self._to_tensor(idx)
        if self.two_views:
            v1 = self._augment_view(hr.clone()) if self.augment else hr
            v2 = self._augment_view(hr.clone()) if self.augment else hr
            return v1, v2, self.video_int_ids[idx]
        else:
            if self.augment:
                hr = self._augment_view(hr)
            return hr

