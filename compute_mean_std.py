import os
import glob
import pickle

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# --- CONFIGURATION ---------------------------------------------------------

base_dir        = "./ood_classifier_data"
success_pattern = os.path.join(base_dir, "*success*.pkl")
failure_pattern = os.path.join(base_dir, "*failure*.pkl")

out_root = "./ood_pngs"
success_out = os.path.join(out_root, 'success')
failure_out = os.path.join(out_root, 'failure')
for d in (success_out, failure_out):
    os.makedirs(d, exist_ok=True)

# --- HELPERS ---------------------------------------------------------------

def save_array_as_png(arr: np.ndarray, out_path: str):
    # Handle channels-first vs channels-last
    if arr.ndim == 3 and arr.shape[0] in (1,3):
        arr = np.transpose(arr, (1,2,0))
    # Float in [0,1] -> [0,255]
    if arr.dtype in (np.float32, np.float64):
        arr = np.clip(arr, 0, 1) * 255
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(out_path)

# --- STEP 1 & 2: CONVERT AND SAVE PNGs, WHILE QUEUING PATHS ---------------

all_paths = []  # list of (png_path, label) for later mean/std

def process_pattern(pattern, label, out_dir):
    for pkl_path in glob.glob(pattern):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        images = []
        for idx, trans in enumerate(data):
            img = trans['observations']["scene"]
            images.append(img)
        base_name = os.path.splitext(os.path.basename(pkl_path))[0]
        for idx, arr in enumerate(images):
            out_name = f"{base_name}_{idx:03d}.png"
            out_path = os.path.join(out_dir, out_name)
            try:
                save_array_as_png(arr, out_path)
                all_paths.append((out_path, label))
            except Exception as e:
                print(f"❌ failed to save {pkl_path}[{idx}]: {e}")

process_pattern(success_pattern, 0, success_out)
process_pattern(failure_pattern, 1, failure_out)

print(f"Converted {len(all_paths)} images to {out_root}/")

# --- STEP 3: COMPUTE MEAN & STD OVER ALL SAVED PNGs -----------------------

class PngDataset(Dataset):
    def __init__(self, path_label_list):
        self.items = path_label_list
        self.to_tensor = transforms.ToTensor()
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        path, lbl = self.items[idx]
        try:
            img = Image.open(path).convert('RGB')
        except UnidentifiedImageError:
            # skip bad ones by returning zeros
            img = Image.new('RGB', (224,224), 0)
        return self.to_tensor(img), lbl

# build DataLoader
ds   = PngDataset(all_paths)
ldr  = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8)

# accumulate
cnt    = 0
sum_   = torch.zeros(3)
sum_sq = torch.zeros(3)

for images, _ in ldr:
    # images.shape = [B, 3, H, W]
    b, c, h, w = images.shape
    cnt    += b * h * w
    sum_   += images.sum(dim=[0,2,3])
    sum_sq += (images**2).sum(dim=[0,2,3])

mean = sum_ / cnt
std  = (sum_sq / cnt - mean**2).sqrt()

print("Channel‑wise mean:", mean.tolist())
print("Channel‑wise std :", std.tolist())
# Channel‑wise mean: [0.5314865708351135, 0.5344920754432678, 0.4852450489997864]
# Channel‑wise std : [0.14621567726135254, 0.15273576974868774, 0.15099382400512695]