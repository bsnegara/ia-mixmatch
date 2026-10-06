import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, RandomSampler
from . import FINDINGS, HEAD_T, TAIL_T


class CXRDataset(Dataset):
    """Returns (uint8 image 1xHxW, float labels C, index). Augmentation is done on GPU."""
    def __init__(self, csv_path, img_dir):
        df = pd.read_csv(csv_path)
        self.names = df["image"].tolist()
        self.labels = df[FINDINGS].values.astype(np.float32)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        im = Image.open(os.path.join(self.img_dir, self.names[i])).convert("L")
        x = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy())[None]
        return x, torch.from_numpy(self.labels[i]), i


def _dl_kwargs(workers):
    kw = dict(num_workers=workers, pin_memory=True)
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return kw


def infinite_loader(ds, batch_size, workers, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = RandomSampler(ds, replacement=True, num_samples=10**9, generator=g)
    dl = DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=True, **_dl_kwargs(workers))
    return iter(dl)


def eval_loader(ds, batch_size, workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=False, **_dl_kwargs(workers))


def prevalence(csv_path):
    return pd.read_csv(csv_path)[FINDINGS].values.mean(0)


def groups_from_train(train_csv):
    """Return dict group -> list of class indices, using the training-set prevalence."""
    p = prevalence(train_csv)
    g = {"head": [], "medium": [], "tail": []}
    for i, v in enumerate(p):
        g["head" if v > HEAD_T else ("tail" if v < TAIL_T else "medium")].append(i)
    return g
