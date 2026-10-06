"""Batch augmentation on GPU. Inputs: float tensor B x 1 x H x W in [0, 1]."""
import math
import torch
import torch.nn.functional as F


def _affine(x, out, scale, rot_deg):
    B, dev = x.shape[0], x.device
    area = torch.empty(B, device=dev).uniform_(*scale)
    side = area.sqrt()                                   # crop side as fraction of image
    ang = torch.empty(B, device=dev).uniform_(-rot_deg, rot_deg) * math.pi / 180
    tx = (torch.rand(B, device=dev) * 2 - 1) * (1 - side)
    ty = (torch.rand(B, device=dev) * 2 - 1) * (1 - side)
    c, s = ang.cos() * side, ang.sin() * side
    theta = torch.stack([torch.stack([c, -s, tx], 1), torch.stack([s, c, ty], 1)], 1)
    grid = F.affine_grid(theta, (B, 1, out, out), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _photometric(x, bright, contrast, gamma=None, noise=0.0):
    B, dev = x.shape[0], x.device
    shape = (B, 1, 1, 1)
    if gamma is not None:
        g = torch.empty(shape, device=dev).uniform_(*gamma)
        x = x.clamp(1e-6, 1).pow(g)
    m = x.mean(dim=(2, 3), keepdim=True)
    c = torch.empty(shape, device=dev).uniform_(*contrast)
    b = torch.empty(shape, device=dev).uniform_(-bright, bright)
    x = (x - m) * c + m + b
    if noise > 0:
        x = x + noise * torch.randn_like(x)
    return x.clamp(0, 1)


def _cutout(x, frac=(0.2, 0.4)):
    B, _, H, W = x.shape
    dev = x.device
    f = torch.empty(B, device=dev).uniform_(*frac)
    h, w = (f * H).long(), (f * W).long()
    cy = torch.randint(0, H, (B,), device=dev)
    cx = torch.randint(0, W, (B,), device=dev)
    yy = torch.arange(H, device=dev).view(1, H, 1)
    xx = torch.arange(W, device=dev).view(1, 1, W)
    mask = ((yy - cy.view(B, 1, 1)).abs() <= h.view(B, 1, 1) // 2) & \
           ((xx - cx.view(B, 1, 1)).abs() <= w.view(B, 1, 1) // 2)
    return x.masked_fill(mask[:, None], 0.0)


def weak_aug(x, out=224):
    """Random resized crop (scale 0.8-1.0), rotation +-10 deg, brightness/contrast jitter.
    No horizontal flip (would move the heart to the wrong side)."""
    x = _affine(x, out, scale=(0.8, 1.0), rot_deg=10)
    return _photometric(x, bright=0.1, contrast=(0.9, 1.1))


def strong_aug(x, out=224):
    """Stronger version for FixMatch/FreeMatch baselines."""
    x = _affine(x, out, scale=(0.5, 1.0), rot_deg=20)
    x = _photometric(x, bright=0.2, contrast=(0.7, 1.3), gamma=(0.7, 1.5), noise=0.03)
    return _cutout(x)


def eval_resize(x, out=224):
    return F.interpolate(x, size=(out, out), mode="bilinear", align_corners=False, antialias=True)
