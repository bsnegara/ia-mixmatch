"""Unit tests (E2). Run: python -m iamm.tests  (CPU is enough)."""
import torch
from iamm.ssl import (PCS, ACW, sharpen_bernoulli, labelwise_temperature, mixup, logit,
                      asymmetric_loss, mean_preserving_sharpen, class_balanced_bce)
from iamm.augment import weak_aug, strong_aug, eval_resize
from iamm.metrics import summarize
import numpy as np

ok = lambda m: print(f"  [OK] {m}")


def test_sharpen():
    p = torch.tensor([0.1, 0.4, 0.5, 0.6, 0.9])
    assert torch.allclose(sharpen_bernoulli(p, 1.0), p, atol=1e-5)
    s = sharpen_bernoulli(p, 0.5)
    assert (s[:2] < p[:2]).all() and (s[3:] > p[3:]).all() and abs(s[2] - 0.5) < 1e-5
    ok("sharpening Bernoulli: T=1 identitas, T<1 menjauhkan dari 0.5")


def test_labelwise_temperature_suppresses():
    p = torch.tensor([0.4])
    std = sharpen_bernoulli(p, 0.5).item()
    lwt = sharpen_bernoulli(p, torch.tensor([0.5 * 0.1])).item()
    assert abs(std - 0.3077) < 1e-3 and abs(lwt - 3.0e-4) < 0.5e-4, (std, lwt)
    prior = torch.tensor([0.16, 0.016])
    Tc = labelwise_temperature(prior, 0.5)
    assert Tc[1] < Tc[0]
    ok(f"analisis Sec 2.2: standar {std:.3f} vs label-wise {lwt:.1e}")


def test_mps():
    torch.manual_seed(0)
    p = torch.cat([torch.full((31, 2), 0.002), torch.tensor([[0.3, 0.002]])])   # 1 likely positive, class 2 uniform
    q, d = mean_preserving_sharpen(p, 0.5)
    assert torch.allclose(q.mean(0), p.mean(0), rtol=1e-3), "batch mean preserved"
    assert torch.allclose(q[:, 1], p[:, 1], rtol=1e-3), "uniform guesses are left unchanged"
    std = sharpen_bernoulli(p, 0.5)
    assert q[-1, 0] > std[-1, 0] and q[:31, 0].mean() < p[:31, 0].mean(), "positive raised relative to negatives"
    r = torch.rand(64, 3)
    qr, _ = mean_preserving_sharpen(r, 0.5)
    assert all((qr[:, c].argsort() == r[:, c].argsort()).all() for c in range(3)), "ranking preserved"
    ok(f"MPS: rata-rata batch terjaga, target positif {q[-1,0]:.3f} (standar {std[-1,0]:.3f}), ranking terjaga")


def test_pcs():
    prior = torch.tensor([0.10, 0.01])
    q = torch.tensor([[0.02, 0.01], [0.08, 0.05], [0.05, 0.03]])
    pcs = PCS(prior, mode="mps")
    qt, qq = pcs.targets(q, 0.5)
    assert torch.allclose(qt, q) and torch.allclose(qq.mean(0), q.mean(0), rtol=1e-3), "mps, eta=0"
    pcs = PCS(prior, mode="integral", kappa=0.05, m=0.9)
    for _ in range(2000):
        qt, qq = pcs.targets(q, 0.5); pcs.update(q, qq)
    assert (pcs.shift() > 0).all() and torch.allclose(qq.mean(0), prior, rtol=0.05), "integral mode"
    pcs = PCS(prior, mode="naive", eta=1.0, m=0.0); pcs.update(q, q)
    qt, qq = pcs.targets(q, 0.5)
    assert torch.allclose(qt.mean(0), prior, rtol=0.3), "distribution alignment"
    ok("PCS: mode mps / naive / integral berjalan sesuai definisi")


def test_acw():
    C = 3
    acw = ACW(C, "cpu", m=0.9)
    q = torch.rand(64, C)
    w0 = acw.weights(q)
    assert torch.allclose(w0, torch.ones_like(w0)), "initial weights = 1"
    for _ in range(200):
        acw.update(torch.cat([torch.full((60, C), 0.02), torch.full((4, C), 0.97)]))
    q2 = torch.tensor([[0.01, 0.3, 0.97], [0.45, 0.6, 0.99]])
    w = acw.weights(q2)
    assert ((w >= 0) & (w <= 1)).all() and w.shape == q2.shape
    assert w[0, 0] == 1 and w[1, 0] < 1, "confident target keeps weight 1, uncertain one is down-weighted"
    hard = ACW(C, "cpu", m=0.9, hard=True); hard.load_state_dict({**acw.state_dict(), "hard": True})
    assert set(hard.weights(q2).unique().tolist()) <= {0.0, 1.0}
    assert acw.thresholds().shape == (2, C)
    ok("ACW: bobot di [0,1], awal = 1, target ragu diturunkan, varian hard biner")


def test_mixup():
    X, Y, W = torch.rand(8, 1, 4, 4), torch.rand(8, 3), torch.rand(8, 3)
    Xm, Ym, Wm, lam = mixup(X, Y, W, 0.75)
    assert lam >= 0.5 and Xm.shape == X.shape and Ym.shape == Y.shape and Wm.shape == W.shape
    ok("MixUp: lambda' >= 0.5, bentuk tensor benar")


def test_aug():
    x = torch.rand(4, 1, 256, 256)
    for f in (weak_aug, strong_aug, eval_resize):
        y = f(x, 224)
        assert y.shape == (4, 1, 224, 224) and y.min() >= 0 and y.max() <= 1
    ok("augmentasi GPU: keluaran 224x224 dalam [0,1]")


def test_losses_metrics():
    lg, y = torch.randn(8, 14), (torch.rand(8, 14) > 0.8).float()
    assert asymmetric_loss(lg, y).item() > 0
    assert class_balanced_bce(lg, y, torch.tensor([100.] * 13 + [2.])).item() > 0
    yy = np.random.rand(500, 3) > 0.7
    r = summarize(yy.astype(float), yy.astype(float) * 0.8 + 0.1, {"head": [0], "tail": [1, 2]})
    assert abs(r["mean_auroc"] - 1) < 1e-9
    ok("loss ASL & metrik AUROC/AUPRC/ECE")


def test_model():
    from iamm.model import CXRNet
    m = CXRNet(14, pretrained=False).eval()
    with torch.no_grad():
        o = m(torch.rand(2, 1, 224, 224))
    assert o.shape == (2, 14)
    ok("DenseNet-121: keluaran (B, 14)")


if __name__ == "__main__":
    for t in [test_sharpen, test_labelwise_temperature_suppresses, test_mps, test_pcs, test_acw, test_mixup,
              test_aug, test_losses_metrics, test_model]:
        t()
    print("SEMUA TES LULUS")
