"""Post-hoc analysis: reproduces the tables, statistics and figures of the paper from the run outputs.

  python scripts/analyze.py summary        # Table 1 and Table 3: mean +- std of test metrics over seeds
  python scripts/analyze.py significance   # paired bootstrap, DeLong (Holm / Benjamini-Hochberg), sign test
  python scripts/analyze.py chexpert       # Table 2: external validation (needs data/chexpert, a GPU helps)
  python scripts/analyze.py figures        # Figure 3 (pseudo-labels, calibration) and Figure 4 (sensitivity)
  python scripts/analyze.py gradcam        # Figure 5: Grad-CAM for three tail findings
  python scripts/analyze.py all

Results are written to <out>/stats and <out>/figures (default: results/).
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from iamm import FINDINGS, analysis as A  # noqa: E402

RATIOS = [2, 5, 10, 15, 20]
SHARED = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]


def holm(p):
    p = np.asarray(p, float); m = len(p); adj = np.empty(m); run = 0.0
    for i, idx in enumerate(np.argsort(p)):
        run = max(run, (m - i) * p[idx]); adj[idx] = min(1.0, run)
    return adj


def bh(p):
    p = np.asarray(p, float); m = len(p); adj = np.empty(m); run = 1.0
    for i, idx in enumerate(np.argsort(p)[::-1]):
        run = min(run, p[idx] * m / (m - i)); adj[idx] = run
    return adj


def test_labels(split_dir):
    return pd.read_csv(f"{split_dir}/test.csv")[FINDINGS].values.astype(int)


# ------------------------------------------------------------------ summary
def summary(a):
    rows = []
    variants = [(m, "p2", "") for m in ["supervised", "supervised_cb", "supervised_asl", "mixmatch", "mixmatch_cb",
                                         "freematch", "ia_mixmatch", "ia_mixmatch_cb"]]
    variants += [(m, "p3", "ablation") for m in ["mixmatch_lwt", "mixmatch_pcsnaive", "mixmatch_pcs_integral",
                                                  "mixmatch_pcs", "mixmatch_acw", "mixmatch_acwhard", "ia_mixmatch_eta1"]]
    for name, tag, note in variants:
        for r in RATIOS:
            met = A.load_metrics(a.runs, name, r, tag)
            if not met:
                continue
            row = {"method": name, "ratio": r, "n_seeds": len(met), "note": note}
            for k in ["mean_auroc", "mean_auprc", "tail_auroc", "tail_auprc", "mean_ece"]:
                v = np.array([100 * m[k] for m in met.values() if k in m])
                if len(v):
                    row[k] = f"{v.mean():.2f} +- {v.std(ddof=1) if len(v) > 1 else 0:.2f}"
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.stats}/summary.csv", index=False)
    print(df.to_string(index=False))


# ------------------------------------------------------------------ significance
def significance(a):
    Y = test_labels(a.split_dir)
    res = {"bootstrap": {}, "delong": {}}
    wins = n = 0
    for r in RATIOS:
        mm, ia = A.load_metrics(a.runs, "mixmatch", r), A.load_metrics(a.runs, "ia_mixmatch", r)
        for s in set(mm) & set(ia):
            n += 1; wins += ia[s]["mean_auroc"] > mm[s]["mean_auroc"]
    res["sign_test"] = {"wins": int(wins), "n": n, "p": A.sign_test(int(wins), n) if n else None}
    print(f"IA-MixMatch better than MixMatch in {wins}/{n} paired runs (supporting evidence only)")

    print("\nPaired bootstrap of the difference in mean AUROC (points), 200 resamples, averaged over seeds")
    for r in RATIOS:
        P = {m: list(A.load_preds(a.runs, m, r).values()) for m in ["supervised", "mixmatch", "ia_mixmatch"]}
        for base in ["mixmatch", "supervised"]:
            if P[base] and P["ia_mixmatch"]:
                d, lo, hi = A.paired_bootstrap(Y, P[base], P["ia_mixmatch"], B=200, seed=r)
                res["bootstrap"][f"r{r}_vs_{base}"] = [100 * d, 100 * lo, 100 * hi]
                print(f"  {r:>2}%  IA-MixMatch - {base:<10} {100*d:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]")

    print("\nDeLong per finding and seed (IA-MixMatch vs MixMatch), 42 tests per ratio")
    for r in [5, 20]:
        Pa, Pb = A.load_preds(a.runs, "mixmatch", r), A.load_preds(a.runs, "ia_mixmatch", r)
        rows = []
        for s in sorted(set(Pa) & set(Pb)):
            for c, f in enumerate(FINDINGS):
                a1, a2, p = A.delong_test(Y[:, c], Pa[s][:, c], Pb[s][:, c])
                rows.append({"seed": s, "finding": f, "auc_mixmatch": a1, "auc_ia": a2, "p": p})
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["p_holm"], df["p_bh"] = holm(df.p), bh(df.p)
        df.to_csv(f"{a.stats}/delong_r{r}.csv", index=False)
        better = df.auc_ia > df.auc_mixmatch
        out = {}
        for col, label in [("p", "uncorrected"), ("p_holm", "Holm"), ("p_bh", "Benjamini-Hochberg")]:
            sig = df[col] < 0.05
            out[label] = {"better": int((sig & better).sum()), "worse": int((sig & ~better).sum()), "total": len(df)}
            print(f"  {r:>2}% {label:<19}: {out[label]['better']} better, {out[label]['worse']} worse")
        res["delong"][f"r{r}"] = out
    json.dump(res, open(f"{a.stats}/significance.json", "w"), indent=1)


# ------------------------------------------------------------------ models (torch only when needed)
def _torch():
    import torch
    return torch, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(run_dir):
    torch, dev = _torch()
    from iamm.model import CXRNet
    m = CXRNet(len(FINDINGS), pretrained=False).to(dev)
    m.load_state_dict(torch.load(f"{run_dir}/best.pt", map_location=dev)); m.eval()
    return m


def predict(run_dir, ds):
    torch, dev = _torch()
    from iamm.data import eval_loader
    from iamm.augment import eval_resize
    m, P = load_model(run_dir), []
    with torch.no_grad():
        for x, _, _ in eval_loader(ds, 64, 2):
            P.append(torch.sigmoid(m(eval_resize(x.to(dev).float() / 255, 224))).float().cpu())
    return torch.cat(P).numpy()


# ------------------------------------------------------------------ CheXpert
def chexpert(a):
    from iamm.data import CXRDataset
    ds = CXRDataset(f"{a.chexpert_dir}/valid.csv", f"{a.chexpert_dir}/images")
    Yc, idx = ds.labels.astype(int), [FINDINGS.index(f) for f in SHARED]
    PC = {}
    for m in ["supervised", "mixmatch", "freematch", "ia_mixmatch"]:
        PC[m] = []
        for s, d in A.run_dirs(a.runs, m, 20).items():
            f = f"{d}/chexpert_pred.npy"
            if not os.path.exists(f):
                print(f"  predicting {os.path.basename(d)} ...")
                np.save(f, predict(d, ds).astype(np.float32))
            PC[m].append(np.load(f))
    rng = np.random.default_rng(0)
    boot = [rng.integers(0, len(Yc), len(Yc)) for _ in range(1000)]
    res = {}
    print("Method (seeds): " + ", ".join(SHARED) + ", mean [95% CI]")
    for m, Ps in PC.items():
        if not Ps:
            continue
        per = np.array([[A.auroc(Yc[:, c], P[:, c]) for c in idx] for P in Ps])
        lo, hi = np.percentile([np.mean([A.mean_auroc(Yc[b], P[b], idx) for P in Ps]) for b in boot], [2.5, 97.5])
        res[m] = {"per_finding": dict(zip(SHARED, (100 * per.mean(0)).tolist())), "mean": float(100 * per.mean()),
                  "ci": [100 * lo, 100 * hi], "n_seeds": len(Ps)}
        print(f"  {m} ({len(Ps)}): " + ", ".join(f"{v:.1f}" for v in 100 * per.mean(0))
              + f", {100*per.mean():.1f} [{100*lo:.1f}, {100*hi:.1f}]")
    res["paired"] = {}
    for base in ["mixmatch", "supervised"]:
        if PC.get(base) and PC.get("ia_mixmatch"):
            d, lo, hi = A.paired_bootstrap(Yc, PC[base], PC["ia_mixmatch"], B=1000, seed=0, cols=idx)
            res["paired"][base] = [100 * d, 100 * lo, 100 * hi]
            print(f"  IA-MixMatch - {base}: {100*d:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]")
    json.dump(res, open(f"{a.stats}/chexpert.json", "w"), indent=1)


# ------------------------------------------------------------------ figures
def figures(a):
    Y = test_labels(a.split_dir)
    print("Figure 3:", A.figure3(a.runs, f"{a.figs}/fig_pl_calibration.png", Y, ratios=(5, 20)))
    print("Figure 4:", A.figure4(a.runs, f"{a.figs}/fig_sensitivity.png"))


def gradcam(a):
    import torch.nn.functional as F
    from PIL import Image
    import matplotlib.pyplot as plt
    from iamm.augment import eval_resize
    torch, dev = _torch()
    Y = test_labels(a.split_dir)
    names = pd.read_csv(f"{a.split_dir}/test.csv")["image"].tolist()
    targets = ["Cardiomegaly", "Emphysema", "Pneumonia"]
    d_mm, d_ia = A.run_dirs(a.runs, "mixmatch", 20)[0], A.run_dirs(a.runs, "ia_mixmatch", 20)[0]
    P_mm, P_ia = np.load(f"{d_mm}/test_pred.npy"), np.load(f"{d_ia}/test_pred.npy")
    pick = {}
    for f in targets:   # positive test image with the highest probability averaged over both models
        c = FINDINGS.index(f); pos = np.where(Y[:, c] == 1)[0]
        pick[f] = int(pos[np.argmax((P_mm[pos, c] + P_ia[pos, c]) / 2)])

    def cam_for(model, x, c):
        xn = (x.expand(-1, 3, -1, -1) - model.mean) / model.std
        feats = model.net.features(xn); feats.retain_grad()
        out = model.net.classifier(torch.flatten(F.adaptive_avg_pool2d(F.relu(feats), 1), 1))
        model.zero_grad(); out[0, c].backward()
        w = feats.grad[0].mean(dim=(1, 2))
        cam = torch.relu((w[:, None, None] * F.relu(feats)[0]).sum(0))
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        return cam.detach().cpu().numpy(), torch.sigmoid(out[0, c]).item()

    models = {"MixMatch": load_model(d_mm), "IA-MixMatch": load_model(d_ia)}
    fig, ax = plt.subplots(len(targets), 3, figsize=(9, 3.1 * len(targets)))
    for i, f in enumerate(targets):
        img = np.asarray(Image.open(f"{a.img_dir}/{names[pick[f]]}").convert("L"), dtype=np.float32) / 255
        x = eval_resize(torch.tensor(img)[None, None].to(dev), 224)
        base = x[0, 0].cpu().numpy()
        ax[i, 0].imshow(base, cmap="gray"); ax[i, 0].set_title(f"{f} (test image)", fontsize=9)
        for j, (name, mdl) in enumerate(models.items(), 1):
            cam, prob = cam_for(mdl, x.clone(), FINDINGS.index(f))
            ax[i, j].imshow(base, cmap="gray"); ax[i, j].imshow(cam, cmap="jet", alpha=0.4)
            ax[i, j].set_title(f"{name}, p = {prob:.2f}", fontsize=9)
        for s in ax[i]:
            s.axis("off")
    fig.tight_layout(); fig.savefig(f"{a.figs}/fig_gradcam.png", dpi=200)
    print("Figure 5:", f"{a.figs}/fig_gradcam.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["summary", "significance", "chexpert", "figures", "gradcam", "all"])
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--split_dir", default="splits")
    ap.add_argument("--img_dir", default="data/cxr14_256")
    ap.add_argument("--chexpert_dir", default="data/chexpert")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    a.stats, a.figs = f"{a.out}/stats", f"{a.out}/figures"
    os.makedirs(a.stats, exist_ok=True); os.makedirs(a.figs, exist_ok=True)
    tasks = ["summary", "significance", "chexpert", "figures", "gradcam"] if a.task == "all" else [a.task]
    for t in tasks:
        print(f"\n===== {t} =====")
        globals()[t](a)


if __name__ == "__main__":
    main()
