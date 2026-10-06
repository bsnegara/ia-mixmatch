"""Post-hoc analysis (Fase 4): statistics, tables and figures from saved run outputs."""
import glob, json, os, re
import numpy as np
from scipy.stats import rankdata, norm, binomtest
from . import FINDINGS

NICE = {f: f.replace("_", " ").capitalize() for f in FINDINGS}


# ---------------- AUROC, DeLong, bootstrap ----------------
def auroc(y, p):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def mean_auroc(Y, P, cols=None):
    cols = range(Y.shape[1]) if cols is None else cols
    return np.nanmean([auroc(Y[:, c], P[:, c]) for c in cols])


def delong_test(y, p1, p2):
    """Two-sided DeLong test for two correlated AUROCs (Sun & Xu, 2014). Returns (auc1, auc2, p)."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="stable")
    m = int(y.sum()); n = len(y) - m
    preds = np.vstack([p1, p2])[:, order].astype(np.float64)
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.array([rankdata(r) for r in pos])
    ty = np.array([rankdata(r) for r in neg])
    tz = np.array([rankdata(r) for r in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    l = np.array([1.0, -1.0])
    var = float(l @ cov @ l)
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return float(aucs[0]), float(aucs[1]), float(2 * norm.sf(abs(z)))


def paired_bootstrap(Y, P_base, P_new, B=200, seed=0, cols=None):
    """Delta mean AUROC (new - base), averaged over seeds; test images resampled jointly.
    P_base, P_new: lists of arrays (one per seed). Returns (delta, lo, hi)."""
    rng = np.random.default_rng(seed)
    N = len(Y)
    point = np.mean([mean_auroc(Y, b, cols) - mean_auroc(Y, a, cols) for a, b in zip(P_base, P_new)])
    boots = []
    for _ in range(B):
        idx = rng.integers(0, N, N)
        Yb = Y[idx]
        boots.append(np.mean([mean_auroc(Yb, b[idx], cols) - mean_auroc(Yb, a[idx], cols)
                              for a, b in zip(P_base, P_new)]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def sign_test(wins, n):
    return float(binomtest(wins, n, 0.5).pvalue)


# ---------------- run loading ----------------
def run_dirs(runs, method, ratio, tag="p2"):
    pat = re.compile(rf"^{re.escape(method)}_r{ratio:02d}_s(\d+)_{tag}$")
    out = {}
    for d in glob.glob(f"{runs}/{method}_r{ratio:02d}_s*_{tag}"):
        m = pat.match(os.path.basename(d))
        if m and os.path.exists(f"{d}/test_metrics.json"):
            out[int(m.group(1))] = d
    return dict(sorted(out.items()))


def load_preds(runs, method, ratio, tag="p2"):
    return {s: np.load(f"{d}/test_pred.npy").astype(np.float32) for s, d in run_dirs(runs, method, ratio, tag).items()}


def load_metrics(runs, method, ratio, tag="p2"):
    return {s: json.load(open(f"{d}/test_metrics.json")) for s, d in run_dirs(runs, method, ratio, tag).items()}


# ---------------- Table 3 ----------------
def table3_rows(runs, prevalence_csv, ratio=20, methods=("mixmatch", "ia_mixmatch")):
    import pandas as pd
    prev = pd.read_csv(prevalence_csv).set_index("finding")
    per = {m: load_metrics(runs, m, ratio) for m in methods}
    lines, stats = [], {}
    for g in ["head", "medium", "tail"]:
        fs = prev[prev.group == g].sort_values("prev_train_%", ascending=False).index.tolist()
        for i, f in enumerate(fs):
            vals = []
            for m in methods:
                a = [per[m][s]["per_class"][f]["auroc"] for s in per[m]]
                b = [per[m][s]["per_class"][f]["auprc"] for s in per[m]]
                vals += [100 * np.mean(a), 100 * np.mean(b)]
                stats[(f, m)] = (100 * np.mean(a), 100 * np.mean(b))
            head = f"\\multirow{{{len(fs)}}}{{*}}{{{g.capitalize()}}}" if i == 0 else ""
            lines.append(f"{head} & {NICE[f]} & {prev.loc[f, 'prev_train_%']:.2f} & " +
                         " & ".join(f"{v:.1f}" for v in vals) + " \\\\")
        lines.append("\\hline")
    mean_vals = []
    for m in methods:
        mean_vals += [100 * np.mean([per[m][s]["mean_auroc"] for s in per[m]]),
                      100 * np.mean([per[m][s]["mean_auprc"] for s in per[m]])]
    lines.append("\\multicolumn{2}{l}{Mean} & -- & " + " & ".join(f"{v:.1f}" for v in mean_vals) + " \\\\")
    return lines, stats


# ---------------- Figure 3 ----------------
def pl_curves(runs, method, ratio, key="pos_mean_target", group="tail", report=None):
    """Mean/std over seeds of a pseudo-label statistic per evaluation step.
    Duplicate iterations (resumed runs) keep the last entry. Seeds whose validation log is
    incomplete (fewer evaluation steps than the longest seed) are excluded; their ids are
    appended to `report` if a list is given."""
    per_seed = {}
    for s, d in run_dirs(runs, method, ratio).items():
        vals = {}
        for l in open(f"{d}/val_log.jsonl"):
            r = json.loads(l)
            if "pseudo_labels" in r and group in r["pseudo_labels"]:
                vals[int(r["it"])] = r["pseudo_labels"][group][key]
        per_seed[s] = vals
    if not per_seed:
        return None
    n = max(len(v) for v in per_seed.values())
    keep = {s: v for s, v in per_seed.items() if len(v) == n and n > 0}
    if report is not None:
        report += [(method, ratio, s, len(v)) for s, v in per_seed.items() if s not in keep]
    its = sorted(set.intersection(*(set(v) for v in keep.values())))
    val = np.array([[v[i] for i in its] for v in keep.values()])
    return np.array(its), val.mean(0), val.std(0)


def check_logs(runs, tag="p2", every=500):
    """Runs whose validation log has fewer entries than iters/eval_every."""
    bad = []
    for d in sorted(glob.glob(f"{runs}/*_{tag}")):
        if not os.path.exists(f"{d}/DONE") or not os.path.exists(f"{d}/config.json"):
            continue
        cfg = json.load(open(f"{d}/config.json"))
        exp = cfg.get("iters", 0) // cfg.get("eval_every", every)
        n = len({json.loads(l)["it"] for l in open(f"{d}/val_log.jsonl")}) if os.path.exists(f"{d}/val_log.jsonl") else 0
        if n < exp:
            bad.append((os.path.basename(d), n, exp))
    return bad


def reliability(Y, Ps, bins=10):
    """Pooled over findings and seeds: (bin confidence, bin frequency, bin count)."""
    y = np.concatenate([Y.ravel()] * len(Ps)); p = np.concatenate([P.ravel() for P in Ps])
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    conf, freq, cnt = [], [], []
    for b in range(bins):
        m = idx == b
        if m.any():
            conf.append(p[m].mean()); freq.append(y[m].mean()); cnt.append(int(m.sum()))
    return np.array(conf), np.array(freq), np.array(cnt)


def figure3(runs, out_png, Y, ratios=(5, 20)):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    styles = {"mixmatch": ("MixMatch", "--"), "ia_mixmatch": ("IA-MixMatch", "-")}
    colors = {ratios[0]: "tab:blue", ratios[1]: "tab:red"}
    excluded = []
    for r in ratios:
        for m, (lab, ls) in styles.items():
            c = pl_curves(runs, m, r, report=excluded)
            if c:
                it, mu, sd = c
                ax[0].plot(it, mu, ls, marker=".", color=colors[r], label=f"{lab}, {r}% labels")
                ax[0].fill_between(it, mu - sd, mu + sd, color=colors[r], alpha=0.12)
    ax[0].set(xlabel="training iteration", ylabel="mean target of true positives",
              title="(a) Pseudo-label targets, tail findings")
    ax[0].legend(fontsize=8)
    ax[1].plot([0, 1], [0, 1], "k:", lw=1)
    for m, (lab, ls) in styles.items():
        Ps = list(load_preds(runs, m, ratios[1]).values())
        if Ps:
            conf, freq, _ = reliability(Y, Ps)
            ax[1].plot(conf, freq, ls + "o", ms=4, label=lab)
    ax[1].set(xlabel="predicted probability", ylabel="observed frequency", xlim=(0, 1), ylim=(0, 1),
              title=f"(b) Reliability, {ratios[1]}% labels (test set)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    for m, r, sd, n in excluded:
        print(f"[note] {m} {r}%: seed {sd} excluded from the curve (validation log has only {n} points)")
    return out_png


# ---------------- Figure 4 (Fase 5) ----------------
def _collect(runs, method, ratio, tag, seeds=None, keys=("mean_auroc", "tail_auprc", "tail_auroc")):
    met = load_metrics(runs, method, ratio, tag)
    if seeds is not None:
        met = {s: v for s, v in met.items() if s in seeds}
    if not met:
        return None
    arr = {k: np.array([100 * v[k] for v in met.values()]) for k in keys}
    return {k: (a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0) for k, a in arr.items()} | {"n": len(met)}


def fase5_results(runs, T_values=(0.3, 0.5, 0.7), imb_values=(1, 2, 4, 8), t_ratio=5, imb_ratio=10, imb_seeds=(0, 1)):
    res = {"T": {}, "imb": {}}
    for m in ["mixmatch", "ia_mixmatch"]:
        for T in T_values:
            name, tag = (m, "p2") if T == 0.5 else (f"{m}_T{T:g}", "p5")
            res["T"][(m, T)] = _collect(runs, name, t_ratio, tag)
        for k in imb_values:
            name, tag = (m, "p2") if k == 1 else (f"{m}_imb{k:g}", "p5")
            res["imb"][(m, k)] = _collect(runs, name, imb_ratio, tag, seeds=set(imb_seeds))
    return res


def figure4(runs, out_png, res=None, t_ratio=5, imb_ratio=10):
    import matplotlib.pyplot as plt
    res = res or fase5_results(runs, t_ratio=t_ratio, imb_ratio=imb_ratio)
    styles = {"mixmatch": ("MixMatch", "--o", "tab:blue"), "ia_mixmatch": ("IA-MixMatch", "-o", "tab:orange")}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for m, (lab, ls, col) in styles.items():
        pts = sorted((T, v) for (mm, T), v in res["T"].items() if mm == m and v)
        if pts:
            x = [p[0] for p in pts]; y = [p[1]["mean_auroc"][0] for p in pts]; e = [p[1]["mean_auroc"][1] for p in pts]
            ax[0].errorbar(x, y, yerr=e, fmt=ls, color=col, capsize=3, label=lab)
        pts = sorted((k, v) for (mm, k), v in res["imb"].items() if mm == m and v)
        if pts:
            x = [p[0] for p in pts]; y = [p[1]["tail_auprc"][0] for p in pts]; e = [p[1]["tail_auprc"][1] for p in pts]
            ax[1].errorbar(x, y, yerr=e, fmt=ls, color=col, capsize=3, label=lab)
    ax[0].set(xlabel="temperature $T$", ylabel="mean AUROC (%)", title=f"(a) Sensitivity to $T$, {t_ratio}% labels")
    ax[1].set_xscale("log", base=2); ax[1].set_xticks([1, 2, 4, 8]); ax[1].set_xticklabels(["1", "2", "4", "8"])
    ax[1].set(xlabel="reduction factor of labeled tail positives", ylabel="tail AUPRC (%)",
              title=f"(b) Stronger imbalance, {imb_ratio}% labels")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png
