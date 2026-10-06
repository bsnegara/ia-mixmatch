import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def per_class(y, p, fn):
    out = np.full(y.shape[1], np.nan)
    for c in range(y.shape[1]):
        if 0 < y[:, c].sum() < len(y):
            out[c] = fn(y[:, c], p[:, c])
    return out


def ece_binary(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def summarize(y, p, groups=None, names=None):
    auroc = per_class(y, p, roc_auc_score)
    auprc = per_class(y, p, average_precision_score)
    ece = np.array([ece_binary(y[:, c], p[:, c]) for c in range(y.shape[1])])
    res = {"mean_auroc": float(np.nanmean(auroc)), "mean_auprc": float(np.nanmean(auprc)),
           "mean_ece": float(ece.mean())}
    if groups:
        for g, idx in groups.items():
            if idx:
                res[f"{g}_auroc"] = float(np.nanmean(auroc[idx]))
                res[f"{g}_auprc"] = float(np.nanmean(auprc[idx]))
    if names:
        res["per_class"] = {n: {"auroc": float(a), "auprc": float(b), "ece": float(e),
                                "prevalence": float(y[:, i].mean())}
                            for i, (n, a, b, e) in enumerate(zip(names, auroc, auprc, ece))}
    return res


def pseudo_label_stats(conf, target, weight, true, groups):
    """Quality of pseudo-labels on unlabeled images (true labels are hidden from training).
    conf: corrected guess q~ ; target: final target q ; weight: w ; true: hidden labels."""
    auroc = per_class(true, conf, roc_auc_score)
    pos = true == 1
    out = {}
    for g, idx in groups.items():
        if not idx:
            continue
        pm = pos[:, idx]
        t, w = target[:, idx], weight[:, idx]
        n = max(pm.sum(), 1)
        out[g] = {"guess_auroc": float(np.nanmean(auroc[idx])),
                  "pos_recall": float(((t >= 0.5) & pm).sum() / n),     # true positives receiving a positive target
                  "pos_mean_target": float((t * pm).sum() / n),
                  "pos_mean_weight": float((w * pm).sum() / n),
                  "neg_mean_target": float((t * ~pm).sum() / max((~pm).sum(), 1))}
    return out
